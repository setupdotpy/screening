"""Semantic segmentation with SegFormer and HSV fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from config import SegmentationConfig


CITYSCAPES_LABELS = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


@dataclass
class SegmentationResult:
    class_masks: Dict[str, np.ndarray]
    confidence_map: np.ndarray
    source: str
    elapsed_seconds: float
    warning: str | None = None


class SegFormerSegmenter:
    """Thin wrapper around Hugging Face SegFormer semantic segmentation."""

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.device = "cpu"
        self.processor = None
        self.model = None
        self.model_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001 - fallback is required for portability.
            self.processor = None
            self.model = None
            self.model_error = str(exc)

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        start = time.perf_counter()
        if self.processor is None or self.model is None:
            warning = f"SegFormer unavailable; using HSV fallback. Reason: {self.model_error}"
            return self._fallback(image_bgr, start, warning)

        try:
            return self._segment_with_model(image_bgr, start)
        except Exception as exc:  # noqa: BLE001 - continue with required fallback.
            warning = f"SegFormer inference failed; using HSV fallback. Reason: {exc}"
            return self._fallback(image_bgr, start, warning)

    def _segment_with_model(self, image_bgr: np.ndarray, start: float) -> SegmentationResult:
        import torch
        import torch.nn.functional as functional

        original_h, original_w = image_bgr.shape[:2]
        resized_bgr = resize_for_inference(image_bgr, self.config.inference_max_size)
        resized_h, resized_w = resized_bgr.shape[:2]
        image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = functional.interpolate(
                outputs.logits,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)
            confidence, labels = probabilities.max(dim=1)

        label_map = labels[0].detach().cpu().numpy().astype(np.int32)
        confidence_map = confidence[0].detach().cpu().numpy().astype(np.float32)
        if (resized_h, resized_w) != (original_h, original_w):
            label_map = cv2.resize(label_map, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            confidence_map = cv2.resize(confidence_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        class_masks = labels_to_class_masks(label_map, self._id_to_label(), image_bgr.shape[:2])
        return SegmentationResult(class_masks, confidence_map, "segformer", time.perf_counter() - start)

    def _fallback(self, image_bgr: np.ndarray, start: float, warning: str) -> SegmentationResult:
        print(f"WARNING: {warning}")
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 35, 30], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        vegetation = cv2.inRange(hsv, lower_green, upper_green)
        confidence = np.full(image_bgr.shape[:2], self.config.fallback_confidence, dtype=np.float32)
        class_masks = {
            "vegetation": vegetation,
            "terrain": np.zeros(image_bgr.shape[:2], dtype=np.uint8),
            "road": np.zeros(image_bgr.shape[:2], dtype=np.uint8),
            "sidewalk": np.zeros(image_bgr.shape[:2], dtype=np.uint8),
        }
        return SegmentationResult(class_masks, confidence, "hsv_fallback", time.perf_counter() - start, warning)

    def _id_to_label(self) -> dict[int, str]:
        id_to_label = getattr(self.model.config, "id2label", None) or {}
        normalized: dict[int, str] = {}
        for index, label in id_to_label.items():
            normalized[int(index)] = normalize_label(str(label))
        if not normalized:
            normalized = {index: label for index, label in enumerate(CITYSCAPES_LABELS)}
        return normalized


def labels_to_class_masks(label_map: np.ndarray, id_to_label: dict[int, str], shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    masks = {label: np.zeros(shape, dtype=np.uint8) for label in CITYSCAPES_LABELS}
    for class_id, label in id_to_label.items():
        if label not in masks:
            continue
        masks[label][label_map == class_id] = 255
    return masks


def resize_for_inference(image_bgr: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    largest = max(height, width)
    if largest <= max_size:
        return image_bgr
    scale = max_size / float(largest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def normalize_label(label: str) -> str:
    label = label.lower().replace("_", " ").replace("-", " ").strip()
    aliases = {
        "trafficlight": "traffic light",
        "traffic light": "traffic light",
        "trafficsign": "traffic sign",
        "traffic sign": "traffic sign",
    }
    return aliases.get(label, label)
