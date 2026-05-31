"""Semantic segmentation with SegFormer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from config import SegmentationConfig
from utils import clean_binary_mask, clamp01


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

UAV_CANOPY_LABELS = (
    "tree",
    "low vegetation",
    "background clutter",
    "static car",
    "moving car",
    "human",
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
        except Exception as exc:  # noqa: BLE001 - report model load failure clearly.
            self.processor = None
            self.model = None
            self.model_error = str(exc)

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        if self.processor is None or self.model is None:
            reason = self.model_error or "SegFormer model failed to load."
            raise RuntimeError(f"SegFormer unavailable: {reason}")

        start = time.perf_counter()
        try:
            return self._segment_with_model(image_bgr, start)
        except Exception as exc:  # noqa: BLE001 - this is a hard failure by design.
            raise RuntimeError(f"SegFormer inference failed: {exc}") from exc

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

        with torch.inference_mode():
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

        class_masks = labels_to_class_masks(label_map, self._id_to_label(), (original_h, original_w))
        return SegmentationResult(
            class_masks=class_masks,
            confidence_map=confidence_map,
            source="segformer",
            elapsed_seconds=time.perf_counter() - start,
        )

    def _id_to_label(self) -> dict[int, str]:
        id_to_label = getattr(self.model.config, "id2label", None) or {}
        normalized: dict[int, str] = {}
        for index, label in id_to_label.items():
            normalized[int(index)] = normalize_label(str(label))
        if not normalized:
            normalized = {index: label for index, label in enumerate(CITYSCAPES_LABELS)}
        return normalized


def labels_to_class_masks(label_map: np.ndarray, id_to_label: dict[int, str], shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    masks = {label: np.zeros(shape, dtype=np.uint8) for label in CITYSCAPES_LABELS + UAV_CANOPY_LABELS}
    for class_id, label in id_to_label.items():
        if label not in masks:
            continue
        masks[label][label_map == class_id] = 255
    masks["tree"] = cv2.bitwise_or(masks["tree"], masks["vegetation"])
    masks["low vegetation"] = cv2.bitwise_or(masks["low vegetation"], masks["terrain"])
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
    normalized = label.lower().replace("_", " ").replace("-", " ").strip()
    aliases = {
        "trafficlight": "traffic light",
        "traffic light": "traffic light",
        "trafficsign": "traffic sign",
        "traffic sign": "traffic sign",
        "low vegetation": "low vegetation",
        "lowvegetation": "low vegetation",
        "static car": "static car",
        "staticcar": "static car",
        "moving car": "moving car",
        "movingcar": "moving car",
        "background clutter": "background clutter",
        "backgroundclutter": "background clutter",
    }
    return aliases.get(normalized, normalized)


def mask_for_labels(class_masks: Dict[str, np.ndarray], labels: tuple[str, ...]) -> np.ndarray:
    if not labels:
        shape = next(iter(class_masks.values())).shape
        return np.zeros(shape, dtype=np.uint8)
    combined = np.zeros(next(iter(class_masks.values())).shape, dtype=np.uint8)
    for label in labels:
        if label in class_masks:
            combined = cv2.bitwise_or(combined, class_masks[label])
    return combined


def mean_confidence(confidence_map: np.ndarray, mask: np.ndarray) -> float:
    pixels = confidence_map[mask > 0]
    if pixels.size == 0:
        return 0.0
    return clamp01(float(np.mean(pixels)))
