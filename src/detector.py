"""Tree and vegetation detection with YOLO segmentation plus HSV fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import DetectorConfig
from utils import bbox_from_mask, clean_binary_mask, split_connected_components


@dataclass
class Detection:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str


class TreeDetector:
    """Run pretrained segmentation when possible, otherwise use vegetation color cues."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = None
        self.model_error: Optional[str] = None
        if not config.force_fallback:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.config.model_name)
        except Exception as exc:  # noqa: BLE001 - fallback is intentional for portability.
            self.model = None
            self.model_error = str(exc)

    def detect(self, image: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        if self.model is not None:
            detections = self._detect_with_yolo(image)

        fallback_detections = self._detect_with_hsv(image)
        if not detections:
            return fallback_detections

        for fallback in fallback_detections:
            if max_mask_iou(fallback.mask, [detection.mask for detection in detections]) < 0.30:
                detections.append(fallback)
        return detections

    def _detect_with_yolo(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(image, conf=self.config.confidence_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        names = result.names or {}
        allowed_names = {name.lower() for name in self.config.tree_class_names}
        detections: List[Detection] = []
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for mask_float, class_id, conf in zip(masks, classes, confs):
            class_name = str(names.get(class_id, class_id)).lower()
            if class_name not in allowed_names:
                continue

            mask = cv2.resize(mask_float, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask = clean_binary_mask(mask, self.config.min_component_area_px)
            for component in split_connected_components(mask, self.config.min_component_area_px):
                detections.append(
                    Detection(
                        mask=component,
                        bbox=bbox_from_mask(component),
                        confidence=float(conf),
                        source="yolo",
                    )
                )

        return detections

    def _detect_with_hsv(self, image: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Conservative green/yellow-green vegetation ranges. This is intentionally
        # simple and explainable for a screening feasibility pipeline.
        lower_green = np.array([25, 35, 30], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Remove likely sky highlights and tiny texture noise.
        mask = clean_binary_mask(mask, self.config.min_component_area_px)
        components = split_connected_components(mask, self.config.min_component_area_px)
        return [
            Detection(
                mask=component,
                bbox=bbox_from_mask(component),
                confidence=self.config.fallback_confidence,
                source="hsv_fallback",
            )
            for component in components
        ]


def max_mask_iou(mask: np.ndarray, existing_masks: List[np.ndarray]) -> float:
    if not existing_masks:
        return 0.0
    mask_bool = mask > 0
    best_iou = 0.0
    for existing in existing_masks:
        existing_bool = existing > 0
        union = np.count_nonzero(mask_bool | existing_bool)
        if union == 0:
            continue
        intersection = np.count_nonzero(mask_bool & existing_bool)
        best_iou = max(best_iou, float(intersection) / float(union))
    return best_iou
