"""Road and sidewalk context extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from config import RoadConfig
from segformer_segmenter import SegmentationResult, mask_for_labels
from utils import mask_centroid, polygon_to_mask


@dataclass
class RoadContext:
    road_mask: np.ndarray
    sidewalk_mask: np.ndarray
    combined_mask: np.ndarray
    source: str
    polygon: list[tuple[int, int]] | None
    centroid_x: float
    centroid_y: float
    area_px: int


def extract_road_context(
    segmentation: SegmentationResult,
    road_config: RoadConfig,
    image_shape: tuple[int, int, int],
) -> RoadContext:
    road_mask = mask_for_labels(segmentation.class_masks, ("road",))
    sidewalk_mask = mask_for_labels(segmentation.class_masks, ("sidewalk",))
    combined_mask = cv2.bitwise_or(road_mask, sidewalk_mask)
    source = "semantic"
    polygon: list[tuple[int, int]] | None = None

    if np.count_nonzero(combined_mask) == 0 and road_config.manual_polygon:
        polygon = [(int(x), int(y)) for x, y in road_config.manual_polygon]
        combined_mask = polygon_to_mask(image_shape[:2], polygon)
        road_mask = combined_mask.copy()
        sidewalk_mask = np.zeros_like(road_mask)
        source = "manual_polygon"

    if np.count_nonzero(combined_mask) == 0:
        polygon = build_heuristic_polygon(image_shape, road_config.heuristic_top_fraction, road_config.heuristic_side_margin_fraction)
        combined_mask = polygon_to_mask(image_shape[:2], polygon)
        road_mask = combined_mask.copy()
        sidewalk_mask = np.zeros_like(road_mask)
        source = "heuristic"

    centroid_x, centroid_y = mask_centroid(combined_mask)
    area_px = int(np.count_nonzero(combined_mask))
    return RoadContext(
        road_mask=road_mask,
        sidewalk_mask=sidewalk_mask,
        combined_mask=combined_mask,
        source=source,
        polygon=polygon,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        area_px=area_px,
    )


def build_heuristic_polygon(
    image_shape: tuple[int, int, int],
    top_fraction: float,
    side_margin_fraction: float,
) -> list[tuple[int, int]]:
    height, width = image_shape[:2]
    top_y = int(height * max(min(top_fraction, 0.95), 0.3))
    margin = int(width * max(min(side_margin_fraction, 0.45), 0.0))
    return [
        (margin, top_y),
        (width - margin - 1, top_y),
        (width - 1, height - 1),
        (0, height - 1),
    ]


def road_centroid_x(road_mask: np.ndarray) -> float:
    centroid_x, _ = mask_centroid(road_mask)
    return centroid_x
