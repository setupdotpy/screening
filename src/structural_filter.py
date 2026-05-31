"""Canopy component filtering for UAV roadside tree screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from config import StructuralFilterConfig
from road_region import RoadContext
from utils import bbox_from_mask, clamp01, mask_centroid


@dataclass
class TreeCandidate:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area_px: int
    canopy_width_px: int
    canopy_height_px: int
    canopy_aspect_ratio: float
    canopy_compactness: float
    canopy_circularity: float
    canopy_centroid_x: float
    canopy_centroid_y: float
    distance_to_road_px: float
    road_buffer_overlap_ratio: float
    segmentation_confidence: float
    candidate_source: str
    accepted: bool

    @property
    def bbox_width_px(self) -> int:
        return self.canopy_width_px

    @property
    def bbox_height_px(self) -> int:
        return self.canopy_height_px


def apply_structural_filter(
    candidates: Sequence[dict],
    road_context: RoadContext,
    image_shape: tuple[int, int, int],
    config: StructuralFilterConfig,
) -> tuple[list[TreeCandidate], list[TreeCandidate]]:
    accepted: list[TreeCandidate] = []
    rejected: list[TreeCandidate] = []

    road_distance_map = distance_map_to_mask(road_context.combined_mask)
    road_buffer = distance_buffer_from_map(road_distance_map, int(config.max_tree_distance_to_road_px))
    for candidate in candidates:
        canopy = score_tree_candidate(
            candidate,
            road_context,
            image_shape,
            config,
            road_distance_map=road_distance_map,
            road_candidate_buffer=road_buffer,
        )
        too_small = (
            canopy.area_px < config.min_tree_area_px
            or canopy.canopy_width_px < config.min_canopy_width_px
            or canopy.canopy_height_px < config.min_canopy_height_px
        )
        roadside = canopy.distance_to_road_px <= config.max_tree_distance_to_road_px or canopy.road_buffer_overlap_ratio > 0
        if np.count_nonzero(road_context.combined_mask) == 0:
            roadside = True
        if np.count_nonzero(road_buffer) == 0:
            roadside = True

        canopy.accepted = (not too_small) and roadside
        if canopy.accepted:
            accepted.append(canopy)
        else:
            rejected.append(canopy)
    return accepted, rejected


def score_tree_candidate(
    candidate: dict,
    road_context: RoadContext,
    image_shape: tuple[int, int, int],
    config: StructuralFilterConfig | None = None,
    road_distance_map: np.ndarray | None = None,
    road_candidate_buffer: np.ndarray | None = None,
) -> TreeCandidate:
    mask = candidate["mask"].astype(np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(v) for v in candidate.get("bbox", bbox_from_mask(mask))]
    width = max(int(bbox_x2 - bbox_x1), 1)
    height = max(int(bbox_y2 - bbox_y1), 1)
    area = int(np.count_nonzero(mask))
    centroid_x, centroid_y = mask_centroid(mask)
    bbox_area = float(width * height)
    compactness = float(area) / max(bbox_area, 1.0)
    circularity = canopy_circularity(mask, area)
    distance_px = minimum_distance_to_mask(mask, road_context.combined_mask, road_distance_map)

    max_distance = config.max_tree_distance_to_road_px if config else 250
    road_buffer = (
        road_candidate_buffer
        if road_candidate_buffer is not None
        else distance_buffer_from_map(distance_map_to_mask(road_context.combined_mask), int(max_distance))
    )
    road_buffer_overlap = float(np.count_nonzero((mask > 0) & (road_buffer > 0))) / max(area, 1)

    return TreeCandidate(
        mask=mask,
        bbox=(bbox_x1, bbox_y1, bbox_x2, bbox_y2),
        area_px=area,
        canopy_width_px=width,
        canopy_height_px=height,
        canopy_aspect_ratio=float(width) / max(float(height), 1.0),
        canopy_compactness=clamp01(compactness),
        canopy_circularity=clamp01(circularity),
        canopy_centroid_x=float(centroid_x),
        canopy_centroid_y=float(centroid_y),
        distance_to_road_px=float(distance_px),
        road_buffer_overlap_ratio=clamp01(road_buffer_overlap),
        segmentation_confidence=clamp01(candidate.get("segmentation_confidence", candidate.get("vegetation_overlap", 0.5))),
        candidate_source=str(candidate.get("source", "components")),
        accepted=False,
    )


def canopy_circularity(mask: np.ndarray, area: int) -> float:
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * float(area) / (perimeter * perimeter))


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if mask is None or np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(int(radius_px), 0)
    if radius == 0:
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(np.where(mask > 0, 255, 0).astype(np.uint8), kernel, iterations=1)


def distance_map_to_mask(target_mask: np.ndarray) -> np.ndarray:
    if target_mask is None or np.count_nonzero(target_mask) == 0:
        return np.empty((0, 0), dtype=np.float32)
    inverse_target = (target_mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inverse_target, cv2.DIST_L2, 5)


def distance_buffer_from_map(distance_map: np.ndarray, radius_px: int) -> np.ndarray:
    if distance_map.size == 0:
        return np.zeros_like(distance_map, dtype=np.uint8)
    return np.where(distance_map <= max(int(radius_px), 0), 255, 0).astype(np.uint8)


def minimum_distance_to_mask(
    mask: np.ndarray,
    target_mask: np.ndarray,
    distance_map: np.ndarray | None = None,
) -> float:
    if np.count_nonzero(mask) == 0:
        return 0.0
    if target_mask is None or np.count_nonzero(target_mask) == 0:
        return float("inf")
    if np.any((mask > 0) & (target_mask > 0)):
        return 0.0
    if distance_map is None or distance_map.size == 0:
        distance_map = distance_map_to_mask(target_mask)
    distances = distance_map[mask > 0]
    if distances.size == 0:
        return 0.0
    return float(np.min(distances))


def proximity_score(mask: np.ndarray, road_mask: np.ndarray, image_shape: tuple[int, int, int]) -> float:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(road_mask) == 0:
        return 0.0
    distance = minimum_distance_to_mask(mask, road_mask)
    diagonal = float(np.hypot(image_shape[0], image_shape[1]))
    return clamp01(1.0 - distance / max(diagonal * 0.35, 1.0))
