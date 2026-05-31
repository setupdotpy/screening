"""UAV canopy feature extraction from accepted tree masks and road context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from config import RiskConfig
from road_region import RoadContext
from structural_filter import TreeCandidate, dilate_mask
from utils import mask_boundary_pixels, safe_float


@dataclass
class FeatureResult:
    row: dict
    distance_line: Optional[tuple[tuple[int, int], tuple[int, int]]]
    lean_arrow: Optional[tuple[tuple[int, int], tuple[int, int]]] = None


def extract_features(
    image_name: str,
    tree_id: int,
    candidate: TreeCandidate,
    road_context: RoadContext,
    image_shape: tuple[int, int, int],
    all_candidates: Sequence[TreeCandidate] | None = None,
    low_vegetation_mask: np.ndarray | None = None,
    risk_config: RiskConfig | None = None,
    road_distance_map: np.ndarray | None = None,
    road_buffer_mask: np.ndarray | None = None,
    road_edge_pixels: np.ndarray | None = None,
) -> FeatureResult:
    risk_config = risk_config or RiskConfig()
    all_candidates = list(all_candidates or [])
    low_vegetation_mask = (
        np.zeros(image_shape[:2], dtype=np.uint8)
        if low_vegetation_mask is None
        else np.where(low_vegetation_mask > 0, 255, 0).astype(np.uint8)
    )

    mask = candidate.mask
    road_mask = road_context.combined_mask
    height, width = image_shape[:2]
    image_area = float(height * width)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = candidate.bbox
    area = int(candidate.area_px)
    distance_px, distance_line = distance_to_road(
        mask,
        road_mask,
        (candidate.canopy_centroid_x, candidate.canopy_centroid_y),
        distance_map=road_distance_map,
        road_edge_pixels=road_edge_pixels,
    )
    road_overlap_ratio = float(np.count_nonzero((mask > 0) & (road_mask > 0))) / max(area, 1)

    road_buffer = road_buffer_mask if road_buffer_mask is not None else dilate_mask(road_mask, risk_config.road_buffer_radius_px)
    road_buffer_overlap_ratio = float(np.count_nonzero((mask > 0) & (road_buffer > 0))) / max(area, 1)
    normalized_canopy_size = float(area) / max(image_area, 1.0)
    nearby_tree_density = compute_nearby_tree_density(candidate, all_candidates, risk_config.nearby_tree_radius_px)
    low_vegetation_context_ratio = compute_low_vegetation_context(
        candidate,
        low_vegetation_mask,
        risk_config.low_vegetation_buffer_radius_px,
    )
    canopy_asymmetry_score = compute_canopy_asymmetry(mask, candidate.canopy_centroid_x, candidate.canopy_centroid_y)
    segmentation_confidence = float(candidate.segmentation_confidence)
    uncertainty = 1.0 - segmentation_confidence

    row = {
        "image_name": image_name,
        "tree_id": tree_id,
        "bbox_x1": int(bbox_x1),
        "bbox_y1": int(bbox_y1),
        "bbox_x2": int(bbox_x2),
        "bbox_y2": int(bbox_y2),
        "canopy_area_px": area,
        "canopy_width_px": int(candidate.canopy_width_px),
        "canopy_height_px": int(candidate.canopy_height_px),
        "canopy_aspect_ratio": safe_float(candidate.canopy_aspect_ratio, 3),
        "canopy_compactness": safe_float(candidate.canopy_compactness),
        "canopy_circularity": safe_float(candidate.canopy_circularity),
        "canopy_centroid_x": safe_float(candidate.canopy_centroid_x, 2),
        "canopy_centroid_y": safe_float(candidate.canopy_centroid_y, 2),
        "distance_to_road_px": safe_float(distance_px, 2),
        "inverse_distance_to_road": safe_float(inverse_distance_to_road(distance_px, risk_config.distance_scale_px)),
        "road_overlap_ratio": safe_float(road_overlap_ratio),
        "road_buffer_overlap_ratio": safe_float(road_buffer_overlap_ratio),
        "normalized_canopy_size": safe_float(normalized_canopy_size),
        "nearby_tree_density": safe_float(nearby_tree_density),
        "low_vegetation_context_ratio": safe_float(low_vegetation_context_ratio),
        "canopy_asymmetry_score": safe_float(canopy_asymmetry_score),
        "segmentation_confidence": safe_float(segmentation_confidence),
        "uncertainty": safe_float(uncertainty),
        "candidate_source": candidate.candidate_source,
    }
    return FeatureResult(row=row, distance_line=distance_line)


def distance_to_road(
    tree_mask: np.ndarray,
    road_mask: np.ndarray,
    tree_centroid: tuple[float, float],
    distance_map: np.ndarray | None = None,
    road_edge_pixels: np.ndarray | None = None,
) -> tuple[float, Optional[tuple[tuple[int, int], tuple[int, int]]]]:
    if np.count_nonzero(tree_mask) == 0 or np.count_nonzero(road_mask) == 0:
        return 0.0, None

    if np.any((tree_mask > 0) & (road_mask > 0)):
        point = (int(tree_centroid[0]), int(tree_centroid[1]))
        return 0.0, (point, point)

    if distance_map is None or distance_map.size == 0:
        inverse_road = (road_mask == 0).astype(np.uint8)
        distance_map = cv2.distanceTransform(inverse_road, cv2.DIST_L2, 5)
    tree_pixels = np.column_stack(np.where(tree_mask > 0))
    tree_distances = distance_map[tree_mask > 0]
    if tree_distances.size == 0:
        return 0.0, None

    min_index = int(np.argmin(tree_distances))
    min_distance = float(tree_distances[min_index])
    tree_y, tree_x = tree_pixels[min_index]
    if road_edge_pixels is None:
        road_edge_pixels = mask_boundary_pixels(road_mask)
    if road_edge_pixels.size == 0:
        return min_distance, None

    deltas = road_edge_pixels - np.array([tree_x, tree_y])
    nearest_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    road_x, road_y = road_edge_pixels[nearest_index]
    line = ((int(tree_x), int(tree_y)), (int(road_x), int(road_y)))
    return min_distance, line


def inverse_distance_to_road(distance_px: float, scale_px: float = 250.0) -> float:
    return float(np.exp(-max(float(distance_px), 0.0) / max(float(scale_px), 1.0)))


def compute_nearby_tree_density(
    candidate: TreeCandidate,
    all_candidates: Sequence[TreeCandidate],
    radius_px: float,
) -> float:
    radius = max(float(radius_px), 1.0)
    count = 0
    for other in all_candidates:
        if other is candidate:
            continue
        distance = float(np.hypot(
            candidate.canopy_centroid_x - other.canopy_centroid_x,
            candidate.canopy_centroid_y - other.canopy_centroid_y,
        ))
        if distance <= radius:
            count += 1
    return min(float(count) / 5.0, 1.0)


def compute_low_vegetation_context(
    candidate: TreeCandidate,
    low_vegetation_mask: np.ndarray,
    buffer_radius_px: int,
) -> float:
    radius = max(int(buffer_radius_px), 0)
    x1, y1, x2, y2 = candidate.bbox
    height, width = candidate.mask.shape[:2]
    rx1 = max(0, x1 - radius)
    ry1 = max(0, y1 - radius)
    rx2 = min(width, x2 + radius)
    ry2 = min(height, y2 + radius)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0

    mask_roi = candidate.mask[ry1:ry2, rx1:rx2]
    low_roi = low_vegetation_mask[ry1:ry2, rx1:rx2]
    local = dilate_mask(mask_roi, radius)
    local[mask_roi > 0] = 0
    area = int(np.count_nonzero(local))
    if area == 0:
        return 0.0
    low_pixels = int(np.count_nonzero((local > 0) & (low_roi > 0)))
    return float(low_pixels) / float(area)


def compute_canopy_asymmetry(mask: np.ndarray, centroid_x: float, centroid_y: float) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0

    left = int(np.count_nonzero(xs < centroid_x))
    right = int(np.count_nonzero(xs >= centroid_x))
    top = int(np.count_nonzero(ys < centroid_y))
    bottom = int(np.count_nonzero(ys >= centroid_y))
    horizontal = abs(left - right) / max(left + right, 1)
    vertical = abs(top - bottom) / max(top + bottom, 1)
    return float(max(horizontal, vertical))
