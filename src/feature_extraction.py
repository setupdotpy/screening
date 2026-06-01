"""UAV canopy feature extraction for inspection-priority screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy import ndimage

from config import RiskConfig
from road_region import RoadContext
from structural_filter import TreeCandidate, dilate_mask
from utils import clamp01, mask_boundary_pixels, safe_float


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
    image_bgr: np.ndarray | None = None,
    segmentation_entropy_map: np.ndarray | None = None,
    tree_probability_map: np.ndarray | None = None,
    uncertainty_source: str = "model_probabilities",
) -> FeatureResult:
    del all_candidates, low_vegetation_mask
    risk_config = risk_config or RiskConfig()

    mask = np.where(candidate.mask > 0, 255, 0).astype(np.uint8)
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

    canopy_diameter_px = max(int(candidate.canopy_width_px), int(candidate.canopy_height_px))
    canopy_asymmetry_score = compute_canopy_asymmetry(mask, candidate.canopy_centroid_x, candidate.canopy_centroid_y)
    canopy_gap_ratio = compute_canopy_gap_ratio(mask)
    canopy_edge_roughness = compute_canopy_edge_roughness(mask, area, risk_config)
    canopy_irregularity = clamp01(
        0.4 * (1.0 - clamp01(candidate.canopy_circularity))
        + 0.3 * canopy_edge_roughness
        + 0.3 * canopy_gap_ratio
    )
    rgb_green_ratio, rgb_brightness_mean, rgb_brightness_std = compute_rgb_canopy_stats(image_bgr, mask)
    segmentation_entropy_uncertainty = mean_map_value(segmentation_entropy_map, mask, default=0.0)
    mean_tree_probability = mean_map_value(tree_probability_map, mask, default=1.0)
    tree_probability_uncertainty = 1.0 - mean_tree_probability
    normalized_canopy_size = float(area) / max(image_area, 1.0)

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
        "canopy_diameter_px": int(canopy_diameter_px),
        "canopy_compactness": safe_float(candidate.canopy_compactness),
        "canopy_circularity": safe_float(candidate.canopy_circularity),
        "canopy_asymmetry_score": safe_float(canopy_asymmetry_score),
        "canopy_gap_ratio": safe_float(canopy_gap_ratio),
        "canopy_edge_roughness": safe_float(canopy_edge_roughness),
        "canopy_irregularity": safe_float(canopy_irregularity),
        "rgb_green_ratio": safe_float(rgb_green_ratio),
        "rgb_brightness_mean": safe_float(rgb_brightness_mean, 2),
        "rgb_brightness_std": safe_float(rgb_brightness_std, 2),
        "distance_to_road_px": safe_float(distance_px, 2),
        "inverse_distance_to_road": safe_float(inverse_distance_to_road(distance_px, risk_config.distance_scale_px)),
        "road_overlap_ratio": safe_float(road_overlap_ratio),
        "road_buffer_overlap_ratio": safe_float(road_buffer_overlap_ratio),
        "normalized_canopy_size": safe_float(normalized_canopy_size),
        "segmentation_entropy_uncertainty": safe_float(segmentation_entropy_uncertainty),
        "mean_tree_probability": safe_float(mean_tree_probability),
        "tree_probability_uncertainty": safe_float(tree_probability_uncertainty),
        "candidate_source": candidate.candidate_source,
        "uncertainty_source": uncertainty_source,
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


def compute_canopy_asymmetry(mask: np.ndarray, centroid_x: float, centroid_y: float) -> float:
    ys, xs = np.where(mask > 0)
    total = int(xs.size)
    if total == 0:
        return 0.0

    left_area = int(np.count_nonzero(xs < centroid_x))
    right_area = int(np.count_nonzero(xs >= centroid_x))
    top_area = int(np.count_nonzero(ys < centroid_y))
    bottom_area = int(np.count_nonzero(ys >= centroid_y))
    lr_asymmetry = abs(left_area - right_area) / max(total, 1)
    tb_asymmetry = abs(top_area - bottom_area) / max(total, 1)
    return clamp01((lr_asymmetry + tb_asymmetry) / 2.0)


def compute_canopy_gap_ratio(mask: np.ndarray) -> float:
    x1, y1, x2, y2 = _bbox_from_binary(mask)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2] > 0
    filled = ndimage.binary_fill_holes(crop)
    filled_area = int(np.count_nonzero(filled))
    if filled_area == 0:
        return 0.0
    original_area = int(np.count_nonzero(crop))
    gap_area = max(filled_area - original_area, 0)
    return clamp01(float(gap_area) / float(filled_area))


def compute_canopy_edge_roughness(mask: np.ndarray, area: int, config: RiskConfig) -> float:
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    raw = perimeter / max(float(np.sqrt(area)), 1.0)
    min_value = float(config.edge_roughness_min)
    max_value = max(float(config.edge_roughness_max), min_value + 1e-6)
    return clamp01((raw - min_value) / (max_value - min_value))


def compute_rgb_canopy_stats(image_bgr: np.ndarray | None, mask: np.ndarray) -> tuple[float, float, float]:
    if image_bgr is None or image_bgr.size == 0 or np.count_nonzero(mask) == 0:
        return 0.0, 0.0, 0.0

    pixels = image_bgr[mask > 0].astype(np.float32)
    if pixels.size == 0:
        return 0.0, 0.0, 0.0
    b = pixels[:, 0]
    g = pixels[:, 1]
    r = pixels[:, 2]
    green_ratio = np.mean(g / np.maximum(r + g + b, 1e-6))
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return clamp01(float(green_ratio)), float(np.mean(gray)), float(np.std(gray))


def mean_map_value(value_map: np.ndarray | None, mask: np.ndarray, default: float) -> float:
    if value_map is None or value_map.size == 0 or np.count_nonzero(mask) == 0:
        return clamp01(default)
    values = value_map[mask > 0]
    if values.size == 0:
        return clamp01(default)
    return clamp01(float(np.mean(values)))


def _bbox_from_binary(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
