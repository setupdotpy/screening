"""Feature extraction from vegetation masks and road masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from detector import Detection
from road_region import RoadRegion, road_centroid_x
from utils import mask_centroid, safe_float


@dataclass
class FeatureResult:
    row: dict
    distance_line: Optional[tuple[tuple[int, int], tuple[int, int]]]
    lean_arrow: Optional[tuple[tuple[int, int], tuple[int, int]]]


def extract_features(
    image_name: str,
    tree_id: int,
    detection: Detection,
    road: RoadRegion,
    image_shape: tuple[int, int, int],
) -> FeatureResult:
    mask = detection.mask
    road_mask = road.mask
    height, width = image_shape[:2]
    image_area = float(height * width)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = detection.bbox

    mask_area = int(np.count_nonzero(mask))
    bbox_height = int(bbox_y2 - bbox_y1)
    canopy_width = int(bbox_x2 - bbox_x1)
    tree_centroid_x, tree_centroid_y = mask_centroid(mask)
    road_cx = road_centroid_x(road_mask)

    distance_px, distance_line = distance_to_road(mask, road_mask)
    lean_dx, lean_toward_road, lean_arrow = estimate_lean(mask, road_cx, (tree_centroid_x, tree_centroid_y))
    overhang_ratio = float(np.count_nonzero((mask > 0) & (road_mask > 0))) / max(mask_area, 1)
    normalized_tree_size = float(mask_area) / max(image_area, 1.0)
    uncertainty = 1.0 - float(detection.confidence)

    row = {
        "image_name": image_name,
        "tree_id": tree_id,
        "bbox_x1": bbox_x1,
        "bbox_y1": bbox_y1,
        "bbox_x2": bbox_x2,
        "bbox_y2": bbox_y2,
        "confidence": safe_float(detection.confidence),
        "mask_area_px": mask_area,
        "bbox_height_px": bbox_height,
        "canopy_width_px": canopy_width,
        "distance_to_road_px": safe_float(distance_px, 2),
        "lean_dx_px": safe_float(lean_dx, 2),
        "lean_toward_road": bool(lean_toward_road),
        "overhang_ratio": safe_float(overhang_ratio),
        "normalized_tree_size": safe_float(normalized_tree_size),
        "uncertainty": safe_float(uncertainty),
        "detection_source": detection.source,
        "road_source": road.source,
    }
    return FeatureResult(row=row, distance_line=distance_line, lean_arrow=lean_arrow)


def distance_to_road(
    tree_mask: np.ndarray,
    road_mask: np.ndarray,
) -> tuple[float, Optional[tuple[tuple[int, int], tuple[int, int]]]]:
    """Minimum distance from tree pixels to road pixels using distance transform."""
    tree_pixels = np.column_stack(np.where(tree_mask > 0))
    if tree_pixels.size == 0 or np.count_nonzero(road_mask) == 0:
        return 0.0, None

    if np.any((tree_mask > 0) & (road_mask > 0)):
        overlap_pixels = np.column_stack(np.where((tree_mask > 0) & (road_mask > 0)))
        y, x = overlap_pixels[0]
        return 0.0, ((int(x), int(y)), (int(x), int(y)))

    inverse_road = (road_mask == 0).astype(np.uint8)
    distance_map = cv2.distanceTransform(inverse_road, cv2.DIST_L2, 5)
    tree_distances = distance_map[tree_mask > 0]
    min_index = int(np.argmin(tree_distances))
    min_distance = float(tree_distances[min_index])

    tree_y, tree_x = tree_pixels[min_index]
    road_edge_pixels = road_boundary_pixels(road_mask)
    if road_edge_pixels.size == 0:
        return min_distance, None

    deltas = road_edge_pixels - np.array([tree_y, tree_x])
    nearest_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    road_y, road_x = road_edge_pixels[nearest_index]
    line = ((int(tree_x), int(tree_y)), (int(road_x), int(road_y)))
    return min_distance, line


def road_boundary_pixels(road_mask: np.ndarray) -> np.ndarray:
    """Return road boundary pixels for efficient distance-line visualization."""
    binary = (road_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)
    contour = max(contours, key=cv2.contourArea)
    points_xy = contour.reshape(-1, 2)
    if points_xy.shape[0] > 2000:
        step = int(np.ceil(points_xy.shape[0] / 2000))
        points_xy = points_xy[::step]
    return points_xy[:, [1, 0]].astype(np.int32)


def estimate_lean(
    tree_mask: np.ndarray,
    road_centroid_x_value: float,
    tree_centroid: tuple[float, float],
) -> tuple[float, bool, Optional[tuple[tuple[int, int], tuple[int, int]]]]:
    ys, xs = np.where(tree_mask > 0)
    if xs.size == 0:
        return 0.0, False, None

    y_min, y_max = int(ys.min()), int(ys.max())
    mask_height = max(y_max - y_min + 1, 1)
    upper_cutoff = y_min + int(mask_height * 0.40)
    lower_cutoff = y_min + int(mask_height * 0.70)

    upper_xs = xs[ys <= upper_cutoff]
    upper_ys = ys[ys <= upper_cutoff]
    lower_xs = xs[ys >= lower_cutoff]
    lower_ys = ys[ys >= lower_cutoff]

    if upper_xs.size == 0 or lower_xs.size == 0:
        return 0.0, False, None

    top_center_x = float(upper_xs.mean())
    top_center_y = float(upper_ys.mean())
    bottom_center_x = float(lower_xs.mean())
    bottom_center_y = float(lower_ys.mean())
    lean_dx = top_center_x - bottom_center_x

    tree_centroid_x, _ = tree_centroid
    road_is_right = road_centroid_x_value > tree_centroid_x
    lean_toward_road = lean_dx > 0 if road_is_right else lean_dx < 0
    arrow = ((int(bottom_center_x), int(bottom_center_y)), (int(top_center_x), int(top_center_y)))
    return lean_dx, bool(lean_toward_road), arrow
