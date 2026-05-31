"""Visualization and mask export for UAV canopy risk screening."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from feature_extraction import FeatureResult
from road_region import RoadContext
from segformer_segmenter import SegmentationResult
from structural_filter import TreeCandidate


RISK_COLORS = {
    "Low": (60, 180, 75),
    "Medium": (0, 165, 255),
    "High": (40, 40, 220),
}

REJECTED_COLOR = (170, 170, 170)
ROAD_COLOR = (255, 90, 20)
TREE_COLOR = (0, 175, 0)
LOW_VEGETATION_COLOR = (130, 220, 110)
TITLE = "UAV Roadside Tree/Canopy Risk Screening"


def save_visualization(
    image: np.ndarray,
    image_name: str,
    segmentation: SegmentationResult,
    road_context: RoadContext,
    tree_mask: np.ndarray,
    low_vegetation_mask: np.ndarray,
    accepted_candidates: Sequence[TreeCandidate],
    rejected_candidates: Sequence[TreeCandidate],
    feature_results: Sequence[FeatureResult],
    splitting_label: str,
    output_dir: Path,
) -> None:
    del segmentation
    vis = image.copy()

    vis = overlay_mask(vis, low_vegetation_mask, LOW_VEGETATION_COLOR, 0.14)
    vis = overlay_mask(vis, tree_mask, TREE_COLOR, 0.16)
    vis = overlay_mask(vis, road_context.road_mask, ROAD_COLOR, 0.28)

    if road_context.polygon:
        cv2.polylines(vis, [np.array(road_context.polygon, dtype=np.int32)], True, ROAD_COLOR, 2)

    draw_banner(vis, TITLE, splitting_label)

    for candidate in rejected_candidates:
        vis = overlay_mask(vis, candidate.mask, REJECTED_COLOR, 0.12)
        x1, y1, x2, y2 = candidate.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), REJECTED_COLOR, 1)

    for candidate, feature in zip(accepted_candidates, feature_results):
        row = feature.row
        color = RISK_COLORS.get(str(row.get("risk_level", "Low")), (255, 255, 255))
        vis = overlay_mask(vis, candidate.mask, color, 0.34)
        x1, y1, x2, y2 = candidate.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        centroid = (int(candidate.canopy_centroid_x), int(candidate.canopy_centroid_y))
        cv2.circle(vis, centroid, 3, (255, 255, 255), -1)
        if feature.distance_line:
            cv2.line(vis, feature.distance_line[0], feature.distance_line[1], (255, 0, 255), 2)
        label = (
            f"T{row['tree_id']} R{row['risk_score']:.2f} {row['risk_level']} "
            f"A{row['canopy_area_px']} D{row['distance_to_road_px']:.0f}"
        )
        draw_label(vis, label, (x1, max(y1 - 8, 18)), color)

    output_path = output_dir / "visualizations" / f"{Path(image_name).stem}_risk.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def save_masks(image_name: str, accepted_candidates: Sequence[TreeCandidate], output_dir: Path) -> None:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    for index, candidate in enumerate(accepted_candidates, start=1):
        cv2.imwrite(str(mask_dir / f"{stem}_canopy_{index:03d}.png"), candidate.mask)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask is None or np.count_nonzero(mask) == 0:
        return image
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1.0, overlay, alpha, 0)


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, image.shape[1] - text_w - 8))
    y = max(text_h + baseline + 5, y)
    cv2.rectangle(image, (x, y - text_h - baseline - 4), (x + text_w + 6, y + 3), color, -1)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_banner(image: np.ndarray, title: str, subtitle: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.72
    sub_scale = 0.52
    title_thickness = 2
    sub_thickness = 1
    (title_w, title_h), title_base = cv2.getTextSize(title, font, title_scale, title_thickness)
    (sub_w, sub_h), sub_base = cv2.getTextSize(subtitle, font, sub_scale, sub_thickness)
    width = max(title_w, sub_w) + 18
    height = title_h + title_base + sub_h + sub_base + 20
    x1, y1 = 12, 14
    cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), (20, 20, 20), -1)
    cv2.putText(image, title, (x1 + 8, y1 + title_h + 7), font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
    cv2.putText(image, subtitle, (x1 + 8, y1 + title_h + sub_h + 17), font, sub_scale, (220, 220, 220), sub_thickness, cv2.LINE_AA)
