"""Visualization and mask export."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from detector import Detection
from road_region import RoadRegion


RISK_COLORS = {
    "Low": (60, 180, 75),
    "Medium": (0, 165, 255),
    "High": (40, 40, 220),
}


def save_visualization(
    image: np.ndarray,
    image_name: str,
    road: RoadRegion,
    detections: List[Detection],
    feature_results: List[dict],
    distance_lines: List[object],
    lean_arrows: List[object],
    output_dir: Path,
) -> None:
    vis = image.copy()

    road_overlay = np.zeros_like(vis)
    road_overlay[road.mask > 0] = (255, 140, 0)
    vis = cv2.addWeighted(vis, 1.0, road_overlay, 0.25, 0)
    if road.polygon:
        cv2.polylines(vis, [np.array(road.polygon, dtype=np.int32)], True, (255, 140, 0), 2)

    for detection, row, distance_line, lean_arrow in zip(detections, feature_results, distance_lines, lean_arrows):
        color = RISK_COLORS.get(str(row["risk_level"]), (255, 255, 255))
        mask_overlay = np.zeros_like(vis)
        mask_overlay[detection.mask > 0] = color
        vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.35, 0)

        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if lean_arrow:
            cv2.arrowedLine(vis, lean_arrow[0], lean_arrow[1], (255, 255, 255), 2, tipLength=0.25)
        if distance_line:
            cv2.line(vis, distance_line[0], distance_line[1], (255, 0, 255), 2)

        label = f"ID {row['tree_id']} {row['risk_level']} {row['risk_score']:.2f}"
        draw_label(vis, label, (x1, max(y1 - 8, 18)), color)

    output_path = output_dir / "visualizations" / f"{Path(image_name).stem}_risk.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def save_masks(image_name: str, detections: List[Detection], output_dir: Path) -> None:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    for index, detection in enumerate(detections, start=1):
        cv2.imwrite(str(mask_dir / f"{stem}_tree_{index:03d}.png"), detection.mask)


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - text_h - baseline - 4), (x + text_w + 6, y + 3), color, -1)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
