"""Shared image, mask, geometry, and CSV utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

from config import IMAGE_EXTENSIONS, ensure_output_dirs


def list_images(image_dir: Path) -> List[Path]:
    """Return supported image files in deterministic order."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_csv(rows: Iterable[Dict[str, object]], csv_path: Path, fieldnames: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def polygon_to_mask(shape: tuple[int, int], polygon: Sequence[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not polygon:
        return mask
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def clean_binary_mask(mask: np.ndarray, min_area_px: int, kernel_size: int = 5, close_iterations: int = 2) -> np.ndarray:
    """Morphologically clean a binary mask and remove tiny components."""
    binary = (mask > 0).astype(np.uint8) * 255
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=max(1, close_iterations))
    return remove_small_components(binary, min_area_px)


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == label] = 255
    return cleaned


def split_connected_components(mask: np.ndarray, min_area_px: int) -> List[np.ndarray]:
    """Split a binary mask into one mask per connected component."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: List[np.ndarray] = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area_px:
            continue
        component = np.zeros_like(binary, dtype=np.uint8)
        component[labels == label] = 255
        components.append(component)
    return components


def component_width_in_band(mask: np.ndarray, top_ratio: float, bottom_ratio: float) -> tuple[int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, 0

    y_min, y_max = int(ys.min()), int(ys.max())
    height = max(y_max - y_min + 1, 1)
    top_cutoff = y_min + int(height * top_ratio)
    bottom_cutoff = y_min + int(height * bottom_ratio)

    top_xs = xs[ys <= top_cutoff]
    bottom_xs = xs[ys >= bottom_cutoff]

    top_width = int(top_xs.max() - top_xs.min() + 1) if top_xs.size else 0
    bottom_width = int(bottom_xs.max() - bottom_xs.min() + 1) if bottom_xs.size else 0
    full_width = int(xs.max() - xs.min() + 1)
    return top_width, bottom_width, full_width


def rows_with_pixels(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(mask > 0, axis=1)))


def mask_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return np.empty((0, 2), dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = cv2.subtract(binary, eroded)
    ys, xs = np.where(boundary > 0)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs, ys]).astype(np.int32)


def ensure_dirs(output_dir: Path) -> None:
    ensure_output_dirs(output_dir)
