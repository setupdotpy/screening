"""Shared image, mask, and CSV utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

from config import IMAGE_EXTENSIONS


def list_images(image_dir: Path) -> List[Path]:
    """Return supported image files in deterministic order."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


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


def write_csv(rows: Iterable[Dict[str, object]], csv_path: Path, fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
