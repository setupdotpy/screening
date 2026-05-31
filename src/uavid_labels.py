"""UAVid label loading and class-mask conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from segformer_segmenter import SegmentationResult


@dataclass(frozen=True)
class UAVidClass:
    name: str
    color_rgb: tuple[int, int, int]
    label_id: int


UAVID_CLASSES: tuple[UAVidClass, ...] = (
    UAVidClass("background clutter", (0, 0, 0), 0),
    UAVidClass("building", (128, 0, 0), 1),
    UAVidClass("road", (128, 64, 128), 2),
    UAVidClass("tree", (0, 128, 0), 3),
    UAVidClass("low vegetation", (128, 128, 0), 4),
    UAVidClass("static car", (192, 0, 192), 5),
    UAVidClass("moving car", (64, 0, 128), 6),
    UAVidClass("human", (64, 64, 0), 7),
)

UAVID_CLASS_NAMES = tuple(item.name for item in UAVID_CLASSES)
UAVID_COLOR_MAP: dict[str, tuple[int, int, int]] = {item.name: item.color_rgb for item in UAVID_CLASSES}


def load_uavid_segmentation(
    image_path: Path,
    label_dir: Path,
    image_shape: tuple[int, int, int],
    tolerance: int = 5,
) -> SegmentationResult:
    label_path = find_label_path(image_path, label_dir)
    if label_path is None:
        raise FileNotFoundError(f"No UAVid label found for {image_path.name} in {label_dir}")

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise ValueError(f"Could not read UAVid label: {label_path}")

    target_shape = image_shape[:2]
    class_masks = uavid_label_to_masks(label, target_shape=target_shape, tolerance=tolerance)
    confidence_map = np.ones(target_shape, dtype=np.float32)
    return SegmentationResult(
        class_masks=class_masks,
        confidence_map=confidence_map,
        source=f"uavid_label:{label_path.name}",
        elapsed_seconds=0.0,
    )


def find_label_path(image_path: Path, label_dir: Path) -> Path | None:
    if not label_dir.exists():
        return None

    stem = image_path.stem
    suffixes = (
        "",
        "_label",
        "_labelTrainIds",
        "_gt",
        "_gtFine_labelIds",
        "_gtFine_color",
        "_color",
    )
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    for suffix in suffixes:
        for extension in extensions:
            candidate = label_dir / f"{stem}{suffix}{extension}"
            if candidate.exists():
                return candidate

    matches = sorted(label_dir.glob(f"{stem}*"))
    return matches[0] if matches else None


def uavid_label_to_masks(
    label_image: np.ndarray,
    target_shape: tuple[int, int] | None = None,
    tolerance: int = 5,
) -> Dict[str, np.ndarray]:
    if label_image.ndim == 2:
        masks = _id_label_to_masks(label_image)
    else:
        if label_image.shape[2] == 4:
            label_image = label_image[:, :, :3]
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        masks = _rgb_label_to_masks(label_rgb, tolerance=tolerance)

    if target_shape is not None:
        masks = {
            name: _resize_mask(mask, target_shape)
            for name, mask in masks.items()
        }
    return masks


def _rgb_label_to_masks(label_rgb: np.ndarray, tolerance: int) -> Dict[str, np.ndarray]:
    masks = _empty_masks(label_rgb.shape[:2])
    label_int = label_rgb.astype(np.int16)
    tolerance = max(int(tolerance), 0)

    for item in UAVID_CLASSES:
        color = np.array(item.color_rgb, dtype=np.int16)
        distance = np.max(np.abs(label_int - color), axis=2)
        masks[item.name][distance <= tolerance] = 255
    return _with_alias_masks(masks)


def _id_label_to_masks(label_ids: np.ndarray) -> Dict[str, np.ndarray]:
    masks = _empty_masks(label_ids.shape[:2])
    for item in UAVID_CLASSES:
        masks[item.name][label_ids == item.label_id] = 255
    return _with_alias_masks(masks)


def _empty_masks(shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    masks = {item.name: np.zeros(shape, dtype=np.uint8) for item in UAVID_CLASSES}
    masks.setdefault("sidewalk", np.zeros(shape, dtype=np.uint8))
    return masks


def _with_alias_masks(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    masks["vegetation"] = masks["tree"].copy()
    masks["terrain"] = masks["low vegetation"].copy()
    masks.setdefault("sidewalk", np.zeros_like(masks["road"], dtype=np.uint8))
    return masks


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
