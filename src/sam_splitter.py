"""SAM-based vegetation blob splitting with connected-component fallback."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from candidate_extraction import build_connected_component_candidates, normalize_candidate_masks
from config import CandidateConfig, SamConfig
from utils import bbox_from_mask, clean_binary_mask, clamp01


_SAM_MODEL_CACHE: dict[str, Any] = {}
_SAM_MODEL_ERROR: dict[str, str] = {}


def split_vegetation_with_sam(
    image_bgr: np.ndarray,
    vegetation_mask: np.ndarray,
    road_context_mask: np.ndarray | None = None,
    config: SamConfig | None = None,
) -> list[dict]:
    """Split vegetation blobs with SAM, falling back to connected components when needed."""
    del road_context_mask  # Context is reserved for future refinement; current splitting is vegetation-driven.

    sam_config = config or SamConfig()
    stats = {
        "used_sam": False,
        "sam_masks_generated": 0,
        "sam_masks_kept_after_overlap": 0,
        "post_split_candidates": 0,
        "connected_components_count": 0,
        "fallback_used": False,
        "fallback_reason": None,
    }

    vegetation_mask = _ensure_binary_mask(vegetation_mask)
    if np.count_nonzero(vegetation_mask) == 0:
        candidates = _fallback_candidates(vegetation_mask, sam_config, stats, reason="Empty vegetation mask")
        split_vegetation_with_sam.last_stats = stats
        return candidates

    if not sam_config.use_sam_splitting:
        candidates = _fallback_candidates(
            vegetation_mask,
            sam_config,
            stats,
            reason="SAM splitting disabled in config",
        )
        split_vegetation_with_sam.last_stats = stats
        return candidates

    model, model_error = _load_sam_model(sam_config.model_name)
    if model is None:
        candidates = _fallback_candidates(
            vegetation_mask,
            sam_config,
            stats,
            reason=model_error or "SAM model unavailable",
        )
        split_vegetation_with_sam.last_stats = stats
        return candidates

    try:
        sam_candidates, sam_mask_count = _sam_candidates(
            model=model,
            image_bgr=image_bgr,
            vegetation_mask=vegetation_mask,
            sam_config=sam_config,
        )
        stats["used_sam"] = True
        stats["sam_masks_generated"] = sam_mask_count
        stats["sam_masks_kept_after_overlap"] = len(sam_candidates)

        sam_candidates = _dedupe_candidates(sam_candidates, sam_config.duplicate_iou_threshold)
        if not sam_candidates:
            candidates = _fallback_candidates(
                vegetation_mask,
                sam_config,
                stats,
                reason="No valid SAM vegetation masks after overlap filtering",
            )
            split_vegetation_with_sam.last_stats = stats
            return candidates

        stats["sam_masks_kept_after_overlap"] = len(sam_candidates)
        sam_candidates = _post_split_merged_candidates(sam_candidates, image_bgr.shape, sam_config)
        stats["post_split_candidates"] = len(sam_candidates)
        split_vegetation_with_sam.last_stats = stats
        return sam_candidates
    except Exception as exc:  # noqa: BLE001 - fallback is required and intentional.
        candidates = _fallback_candidates(vegetation_mask, sam_config, stats, reason=str(exc))
        split_vegetation_with_sam.last_stats = stats
        return candidates


def _sam_candidates(
    model: Any,
    image_bgr: np.ndarray,
    vegetation_mask: np.ndarray,
    sam_config: SamConfig,
) -> tuple[list[dict], int]:
    height, width = image_bgr.shape[:2]
    image_area = float(height * width)

    results = model.predict(source=image_bgr, imgsz=sam_config.image_size, verbose=False)
    if not results:
        return [], 0

    result = results[0]
    if getattr(result, "masks", None) is None or getattr(result.masks, "data", None) is None:
        return [], 0

    mask_data = result.masks.data.detach().cpu().numpy()
    if mask_data.ndim == 2:
        mask_data = mask_data[None, ...]

    candidates: list[dict] = []
    for mask in mask_data:
        mask_binary = _mask_to_binary(mask, (height, width))
        mask_area = int(np.count_nonzero(mask_binary))
        if mask_area < sam_config.min_mask_area:
            continue
        if mask_area > int(sam_config.max_mask_area_ratio * image_area):
            continue

        overlap = float(np.count_nonzero((mask_binary > 0) & (vegetation_mask > 0))) / max(mask_area, 1)
        if overlap < sam_config.vegetation_overlap_threshold:
            continue

        clipped_mask = np.where((mask_binary > 0) & (vegetation_mask > 0), 255, 0).astype(np.uint8)
        clipped_area = int(np.count_nonzero(clipped_mask))
        if clipped_area < sam_config.min_mask_area:
            continue

        candidates.append(
            {
                "mask": clipped_mask,
                "bbox": list(bbox_from_mask(clipped_mask)),
                "area": clipped_area,
                "vegetation_overlap": clamp01(overlap),
                "segmentation_confidence": clamp01(overlap),
                "source": "sam",
                "used_sam_splitting": True,
            }
        )

    return normalize_candidate_masks(candidates), int(mask_data.shape[0])


def _dedupe_candidates(candidates: list[dict], duplicate_iou_threshold: float) -> list[dict]:
    if len(candidates) < 2:
        return candidates

    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("vegetation_overlap", 0.0)), int(item.get("area", 0))),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in ranked:
        mask = candidate["mask"]
        if any(_mask_iou(mask, existing["mask"]) > duplicate_iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _post_split_merged_candidates(
    candidates: list[dict],
    image_shape: tuple[int, ...],
    sam_config: SamConfig,
) -> list[dict]:
    """Split wide merged vegetation masks using vertical low-density valleys.

    SAM can still return one mask for adjacent canopies. This pass is intentionally
    conservative: it only cuts masks that are large, wide, and have clear column
    valleys between vegetation-dense regions.
    """
    if not sam_config.post_split_merged_vegetation:
        return candidates

    image_area = float(image_shape[0] * image_shape[1])
    split_candidates: list[dict] = []
    for candidate in candidates:
        children = _split_candidate_by_vertical_valleys(candidate, image_area, sam_config)
        split_candidates.extend(children if len(children) > 1 else [candidate])
    return normalize_candidate_masks(split_candidates)


def _split_candidate_by_vertical_valleys(
    candidate: dict,
    image_area: float,
    sam_config: SamConfig,
) -> list[dict]:
    mask = _ensure_binary_mask(candidate["mask"])
    x1, y1, x2, y2 = [int(v) for v in candidate["bbox"]]
    width = max(x2 - x1, 1)
    area = int(np.count_nonzero(mask))

    if width < sam_config.post_split_min_width_px:
        return [candidate]
    if area / max(image_area, 1.0) < sam_config.post_split_min_area_ratio:
        return [candidate]

    crop = mask[y1:y2, x1:x2] > 0
    if crop.size == 0:
        return [candidate]

    projection = crop.sum(axis=0).astype(np.float32)
    if projection.max(initial=0.0) <= 0:
        return [candidate]

    kernel_width = max(9, int(width * 0.04))
    if kernel_width % 2 == 0:
        kernel_width += 1
    smoothed = cv2.GaussianBlur(projection.reshape(1, -1), (kernel_width, 1), 0).ravel()

    split_columns = _find_projection_valleys(smoothed, sam_config)
    if not split_columns:
        return [candidate]

    pieces = _candidate_pieces_from_columns(candidate, mask, split_columns, sam_config)
    if len(pieces) <= 1:
        return [candidate]
    return pieces


def _find_projection_valleys(projection: np.ndarray, sam_config: SamConfig) -> list[int]:
    width = int(projection.size)
    if width < sam_config.post_split_min_width_px:
        return []

    try:
        from scipy.signal import find_peaks
    except Exception:  # noqa: BLE001 - scipy is optional at runtime.
        return []

    max_value = float(projection.max(initial=0.0))
    if max_value <= 0:
        return []

    min_peak_distance = max(sam_config.post_split_min_child_width_px, int(width * 0.12))
    peaks, _ = find_peaks(
        projection,
        distance=min_peak_distance,
        prominence=max(max_value * 0.035, 3.0),
        height=max(max_value * 0.08, 5.0),
    )
    if len(peaks) < 2:
        return []

    valleys: list[int] = []
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        left_peak = int(left_peak)
        right_peak = int(right_peak)
        if right_peak - left_peak < sam_config.post_split_min_child_width_px:
            continue

        between = projection[left_peak:right_peak + 1]
        valley_offset = int(np.argmin(between))
        valley_index = left_peak + valley_offset
        if (
            valley_index < sam_config.post_split_min_child_width_px
            or width - valley_index < sam_config.post_split_min_child_width_px
        ):
            continue
        valley_value = float(projection[valley_index])
        lower_peak = float(min(projection[left_peak], projection[right_peak]))
        if lower_peak <= 0:
            continue
        if valley_value <= lower_peak * sam_config.post_split_valley_ratio:
            valleys.append(valley_index)

    if len(valleys) > max(sam_config.post_split_max_children - 1, 0):
        valley_scores = [(float(projection[index]), index) for index in valleys]
        valleys = [index for _, index in sorted(valley_scores)[: sam_config.post_split_max_children - 1]]
    return sorted(set(valleys))


def _candidate_pieces_from_columns(
    candidate: dict,
    mask: np.ndarray,
    split_columns: list[int],
    sam_config: SamConfig,
) -> list[dict]:
    x1, _, x2, _ = [int(v) for v in candidate["bbox"]]
    boundaries = [x1] + [x1 + column for column in split_columns] + [x2]
    pieces: list[dict] = []

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left < sam_config.post_split_min_child_width_px:
            continue
        piece_mask = np.zeros_like(mask, dtype=np.uint8)
        piece_mask[:, left:right] = mask[:, left:right]
        piece_mask = clean_binary_mask(
            piece_mask,
            min_area_px=sam_config.post_split_min_child_area,
            kernel_size=3,
            close_iterations=1,
        )
        piece_area = int(np.count_nonzero(piece_mask))
        if piece_area < sam_config.post_split_min_child_area:
            continue
        px1, py1, px2, py2 = bbox_from_mask(piece_mask)
        if px2 - px1 < sam_config.post_split_min_child_width_px:
            continue

        source = str(candidate.get("source", "components"))
        if not source.endswith("_split"):
            source = f"{source}_split"
        pieces.append(
            {
                "mask": piece_mask,
                "bbox": [px1, py1, px2, py2],
                "area": piece_area,
                "vegetation_overlap": float(candidate.get("vegetation_overlap", 1.0)),
                "segmentation_confidence": float(candidate.get("segmentation_confidence", 0.5)),
                "source": source,
                "used_sam_splitting": bool(candidate.get("used_sam_splitting", False)),
            }
        )

    if len(pieces) <= 1:
        return [candidate]
    return pieces


def _fallback_candidates(
    vegetation_mask: np.ndarray,
    sam_config: SamConfig,
    stats: dict,
    reason: str,
) -> list[dict]:
    print("SAM splitting unavailable. Falling back to connected components.")
    stats["used_sam"] = False
    stats["fallback_used"] = True
    stats["fallback_reason"] = reason

    candidate_config = CandidateConfig()
    candidates = build_connected_component_candidates(
        vegetation_mask=clean_binary_mask(
            vegetation_mask,
            min_area_px=candidate_config.min_component_area_px,
            kernel_size=candidate_config.morph_kernel_size,
            close_iterations=candidate_config.morph_close_iterations,
        ),
        candidate_config=candidate_config,
        vegetation_overlap=1.0,
        used_sam_splitting=False,
    )
    candidates = _post_split_merged_candidates(candidates, vegetation_mask.shape, sam_config)
    stats["connected_components_count"] = len(candidates)
    stats["post_split_candidates"] = len(candidates)
    stats["sam_masks_kept_after_overlap"] = 0
    return candidates


def _load_sam_model(model_name: str) -> tuple[Any | None, str | None]:
    if model_name in _SAM_MODEL_CACHE:
        return _SAM_MODEL_CACHE[model_name], None
    if model_name in _SAM_MODEL_ERROR:
        return None, _SAM_MODEL_ERROR[model_name]

    try:
        from ultralytics import SAM

        model = SAM(model_name)
        _SAM_MODEL_CACHE[model_name] = model
        return model, None
    except Exception as exc:  # noqa: BLE001 - connected-component fallback is required.
        message = f"{exc}"
        _SAM_MODEL_ERROR[model_name] = message
        return None, message


def _mask_to_binary(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    mask = mask.astype(np.float32)
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8) * 255


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    intersection = np.count_nonzero(a & b)
    return float(intersection) / float(union)


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


split_vegetation_with_sam.last_stats = {
    "used_sam": False,
    "sam_masks_generated": 0,
    "sam_masks_kept_after_overlap": 0,
    "connected_components_count": 0,
    "fallback_used": False,
    "fallback_reason": None,
}
