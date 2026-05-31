"""Connected-component candidate extraction utilities."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from config import CandidateConfig
from utils import bbox_from_mask, clean_binary_mask, split_connected_components


def build_connected_component_candidates(
    vegetation_mask: np.ndarray,
    candidate_config: CandidateConfig,
    vegetation_overlap: float = 1.0,
    used_sam_splitting: bool = False,
) -> List[dict]:
    """Split a vegetation mask into connected components and normalize them as candidate dicts."""
    if vegetation_mask is None or np.count_nonzero(vegetation_mask) == 0:
        return []

    cleaned = clean_binary_mask(
        vegetation_mask,
        min_area_px=candidate_config.min_component_area_px,
        kernel_size=candidate_config.morph_kernel_size,
        close_iterations=candidate_config.morph_close_iterations,
    )
    candidates: List[dict] = []
    for component in split_connected_components(cleaned, candidate_config.min_component_area_px):
        candidates.append(
            {
                "mask": component.astype(np.uint8),
                "bbox": list(bbox_from_mask(component)),
                "area": int(np.count_nonzero(component)),
                "vegetation_overlap": float(vegetation_overlap),
                "segmentation_confidence": 0.5,
                "source": "components",
                "used_sam_splitting": bool(used_sam_splitting),
            }
        )
    return candidates


def normalize_candidate_masks(candidates: Sequence[dict]) -> List[dict]:
    normalized: List[dict] = []
    for candidate in candidates:
        mask = candidate.get("mask")
        if mask is None:
            continue
        bbox = candidate.get("bbox")
        if bbox is None:
            bbox = list(bbox_from_mask(mask))
        normalized.append(
            {
                "mask": mask.astype(np.uint8),
                "bbox": [int(v) for v in bbox],
                "area": int(candidate.get("area", int(np.count_nonzero(mask)))),
                "vegetation_overlap": float(candidate.get("vegetation_overlap", 1.0)),
                "segmentation_confidence": float(candidate.get("segmentation_confidence", 0.5)),
                "source": str(candidate.get("source", "components")),
                "used_sam_splitting": bool(candidate.get("used_sam_splitting", False)),
            }
        )
    return normalized
