"""Inspection-priority scoring for accepted UAV canopy candidates."""

from __future__ import annotations

import math

from config import RiskConfig
from utils import clamp01, safe_float


def score_inspection_priority(row: dict, config: RiskConfig, median_canopy_area: float | None = None) -> dict:
    inverse_distance = clamp01(float(row["inverse_distance_to_road"]))
    road_buffer_overlap = clamp01(float(row["road_buffer_overlap_ratio"]))
    normalized_size = clamp01(float(row["normalized_canopy_size"]))
    canopy_irregularity = clamp01(float(row["canopy_irregularity"]))
    canopy_gap_ratio = clamp01(float(row["canopy_gap_ratio"]))
    canopy_asymmetry = clamp01(float(row["canopy_asymmetry_score"]))

    w_distance, w_buffer, w_size, w_irregularity, w_gap, w_asymmetry = config.weights
    base_score = clamp01(
        w_distance * inverse_distance
        + w_buffer * road_buffer_overlap
        + w_size * normalized_size
        + w_irregularity * canopy_irregularity
        + w_gap * canopy_gap_ratio
        + w_asymmetry * canopy_asymmetry
    )

    segmentation_entropy_uncertainty = clamp01(float(row.get("segmentation_entropy_uncertainty", 0.0)))
    tree_probability_uncertainty = clamp01(float(row.get("tree_probability_uncertainty", 0.0)))
    instance_merge_uncertainty = compute_instance_merge_uncertainty(
        canopy_area_px=float(row.get("canopy_area_px", 0.0)),
        median_canopy_area=median_canopy_area,
    )

    w_entropy, w_tree_probability, w_instance = config.uncertainty_weights
    uncertainty_score = clamp01(
        w_entropy * segmentation_entropy_uncertainty
        + w_tree_probability * tree_probability_uncertainty
        + w_instance * instance_merge_uncertainty
    )
    if config.use_uncertainty_aware_priority:
        final_score = clamp01(base_score + float(config.alpha_uncertainty) * uncertainty_score)
    else:
        final_score = base_score

    row["base_inspection_score"] = safe_float(base_score)
    row["segmentation_entropy_uncertainty"] = safe_float(segmentation_entropy_uncertainty)
    row["tree_probability_uncertainty"] = safe_float(tree_probability_uncertainty)
    row["instance_merge_uncertainty"] = safe_float(instance_merge_uncertainty)
    row["uncertainty_score"] = safe_float(uncertainty_score)
    row["final_priority_score"] = safe_float(final_score)
    row["inspection_priority_level"] = inspection_priority_level(final_score)
    return row


def compute_instance_merge_uncertainty(canopy_area_px: float, median_canopy_area: float | None) -> float:
    if median_canopy_area is None or median_canopy_area <= 0:
        return 0.0
    area_ratio = max(float(canopy_area_px), 0.0) / float(median_canopy_area)
    return clamp01((area_ratio - 1.0) / 4.0)


def score_risk(row: dict, config: RiskConfig, median_canopy_area: float | None = None) -> dict:
    """Compatibility wrapper; writes inspection-priority fields."""
    return score_inspection_priority(row, config, median_canopy_area=median_canopy_area)


def add_risk_score(row: dict, config: RiskConfig, median_canopy_area: float | None = None) -> dict:
    """Compatibility alias for older imports."""
    return score_inspection_priority(row, config, median_canopy_area=median_canopy_area)


def distance_risk(distance_px: float, scale_px: float) -> float:
    """Convert distance to a smooth bounded proximity score."""
    scale = max(float(scale_px), 1.0)
    distance = max(float(distance_px), 0.0)
    return clamp01(math.exp(-distance / scale))


def inspection_priority_level(score: float) -> str:
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"


def risk_level(score: float) -> str:
    """Compatibility alias for older imports."""
    return inspection_priority_level(score)
