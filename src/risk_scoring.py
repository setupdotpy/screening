"""Inspection-priority scoring for accepted UAV canopy candidates."""

from __future__ import annotations

import math

from config import RiskConfig
from utils import clamp01, safe_float


def score_inspection_priority(row: dict, config: RiskConfig) -> dict:
    inverse_distance = clamp01(float(row["inverse_distance_to_road"]))
    road_buffer_overlap = clamp01(float(row["road_buffer_overlap_ratio"]))
    normalized_size = clamp01(float(row["normalized_canopy_size"]))
    canopy_irregularity = clamp01(float(row["canopy_irregularity"]))
    canopy_gap_ratio = clamp01(float(row["canopy_gap_ratio"]))
    canopy_asymmetry = clamp01(float(row["canopy_asymmetry_score"]))

    w_distance, w_buffer, w_size, w_irregularity, w_gap, w_asymmetry = config.weights
    score = clamp01(
        w_distance * inverse_distance
        + w_buffer * road_buffer_overlap
        + w_size * normalized_size
        + w_irregularity * canopy_irregularity
        + w_gap * canopy_gap_ratio
        + w_asymmetry * canopy_asymmetry
    )
    row["inspection_priority_score"] = safe_float(score)
    row["inspection_priority_level"] = inspection_priority_level(score)
    return row


def score_risk(row: dict, config: RiskConfig) -> dict:
    """Compatibility wrapper; writes inspection-priority fields."""
    return score_inspection_priority(row, config)


def add_risk_score(row: dict, config: RiskConfig) -> dict:
    """Compatibility alias for older imports."""
    return score_inspection_priority(row, config)


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
