"""Explainable risk scoring for accepted tree candidates."""

from __future__ import annotations

import math

from config import RiskConfig
from utils import clamp01, safe_float


def score_risk(row: dict, config: RiskConfig) -> dict:
    inverse_distance = clamp01(float(row["inverse_distance_to_road"]))
    road_buffer_overlap = clamp01(float(row["road_buffer_overlap_ratio"]))
    normalized_size = clamp01(float(row["normalized_canopy_size"]))
    nearby_density = clamp01(float(row["nearby_tree_density"]))
    canopy_asymmetry = clamp01(float(row["canopy_asymmetry_score"]))
    uncertainty = clamp01(float(row["uncertainty"]))

    w_distance, w_buffer, w_size, w_density, w_asymmetry, w_uncertainty = config.weights
    risk_score = clamp01(
        w_distance * inverse_distance
        + w_buffer * road_buffer_overlap
        + w_size * normalized_size
        + w_density * nearby_density
        + w_asymmetry * canopy_asymmetry
        + w_uncertainty * uncertainty
    )
    row["risk_score"] = safe_float(risk_score)
    row["risk_level"] = risk_level(risk_score)
    return row


def add_risk_score(row: dict, config: RiskConfig) -> dict:
    """Compatibility alias for older imports."""
    return score_risk(row, config)


def distance_risk(distance_px: float, scale_px: float) -> float:
    """Convert distance to a smooth bounded proximity risk."""
    scale = max(float(scale_px), 1.0)
    distance = max(float(distance_px), 0.0)
    return clamp01(math.exp(-distance / scale))


def risk_level(score: float) -> str:
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"
