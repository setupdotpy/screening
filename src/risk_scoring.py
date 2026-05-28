"""Explainable risk scoring."""

from __future__ import annotations

import math

from config import RiskConfig
from utils import safe_float


def add_risk_score(row: dict, config: RiskConfig) -> dict:
    distance = float(row["distance_to_road_px"])
    inverse_distance = distance_risk(distance, config.distance_normalizer_px)
    lean = 1.0 if row["lean_toward_road"] else 0.0
    overhang = clamp01(float(row["overhang_ratio"]))
    normalized_size = clamp01(float(row["normalized_tree_size"]))
    uncertainty = clamp01(float(row["uncertainty"]))

    w_distance, w_lean, w_overhang, w_size, w_uncertainty = config.weights
    risk_score = (
        w_distance * inverse_distance
        + w_lean * lean
        + w_overhang * overhang
        + w_size * normalized_size
        + w_uncertainty * uncertainty
    )
    risk_score = clamp01(risk_score)

    row["inverse_distance_to_road"] = safe_float(inverse_distance)
    row["risk_score"] = safe_float(risk_score)
    row["risk_level"] = risk_level(risk_score)
    return row


def distance_risk(distance_px: float, scale_px: float) -> float:
    """Convert distance to a smooth bounded proximity risk.

    The exponential form is monotonic, bounded in [0, 1], equals 1 at zero
    distance, and decays smoothly instead of imposing an arbitrary hard cutoff.
    """
    scale = max(float(scale_px), 1.0)
    distance = max(float(distance_px), 0.0)
    return clamp01(math.exp(-distance / scale))


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def risk_level(score: float) -> str:
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"
