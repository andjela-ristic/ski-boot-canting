from __future__ import annotations

import math

import numpy as np

from . import context
from .geometry import (
    acute_line_angle_deg,
    axis_up_vector,
    signed_angle_deg,
    table_direction_vector,
    table_up_normal,
)


def compute_line_angle_uncertainty(table_line: dict) -> dict:
    refinement = table_line.get("refinement", {})
    rmse = refinement.get("fit_rmse_px")
    p90 = refinement.get("fit_p90_abs_residual_px")
    coverage = table_line.get("coverage", {})
    support_points = table_line.get("support_points")
    support_span = 0.0
    if isinstance(support_points, np.ndarray) and support_points.shape[0] >= 2:
        x_values = support_points[:, 0].astype(np.float64)
        support_span = float(np.quantile(x_values, 0.98) - np.quantile(x_values, 0.02))
    effective_span = max(
        1.0,
        support_span,
        float(coverage.get("left_union_length_px", 0.0))
        + float(coverage.get("right_union_length_px", 0.0)),
    )
    if rmse is None:
        return {
            "table_angle_standard_error_deg": None,
            "table_angle_uncertainty_95_deg": None,
            "effective_support_span_px": effective_span,
        }
    residual_scale = max(float(rmse), 0.5 * float(p90 or rmse))
    # Small-angle approximation for the slope uncertainty caused by vertical
    # residuals distributed over the effective horizontal support span.
    standard_error = math.degrees(math.atan2(residual_scale, effective_span / math.sqrt(12.0)))
    uncertainty95 = 1.96 * standard_error
    return {
        "table_angle_standard_error_deg": float(standard_error),
        "table_angle_uncertainty_95_deg": float(uncertainty95),
        "effective_support_span_px": float(effective_span),
    }


def compute_canting(axis_candidate: dict, table_line: dict, y_min: int, y_max: int) -> dict:
    axis_vector = axis_up_vector(axis_candidate, y_min, y_max)
    table_vector = table_direction_vector(float(table_line["slope"]))
    normal_vector = table_up_normal(float(table_line["slope"]))
    canting_signed = signed_angle_deg(normal_vector, axis_vector)
    axis_table_angle = acute_line_angle_deg(axis_vector, table_vector)
    direction_cfg = context.STEP_CONFIG.get("direction", {})
    neutral_threshold = float(direction_cfg.get("neutral_threshold_deg", 0.05))
    if canting_signed > neutral_threshold:
        direction = str(direction_cfg.get("positive_label", "right"))
    elif canting_signed < -neutral_threshold:
        direction = str(direction_cfg.get("negative_label", "left"))
    else:
        direction = str(direction_cfg.get("neutral_label", "neutral"))
    return {
        "axis_table_angle_deg": float(axis_table_angle),
        "canting_angle_deg": float(canting_signed),
        "absolute_canting_angle_deg": float(abs(canting_signed)),
        "canting_direction": direction,
        "axis_up_vector": [float(axis_vector[0]), float(axis_vector[1])],
        "table_direction_vector": [float(table_vector[0]), float(table_vector[1])],
        "table_up_normal_vector": [float(normal_vector[0]), float(normal_vector[1])],
    }


def combine_measurement_confidence(axis_quality: dict, table_result: dict, uncertainty: dict) -> dict:
    cfg = context.STEP_CONFIG.get("measurement_confidence", {})
    table_line = table_result["winner"]
    axis_score = context.clip01(float(axis_quality.get("axis_quality_score", 0.0)))
    table_score = context.clip01(float(table_line.get("table_line_quality_score", table_line.get("score", 0.0))))
    axis_stability = axis_quality.get("component_scores", {}).get("ensemble_stability")
    if axis_stability is None:
        axis_stability = axis_quality.get("component_scores", {}).get("winner_validator_evidence", axis_score)
    table_stability = context.clip01(float(table_result.get("stability", {}).get("score", 0.0)))
    joint_stability = math.sqrt(max(0.0, context.clip01(float(axis_stability)) * table_stability))
    axis_margin = axis_quality.get("component_scores", {}).get("distinct_margin")
    if axis_margin is None:
        axis_margin = axis_score
    table_margin = context.clip01(float(table_line.get("candidate_margin_component", 0.0)))
    joint_margin = math.sqrt(max(0.0, context.clip01(float(axis_margin)) * table_margin))

    weights = {
        "axis_quality": float(cfg.get("axis_quality_weight", 0.43)),
        "table_quality": float(cfg.get("table_quality_weight", 0.39)),
        "joint_stability": float(cfg.get("joint_stability_weight", 0.12)),
        "joint_margin": float(cfg.get("joint_margin_weight", 0.06)),
    }
    components = {
        "axis_quality": axis_score,
        "table_quality": table_score,
        "joint_stability": joint_stability,
        "joint_margin": joint_margin,
    }
    denominator = sum(weights.values())
    confidence = sum(components[name] * weights[name] for name in components) / max(1e-9, denominator)

    uncertainty95 = uncertainty.get("table_angle_uncertainty_95_deg")
    max_uncertainty = float(cfg.get("max_accepted_uncertainty_deg", 0.55))
    uncertainty_penalty = 1.0
    if uncertainty95 is not None and float(uncertainty95) > max_uncertainty:
        uncertainty_penalty = float(math.exp(-(float(uncertainty95) - max_uncertainty) / max(0.1, max_uncertainty)))
    confidence = context.clip01(confidence * uncertainty_penalty)
    percent = 100.0 * confidence

    if table_score * 100.0 < float(cfg.get("reference_line_min_percent", 52.0)) or not bool(table_line.get("valid", False)):
        decision = "reference_line_not_reliable"
    elif percent >= float(cfg.get("accepted_min_percent", 82.0)):
        decision = "accepted"
    elif percent >= float(cfg.get("accepted_low_confidence_min_percent", 68.0)):
        decision = "accepted_low_confidence"
    elif percent >= float(cfg.get("manual_review_min_percent", 52.0)):
        decision = "manual_review"
    else:
        decision = "recapture_required"

    return {
        "measurement_confidence_score": confidence,
        "measurement_confidence_percent": percent,
        "measurement_quality_components": components,
        "measurement_quality_weights": weights,
        "uncertainty_penalty": float(uncertainty_penalty),
        "decision": decision,
        "calibration_status": str(cfg.get("calibration_status", "uncalibrated_estimated_measurement_confidence")),
    }
