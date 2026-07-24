from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import context


def _fraction(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    if numeric > 1.0:
        numeric /= 100.0
    return context.clip01(numeric)


def _validator_score(validator: Any) -> float | None:
    if not isinstance(validator, dict):
        return None
    if validator.get("available") is False:
        return None
    for key in ("score", "quality", "value"):
        if key in validator:
            return _fraction(validator.get(key))
    return None


def extract_winner_validator_scores(step08: dict, winner: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    validators = winner.get("validators", {}) if isinstance(winner, dict) else {}
    if isinstance(validators, dict):
        for name, payload in validators.items():
            score = _validator_score(payload)
            if score is not None:
                scores[str(name)] = score
    return scores


def _geometric_mean(values: list[float], floor: float = 1e-6) -> float | None:
    valid = [context.clip01(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return None
    return float(math.exp(sum(math.log(max(floor, value)) for value in valid) / len(valid)))


def build_axis_quality(step08: dict, winner: dict) -> dict:
    validator_scores = extract_winner_validator_scores(step08, winner)
    validator_evidence = _geometric_mean(list(validator_scores.values()))

    components = {
        "symmetry": _fraction(step08.get("symmetry_percent", winner.get("symmetry_percent"))),
        "multi_validation": _fraction(step08.get("multi_validation_percent", winner.get("multi_validation_percent"))),
        "step08_confidence": _fraction(step08.get("confidence_percent")),
        "winner_validator_evidence": validator_evidence,
        "validator_agreement": _fraction(step08.get("validator_agreement_percent", step08.get("validator_agreement"))),
        "ensemble_stability": _fraction(step08.get("ensemble_stability_percent", step08.get("ensemble_stability"))),
        "validator_availability": _fraction(step08.get("validator_availability_percent", step08.get("validator_availability"))),
        "distinct_margin": _fraction(step08.get("winner_margin_percent", step08.get("distinct_margin_component"))),
    }

    cfg = context.STEP_CONFIG.get("axis_quality", {})
    weights = {
        "symmetry": float(cfg.get("symmetry_weight", 0.20)),
        "multi_validation": float(cfg.get("multi_validation_weight", 0.20)),
        "step08_confidence": float(cfg.get("step08_confidence_weight", 0.11)),
        "winner_validator_evidence": float(cfg.get("winner_validator_evidence_weight", 0.17)),
        "validator_agreement": float(cfg.get("validator_agreement_weight", 0.11)),
        "ensemble_stability": float(cfg.get("ensemble_stability_weight", 0.10)),
        "validator_availability": float(cfg.get("validator_availability_weight", 0.05)),
        "distinct_margin": float(cfg.get("distinct_margin_weight", 0.06)),
    }
    available = [(name, value, weights[name]) for name, value in components.items() if value is not None and weights[name] > 0]
    denominator = sum(weight for _, _, weight in available)
    base_quality = (
        sum(float(value) * weight for _, value, weight in available) / denominator
        if denominator > 0
        else 0.0
    )

    step07_agrees = bool(step08.get("step_07_agrees", True))
    agreement_factor = 1.0 if step07_agrees else float(cfg.get("step07_disagreement_factor", 0.90))
    decision = str(step08.get("decision", "unknown"))
    decision_factor = float(cfg.get("decision_factors", {}).get(decision, 0.88))
    axis_quality = context.clip01(base_quality * agreement_factor * decision_factor)

    return {
        "axis_quality_score": axis_quality,
        "axis_quality_percent": 100.0 * axis_quality,
        "axis_quality_before_factors": float(base_quality),
        "step07_agreement_factor": float(agreement_factor),
        "source_decision_factor": float(decision_factor),
        "source_decision": decision,
        "step_07_agrees": step07_agrees,
        "component_scores": components,
        "component_weights": weights,
        "available_component_count": int(len(available)),
        "winner_validator_scores": validator_scores,
        "winner_validator_geometric_mean": validator_evidence,
        "raw_step08_metrics": {
            key: step08.get(key)
            for key in (
                "symmetry_percent",
                "multi_validation_score",
                "multi_validation_percent",
                "confidence_percent",
                "evidence_quality",
                "validator_agreement",
                "validator_agreement_percent",
                "validator_top_vote_ratio",
                "winner_margin",
                "winner_margin_percent",
                "distinct_margin_component",
                "ensemble_stability",
                "ensemble_stability_percent",
                "validator_availability",
                "validator_availability_percent",
                "distinct_runner_up_label",
                "decision",
                "calibration_status",
            )
            if key in step08
        },
    }
