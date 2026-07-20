from __future__ import annotations

import math

import numpy as np

from .context import cfg, clip01
from .geometry import candidates_equivalent


VALIDATOR_NAMES = (
    "step_07_symmetry",
    "segment_consistency",
    "perturbation_stability",
    "fragment_evidence",
)


def _weighted_geometric_mean(values: list[float], weights: list[float], floor: float = 1e-6) -> float:
    if not values:
        return 0.0
    array = np.maximum(np.asarray(values, dtype=np.float64), floor)
    weight_array = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if float(np.sum(weight_array)) <= 0:
        weight_array = np.ones_like(array)
    weight_array /= float(np.sum(weight_array))
    return float(math.exp(float(np.sum(weight_array * np.log(array)))))


def _validator_weights(section: str) -> dict[str, float]:
    prefix = {
        "step_07_symmetry": "symmetry_weight",
        "segment_consistency": "segment_consistency_weight",
        "perturbation_stability": "perturbation_stability_weight",
        "fragment_evidence": "fragment_evidence_weight",
    }
    return {name: float(cfg(section, key, default=0.0)) for name, key in prefix.items()}


def _assign_validator_ranks(candidates: list[dict]) -> None:
    for validator_name in VALIDATOR_NAMES:
        available = [
            candidate
            for candidate in candidates
            if candidate.get("validators", {}).get(validator_name, {}).get("available")
            and candidate["validators"][validator_name].get("score") is not None
        ]
        available.sort(
            key=lambda candidate: (
                float(candidate["validators"][validator_name]["score"]),
                -int(candidate.get("source_rank", 9999)),
            ),
            reverse=True,
        )
        denominator = max(1, len(available) - 1)
        for rank, candidate in enumerate(available, start=1):
            result = candidate["validators"][validator_name]
            result["rank"] = int(rank)
            result["rank_percentile"] = float(1.0 - (rank - 1) / denominator) if len(available) > 1 else 1.0


def compute_candidate_validation_scores(candidates: list[dict]) -> list[dict]:
    _assign_validator_ranks(candidates)
    weights = _validator_weights("candidate_score")
    floor = float(cfg("candidate_score", "missing_component_floor", default=0.0001))
    require_step07_valid = bool(cfg("candidate_validation", "require_step_07_valid", default=True))
    for candidate in candidates:
        values = []
        value_weights = []
        available_names = []
        rejection_reasons = []
        for name in VALIDATOR_NAMES:
            result = candidate.get("validators", {}).get(name, {})
            weight = max(0.0, weights.get(name, 0.0))
            if weight <= 0 or not result.get("available") or result.get("score") is None:
                continue
            values.append(max(floor, clip01(float(result["score"]))))
            value_weights.append(weight)
            available_names.append(name)
        availability = sum(value_weights) / max(sum(max(0.0, weights[name]) for name in VALIDATOR_NAMES), 1e-9)
        score = _weighted_geometric_mean(values, value_weights, floor) * availability
        step07_result = candidate.get("validators", {}).get("step_07_symmetry", {})
        if require_step07_valid and step07_result.get("available") and not step07_result.get("step_07_valid", True):
            rejection_reasons.append("step_07_candidate_invalid")
        mask_result = candidate.get("validators", {}).get("evaluation_mask_support", {})
        if mask_result.get("available") and not mask_result.get("valid", False):
            rejection_reasons.append("axis_leaves_step_07_evaluation_mask")
        valid = not rejection_reasons
        if not valid:
            score *= 0.45
        candidate.update({
            "available_validators": available_names,
            "validator_availability": float(availability),
            "multi_validation_score": clip01(score),
            "multi_validation_percent": 100.0 * clip01(score),
            "validation_valid": bool(valid),
            "rejection_reasons": rejection_reasons,
        })
    return sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate["validation_valid"]),
            float(candidate["multi_validation_score"]),
            float(candidate.get("validators", {}).get("step_07_symmetry", {}).get("score") or 0.0),
            -int(candidate.get("source_rank", 9999)),
        ),
        reverse=True,
    )


def select_final_candidate(ranked: list[dict], step07_winner_label: str) -> tuple[dict, dict]:
    step07_winner = next((candidate for candidate in ranked if str(candidate["candidate_label"]) == str(step07_winner_label)), None)
    if step07_winner is None:
        step07_winner = max(
            ranked,
            key=lambda candidate: float(candidate.get("validators", {}).get("step_07_symmetry", {}).get("score") or 0.0),
        )
    ensemble_winner = ranked[0]
    mode = str(cfg("selection", "mode", default="validate_step_07_winner"))
    allow_override = bool(cfg("selection", "allow_candidate_override", default=False))
    if mode == "ensemble_override" and allow_override:
        final = ensemble_winner
        reason = "ensemble_override_enabled"
    else:
        final = step07_winner
        reason = "step_07_winner_validated_without_override"
    return final, {
        "selection_mode": mode,
        "allow_candidate_override": allow_override,
        "selection_reason": reason,
        "step_07_winner_label": str(step07_winner["candidate_label"]),
        "ensemble_recommendation_label": str(ensemble_winner["candidate_label"]),
        "ensemble_recommends_same_or_equivalent": None,
    }


def _distinct_step07_margin(final_candidate: dict, candidates: list[dict], y_min: int, y_max: int) -> tuple[float, str | None]:
    final_score = float(final_candidate.get("validators", {}).get("step_07_symmetry", {}).get("score") or 0.0)
    alternatives = sorted(
        candidates,
        key=lambda candidate: float(candidate.get("validators", {}).get("step_07_symmetry", {}).get("score") or 0.0),
        reverse=True,
    )
    runner = next(
        (candidate for candidate in alternatives if not candidates_equivalent(final_candidate, candidate, y_min, y_max)),
        None,
    )
    if runner is None:
        return final_score, None
    runner_score = float(runner.get("validators", {}).get("step_07_symmetry", {}).get("score") or 0.0)
    return max(0.0, final_score - runner_score), str(runner["candidate_label"])


def compute_confidence(final_candidate: dict, ranked: list[dict], y_min: int, y_max: int) -> dict:
    weights = _validator_weights("confidence")
    values = []
    value_weights = []
    component_values = {}
    for name in VALIDATOR_NAMES:
        result = final_candidate.get("validators", {}).get(name, {})
        weight = max(0.0, weights.get(name, 0.0))
        if weight <= 0 or not result.get("available") or result.get("score") is None:
            continue
        value = clip01(float(result["score"]))
        component_values[name] = value
        values.append(max(1e-6, value))
        value_weights.append(weight)

    raw_margin, runner_label = _distinct_step07_margin(final_candidate, ranked, y_min, y_max)
    margin_saturation = max(1e-6, float(cfg("confidence", "distinct_margin_saturation", default=0.08)))
    margin_component = clip01(raw_margin / margin_saturation)
    margin_weight = max(0.0, float(cfg("confidence", "distinct_margin_weight", default=0.10)))
    values.append(max(1e-6, margin_component))
    value_weights.append(margin_weight)
    component_values["distinct_step_07_margin"] = margin_component

    # Agreement is reported and used only as a mild confidence modifier. It
    # cannot replace the Step 07 winner or dominate the actual evidence.
    votes = 0
    available_votes = 0
    rank_percentiles = []
    for name in VALIDATOR_NAMES:
        final_result = final_candidate.get("validators", {}).get(name, {})
        if not final_result.get("available") or final_result.get("score") is None:
            continue
        available_votes += 1
        rank_percentiles.append(float(final_result.get("rank_percentile", 0.0)))
        available_candidates = [
            candidate for candidate in ranked
            if candidate.get("validators", {}).get(name, {}).get("available")
            and candidate["validators"][name].get("score") is not None
        ]
        if available_candidates:
            validator_winner = max(available_candidates, key=lambda candidate: float(candidate["validators"][name]["score"]))
            if candidates_equivalent(final_candidate, validator_winner, y_min, y_max):
                votes += 1
    vote_ratio = votes / max(1, available_votes)
    mean_rank = float(np.mean(rank_percentiles)) if rank_percentiles else 0.0
    agreement = clip01(0.5 * vote_ratio + 0.5 * mean_rank)

    configured_total_weight = sum(max(0.0, weights[name]) for name in VALIDATOR_NAMES) + margin_weight
    available_weight = sum(value_weights)
    availability = available_weight / max(configured_total_weight, 1e-9)
    confidence = _weighted_geometric_mean(values, value_weights) * availability
    confidence *= 0.82 + 0.18 * agreement

    mask_result = final_candidate.get("validators", {}).get("evaluation_mask_support", {})
    if mask_result.get("available") and not mask_result.get("valid", False):
        confidence *= 0.45
    if not final_candidate.get("validation_valid", False):
        confidence *= 0.55
    confidence_percent = 100.0 * clip01(confidence)

    accepted = float(cfg("confidence", "accepted_min_percent", default=82.0))
    accepted_low = float(cfg("confidence", "accepted_low_confidence_min_percent", default=68.0))
    manual = float(cfg("confidence", "manual_review_min_percent", default=52.0))
    if confidence_percent >= accepted and final_candidate.get("validation_valid", False):
        decision = "accepted"
    elif confidence_percent >= accepted_low and final_candidate.get("validation_valid", False):
        decision = "accepted_low_confidence"
    elif confidence_percent >= manual:
        decision = "manual_review"
    else:
        decision = "recapture_required"

    return {
        "confidence_percent": float(confidence_percent),
        "calibration_status": "uncalibrated_estimated_confidence",
        "decision": decision,
        "confidence_components": component_values,
        "validator_agreement": float(agreement),
        "validator_agreement_percent": 100.0 * float(agreement),
        "validator_top_vote_ratio": float(vote_ratio),
        "validator_mean_rank_percentile": float(mean_rank),
        "validator_availability": float(availability),
        "validator_availability_percent": 100.0 * float(availability),
        "distinct_step_07_margin": float(raw_margin),
        "distinct_step_07_margin_percent": 100.0 * float(raw_margin),
        "distinct_margin_component": float(margin_component),
        "distinct_runner_up_label": runner_label,
    }
