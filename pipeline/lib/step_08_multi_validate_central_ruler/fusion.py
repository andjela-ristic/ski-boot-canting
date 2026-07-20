from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from .context import cfg, clip01
from .geometry import axis_tilt_deg, mean_axis_distance


def _geometric_mean(values: list[float], floor: float = 1e-6) -> float:
    if not values:
        return 0.0
    array = np.maximum(np.asarray(values, dtype=np.float64), floor)
    return float(math.exp(float(np.mean(np.log(array)))))


def _score_table(candidates: list[dict], validators: list[str]) -> None:
    count = len(candidates)
    for validator in validators:
        available = [
            (index, float(candidate["validators"][validator]["score"]))
            for index, candidate in enumerate(candidates)
            if validator in candidate["validators"]
            and candidate["validators"][validator].get("available")
            and candidate["validators"][validator].get("score") is not None
        ]
        available.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        ranks = {index: rank for rank, (index, _) in enumerate(available, start=1)}
        denominator = max(1, len(available))
        for index, candidate in enumerate(candidates):
            result = candidate["validators"].get(validator)
            if not result or not result.get("available") or result.get("score") is None:
                continue
            rank = ranks[index]
            result["rank"] = int(rank)
            result["rank_percentile"] = clip01(1.0 - (rank - 1) / denominator)


def compute_candidate_fusion(candidates: list[dict], validator_names: list[str] | None = None) -> list[dict]:
    validators = list(validator_names or cfg("fusion", "validator_names", default=[]))
    _score_table(candidates, validators)
    minimum = int(cfg("fusion", "minimum_available_validators", default=4))
    floor = float(cfg("fusion", "absolute_score_floor", default=0.0001))
    invalid_scale = float(cfg("fusion", "invalid_candidate_scale", default=0.45))
    require_step07_valid = bool(cfg("fusion", "require_step_07_valid", default=True))
    for candidate in candidates:
        absolute, ranks, names = [], [], []
        for validator in validators:
            result = candidate["validators"].get(validator)
            if not result or not result.get("available") or result.get("score") is None:
                continue
            absolute.append(max(floor, clip01(result["score"])))
            ranks.append(max(floor, clip01(result.get("rank_percentile", 0.0))))
            names.append(validator)
        evidence = _geometric_mean(absolute, floor)
        rank_consensus = float(np.mean(ranks)) if ranks else 0.0
        score = math.sqrt(max(0.0, evidence * rank_consensus))
        rejection = []
        if len(names) < minimum:
            rejection.append("insufficient_available_validators")
        step07 = candidate["validators"].get("step_07_symmetry", {})
        if require_step07_valid and step07.get("available") and not step07.get("step_07_valid", True):
            rejection.append("step_07_candidate_invalid")
        balance = candidate["validators"].get("roi_balance", {})
        if balance.get("available") and balance.get("axis_inside_ratio", 1.0) < float(cfg("roi_balance", "minimum_axis_inside_ratio", default=0.70)):
            rejection.append("axis_leaves_roi_core")
        valid = not rejection
        if not valid:
            score *= invalid_scale
        candidate.update({
            "available_validator_count": int(len(names)),
            "available_validators": names,
            "evidence_quality": clip01(evidence),
            "rank_consensus": clip01(rank_consensus),
            "multi_validation_score": clip01(score),
            "multi_validation_percent": 100.0 * clip01(score),
            "validation_valid": bool(valid),
            "rejection_reasons": rejection,
        })
    return sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate["validation_valid"]),
            float(candidate["multi_validation_score"]),
            float(candidate["evidence_quality"]),
            float(candidate["rank_consensus"]),
            -int(candidate.get("source_rank", 9999)),
        ),
        reverse=True,
    )


def candidates_equivalent(first: dict, second: dict, y_min: int, y_max: int) -> bool:
    distance = mean_axis_distance(
        first,
        second,
        y_min,
        y_max,
        int(cfg("equivalence", "sample_count", default=24)),
    )
    tilt = abs(axis_tilt_deg(first) - axis_tilt_deg(second))
    return (
        distance <= float(cfg("equivalence", "max_mean_axis_distance_px", default=5.0))
        and tilt <= float(cfg("equivalence", "max_tilt_difference_deg", default=0.25))
    )


def _winner_for_subset(candidates: list[dict], validator_subset: list[str]) -> dict:
    copied = deepcopy(candidates)
    ranked = compute_candidate_fusion(copied, validator_subset)
    return ranked[0]


def compute_confidence(ranked: list[dict], validator_names: list[str], y_min: int, y_max: int) -> dict:
    winner = ranked[0]
    distinct_runner = None
    for candidate in ranked[1:]:
        if not candidates_equivalent(winner, candidate, y_min, y_max):
            distinct_runner = candidate
            break
    if distinct_runner is None:
        raw_margin = float(winner["multi_validation_score"])
        margin_component = 1.0
    else:
        raw_margin = max(0.0, float(winner["multi_validation_score"]) - float(distinct_runner["multi_validation_score"]))
        scale = max(1e-6, float(cfg("confidence", "distinct_margin_saturation", default=0.10)))
        margin_component = clip01(raw_margin / scale)

    top_votes = 0
    available_votes = 0
    winner_rank_values = []
    for validator in validator_names:
        result = winner["validators"].get(validator)
        if not result or not result.get("available"):
            continue
        available_votes += 1
        winner_rank_values.append(float(result.get("rank_percentile", 0.0)))
        top_candidate = max(
            (candidate for candidate in ranked if candidate["validators"].get(validator, {}).get("available")),
            key=lambda candidate: float(candidate["validators"][validator]["score"]),
            default=None,
        )
        if top_candidate is not None and candidates_equivalent(winner, top_candidate, y_min, y_max):
            top_votes += 1
    vote_ratio = top_votes / max(1, available_votes)
    mean_rank = float(np.mean(winner_rank_values)) if winner_rank_values else 0.0
    agreement = math.sqrt(max(0.0, mean_rank * (0.5 + 0.5 * vote_ratio)))

    scenarios: list[tuple[str, list[str]]] = [("all_validators", list(validator_names))]
    if len(validator_names) > 2:
        scenarios.extend((f"without_{name}", [v for v in validator_names if v != name]) for name in validator_names)
    stable = 0
    scenario_results = []
    for scenario_name, subset in scenarios:
        scenario_winner = _winner_for_subset(ranked, subset)
        equivalent = candidates_equivalent(winner, scenario_winner, y_min, y_max)
        stable += int(equivalent)
        scenario_results.append({
            "scenario": scenario_name,
            "validators": subset,
            "winner_label": scenario_winner["candidate_label"],
            "equivalent_to_final_winner": bool(equivalent),
        })
    ensemble_stability = stable / max(1, len(scenarios))

    availability = len(winner.get("available_validators", [])) / max(1, len(validator_names))
    components = [
        max(1e-6, float(winner["evidence_quality"])),
        max(1e-6, clip01(agreement)),
        max(1e-6, clip01(margin_component)),
        max(1e-6, clip01(ensemble_stability)),
    ]
    confidence = 100.0 * availability * _geometric_mean(components)
    confidence = float(np.clip(confidence, 0.0, 100.0))
    if not winner.get("validation_valid", False):
        confidence *= 0.50

    accepted = float(cfg("confidence", "accepted_min_percent", default=82.0))
    accepted_low = float(cfg("confidence", "accepted_low_confidence_min_percent", default=68.0))
    manual = float(cfg("confidence", "manual_review_min_percent", default=52.0))
    if confidence >= accepted and winner.get("validation_valid"):
        decision = "accepted"
    elif confidence >= accepted_low and winner.get("validation_valid"):
        decision = "accepted_low_confidence"
    elif confidence >= manual:
        decision = "manual_review"
    else:
        decision = "recapture_required"

    return {
        "confidence_percent": float(confidence),
        "calibration_status": "uncalibrated_estimated_confidence",
        "decision": decision,
        "evidence_quality": float(winner["evidence_quality"]),
        "validator_agreement": clip01(agreement),
        "validator_agreement_percent": 100.0 * clip01(agreement),
        "validator_top_vote_ratio": float(vote_ratio),
        "winner_margin": float(raw_margin),
        "winner_margin_percent": 100.0 * float(raw_margin),
        "distinct_margin_component": clip01(margin_component),
        "ensemble_stability": float(ensemble_stability),
        "ensemble_stability_percent": 100.0 * float(ensemble_stability),
        "validator_availability": float(availability),
        "validator_availability_percent": 100.0 * float(availability),
        "distinct_runner_up_label": None if distinct_runner is None else distinct_runner["candidate_label"],
        "leave_one_validator_out": scenario_results,
    }
