from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .context import cfg, clip01
from .geometry import candidates_equivalent


def _first_numeric(source: dict, *keys: str) -> float | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def step07_symmetry_validator(candidate: dict) -> dict:
    score = _first_numeric(candidate, "verification_score")
    if score is None:
        percent = _first_numeric(candidate, "verification_percent", "symmetry_percent")
        score = None if percent is None else percent / 100.0
    if score is None:
        return {"available": False, "score": None, "reason": "step_07_score_missing"}
    return {
        "available": True,
        "score": clip01(score),
        "step_07_valid": bool(candidate.get("verification_valid", True)),
        "step_07_rank": int(candidate.get("verification_rank", candidate.get("source_rank", 0) or 0)),
        "mirror_symmetry_percent": _first_numeric(candidate, "mirror_symmetry_percent"),
        "bilateral_coverage_score": _first_numeric(candidate, "bilateral_coverage_score"),
    }


def _segment_numeric_score(segment: dict) -> float | None:
    value = _first_numeric(segment, "score", "mirror_symmetry_score")
    if value is None:
        percent = _first_numeric(segment, "mirror_symmetry_percent", "symmetry_percent")
        value = None if percent is None else percent / 100.0
    return None if value is None else clip01(value)


def _segment_weight(segment: dict) -> float:
    value = _first_numeric(segment, "evidence_weight")
    if value is None or value <= 0:
        value = 1.0
    return float(value)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[min(len(values) - 1, int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")))])


def _subset_aggregate(candidate: dict, included_indices: Iterable[int]) -> tuple[float | None, int]:
    segments = list(candidate.get("segments", []))
    selected = []
    weights = []
    for index in included_indices:
        if index < 0 or index >= len(segments):
            continue
        segment = segments[index]
        if not bool(segment.get("valid", True)):
            continue
        score = _segment_numeric_score(segment)
        if score is None:
            continue
        selected.append(score)
        weights.append(_segment_weight(segment))
    minimum = int(cfg("perturbation_stability", "minimum_segments_per_scenario", default=6))
    if len(selected) < minimum:
        return None, len(selected)
    values = np.asarray(selected, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w /= max(float(np.sum(w)), 1e-9)
    mean = float(np.sum(values * w))
    median = _weighted_median(values, w)
    lower = float(np.quantile(values, 0.25))
    return clip01(0.45 * mean + 0.35 * median + 0.20 * lower), len(selected)


def segment_consistency_validator(candidate: dict) -> dict:
    if not bool(cfg("segment_consistency", "enabled", default=True)):
        return {"available": False, "score": None, "reason": "disabled"}
    segments = list(candidate.get("segments", []))
    if not segments:
        # V2 Step 07 also stores aggregate zone values. They are not enough for
        # leave-one-segment-out stability, but can still support a basic score.
        zone_container = candidate.get("zones", {})
        zone_scores = [
            _first_numeric(zone_container.get(name, {}), "score")
            for name in ("top", "middle", "bottom")
        ]
        zone_scores = [value for value in zone_scores if value is not None]
        if len(zone_scores) < 3:
            return {"available": False, "score": None, "reason": "step_07_segments_missing"}
        harmonic = 3.0 / sum(1.0 / max(1e-6, value) for value in zone_scores)
        return {
            "available": True,
            "score": clip01(harmonic),
            "zone_scores": {name: float(value) for name, value in zip(("top", "middle", "bottom"), zone_scores)},
            "fallback_from_zone_aggregates": True,
        }

    values = []
    weights = []
    valid_indices = []
    for index, segment in enumerate(segments):
        if not bool(segment.get("valid", True)):
            continue
        score = _segment_numeric_score(segment)
        if score is None:
            continue
        values.append(score)
        weights.append(_segment_weight(segment))
        valid_indices.append(index)
    minimum = int(cfg("segment_consistency", "minimum_valid_segments", default=8))
    if len(values) < minimum:
        return {
            "available": False,
            "score": None,
            "reason": "insufficient_valid_step_07_segments",
            "valid_segment_count": int(len(values)),
            "segment_count": int(len(segments)),
        }

    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w /= max(float(np.sum(w)), 1e-9)
    median = _weighted_median(arr, w)
    lower_quantile = float(np.quantile(arr, float(cfg("segment_consistency", "lower_quantile", default=0.25))))
    valid_ratio = len(values) / max(1, len(segments))

    zone_indices = np.array_split(np.arange(len(segments), dtype=np.int32), 3)
    zone_names = ("top", "middle", "bottom")
    zone_results: dict[str, dict] = {}
    zone_scores = []
    min_zone = int(cfg("segment_consistency", "minimum_valid_segments_per_zone", default=2))
    for name, indexes in zip(zone_names, zone_indices):
        zone_values = []
        zone_weights = []
        for index in indexes:
            segment = segments[int(index)]
            if not bool(segment.get("valid", True)):
                continue
            score = _segment_numeric_score(segment)
            if score is None:
                continue
            zone_values.append(score)
            zone_weights.append(_segment_weight(segment))
        if len(zone_values) < min_zone:
            zone_results[name] = {"available": False, "score": 0.0, "valid_segment_count": len(zone_values)}
            zone_scores.append(1e-6)
            continue
        zv = np.asarray(zone_values, dtype=np.float64)
        zw = np.asarray(zone_weights, dtype=np.float64)
        zw /= max(float(np.sum(zw)), 1e-9)
        zscore = 0.55 * float(np.sum(zv * zw)) + 0.45 * _weighted_median(zv, zw)
        zone_results[name] = {"available": True, "score": clip01(zscore), "valid_segment_count": len(zone_values)}
        zone_scores.append(max(1e-6, clip01(zscore)))
    zone_harmonic = 3.0 / sum(1.0 / value for value in zone_scores)

    dispersion = float(np.std(arr))
    dispersion_scale = max(1e-6, float(cfg("segment_consistency", "dispersion_scale", default=0.24)))
    dispersion_score = float(math.exp(-dispersion / dispersion_scale))
    component_weights = np.asarray([
        float(cfg("segment_consistency", "median_weight", default=0.24)),
        float(cfg("segment_consistency", "lower_quantile_weight", default=0.26)),
        float(cfg("segment_consistency", "zone_harmonic_weight", default=0.34)),
        float(cfg("segment_consistency", "valid_ratio_weight", default=0.16)),
    ], dtype=np.float64)
    component_weights /= max(float(np.sum(component_weights)), 1e-9)
    base = float(np.dot(component_weights, np.asarray([median, lower_quantile, zone_harmonic, valid_ratio])))
    score = base * (0.75 + 0.25 * dispersion_score)
    return {
        "available": True,
        "score": clip01(score),
        "weighted_median_score": clip01(median),
        "lower_quantile_score": clip01(lower_quantile),
        "zone_harmonic_score": clip01(zone_harmonic),
        "valid_segment_ratio": float(valid_ratio),
        "valid_segment_count": int(len(values)),
        "segment_count": int(len(segments)),
        "segment_score_std": float(dispersion),
        "dispersion_score": clip01(dispersion_score),
        "zones": zone_results,
    }


def fragment_evidence_validator(step06_candidate: dict | None) -> dict:
    if not bool(cfg("fragment_evidence", "enabled", default=True)):
        return {"available": False, "score": None, "reason": "disabled"}
    if step06_candidate is None:
        return {"available": False, "score": None, "reason": "step_06_candidate_missing"}
    span = _first_numeric(step06_candidate, "chain_span_ratio")
    coverage = _first_numeric(step06_candidate, "unique_vertical_coverage", "vertical_coverage_score")
    fit = _first_numeric(step06_candidate, "fit_consistency_score")
    alignment = _first_numeric(step06_candidate, "fragment_alignment_score")
    balance = _first_numeric(step06_candidate, "above_below_balance_score")
    values = []
    breakdown = {}
    if span is not None:
        value = clip01(span / max(float(cfg("fragment_evidence", "span_saturation", default=0.65)), 1e-6))
        values.append(value)
        breakdown["chain_span"] = value
    if coverage is not None:
        value = clip01(coverage / max(float(cfg("fragment_evidence", "coverage_saturation", default=0.35)), 1e-6))
        values.append(value)
        breakdown["unique_vertical_coverage"] = value
    for name, raw in (("fit_consistency", fit), ("fragment_alignment", alignment), ("above_below_balance", balance)):
        if raw is not None:
            value = clip01(raw)
            values.append(value)
            breakdown[name] = value
    minimum = int(cfg("fragment_evidence", "minimum_available_metrics", default=3))
    if len(values) < minimum:
        return {"available": False, "score": None, "reason": "insufficient_step_06_metrics", "metrics": breakdown}
    floor = 1e-6
    score = float(math.exp(float(np.mean(np.log(np.maximum(np.asarray(values), floor))))))
    return {
        "available": True,
        "score": clip01(score),
        "metrics": breakdown,
        "selected_fragment_count": int(step06_candidate.get("selected_fragment_count", 0) or 0),
        "support_distribution_mode": step06_candidate.get("support_distribution_mode"),
    }


def compute_segment_perturbation_stability(candidates: list[dict], y_min: int, y_max: int) -> tuple[dict[str, dict], list[dict]]:
    if not bool(cfg("perturbation_stability", "enabled", default=True)):
        return ({str(candidate["candidate_label"]): {"available": False, "score": None, "reason": "disabled"} for candidate in candidates}, [])
    max_segments = max((len(candidate.get("segments", [])) for candidate in candidates), default=0)
    if max_segments < 3:
        return ({str(candidate["candidate_label"]): {"available": False, "score": None, "reason": "step_07_segments_missing"} for candidate in candidates}, [])

    all_indices = tuple(range(max_segments))
    scenarios: list[tuple[str, tuple[int, ...]]] = [("all_segments", all_indices)]
    if bool(cfg("perturbation_stability", "include_leave_one_segment_out", default=True)):
        scenarios.extend((f"without_segment_{i + 1:02d}", tuple(j for j in all_indices if j != i)) for i in all_indices)
    zones = np.array_split(np.arange(max_segments, dtype=np.int32), 3)
    if bool(cfg("perturbation_stability", "include_leave_one_zone_out", default=True)):
        for name, zone in zip(("top", "middle", "bottom"), zones):
            removed = set(int(i) for i in zone)
            scenarios.append((f"without_{name}_zone", tuple(i for i in all_indices if i not in removed)))
    if bool(cfg("perturbation_stability", "include_odd_even_subsets", default=True)):
        scenarios.append(("odd_segments", tuple(i for i in all_indices if i % 2 == 0)))
        scenarios.append(("even_segments", tuple(i for i in all_indices if i % 2 == 1)))

    per_candidate = {
        str(candidate["candidate_label"]): {
            "available": True,
            "equivalent_winner_count": 0,
            "rank_percentiles": [],
            "valid_scenario_count": 0,
        }
        for candidate in candidates
    }
    scenario_results = []
    for scenario_name, included in scenarios:
        scored = []
        for candidate in candidates:
            aggregate, valid_count = _subset_aggregate(candidate, included)
            if aggregate is None:
                continue
            step07 = _first_numeric(candidate, "verification_score") or 0.0
            scored.append((candidate, float(aggregate), int(valid_count), float(step07)))
        if not scored:
            scenario_results.append({"scenario": scenario_name, "available": False, "reason": "no_candidate_has_enough_segments"})
            continue
        scored.sort(key=lambda item: (item[1], item[3], -int(item[0].get("source_rank", 9999))), reverse=True)
        winner = scored[0][0]
        denominator = max(1, len(scored) - 1)
        rank_map = {str(item[0]["candidate_label"]): rank for rank, item in enumerate(scored, start=1)}
        for candidate in candidates:
            label = str(candidate["candidate_label"])
            if label not in rank_map:
                continue
            state = per_candidate[label]
            state["valid_scenario_count"] += 1
            rank = rank_map[label]
            state["rank_percentiles"].append(1.0 - (rank - 1) / denominator if len(scored) > 1 else 1.0)
            if candidates_equivalent(candidate, winner, y_min, y_max):
                state["equivalent_winner_count"] += 1
        scenario_results.append({
            "scenario": scenario_name,
            "available": True,
            "included_segment_indices": [int(i) for i in included],
            "winner_label": str(winner["candidate_label"]),
            "ranking": [
                {"candidate_label": str(item[0]["candidate_label"]), "scenario_score": float(item[1]), "valid_segment_count": int(item[2])}
                for item in scored
            ],
        })

    results: dict[str, dict] = {}
    for candidate in candidates:
        label = str(candidate["candidate_label"])
        state = per_candidate[label]
        count = int(state["valid_scenario_count"])
        if count == 0:
            results[label] = {"available": False, "score": None, "reason": "no_valid_perturbation_scenarios"}
            continue
        win_ratio = int(state["equivalent_winner_count"]) / count
        mean_rank = float(np.mean(state["rank_percentiles"])) if state["rank_percentiles"] else 0.0
        score = math.sqrt(max(0.0, win_ratio * mean_rank))
        results[label] = {
            "available": True,
            "score": clip01(score),
            "equivalent_winner_ratio": float(win_ratio),
            "mean_rank_percentile": clip01(mean_rank),
            "equivalent_winner_count": int(state["equivalent_winner_count"]),
            "valid_scenario_count": int(count),
        }
    return results, scenario_results
