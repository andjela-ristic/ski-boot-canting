from __future__ import annotations

import math

import numpy as np

from .context import cfg, clip01


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
    valid = bool(candidate.get("verification_valid", True))
    return {
        "available": True,
        "score": clip01(score),
        "step_07_valid": valid,
        "step_07_rank": int(candidate.get("verification_rank", candidate.get("source_rank", 0) or 0)),
        "mirror_symmetry_percent": _first_numeric(candidate, "mirror_symmetry_percent"),
    }


def fragment_evidence_validator(step06_candidate: dict | None) -> dict:
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
    for name, raw in [("fit_consistency", fit), ("fragment_alignment", alignment), ("above_below_balance", balance)]:
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
