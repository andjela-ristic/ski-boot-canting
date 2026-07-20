from __future__ import annotations

import numpy as np

from .context import cfg
from .calculations import candidate_endpoint_strengths


def mean_axis_distance_px(
    candidate_a: dict,
    candidate_b: dict,
    roi_profile: dict | None = None,
    probe_rows: np.ndarray | None = None,
) -> float:
    if probe_rows is None:
        if roi_profile is None:
            raise ValueError("roi_profile or probe_rows must be provided")
        probe_rows = np.linspace(
            float(roi_profile["trimmed_y_min"]),
            float(roi_profile["trimmed_y_max"]),
            8,
            dtype=np.float64,
        )
    delta_a = float(candidate_a["a"]) - float(candidate_b["a"])
    delta_b = float(candidate_a["b"]) - float(candidate_b["b"])
    return float(np.mean(np.abs(delta_a * probe_rows + delta_b)))

def deduplicate_candidates(
    candidates: list[dict],
    roi_profile: dict,
    max_candidates: int | None = None,
    sort_key=None,
) -> list[dict]:
    kept: list[dict] = []
    max_mean_axis_distance_px = float(cfg("candidate_deduplication", "max_mean_axis_distance_px", default=8.0))
    max_angle_difference_deg = float(cfg("candidate_deduplication", "max_angle_difference_deg", default=0.45))
    resolved_max_candidates = (
        int(cfg("candidate_deduplication", "max_saved_candidates", default=8))
        if max_candidates is None
        else max(0, int(max_candidates))
    )
    if resolved_max_candidates <= 0 or not candidates:
        return []
    probe_rows = np.linspace(
        float(roi_profile["trimmed_y_min"]),
        float(roi_profile["trimmed_y_max"]),
        8,
        dtype=np.float64,
    )
    angle_bucket_size = max(1e-6, max_angle_difference_deg)
    x_bucket_size = max(1e-6, max_mean_axis_distance_px)
    kept_buckets: dict[tuple[int, int], list[dict]] = {}

    if sort_key is None:
        ordered_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    else:
        ordered_candidates = sorted(candidates, key=sort_key, reverse=True)

    for candidate in ordered_candidates:
        is_duplicate = False
        angle_bucket = int(round(float(candidate["tilt_deg"]) / angle_bucket_size))
        x_bucket = int(round(float(candidate["x_ref"]) / x_bucket_size))
        for angle_offset in (-1, 0, 1):
            if is_duplicate:
                break
            for x_offset in (-1, 0, 1):
                bucket_key = (angle_bucket + angle_offset, x_bucket + x_offset)
                for existing in kept_buckets.get(bucket_key, []):
                    if abs(float(candidate["tilt_deg"]) - float(existing["tilt_deg"])) > max_angle_difference_deg:
                        continue
                    if (
                        mean_axis_distance_px(candidate, existing, probe_rows=probe_rows)
                        <= max_mean_axis_distance_px
                    ):
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
        if not is_duplicate:
            kept.append(candidate)
            kept_buckets.setdefault((angle_bucket, x_bucket), []).append(candidate)
            if len(kept) >= resolved_max_candidates:
                break

    return kept[:resolved_max_candidates]

def sort_candidates(
    candidates: list[dict],
    sort_key=None,
) -> list[dict]:
    if sort_key is None:
        return sorted(candidates, key=lambda item: item["score"], reverse=True)
    return sorted(candidates, key=sort_key, reverse=True)

def unique_candidates_by_axis(
    candidates: list[dict],
) -> list[dict]:
    unique: list[dict] = []
    seen_axes: set[tuple[float, float]] = set()
    for candidate in candidates:
        axis_key = (
            round(float(candidate["x_ref"]), 6),
            round(float(candidate["tilt_deg"]), 6),
        )
        if axis_key in seen_axes:
            continue
        seen_axes.add(axis_key)
        unique.append(candidate)
    return unique

def filter_candidates_by_score_ratio(
    candidates: list[dict],
    score_ratio_floor: float,
) -> list[dict]:
    if not candidates:
        return []
    if score_ratio_floor <= 0.0:
        return candidates

    best_score = float(candidates[0]["score"])
    min_score = best_score * score_ratio_floor
    kept = [candidate for candidate in candidates if float(candidate["score"]) >= min_score]
    return kept if kept else candidates[:1]

def select_diverse_candidates_by_angle(
    candidates: list[dict],
    max_candidates: int,
    angle_bucket_deg: float,
    max_per_bucket: int,
) -> list[dict]:
    if max_candidates <= 0 or not candidates:
        return []
    if angle_bucket_deg <= 0.0 or max_per_bucket <= 0:
        return candidates[:max_candidates]

    selected: list[dict] = []
    selected_ids: set[int] = set()
    bucket_counts: dict[int, int] = {}

    for candidate in candidates:
        bucket = int(round(float(candidate["tilt_deg"]) / angle_bucket_deg))
        current_count = bucket_counts.get(bucket, 0)
        if current_count >= max_per_bucket:
            continue

        selected.append(candidate)
        selected_ids.add(id(candidate))
        bucket_counts[bucket] = current_count + 1
        if len(selected) >= max_candidates:
            return selected

    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    return selected

def select_diverse_candidates(
    candidates: list[dict],
    max_candidates: int,
    angle_bucket_deg: float,
    max_per_angle_bucket: int,
    x_bucket_px: float,
    max_per_x_bucket: int,
) -> list[dict]:
    if max_candidates <= 0 or not candidates:
        return []
    if (
        angle_bucket_deg <= 0.0
        or max_per_angle_bucket <= 0
        or x_bucket_px <= 0.0
        or max_per_x_bucket <= 0
    ):
        return candidates[:max_candidates]

    selected: list[dict] = []
    selected_ids: set[int] = set()
    angle_bucket_counts: dict[int, int] = {}
    x_bucket_counts: dict[int, int] = {}

    for candidate in candidates:
        angle_bucket = int(round(float(candidate["tilt_deg"]) / angle_bucket_deg))
        x_bucket = int(round(float(candidate["x_ref"]) / x_bucket_px))
        current_angle_count = angle_bucket_counts.get(angle_bucket, 0)
        current_x_count = x_bucket_counts.get(x_bucket, 0)
        if current_angle_count >= max_per_angle_bucket or current_x_count >= max_per_x_bucket:
            continue

        selected.append(candidate)
        selected_ids.add(id(candidate))
        angle_bucket_counts[angle_bucket] = current_angle_count + 1
        x_bucket_counts[x_bucket] = current_x_count + 1
        if len(selected) >= max_candidates:
            return selected

    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    return selected

def compute_best_fit_selection_score(candidate: dict) -> float:
    """Return the single normalized score used for final ranking."""
    return float(candidate.get("final_score", candidate.get("score", 0.0)))

def annotate_candidate_selection(
    candidate: dict,
    hypothesis: dict,
    hypothesis_rank: int,
    stage_name: str,
) -> dict:
    result = dict(candidate)
    result["search_stage"] = stage_name
    result["hypothesis_x_ref"] = float(hypothesis["x_ref"])
    result["hypothesis_tilt_deg"] = float(hypothesis["tilt_deg"])
    result["hypothesis_score"] = float(hypothesis.get("final_score", hypothesis.get("score", 0.0)))
    result["hypothesis_x_ref_delta_px"] = abs(float(result["x_ref"]) - float(hypothesis["x_ref"]))
    result["hypothesis_tilt_delta_deg"] = abs(float(result["tilt_deg"]) - float(hypothesis["tilt_deg"]))
    result["source_hypothesis_rank"] = int(hypothesis_rank)
    result["source_hypothesis_label"] = f"C{hypothesis_rank:02d}"
    result["selection_score"] = float(compute_best_fit_selection_score(result))
    return result

def candidate_selection_key(candidate: dict) -> tuple[float, ...]:
    return candidate_ranking_key(candidate)

def candidate_ranking_key(candidate: dict) -> tuple[float, ...]:
    """Final score is authoritative; remaining fields are deterministic tie-breakers."""
    return (
        float(candidate.get("final_score", candidate.get("selection_score", candidate.get("score", 0.0)))),
        1.0 if bool(candidate.get("validation_passed", True)) else 0.0,
        float(candidate.get("geometry_score", 0.0)),
        float(candidate.get("mirror_symmetry_score", 0.5)),
        float(candidate.get("unique_vertical_coverage", candidate.get("vertical_coverage_score", 0.0))),
        float(candidate.get("chain_span_ratio", 0.0)),
        float(candidate.get("chain_continuity_ratio", 0.0)),
        -float(candidate.get("fit_rmse_px", 1e9)),
        -float(candidate.get("largest_gap_px", 1e9)),
    )

