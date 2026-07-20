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
    def formula_weight(key: str, default: float) -> float:
        return float(cfg("best_fit_selection", "formula_weights", key, default=default))

    def formula_bucket(key: str, default: float) -> float:
        return float(cfg("best_fit_selection", "formula_buckets", key, default=default))

    longest_interval_px = float(candidate.get("longest_merged_interval_px", 0.0))
    chain_continuity_ratio = float(candidate.get("chain_continuity_ratio", 0.0))
    longest_interval_bucket_px = max(1e-6, formula_bucket("longest_interval_px", 20.0))
    continuity_ratio_scale = max(1e-6, formula_bucket("continuity_ratio_scale", 20.0))
    longest_interval_bucket = float(int(longest_interval_px / longest_interval_bucket_px))
    continuity_bucket = float(int(chain_continuity_ratio * continuity_ratio_scale))
    endpoint_strengths = candidate_endpoint_strengths(candidate)
    paired_anchor_strength = float(endpoint_strengths["paired_anchor_strength"])
    paired_original_anchor_strength = float(endpoint_strengths["paired_original_anchor_strength"])
    paired_endpoint_coverage = float(endpoint_strengths["paired_endpoint_coverage"])
    paired_original_endpoint_coverage = float(endpoint_strengths["paired_original_endpoint_coverage"])
    total_reach_gap_px = float(candidate.get("top_reach_gap_px", 0.0)) + float(candidate.get("bottom_reach_gap_px", 0.0))
    hypothesis_x_ref_delta_px = float(candidate.get("hypothesis_x_ref_delta_px", 0.0))
    hypothesis_tilt_delta_deg = float(candidate.get("hypothesis_tilt_delta_deg", 0.0))
    side_clearance_row_ratio = float(candidate.get("side_clearance_row_ratio", 0.0))
    side_clearance_score = float(candidate.get("side_clearance_score", 0.0))
    return (
        formula_weight("has_top_bottom_anchor", 1000.0) * float(bool(candidate.get("has_top_bottom_anchor", False)))
        + formula_weight("has_top_anchor", 160.0) * float(bool(candidate.get("has_top_anchor", False)))
        + formula_weight("has_bottom_anchor", 160.0) * float(bool(candidate.get("has_bottom_anchor", False)))
        + formula_weight("paired_anchor_strength", 340.0) * paired_anchor_strength
        + formula_weight("paired_endpoint_coverage", 220.0) * paired_endpoint_coverage
        + formula_weight("longest_interval_bucket", 22.0) * longest_interval_bucket
        + formula_weight("continuity_bucket", 14.0) * continuity_bucket
        + formula_weight("longest_interval_px", 0.08) * longest_interval_px
        + formula_weight("chain_continuity_ratio", 14.0) * chain_continuity_ratio
        + formula_weight("has_top_bottom_original_anchor", 520.0)
        * float(bool(candidate.get("has_top_bottom_original_anchor", False)))
        + formula_weight("has_top_original_anchor", 110.0)
        * float(bool(candidate.get("has_top_original_anchor", False)))
        + formula_weight("has_bottom_original_anchor", 110.0)
        * float(bool(candidate.get("has_bottom_original_anchor", False)))
        + formula_weight("paired_original_anchor_strength", 260.0) * paired_original_anchor_strength
        + formula_weight("paired_original_endpoint_coverage", 180.0) * paired_original_endpoint_coverage
        + formula_weight("endpoint_anchor_score", 0.10) * float(candidate["endpoint_anchor_score"])
        + formula_weight("has_min_side_clearance", 180.0)
        * float(bool(candidate.get("has_min_side_clearance", False)))
        + formula_weight("side_clearance_row_ratio", 120.0) * side_clearance_row_ratio
        + formula_weight("side_clearance_score", 90.0) * side_clearance_score
        - formula_weight("outside_chain_length_ratio", 140.0)
        * float(candidate.get("outside_chain_length_ratio", 0.0))
        - formula_weight("outside_chain_fragment_ratio", 80.0)
        * float(candidate.get("outside_chain_fragment_ratio", 0.0))
        - formula_weight("total_reach_gap_px", 0.10) * total_reach_gap_px
        - formula_weight("chain_total_gap_px", 0.03) * float(candidate.get("chain_total_gap_px", 0.0))
        - formula_weight("gap_penalty", 0.12) * float(candidate["gap_penalty"])
        - formula_weight("merged_interval_count", 4.0) * float(candidate.get("merged_interval_count", 0))
        - formula_weight("adjusted_fragment_ratio", 80.0)
        * float(candidate.get("adjusted_fragment_ratio", 0.0))
        - formula_weight("support_adjustment_penalty", 0.18) * float(candidate["support_adjustment_penalty"])
        - formula_weight("length_weighted_mean_abs_support_shift_px", 0.08)
        * float(candidate.get("length_weighted_mean_abs_support_shift_px", 0.0))
        - formula_weight("max_abs_support_shift_px", 0.12) * float(candidate.get("max_abs_support_shift_px", 0.0))
        - formula_weight("outside_mask_penalty", 0.08) * float(candidate["outside_mask_penalty"])
        - formula_weight("hypothesis_x_ref_delta_px", 8.0) * hypothesis_x_ref_delta_px
        - formula_weight("hypothesis_tilt_delta_deg", 24.0) * hypothesis_tilt_delta_deg
        + formula_weight("score", 60.0) * float(candidate["score"])
    )

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
    result["hypothesis_score"] = float(hypothesis["score"])
    result["hypothesis_x_ref_delta_px"] = abs(float(result["x_ref"]) - float(hypothesis["x_ref"]))
    result["hypothesis_tilt_delta_deg"] = abs(float(result["tilt_deg"]) - float(hypothesis["tilt_deg"]))
    result["source_hypothesis_rank"] = int(hypothesis_rank)
    result["source_hypothesis_label"] = f"C{hypothesis_rank:02d}"
    result["selection_score"] = float(compute_best_fit_selection_score(result))
    return result

def candidate_selection_key(candidate: dict) -> tuple[float, ...]:
    longest_interval_px = float(candidate.get("longest_merged_interval_px", 0.0))
    chain_continuity_ratio = float(candidate.get("chain_continuity_ratio", 0.0))
    endpoint_strengths = candidate_endpoint_strengths(candidate)
    return (
        1 if bool(candidate.get("has_top_bottom_anchor", False)) else 0,
        1 if bool(candidate.get("has_top_bottom_original_anchor", False)) else 0,
        1 if bool(candidate.get("has_min_side_clearance", False)) else 0,
        int(endpoint_strengths["paired_anchor_strength"] * 100.0),
        int(endpoint_strengths["paired_endpoint_coverage"] * 100.0),
        int(endpoint_strengths["paired_original_anchor_strength"] * 100.0),
        int(endpoint_strengths["paired_original_endpoint_coverage"] * 100.0),
        float(candidate.get("side_clearance_row_ratio", 0.0)),
        float(candidate.get("side_clearance_score", 0.0)),
        int(longest_interval_px / 20.0),
        int(chain_continuity_ratio * 20.0),
        float(candidate["score"]),
        float(candidate["symmetry_score"]),
        float(candidate["roi_center_score"]),
        -float(candidate.get("hypothesis_x_ref_delta_px", 0.0)),
        -float(candidate.get("hypothesis_tilt_delta_deg", 0.0)),
        longest_interval_px,
        chain_continuity_ratio,
        -float(candidate.get("outside_chain_length_ratio", 0.0)),
        -float(candidate.get("outside_chain_fragment_ratio", 0.0)),
        -float(candidate.get("top_reach_gap_px", 0.0)),
        -float(candidate.get("bottom_reach_gap_px", 0.0)),
        -float(candidate.get("largest_gap_px", 0.0)),
        -float(candidate.get("chain_total_gap_px", 0.0)),
        float(candidate.get("total_merged_length_px", 0.0)),
        -float(candidate.get("merged_interval_count", 0)),
        -float(candidate.get("adjusted_fragment_ratio", 0.0)),
        -float(candidate.get("support_adjustment_penalty", 0.0)),
        -float(candidate.get("length_weighted_mean_abs_support_shift_px", 0.0)),
        -float(candidate.get("max_abs_support_shift_px", 0.0)),
        float(candidate["endpoint_anchor_score"]),
    )

def candidate_ranking_key(candidate: dict) -> tuple[float, ...]:
    return (
        *candidate_selection_key(candidate),
        float(candidate.get("selection_score", candidate.get("score", 0.0))),
    )
