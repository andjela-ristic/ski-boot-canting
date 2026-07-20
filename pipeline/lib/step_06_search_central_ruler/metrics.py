from __future__ import annotations

import numpy as np

from .context import cfg, clip01
from .geometry import line_x_at_y, make_axis_signature
from .calculations import (
    build_support_analysis,
    compute_gap_penalty,
    has_endpoint_anchor,
    make_support_signature,
    select_support_fragments,
)


def calculate_symmetry_score(axis: dict[str, float], roi_profile: dict) -> float:
    return float(compute_row_balance_metrics(axis, roi_profile)["symmetry_score"])


def compute_vertical_coverage(
    selected_support: list[dict],
    y_min: float,
    y_max: float,
) -> dict[str, float | int]:
    bin_count = int(cfg("coverage", "bin_count", default=12))
    total_span = max(1.0, float(y_max) - float(y_min))
    supported_bins = np.zeros(bin_count, dtype=bool)

    for item in selected_support:
        line = item["line"]
        start_bin = int(np.floor((float(line["y_min"]) - float(y_min)) / total_span * bin_count))
        end_bin = int(np.floor((float(line["y_max"]) - float(y_min)) / total_span * bin_count))
        start_bin = max(0, min(bin_count - 1, start_bin))
        end_bin = max(0, min(bin_count - 1, end_bin))
        supported_bins[start_bin : end_bin + 1] = True

    supported_bin_count = int(np.count_nonzero(supported_bins))
    return {
        "bin_count": bin_count,
        "supported_bin_count": supported_bin_count,
        "coverage_score": clip01(supported_bin_count / max(1, bin_count)),
    }

def compute_row_balance_metrics(axis: dict[str, float], roi_profile: dict) -> dict[str, float]:
    return compute_row_balance_metrics_cached(axis, roi_profile, row_metrics_cache=None)

def compute_row_balance_metrics_cached(
    axis: dict[str, float],
    roi_profile: dict,
    row_metrics_cache: dict | None = None,
) -> dict[str, float]:
    cache_key = None
    if row_metrics_cache is not None:
        cache_key = make_axis_signature(axis)
        cached_metrics = row_metrics_cache.get(cache_key)
        if cached_metrics is not None:
            return cached_metrics

    sample_step_px = int(cfg("roi_profile", "sample_step_px", default=10))
    trimmed_rows = roi_profile["trimmed_rows"]
    sampled_rows = trimmed_rows[::max(1, sample_step_px)]
    if sampled_rows.size == 0:
        sampled_rows = trimmed_rows

    left_bounds = roi_profile["left_bounds"]
    right_bounds = roi_profile["right_bounds"]
    row_widths = roi_profile["row_widths"]
    center_fit = roi_profile["center_fit"]
    min_side_ratio_threshold = float(
        cfg("best_fit_selection", "minimal_symmetry", "min_side_ratio", default=0.035)
    )
    min_clearance_row_ratio_threshold = float(
        cfg("best_fit_selection", "minimal_symmetry", "min_clearance_row_ratio", default=0.65)
    )
    target_side_ratio = float(
        cfg("best_fit_selection", "minimal_symmetry", "target_side_ratio", default=0.12)
    )
    total_rows = len(sampled_rows)
    row_indices = sampled_rows.astype(np.int32, copy=False)
    widths = row_widths[row_indices].astype(np.float64, copy=False)
    valid_width_mask = widths > 0.0
    row_values = row_indices.astype(np.float64, copy=False)
    axis_x = float(axis["a"]) * row_values + float(axis["b"])
    left = left_bounds[row_indices].astype(np.float64, copy=False)
    right = right_bounds[row_indices].astype(np.float64, copy=False)
    inside_mask = valid_width_mask & (axis_x >= left) & (axis_x <= right)
    inside_rows = int(np.count_nonzero(inside_mask))

    if inside_rows > 0:
        inside_axis_x = axis_x[inside_mask]
        inside_left = left[inside_mask]
        inside_right = right[inside_mask]
        inside_widths = np.maximum(1.0, widths[inside_mask])
        left_widths = inside_axis_x - inside_left
        right_widths = inside_right - inside_axis_x
        min_side_ratios = np.minimum(left_widths, right_widths) / inside_widths
        symmetry_errors = np.abs((inside_axis_x - inside_left) - (inside_right - inside_axis_x)) / inside_widths
        center_x = float(center_fit["a"]) * row_values[inside_mask] + float(center_fit["b"])
        center_errors = np.abs(inside_axis_x - center_x) / np.maximum(1.0, inside_widths * 0.5)
        symmetry_score = clip01(1.0 - float(np.median(symmetry_errors)))
        center_score = clip01(1.0 - float(np.median(center_errors)))
        median_min_side_ratio = float(np.median(min_side_ratios))
        side_clearance_row_ratio = float(np.mean(min_side_ratios >= min_side_ratio_threshold))
        side_clearance_score = clip01(
            (median_min_side_ratio - min_side_ratio_threshold)
            / max(1e-6, target_side_ratio - min_side_ratio_threshold)
        )
        has_min_side_clearance = bool(side_clearance_row_ratio >= min_clearance_row_ratio_threshold)
    else:
        symmetry_score = 0.0
        center_score = 0.0
        median_min_side_ratio = 0.0
        side_clearance_row_ratio = 0.0
        side_clearance_score = 0.0
        has_min_side_clearance = False

    outside_mask_penalty = clip01(1.0 - inside_rows / max(1, total_rows))

    result = {
        "sampled_row_count": int(total_rows),
        "rows_inside_mask_count": int(inside_rows),
        "outside_mask_penalty": float(outside_mask_penalty),
        "symmetry_score": float(symmetry_score),
        "roi_center_score": float(center_score),
        "median_min_side_ratio": float(median_min_side_ratio),
        "side_clearance_row_ratio": float(side_clearance_row_ratio),
        "side_clearance_score": float(side_clearance_score),
        "has_min_side_clearance": bool(has_min_side_clearance),
    }
    if row_metrics_cache is not None and cache_key is not None:
        row_metrics_cache[cache_key] = result
    return result

def summarize_candidate_from_support(
    axis: dict[str, float],
    selected_support: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
    support_cache: dict | None = None,
    row_metrics_cache: dict | None = None,
    candidate_summary_cache: dict | None = None,
) -> dict:
    cache_key = None
    if candidate_summary_cache is not None:
        cache_key = (
            make_axis_signature(axis),
            make_support_signature(selected_support),
        )
        cached_candidate = candidate_summary_cache.get(cache_key)
        if cached_candidate is not None:
            return dict(cached_candidate)

    support_analysis = build_support_analysis(selected_support, roi_profile, support_cache=support_cache)
    selected_total_length_px = float(support_analysis["selected_total_length_px"])
    selected_total_support_strength = float(support_analysis["selected_total_support_strength"])
    fragment_support_score = clip01(selected_total_support_strength / max(1.0, total_available_length_px))
    chain_support = support_analysis["chain_support"]
    chain_total_length_px = float(support_analysis["chain_total_length_px"])
    outside_chain_length_ratio = float(support_analysis["outside_chain_length_ratio"])
    outside_chain_fragment_ratio = float(support_analysis["outside_chain_fragment_ratio"])

    coverage_metrics = compute_vertical_coverage(
        selected_support=selected_support,
        y_min=float(roi_profile["trimmed_y_min"]),
        y_max=float(roi_profile["trimmed_y_max"]),
    )
    gap_metrics = compute_gap_penalty(
        selected_support=selected_support,
        y_min=float(roi_profile["trimmed_y_min"]),
        y_max=float(roi_profile["trimmed_y_max"]),
    )
    endpoint_metrics = support_analysis["endpoint_metrics"]
    chain_metrics = support_analysis["chain_metrics"]
    row_metrics = compute_row_balance_metrics_cached(
        axis=axis,
        roi_profile=roi_profile,
        row_metrics_cache=row_metrics_cache,
    )
    adjustment_metrics = support_analysis["adjustment_metrics"]
    has_top_anchor = has_endpoint_anchor(endpoint_metrics, "top", require_original=False)
    has_bottom_anchor = has_endpoint_anchor(endpoint_metrics, "bottom", require_original=False)
    has_top_original_anchor = has_endpoint_anchor(endpoint_metrics, "top", require_original=True)
    has_bottom_original_anchor = has_endpoint_anchor(endpoint_metrics, "bottom", require_original=True)

    score = (
        float(cfg("scoring", "fragment_support_weight", default=0.34)) * fragment_support_score
        + float(cfg("scoring", "vertical_coverage_weight", default=0.22)) * float(coverage_metrics["coverage_score"])
        + float(cfg("scoring", "symmetry_weight", default=0.24)) * float(row_metrics["symmetry_score"])
        + float(cfg("scoring", "roi_center_weight", default=0.08)) * float(row_metrics["roi_center_score"])
        + float(cfg("scoring", "endpoint_anchor_weight", default=0.10))
        * float(endpoint_metrics["endpoint_anchor_score"])
        - float(cfg("scoring", "gap_penalty_weight", default=0.14)) * float(gap_metrics["gap_penalty"])
        - float(cfg("scoring", "outside_mask_penalty_weight", default=0.16)) * float(row_metrics["outside_mask_penalty"])
        - float(cfg("scoring", "support_adjustment_penalty_weight", default=0.08))
        * float(adjustment_metrics["adjustment_penalty"])
    )

    if len(selected_support) < int(cfg("search", "min_support_fragments", default=2)):
        score -= float(cfg("scoring", "low_support_penalty", default=0.20))
    if int(coverage_metrics["supported_bin_count"]) < int(cfg("coverage", "min_supported_bins", default=4)):
        score -= float(cfg("scoring", "low_coverage_penalty", default=0.20))

    result = {
        **axis,
        "score": float(score),
        "selected_support": selected_support,
        "selected_fragment_line_indices": [int(item["line"]["line_index"]) for item in selected_support],
        "selected_fragment_count": len(selected_support),
        "selected_total_length_px": float(selected_total_length_px),
        "selected_total_support_strength": float(selected_total_support_strength),
        "fragment_support_score": float(fragment_support_score),
        "vertical_coverage_score": float(coverage_metrics["coverage_score"]),
        "supported_bin_count": int(coverage_metrics["supported_bin_count"]),
        "bin_count": int(coverage_metrics["bin_count"]),
        "gap_penalty": float(gap_metrics["gap_penalty"]),
        "largest_gap_px": float(gap_metrics["largest_gap_px"]),
        "support_y_min": float(endpoint_metrics["support_y_min"]),
        "support_y_max": float(endpoint_metrics["support_y_max"]),
        "support_span_px": float(endpoint_metrics["support_span_px"]),
        "endpoint_band_px": float(endpoint_metrics["endpoint_band_px"]),
        "top_endpoint_coverage": float(endpoint_metrics["top_endpoint_coverage"]),
        "bottom_endpoint_coverage": float(endpoint_metrics["bottom_endpoint_coverage"]),
        "top_endpoint_alignment_score": float(endpoint_metrics["top_endpoint_alignment_score"]),
        "bottom_endpoint_alignment_score": float(endpoint_metrics["bottom_endpoint_alignment_score"]),
        "top_endpoint_best_fragment_overlap_px": float(endpoint_metrics["top_endpoint_best_fragment_overlap_px"]),
        "bottom_endpoint_best_fragment_overlap_px": float(endpoint_metrics["bottom_endpoint_best_fragment_overlap_px"]),
        "top_endpoint_best_fragment_ratio": float(endpoint_metrics["top_endpoint_best_fragment_ratio"]),
        "bottom_endpoint_best_fragment_ratio": float(endpoint_metrics["bottom_endpoint_best_fragment_ratio"]),
        "top_original_endpoint_coverage": float(endpoint_metrics["top_original_endpoint_coverage"]),
        "bottom_original_endpoint_coverage": float(endpoint_metrics["bottom_original_endpoint_coverage"]),
        "top_original_endpoint_best_fragment_overlap_px": float(
            endpoint_metrics["top_original_endpoint_best_fragment_overlap_px"]
        ),
        "bottom_original_endpoint_best_fragment_overlap_px": float(
            endpoint_metrics["bottom_original_endpoint_best_fragment_overlap_px"]
        ),
        "top_original_endpoint_fragment_ratio": float(endpoint_metrics["top_original_endpoint_fragment_ratio"]),
        "bottom_original_endpoint_fragment_ratio": float(endpoint_metrics["bottom_original_endpoint_fragment_ratio"]),
        "endpoint_anchor_score": float(endpoint_metrics["endpoint_anchor_score"]),
        "top_reach_gap_px": float(endpoint_metrics["top_reach_gap_px"]),
        "bottom_reach_gap_px": float(endpoint_metrics["bottom_reach_gap_px"]),
        "has_top_anchor": bool(has_top_anchor),
        "has_bottom_anchor": bool(has_bottom_anchor),
        "has_top_bottom_anchor": bool(has_top_anchor and has_bottom_anchor),
        "has_top_original_anchor": bool(has_top_original_anchor),
        "has_bottom_original_anchor": bool(has_bottom_original_anchor),
        "has_top_bottom_original_anchor": bool(has_top_original_anchor and has_bottom_original_anchor),
        "merged_interval_count": int(chain_metrics["merged_interval_count"]),
        "total_merged_length_px": float(chain_metrics["total_merged_length_px"]),
        "longest_merged_interval_px": float(chain_metrics["longest_merged_interval_px"]),
        "chain_total_gap_px": float(chain_metrics["chain_total_gap_px"]),
        "chain_continuity_ratio": float(chain_metrics["chain_continuity_ratio"]),
        "chain_fragment_count": int(len(chain_support)),
        "chain_total_length_px": float(chain_total_length_px),
        "outside_chain_length_ratio": float(clip01(outside_chain_length_ratio)),
        "outside_chain_fragment_ratio": float(clip01(outside_chain_fragment_ratio)),
        "outside_mask_penalty": float(row_metrics["outside_mask_penalty"]),
        "symmetry_score": float(row_metrics["symmetry_score"]),
        "roi_center_score": float(row_metrics["roi_center_score"]),
        "median_min_side_ratio": float(row_metrics["median_min_side_ratio"]),
        "side_clearance_row_ratio": float(row_metrics["side_clearance_row_ratio"]),
        "side_clearance_score": float(row_metrics["side_clearance_score"]),
        "has_min_side_clearance": bool(row_metrics["has_min_side_clearance"]),
        "rows_inside_mask_count": int(row_metrics["rows_inside_mask_count"]),
        "sampled_row_count": int(row_metrics["sampled_row_count"]),
        "adjusted_fragment_count": int(adjustment_metrics["adjusted_fragment_count"]),
        "adjusted_fragment_ratio": float(adjustment_metrics["adjusted_fragment_ratio"]),
        "mean_abs_support_shift_px": float(adjustment_metrics["mean_abs_shift_px"]),
        "length_weighted_mean_abs_support_shift_px": float(adjustment_metrics["length_weighted_mean_abs_shift_px"]),
        "max_abs_support_shift_px": float(adjustment_metrics["max_abs_shift_px"]),
        "mean_abs_support_tilt_delta_deg": float(adjustment_metrics["mean_abs_tilt_delta_deg"]),
        "max_abs_support_tilt_delta_deg": float(adjustment_metrics["max_abs_tilt_delta_deg"]),
        "support_adjustment_penalty": float(adjustment_metrics["adjustment_penalty"]),
    }
    if candidate_summary_cache is not None and cache_key is not None:
        candidate_summary_cache[cache_key] = dict(result)
    return result

def evaluate_candidate(
    axis: dict[str, float],
    lines: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
    band_half_width_px: float,
    max_angle_error_deg: float,
    allow_adjustment: bool = False,
    support_cache: dict | None = None,
    row_metrics_cache: dict | None = None,
    candidate_summary_cache: dict | None = None,
    selection_cache: dict | None = None,
    line_selection_cache: dict | None = None,
) -> dict:
    selected_support = select_support_fragments(
        lines=lines,
        axis=axis,
        band_half_width_px=band_half_width_px,
        max_angle_error_deg=max_angle_error_deg,
        allow_adjustment=allow_adjustment,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )
    return summarize_candidate_from_support(
        axis,
        selected_support,
        roi_profile,
        total_available_length_px,
        support_cache=support_cache,
        row_metrics_cache=row_metrics_cache,
        candidate_summary_cache=candidate_summary_cache,
    )
