from __future__ import annotations

import math

import cv2
import numpy as np

from .context import cfg, clip01
from .geometry import line_x_at_y, make_axis_signature
from .calculations import (
    build_support_analysis,
    compute_axis_fit_metrics,
    compute_gap_penalty,
    has_endpoint_anchor,
    interval_union_length,
    make_support_signature,
    merge_numeric_intervals,
    select_support_fragments,
)


def calculate_symmetry_score(axis: dict[str, float], roi_profile: dict) -> float:
    """Compatibility API: ROI balance, not image mirror symmetry."""
    return float(compute_row_balance_metrics(axis, roi_profile)["roi_balance_score"])


def compute_vertical_coverage(
    selected_support: list[dict],
    y_min: float,
    y_max: float,
) -> dict[str, float | int]:
    bin_count = int(cfg("coverage", "bin_count", default=12))
    total_span = max(1.0, float(y_max) - float(y_min))
    supported_bins = np.zeros(bin_count, dtype=bool)
    clipped_intervals: list[tuple[float, float]] = []

    for item in selected_support:
        line = item["line"]
        start = max(float(y_min), float(line["y_min"]))
        end = min(float(y_max), float(line["y_max"]))
        if end <= start:
            continue
        clipped_intervals.append((start, end))
        start_bin = int(np.floor((start - float(y_min)) / total_span * bin_count))
        end_bin = int(np.floor((end - float(y_min)) / total_span * bin_count))
        start_bin = max(0, min(bin_count - 1, start_bin))
        end_bin = max(0, min(bin_count - 1, end_bin))
        supported_bins[start_bin : end_bin + 1] = True

    unique_length = interval_union_length(clipped_intervals)
    supported_bin_count = int(np.count_nonzero(supported_bins))
    return {
        "bin_count": bin_count,
        "supported_bin_count": supported_bin_count,
        "coverage_score": float(clip01(unique_length / total_span)),
        "unique_vertical_coverage": float(clip01(unique_length / total_span)),
        "unique_covered_length_px": float(unique_length),
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
    sampled_rows = trimmed_rows[:: max(1, sample_step_px)]
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
        balance_errors = np.abs(left_widths - right_widths) / inside_widths
        center_x = float(center_fit["a"]) * row_values[inside_mask] + float(center_fit["b"])
        center_errors = np.abs(inside_axis_x - center_x) / np.maximum(1.0, inside_widths * 0.5)
        roi_balance_score = clip01(1.0 - float(np.median(balance_errors)))
        center_score = clip01(1.0 - float(np.median(center_errors)))
        median_min_side_ratio = float(np.median(min_side_ratios))
        side_clearance_row_ratio = float(np.mean(min_side_ratios >= min_side_ratio_threshold))
        side_clearance_score = clip01(
            (median_min_side_ratio - min_side_ratio_threshold)
            / max(1e-6, target_side_ratio - min_side_ratio_threshold)
        )
        has_min_side_clearance = bool(
            side_clearance_row_ratio >= min_clearance_row_ratio_threshold
        )
    else:
        roi_balance_score = 0.0
        center_score = 0.0
        median_min_side_ratio = 0.0
        side_clearance_row_ratio = 0.0
        side_clearance_score = 0.0
        has_min_side_clearance = False

    outside_mask_penalty = clip01(1.0 - inside_rows / max(1, total_rows))
    result = {
        "sampled_row_count": int(total_rows),
        "rows_inside_mask_count": int(inside_rows),
        "axis_inside_roi_ratio": float(clip01(inside_rows / max(1, total_rows))),
        "outside_mask_penalty": float(outside_mask_penalty),
        "roi_balance_score": float(roi_balance_score),
        "symmetry_score": float(roi_balance_score),  # compatibility alias
        "roi_center_score": float(center_score),
        "median_min_side_ratio": float(median_min_side_ratio),
        "side_clearance_row_ratio": float(side_clearance_row_ratio),
        "side_clearance_score": float(side_clearance_score),
        "has_min_side_clearance": bool(has_min_side_clearance),
    }
    if row_metrics_cache is not None and cache_key is not None:
        row_metrics_cache[cache_key] = result
    return result


def _nearest_distances(
    values: np.ndarray,
    targets: np.ndarray,
    *,
    assume_sorted: bool = False,
) -> np.ndarray:
    if values.size == 0 or targets.size == 0:
        return np.empty(0, dtype=np.float64)
    values = values.astype(np.float64, copy=False)
    if not assume_sorted:
        values = np.sort(values)
    targets = targets.astype(np.float64, copy=False)
    positions = np.searchsorted(values, targets)
    left_positions = np.clip(positions - 1, 0, values.size - 1)
    right_positions = np.clip(positions, 0, values.size - 1)
    left_distance = np.abs(targets - values[left_positions])
    right_distance = np.abs(targets - values[right_positions])
    return np.minimum(left_distance, right_distance)


def _prepare_edge_mask(edge_image: np.ndarray, roi_profile: dict) -> np.ndarray:
    if edge_image.ndim == 3:
        gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_image.astype(np.uint8, copy=False)
    roi_mask = roi_profile["mask"] > 0
    roi_pixels = gray[roi_mask]
    if roi_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    threshold = int(cfg("mirror_symmetry", "edge_threshold", default=24))
    nonzero_ratio = float(np.mean(roi_pixels > threshold))
    unique_count = int(np.unique(roi_pixels[:: max(1, roi_pixels.size // 50000)]).size)
    if nonzero_ratio > 0.18 or unique_count > 32:
        edge_mask = cv2.Canny(gray, 50, 150)
    else:
        edge_mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return np.where(roi_mask, edge_mask, 0).astype(np.uint8)


def prepare_mirror_symmetry_context(
    edge_image: np.ndarray | None,
    roi_profile: dict,
) -> dict | None:
    """Precompute image-only mirror data once per analyzed image.

    Candidate axes only change the split between the already sorted left/right
    edge coordinates. The expensive Canny pass and row scans are independent
    of the candidate and therefore must not be repeated for every axis.
    """
    if edge_image is None or not bool(cfg("mirror_symmetry", "enabled", default=True)):
        return None

    edge_mask = _prepare_edge_mask(edge_image, roi_profile)
    row_step = max(1, int(cfg("mirror_symmetry", "row_step_px", default=6)))
    sampled_rows = roi_profile["trimmed_rows"][::row_step]
    row_edges_by_sample: list[np.ndarray] = []
    for y_value in sampled_rows.tolist():
        left_bound = int(roi_profile["left_bounds"][y_value])
        right_bound = int(roi_profile["right_bounds"][y_value])
        if right_bound <= left_bound:
            row_edges_by_sample.append(np.empty(0, dtype=np.float64))
            continue
        row_edges = (
            np.flatnonzero(edge_mask[y_value, left_bound : right_bound + 1] > 0)
            + left_bound
        )
        row_edges_by_sample.append(row_edges.astype(np.float64, copy=False))

    return {
        "sampled_rows": sampled_rows,
        "row_edges_by_sample": tuple(row_edges_by_sample),
        "max_distance_px": max(
            1.0,
            float(cfg("mirror_symmetry", "max_distance_px", default=12.0)),
        ),
        "center_exclusion_px": float(
            cfg("mirror_symmetry", "center_exclusion_px", default=3.0)
        ),
        "min_edge_pixels": max(
            1,
            int(cfg("mirror_symmetry", "min_edge_pixels_per_side", default=2)),
        ),
        "min_valid_row_ratio": float(
            cfg("mirror_symmetry", "min_valid_row_ratio", default=0.08)
        ),
        "trim_fraction": clip01(
            float(cfg("mirror_symmetry", "trim_fraction", default=0.10))
        ),
    }


def compute_mirror_symmetry_score(
    axis: dict[str, float],
    edge_image: np.ndarray | None,
    roi_profile: dict,
    mirror_context: dict | None = None,
) -> dict[str, float | bool | int]:
    resolved_context = mirror_context
    if resolved_context is None:
        resolved_context = prepare_mirror_symmetry_context(edge_image, roi_profile)

    max_distance_px = (
        float(resolved_context["max_distance_px"])
        if resolved_context is not None
        else float(cfg("mirror_symmetry", "max_distance_px", default=12.0))
    )
    neutral = {
        "mirror_left_to_right_score": 0.5,
        "mirror_right_to_left_score": 0.5,
        "mirror_symmetry_score": 0.5,
        "mirror_valid_row_ratio": 0.0,
        "mirror_median_distance_px": max_distance_px,
        "mirror_valid_row_count": 0,
        "mirror_is_reliable": False,
    }
    if resolved_context is None:
        return neutral

    sampled_rows = resolved_context["sampled_rows"]
    row_edges_by_sample = resolved_context["row_edges_by_sample"]
    center_exclusion_px = float(resolved_context["center_exclusion_px"])
    min_edge_pixels = int(resolved_context["min_edge_pixels"])
    min_valid_row_ratio = float(resolved_context["min_valid_row_ratio"])
    trim_fraction = float(resolved_context["trim_fraction"])

    ltr_scores: list[float] = []
    rtl_scores: list[float] = []
    all_distances: list[float] = []

    axis_a = float(axis["a"])
    axis_b = float(axis["b"])
    for y_value, row_edges in zip(sampled_rows.tolist(), row_edges_by_sample):
        if row_edges.size == 0:
            continue
        axis_x = axis_a * float(y_value) + axis_b
        left_x = row_edges[row_edges < axis_x - center_exclusion_px]
        right_x = row_edges[row_edges > axis_x + center_exclusion_px]
        if left_x.size < min_edge_pixels or right_x.size < min_edge_pixels:
            continue

        mirrored_left = 2.0 * axis_x - left_x
        mirrored_right = 2.0 * axis_x - right_x
        ltr_distances = np.clip(
            _nearest_distances(right_x, mirrored_left, assume_sorted=True), 0.0, max_distance_px
        )
        rtl_distances = np.clip(
            _nearest_distances(left_x, mirrored_right, assume_sorted=True), 0.0, max_distance_px
        )
        if ltr_distances.size == 0 or rtl_distances.size == 0:
            continue
        ltr_scores.append(float(1.0 - np.mean(ltr_distances) / max_distance_px))
        rtl_scores.append(float(1.0 - np.mean(rtl_distances) / max_distance_px))
        all_distances.extend(ltr_distances.tolist())
        all_distances.extend(rtl_distances.tolist())

    valid_row_count = len(ltr_scores)
    valid_row_ratio = valid_row_count / max(1, len(sampled_rows))
    if valid_row_count == 0 or valid_row_ratio < min_valid_row_ratio:
        return {**neutral, "mirror_valid_row_ratio": float(valid_row_ratio)}

    def robust_mean(values: list[float]) -> float:
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        trim_count = int(len(ordered) * trim_fraction)
        if trim_count > 0 and 2 * trim_count < len(ordered):
            ordered = ordered[trim_count:-trim_count]
        return float(np.mean(ordered))

    ltr_score = clip01(robust_mean(ltr_scores))
    rtl_score = clip01(robust_mean(rtl_scores))
    return {
        "mirror_left_to_right_score": float(ltr_score),
        "mirror_right_to_left_score": float(rtl_score),
        "mirror_symmetry_score": float(0.5 * (ltr_score + rtl_score)),
        "mirror_valid_row_ratio": float(valid_row_ratio),
        "mirror_median_distance_px": float(np.median(all_distances)),
        "mirror_valid_row_count": int(valid_row_count),
        "mirror_is_reliable": True,
    }


def _weighted_component_score(components: dict[str, float], config_path: tuple[str, ...]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, value in components.items():
        weight = float(cfg(*config_path, name, default=0.0))
        if weight <= 0.0:
            continue
        weighted_sum += weight * clip01(value)
        weight_sum += weight
    return float(weighted_sum / max(1e-9, weight_sum))


def _coverage_around_reference(
    selected_support: list[dict],
    y_min: float,
    y_ref: float,
    y_max: float,
) -> tuple[float, float, float]:
    above_intervals: list[tuple[float, float]] = []
    below_intervals: list[tuple[float, float]] = []
    for item in selected_support:
        start = float(item["line"]["y_min"])
        end = float(item["line"]["y_max"])
        if min(end, y_ref) > max(start, y_min):
            above_intervals.append((max(start, y_min), min(end, y_ref)))
        if min(end, y_max) > max(start, y_ref):
            below_intervals.append((max(start, y_ref), min(end, y_max)))
    roi_span = max(1.0, y_max - y_min)
    above_ratio = interval_union_length(above_intervals) / roi_span
    below_ratio = interval_union_length(below_intervals) / roi_span
    balance = (
        2.0 * min(above_ratio, below_ratio) / max(1e-9, above_ratio + below_ratio)
        if above_ratio + below_ratio > 0.0
        else 0.0
    )
    return float(clip01(above_ratio)), float(clip01(below_ratio)), float(clip01(balance))


def _fragment_alignment_score(selected_support: list[dict]) -> float:
    if not selected_support:
        return 0.0
    values = [
        0.7 * float(item.get("distance_alignment", 0.0))
        + 0.3 * float(item.get("angle_alignment", 0.0))
        for item in selected_support
    ]
    weights = [max(1.0, float(item["line"]["length"])) for item in selected_support]
    return float(clip01(np.average(values, weights=weights)))


def update_candidate_scores(candidate: dict) -> dict:
    result = dict(candidate)
    geometry_components = {
        "chain_span_ratio": float(result.get("chain_span_ratio", 0.0)),
        "unique_vertical_coverage": float(result.get("unique_vertical_coverage", 0.0)),
        "chain_continuity_ratio": float(result.get("chain_continuity_ratio", 0.0)),
        "fit_consistency_score": float(result.get("fit_consistency_score", 0.0)),
        "fragment_alignment_score": float(result.get("fragment_alignment_score", 0.0)),
    }
    geometry_score = _weighted_component_score(
        geometry_components,
        ("normalized_scoring", "geometry"),
    )
    result["geometry_score"] = float(geometry_score)

    final_components = {
        "geometry_score": geometry_score,
        "above_below_balance_score": float(result.get("above_below_balance_score", 0.0)),
        "roi_balance_score": float(result.get("roi_balance_score", 0.0)),
    }
    if bool(result.get("mirror_is_reliable", False)):
        final_components["mirror_symmetry_score"] = float(result.get("mirror_symmetry_score", 0.5))

    final_score = _weighted_component_score(
        final_components,
        ("normalized_scoring", "final"),
    )
    gap_weight = float(
        cfg("normalized_scoring", "final", "gap_outlier_penalty", default=0.10)
    )
    final_score -= gap_weight * float(result.get("gap_outlier_penalty", 0.0))
    if not bool(result.get("validation_passed", True)):
        final_score -= float(
            cfg("normalized_scoring", "invalid_candidate_penalty", default=0.45)
        )

    result["final_score"] = float(final_score)
    result["selection_score"] = float(final_score)
    result["score"] = float(final_score)
    result["symmetry_score"] = float(
        result.get("mirror_symmetry_score", result.get("roi_balance_score", 0.0))
        if bool(result.get("mirror_is_reliable", False))
        else result.get("roi_balance_score", 0.0)
    )
    return result


def apply_mirror_symmetry(
    candidate: dict,
    edge_image: np.ndarray | None,
    roi_profile: dict,
    mirror_context: dict | None = None,
) -> dict:
    result = dict(candidate)
    result.update(
        compute_mirror_symmetry_score(
            result,
            edge_image,
            roi_profile,
            mirror_context=mirror_context,
        )
    )
    return update_candidate_scores(result)


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
        cache_key = (make_axis_signature(axis), make_support_signature(selected_support))
        cached_candidate = candidate_summary_cache.get(cache_key)
        if cached_candidate is not None:
            return dict(cached_candidate)

    support_analysis = build_support_analysis(selected_support, roi_profile, support_cache=support_cache)
    chain_support = list(support_analysis["chain_support"])
    if not chain_support:
        chain_support = list(selected_support)

    selected_total_length_px = float(sum(float(item["line"]["length"]) for item in chain_support))
    selected_total_support_strength = float(sum(float(item["support_strength"]) for item in chain_support))
    fragment_support_score = clip01(
        selected_total_support_strength / max(1.0, total_available_length_px)
    )
    chain_total_length_px = float(support_analysis["chain_total_length_px"])
    outside_chain_length_ratio = float(support_analysis["outside_chain_length_ratio"])
    outside_chain_fragment_ratio = float(support_analysis["outside_chain_fragment_ratio"])

    y_min = float(roi_profile["trimmed_y_min"])
    y_max = float(roi_profile["trimmed_y_max"])
    y_ref = float(roi_profile["y_ref"])
    roi_span = max(1.0, y_max - y_min)
    coverage_metrics = compute_vertical_coverage(chain_support, y_min, y_max)
    gap_metrics = compute_gap_penalty(chain_support, y_min, y_max)
    endpoint_metrics = support_analysis["endpoint_metrics"]
    chain_metrics = support_analysis["chain_metrics"]
    row_metrics = compute_row_balance_metrics_cached(axis, roi_profile, row_metrics_cache)
    adjustment_metrics = support_analysis["adjustment_metrics"]
    fit_metrics = compute_axis_fit_metrics(axis, chain_support)

    has_top_anchor = has_endpoint_anchor(endpoint_metrics, "top", require_original=False)
    has_bottom_anchor = has_endpoint_anchor(endpoint_metrics, "bottom", require_original=False)
    has_top_original_anchor = has_endpoint_anchor(endpoint_metrics, "top", require_original=True)
    has_bottom_original_anchor = has_endpoint_anchor(endpoint_metrics, "bottom", require_original=True)

    chain_span_ratio = clip01(float(endpoint_metrics["support_span_px"]) / roi_span)
    unique_vertical_coverage = float(coverage_metrics["unique_vertical_coverage"])
    fit_rmse_px = float(fit_metrics["fit_rmse_px"])
    fit_rmse_scale = max(
        1.0,
        float(cfg("normalized_scoring", "fit_rmse_scale_px", default=10.0)),
    )
    fit_consistency_score = (
        float(math.exp(-fit_rmse_px / fit_rmse_scale))
        if math.isfinite(fit_rmse_px)
        else 0.0
    )
    fragment_alignment_score = _fragment_alignment_score(chain_support)
    above_ratio, below_ratio, above_below_balance_score = _coverage_around_reference(
        chain_support, y_min, y_ref, y_max
    )
    largest_gap_ratio = float(gap_metrics["largest_gap_px"]) / roi_span
    soft_gap_ratio = float(cfg("normalized_scoring", "gap_soft_ratio", default=0.08))
    hard_gap_ratio = float(cfg("normalized_scoring", "gap_hard_ratio", default=0.24))
    gap_outlier_penalty = clip01(
        (largest_gap_ratio - soft_gap_ratio) / max(1e-6, hard_gap_ratio - soft_gap_ratio)
    )

    dense_support_passed = (
        chain_span_ratio
        >= float(cfg("candidate_validation", "min_chain_span_ratio", default=0.22))
        and unique_vertical_coverage
        >= float(
            cfg(
                "candidate_validation",
                "min_unique_vertical_coverage",
                default=0.18,
            )
        )
        and above_ratio
        >= float(
            cfg(
                "candidate_validation",
                "min_support_above_y_ref_ratio",
                default=0.04,
            )
        )
        and below_ratio
        >= float(
            cfg(
                "candidate_validation",
                "min_support_below_y_ref_ratio",
                default=0.04,
            )
        )
        and fit_rmse_px
        <= float(cfg("candidate_validation", "max_fit_rmse_px", default=14.0))
        and len(chain_support)
        >= int(cfg("search", "min_support_fragments", default=2))
    )

    sparse_support_enabled = bool(
        cfg(
            "candidate_validation",
            "allow_sparse_spanning_support",
            default=True,
        )
    )
    sparse_support_passed = bool(
        sparse_support_enabled
        and chain_span_ratio
        >= float(
            cfg(
                "candidate_validation",
                "sparse_min_chain_span_ratio",
                default=0.52,
            )
        )
        and unique_vertical_coverage
        >= float(
            cfg(
                "candidate_validation",
                "sparse_min_unique_vertical_coverage",
                default=0.10,
            )
        )
        and above_ratio
        >= float(
            cfg(
                "candidate_validation",
                "sparse_min_support_above_y_ref_ratio",
                default=0.025,
            )
        )
        and below_ratio
        >= float(
            cfg(
                "candidate_validation",
                "sparse_min_support_below_y_ref_ratio",
                default=0.025,
            )
        )
        and fit_rmse_px
        <= float(
            cfg(
                "candidate_validation",
                "sparse_max_fit_rmse_px",
                default=10.0,
            )
        )
        and len(chain_support)
        >= int(
            cfg(
                "candidate_validation",
                "sparse_min_support_fragments",
                default=2,
            )
        )
    )
    support_distribution_passed = bool(
        dense_support_passed or sparse_support_passed
    )

    rejection_reasons: list[str] = []
    if not support_distribution_passed:
        if chain_span_ratio < float(
            cfg("candidate_validation", "min_chain_span_ratio", default=0.22)
        ):
            rejection_reasons.append("chain_span_too_short")
        if unique_vertical_coverage < float(
            cfg(
                "candidate_validation",
                "min_unique_vertical_coverage",
                default=0.18,
            )
        ):
            rejection_reasons.append("vertical_coverage_too_low")
        if above_ratio < float(
            cfg(
                "candidate_validation",
                "min_support_above_y_ref_ratio",
                default=0.04,
            )
        ):
            rejection_reasons.append("insufficient_support_above_y_ref")
        if below_ratio < float(
            cfg(
                "candidate_validation",
                "min_support_below_y_ref_ratio",
                default=0.04,
            )
        ):
            rejection_reasons.append("insufficient_support_below_y_ref")
        if fit_rmse_px > float(
            cfg("candidate_validation", "max_fit_rmse_px", default=14.0)
        ):
            rejection_reasons.append("fit_rmse_too_high")
        if len(chain_support) < int(
            cfg("search", "min_support_fragments", default=2)
        ):
            rejection_reasons.append("too_few_support_fragments")

    if float(row_metrics["axis_inside_roi_ratio"]) < float(
        cfg("candidate_validation", "min_axis_inside_roi_ratio", default=0.88)
    ):
        rejection_reasons.append("axis_outside_roi")

    result = {
        **axis,
        "selected_support": chain_support,
        "selected_fragment_line_indices": [int(item["line"]["line_index"]) for item in chain_support],
        "selected_fragment_count": len(chain_support),
        "selected_total_length_px": selected_total_length_px,
        "selected_total_support_strength": selected_total_support_strength,
        "fragment_support_score": float(fragment_support_score),
        "vertical_coverage_score": float(unique_vertical_coverage),
        "unique_vertical_coverage": float(unique_vertical_coverage),
        "unique_covered_length_px": float(coverage_metrics["unique_covered_length_px"]),
        "supported_bin_count": int(coverage_metrics["supported_bin_count"]),
        "bin_count": int(coverage_metrics["bin_count"]),
        "gap_penalty": float(gap_metrics["gap_penalty"]),
        "gap_outlier_penalty": float(gap_outlier_penalty),
        "largest_gap_px": float(gap_metrics["largest_gap_px"]),
        "support_y_min": float(endpoint_metrics["support_y_min"]),
        "support_y_max": float(endpoint_metrics["support_y_max"]),
        "support_span_px": float(endpoint_metrics["support_span_px"]),
        "chain_span_ratio": float(chain_span_ratio),
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
        "top_original_endpoint_best_fragment_overlap_px": float(endpoint_metrics["top_original_endpoint_best_fragment_overlap_px"]),
        "bottom_original_endpoint_best_fragment_overlap_px": float(endpoint_metrics["bottom_original_endpoint_best_fragment_overlap_px"]),
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
        "axis_inside_roi_ratio": float(row_metrics["axis_inside_roi_ratio"]),
        "roi_balance_score": float(row_metrics["roi_balance_score"]),
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
        "fit_rmse_px": float(fit_metrics["fit_rmse_px"]),
        "fit_median_abs_residual_px": float(fit_metrics["fit_median_abs_residual_px"]),
        "fit_p90_abs_residual_px": float(fit_metrics["fit_p90_abs_residual_px"]),
        "fit_consistency_score": float(clip01(fit_consistency_score)),
        "fragment_alignment_score": float(fragment_alignment_score),
        "support_above_y_ref_ratio": float(above_ratio),
        "support_below_y_ref_ratio": float(below_ratio),
        "above_below_balance_score": float(above_below_balance_score),
        "dense_support_validation_passed": bool(dense_support_passed),
        "sparse_support_validation_passed": bool(sparse_support_passed),
        "support_distribution_passed": bool(support_distribution_passed),
        "support_distribution_mode": (
            "dense"
            if dense_support_passed
            else "sparse_spanning"
            if sparse_support_passed
            else "rejected"
        ),
        "mirror_left_to_right_score": 0.5,
        "mirror_right_to_left_score": 0.5,
        "mirror_symmetry_score": 0.5,
        "mirror_valid_row_ratio": 0.0,
        "mirror_median_distance_px": float(cfg("mirror_symmetry", "max_distance_px", default=12.0)),
        "mirror_valid_row_count": 0,
        "mirror_is_reliable": False,
        "validation_passed": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
    }
    result = update_candidate_scores(result)
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
    # Candidate selection intentionally uses original evidence only. The legacy
    # argument is retained for API compatibility but is never enabled here.
    del allow_adjustment
    selected_support = select_support_fragments(
        lines=lines,
        axis=axis,
        band_half_width_px=band_half_width_px,
        max_angle_error_deg=max_angle_error_deg,
        allow_adjustment=False,
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
