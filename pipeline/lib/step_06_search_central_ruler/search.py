from __future__ import annotations

import math

import numpy as np

from . import context
from .context import cfg
from .geometry import line_from_angle_and_anchor, line_x_at_y
from .calculations import (
    build_line_selection_cache,
    fit_axis_from_support,
    select_support_fragments,
    suppress_redundant_fragments,
)
from .metrics import (
    apply_mirror_symmetry,
    evaluate_candidate,
    summarize_candidate_from_support,
)
from .sorting import (
    annotate_candidate_selection,
    candidate_ranking_key,
    deduplicate_candidates,
    select_diverse_candidates,
    sort_candidates,
    unique_candidates_by_axis,
)


def search_central_ruler(
    lines: list[dict],
    roi_profile: dict,
    edge_image: np.ndarray | None = None,
) -> dict:
    return search_best_candidate(lines, roi_profile, edge_image=edge_image)


def build_fast_search_cache(lines: list[dict], roi_profile: dict) -> dict:
    bin_count = int(cfg("coverage", "bin_count", default=12))
    trimmed_y_min = float(roi_profile["trimmed_y_min"])
    trimmed_y_max = float(roi_profile["trimmed_y_max"])
    total_span = max(1.0, trimmed_y_max - trimmed_y_min)
    line_count = len(lines)

    line_a = np.asarray([float(line["a"]) for line in lines], dtype=np.float64)
    line_b = np.asarray([float(line["b"]) for line in lines], dtype=np.float64)
    line_tilt_deg = np.asarray([float(line["signed_tilt_deg"]) for line in lines], dtype=np.float64)
    line_length = np.asarray([float(line["length"]) for line in lines], dtype=np.float64)
    probe_y_min = np.asarray([float(line["y_min"]) for line in lines], dtype=np.float64)
    probe_y_mid = np.asarray([float(line["y_mid"]) for line in lines], dtype=np.float64)
    probe_y_max = np.asarray([float(line["y_max"]) for line in lines], dtype=np.float64)

    bin_coverage = np.zeros((line_count, bin_count), dtype=np.uint8)
    for line_index, line in enumerate(lines):
        start_bin = int(np.floor((float(line["y_min"]) - trimmed_y_min) / total_span * bin_count))
        end_bin = int(np.floor((float(line["y_max"]) - trimmed_y_min) / total_span * bin_count))
        start_bin = max(0, min(bin_count - 1, start_bin))
        end_bin = max(0, min(bin_count - 1, end_bin))
        bin_coverage[line_index, start_bin : end_bin + 1] = 1

    probe_count = 9
    probe_rows = np.linspace(trimmed_y_min, trimmed_y_max, num=probe_count, dtype=np.float64)
    probe_row_indices = np.clip(
        np.round(probe_rows).astype(np.int32), 0, roi_profile["height"] - 1
    )
    probe_left = roi_profile["left_bounds"][probe_row_indices].astype(np.float64)
    probe_right = roi_profile["right_bounds"][probe_row_indices].astype(np.float64)
    probe_width = np.maximum(
        1.0, roi_profile["row_widths"][probe_row_indices].astype(np.float64)
    )
    probe_center = np.asarray(
        [line_x_at_y(roi_profile["center_fit"], float(row)) for row in probe_rows],
        dtype=np.float64,
    )
    return {
        "line_a": line_a,
        "line_b": line_b,
        "line_tilt_deg": line_tilt_deg,
        "line_length": line_length,
        "probe_y_min": probe_y_min,
        "probe_y_mid": probe_y_mid,
        "probe_y_max": probe_y_max,
        "bin_count": bin_count,
        "bin_coverage_t": bin_coverage.T.astype(np.uint8),
        "roi_probe_rows": probe_rows,
        "roi_probe_left": probe_left,
        "roi_probe_right": probe_right,
        "roi_probe_width": probe_width,
        "roi_probe_center": probe_center,
    }


def make_candidate_grid_fast(
    angle_values: np.ndarray,
    x_ref_values: np.ndarray,
    y_ref: float,
    total_available_length_px: float,
    fast_cache: dict,
    band_half_width_px: float,
    max_angle_error_deg: float,
) -> list[dict]:
    """Cheap geometry-only screening. Endpoint bonuses are deliberately absent."""
    if angle_values.size == 0 or x_ref_values.size == 0:
        return []

    line_a = fast_cache["line_a"]
    line_b = fast_cache["line_b"]
    line_tilt_deg = fast_cache["line_tilt_deg"]
    line_length = fast_cache["line_length"]
    probe_y_min = fast_cache["probe_y_min"]
    probe_y_mid = fast_cache["probe_y_mid"]
    probe_y_max = fast_cache["probe_y_max"]
    bin_count = int(fast_cache["bin_count"])
    bin_coverage_t = fast_cache["bin_coverage_t"]
    roi_probe_rows = fast_cache["roi_probe_rows"]
    roi_probe_left = fast_cache["roi_probe_left"]
    roi_probe_right = fast_cache["roi_probe_right"]
    roi_probe_width = fast_cache["roi_probe_width"]
    roi_probe_center = fast_cache["roi_probe_center"]
    line_length_column = line_length[:, None]
    total_available_length_px = max(1.0, float(total_available_length_px))

    min_support_fragments = int(cfg("search", "min_support_fragments", default=2))
    min_supported_bins = int(cfg("coverage", "min_supported_bins", default=4))
    candidates: list[dict] = []

    for angle_deg_value in angle_values:
        angle_deg = float(angle_deg_value)
        axis_a = math.tan(math.radians(angle_deg))
        axis_b_values = np.asarray(x_ref_values, dtype=np.float64) - axis_a * float(y_ref)

        delta_a = line_a - axis_a
        base_min = delta_a * probe_y_min + line_b
        base_mid = delta_a * probe_y_mid + line_b
        base_max = delta_a * probe_y_max + line_b
        axis_distance = (
            np.abs(base_min[:, None] - axis_b_values[None, :])
            + np.abs(base_mid[:, None] - axis_b_values[None, :])
            + np.abs(base_max[:, None] - axis_b_values[None, :])
        ) / 3.0
        angle_error = np.abs(line_tilt_deg - angle_deg)[:, None]
        support_mask = (angle_error <= max_angle_error_deg) & (
            axis_distance <= band_half_width_px
        )
        distance_alignment = np.clip(
            1.0 - axis_distance / max(1e-6, band_half_width_px), 0.0, 1.0
        )
        angle_alignment = np.clip(
            1.0 - angle_error / max(1e-6, max_angle_error_deg), 0.0, 1.0
        )
        support_strength = (
            line_length_column
            * (0.72 * distance_alignment + 0.28 * angle_alignment)
            * support_mask
        )

        selected_fragment_count = np.sum(support_mask, axis=0)
        selected_total_support_strength = np.sum(support_strength, axis=0)
        fragment_support_score = np.clip(
            selected_total_support_strength / total_available_length_px, 0.0, 1.0
        )
        covered_bins = (bin_coverage_t @ support_mask.astype(np.uint8)) > 0
        supported_bin_count = np.sum(covered_bins, axis=0)
        coverage_score = supported_bin_count.astype(np.float64) / max(1, bin_count)
        has_support = supported_bin_count > 0
        first_supported = np.where(has_support, np.argmax(covered_bins, axis=0), 0)
        last_supported = np.where(
            has_support,
            bin_count - 1 - np.argmax(covered_bins[::-1], axis=0),
            0,
        )
        support_span_bins = np.where(
            has_support, np.maximum(1, last_supported - first_supported + 1), 1
        )
        continuity_score = supported_bin_count.astype(np.float64) / support_span_bins

        axis_probe_x = axis_a * roi_probe_rows[:, None] + axis_b_values[None, :]
        inside_mask = (axis_probe_x >= roi_probe_left[:, None]) & (
            axis_probe_x <= roi_probe_right[:, None]
        )
        inside_ratio = np.mean(inside_mask.astype(np.float64), axis=0)
        center_errors = np.abs(axis_probe_x - roi_probe_center[:, None]) / np.maximum(
            1.0, roi_probe_width[:, None] * 0.5
        )
        center_score = 1.0 - np.clip(np.median(center_errors, axis=0), 0.0, 1.0)

        score = (
            0.32 * coverage_score
            + 0.28 * fragment_support_score
            + 0.22 * continuity_score
            + 0.13 * inside_ratio
            + 0.05 * center_score
        )
        score = np.where(
            selected_fragment_count < min_support_fragments, score - 0.25, score
        )
        score = np.where(supported_bin_count < min_supported_bins, score - 0.20, score)

        for candidate_index, x_ref in enumerate(x_ref_values):
            candidates.append(
                {
                    "a": float(axis_a),
                    "b": float(axis_b_values[candidate_index]),
                    "tilt_deg": angle_deg,
                    "x_ref": float(x_ref),
                    "y_ref": float(y_ref),
                    "score": float(score[candidate_index]),
                    "final_score": float(score[candidate_index]),
                    "selected_fragment_count": int(selected_fragment_count[candidate_index]),
                    "fragment_support_score": float(fragment_support_score[candidate_index]),
                    "vertical_coverage_score": float(coverage_score[candidate_index]),
                    "chain_continuity_ratio": float(continuity_score[candidate_index]),
                    "supported_bin_count": int(supported_bin_count[candidate_index]),
                    "bin_count": bin_count,
                    "axis_inside_roi_ratio": float(inside_ratio[candidate_index]),
                    "roi_center_score": float(center_score[candidate_index]),
                }
            )
    return candidates


def make_candidate_grid(
    angle_values: np.ndarray,
    x_ref_values: np.ndarray,
    y_ref: float,
    lines: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
    band_half_width_px: float,
    max_angle_error_deg: float,
    allow_adjustment: bool = False,
) -> list[dict]:
    del allow_adjustment
    return [
        evaluate_candidate(
            axis=line_from_angle_and_anchor(float(angle), float(x_ref), float(y_ref)),
            lines=lines,
            roi_profile=roi_profile,
            total_available_length_px=total_available_length_px,
            band_half_width_px=band_half_width_px,
            max_angle_error_deg=max_angle_error_deg,
            allow_adjustment=False,
        )
        for angle in angle_values
        for x_ref in x_ref_values
    ]


def refresh_support_for_axis(
    lines: list[dict],
    axis: dict[str, float],
    seed_support: list[dict],
    band_half_width_px: float,
    max_angle_error_deg: float,
    allow_adjustment: bool,
    selection_cache: dict | None = None,
    line_selection_cache: dict | None = None,
) -> list[dict]:
    del seed_support, allow_adjustment
    return select_support_fragments(
        lines=lines,
        axis=axis,
        band_half_width_px=band_half_width_px,
        max_angle_error_deg=max_angle_error_deg,
        allow_adjustment=False,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )


def build_axis_harmonized_candidates(*args, **kwargs) -> list[dict]:
    """Legacy compatibility: harmonization is intentionally excluded from ranking."""
    del args, kwargs
    return []


def evaluate_hypothesis_variants(
    hypothesis: dict,
    hypothesis_rank: int,
    lines: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
    final_band_half_width_px: float,
    final_max_angle_error_deg: float,
    use_support_adjustment: bool,
    y_ref: float,
    support_cache: dict | None = None,
    fit_cache: dict | None = None,
    row_metrics_cache: dict | None = None,
    candidate_summary_cache: dict | None = None,
    selection_cache: dict | None = None,
    line_selection_cache: dict | None = None,
) -> list[dict]:
    """Build only original-evidence variants: hypothesis, chain fit, and one refit."""
    del use_support_adjustment
    stage_candidates: list[dict] = [
        annotate_candidate_selection(
            candidate=hypothesis,
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            stage_name="detailed_hypothesis",
        )
    ]

    seed_support = list(hypothesis.get("selected_support", []))
    fitted_axis = fit_axis_from_support(seed_support, y_ref=y_ref, fit_cache=fit_cache)
    if fitted_axis is None:
        return stage_candidates

    fitted_support = refresh_support_for_axis(
        lines=lines,
        axis=fitted_axis,
        seed_support=seed_support,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=False,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )
    fitted_candidate = summarize_candidate_from_support(
        axis=fitted_axis,
        selected_support=fitted_support,
        roi_profile=roi_profile,
        total_available_length_px=total_available_length_px,
        support_cache=support_cache,
        row_metrics_cache=row_metrics_cache,
        candidate_summary_cache=candidate_summary_cache,
    )
    stage_candidates.append(
        annotate_candidate_selection(
            fitted_candidate,
            hypothesis,
            hypothesis_rank,
            "chain_fit",
        )
    )

    refitted_axis = fit_axis_from_support(
        fitted_candidate.get("selected_support", fitted_support),
        y_ref=y_ref,
        fit_cache=fit_cache,
    )
    if refitted_axis is not None:
        refitted_support = refresh_support_for_axis(
            lines=lines,
            axis=refitted_axis,
            seed_support=fitted_candidate.get("selected_support", fitted_support),
            band_half_width_px=final_band_half_width_px,
            max_angle_error_deg=final_max_angle_error_deg,
            allow_adjustment=False,
            selection_cache=selection_cache,
            line_selection_cache=line_selection_cache,
        )
        refitted_candidate = summarize_candidate_from_support(
            axis=refitted_axis,
            selected_support=refitted_support,
            roi_profile=roi_profile,
            total_available_length_px=total_available_length_px,
            support_cache=support_cache,
            row_metrics_cache=row_metrics_cache,
            candidate_summary_cache=candidate_summary_cache,
        )
        stage_candidates.append(
            annotate_candidate_selection(
                refitted_candidate,
                hypothesis,
                hypothesis_rank,
                "chain_refit",
            )
        )

    return sort_candidates(
        unique_candidates_by_axis(stage_candidates), sort_key=candidate_ranking_key
    )


def _empty_result(raw_line_count: int = 0, nms_line_count: int = 0) -> dict:
    return {
        "coarse_candidates": [],
        "fine_candidates": [],
        "ranked_candidates": [],
        "best_hypothesis": None,
        "best_candidate": None,
        "raw_search_line_count": int(raw_line_count),
        "nms_line_count": int(nms_line_count),
    }


def search_best_candidate(
    lines: list[dict],
    roi_profile: dict,
    edge_image: np.ndarray | None = None,
) -> dict:
    raw_line_count = len(lines)
    lines = suppress_redundant_fragments(lines)
    if not lines:
        return _empty_result(raw_line_count=raw_line_count, nms_line_count=0)

    total_available_length_px = float(sum(float(line["length"]) for line in lines))
    support_selection_cache: dict = {}
    support_analysis_cache: dict = {}
    fit_cache: dict = {}
    row_metrics_cache: dict = {}
    candidate_summary_cache: dict = {}
    line_selection_cache = build_line_selection_cache(lines)

    max_candidate_tilt_deg = float(cfg("search", "max_candidate_tilt_deg", default=12.0))
    coarse_angle_step_deg = float(cfg("search", "coarse_angle_step_deg", default=1.0))
    fine_angle_step_deg = float(cfg("search", "fine_angle_step_deg", default=0.2))
    coarse_x_step_px = int(cfg("search", "coarse_x_step_px", default=14))
    fine_x_step_px = int(cfg("search", "fine_x_step_px", default=2))
    coarse_band_half_width_px = float(cfg("search", "coarse_band_half_width_px", default=16.0))
    final_band_half_width_px = float(cfg("search", "final_band_half_width_px", default=9.0))
    coarse_max_angle_error_deg = float(cfg("search", "coarse_max_angle_error_deg", default=6.0))
    final_max_angle_error_deg = float(cfg("search", "final_max_angle_error_deg", default=4.0))
    fine_window_x_px = int(cfg("search", "fine_window_x_px", default=40))
    fine_window_angle_deg = float(cfg("search", "fine_window_angle_deg", default=2.2))

    trimmed_rows = roi_profile["trimmed_rows"]
    left_bounds = roi_profile["left_bounds"][trimmed_rows]
    right_bounds = roi_profile["right_bounds"][trimmed_rows]
    x_min = int(np.min(left_bounds))
    x_max = int(np.max(right_bounds))
    y_ref = float(roi_profile["y_ref"])
    fast_cache = build_fast_search_cache(lines, roi_profile)

    coarse_angles = np.arange(
        -max_candidate_tilt_deg,
        max_candidate_tilt_deg + 0.5 * coarse_angle_step_deg,
        coarse_angle_step_deg,
    )
    coarse_x_values = np.arange(x_min, x_max + 1, max(1, coarse_x_step_px))
    coarse_candidates = make_candidate_grid_fast(
        angle_values=coarse_angles,
        x_ref_values=coarse_x_values,
        y_ref=y_ref,
        total_available_length_px=total_available_length_px,
        fast_cache=fast_cache,
        band_half_width_px=coarse_band_half_width_px,
        max_angle_error_deg=coarse_max_angle_error_deg,
    )
    coarse_candidates = sort_candidates(unique_candidates_by_axis(coarse_candidates))
    coarse_pool_limit = max(1, int(cfg("search", "coarse_candidate_pool_limit", default=120)))
    coarse_pool = select_diverse_candidates(
        coarse_candidates,
        max_candidates=coarse_pool_limit,
        angle_bucket_deg=float(cfg("search", "coarse_angle_bucket_deg", default=1.0)),
        max_per_angle_bucket=int(
            cfg("search", "max_coarse_candidates_per_angle_bucket", default=3)
        ),
        x_bucket_px=float(cfg("search", "coarse_x_bucket_px", default=18.0)),
        max_per_x_bucket=int(cfg("search", "max_coarse_candidates_per_x_bucket", default=3)),
    )
    top_coarse_count = max(1, int(cfg("search", "top_coarse_candidates", default=24)))
    top_coarse = coarse_pool[:top_coarse_count]

    fine_screened: list[dict] = []
    for coarse_candidate in top_coarse:
        fine_angles = np.arange(
            float(coarse_candidate["tilt_deg"]) - fine_window_angle_deg,
            float(coarse_candidate["tilt_deg"]) + fine_window_angle_deg + 0.5 * fine_angle_step_deg,
            fine_angle_step_deg,
        )
        fine_x_values = np.arange(
            int(round(float(coarse_candidate["x_ref"]) - fine_window_x_px)),
            int(round(float(coarse_candidate["x_ref"]) + fine_window_x_px)) + 1,
            max(1, fine_x_step_px),
        )
        fine_screened.extend(
            make_candidate_grid_fast(
                angle_values=fine_angles,
                x_ref_values=fine_x_values,
                y_ref=y_ref,
                total_available_length_px=total_available_length_px,
                fast_cache=fast_cache,
                band_half_width_px=coarse_band_half_width_px,
                max_angle_error_deg=coarse_max_angle_error_deg,
            )
        )

    fine_screened = sort_candidates(
        unique_candidates_by_axis(fine_screened or coarse_pool)
    )
    fine_pool_limit = max(1, int(cfg("search", "fine_candidate_pool_limit", default=144)))
    fine_screened = select_diverse_candidates(
        fine_screened,
        max_candidates=fine_pool_limit,
        angle_bucket_deg=max(0.25, float(cfg("search", "coarse_angle_bucket_deg", default=1.0)) / 2.0),
        max_per_angle_bucket=max(
            2, int(cfg("search", "max_coarse_candidates_per_angle_bucket", default=3)) * 2
        ),
        x_bucket_px=max(4.0, float(cfg("search", "coarse_x_bucket_px", default=18.0)) / 2.0),
        max_per_x_bucket=max(
            2, int(cfg("search", "max_coarse_candidates_per_x_bucket", default=3)) * 2
        ),
    )

    detailed_hypotheses = [
        evaluate_candidate(
            axis=screened,
            lines=lines,
            roi_profile=roi_profile,
            total_available_length_px=total_available_length_px,
            band_half_width_px=coarse_band_half_width_px,
            max_angle_error_deg=coarse_max_angle_error_deg,
            allow_adjustment=False,
            support_cache=support_analysis_cache,
            row_metrics_cache=row_metrics_cache,
            candidate_summary_cache=candidate_summary_cache,
            selection_cache=support_selection_cache,
            line_selection_cache=line_selection_cache,
        )
        for screened in fine_screened
    ]
    detailed_hypotheses = sort_candidates(
        unique_candidates_by_axis(detailed_hypotheses),
        sort_key=candidate_ranking_key,
    )
    if not detailed_hypotheses:
        return _empty_result(raw_line_count=raw_line_count, nms_line_count=len(lines))

    top_hypothesis_count = max(
        1, int(cfg("best_fit_selection", "top_hypothesis_count", default=24))
    )
    evaluated_hypotheses = deduplicate_candidates(
        detailed_hypotheses,
        roi_profile,
        max_candidates=top_hypothesis_count,
        sort_key=candidate_ranking_key,
    )

    final_candidate_pool: list[dict] = []
    for hypothesis_rank, hypothesis in enumerate(evaluated_hypotheses, start=1):
        final_candidate_pool.extend(
            evaluate_hypothesis_variants(
                hypothesis=hypothesis,
                hypothesis_rank=hypothesis_rank,
                lines=lines,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                final_band_half_width_px=final_band_half_width_px,
                final_max_angle_error_deg=final_max_angle_error_deg,
                use_support_adjustment=False,
                y_ref=y_ref,
                support_cache=support_analysis_cache,
                fit_cache=fit_cache,
                row_metrics_cache=row_metrics_cache,
                candidate_summary_cache=candidate_summary_cache,
                selection_cache=support_selection_cache,
                line_selection_cache=line_selection_cache,
            )
        )

    final_candidate_pool = sort_candidates(
        unique_candidates_by_axis(final_candidate_pool or evaluated_hypotheses),
        sort_key=candidate_ranking_key,
    )
    mirror_pool_limit = max(
        1, int(cfg("mirror_symmetry", "evaluation_pool_limit", default=160))
    )
    mirror_pool = deduplicate_candidates(
        final_candidate_pool,
        roi_profile,
        max_candidates=min(mirror_pool_limit, len(final_candidate_pool)),
        sort_key=candidate_ranking_key,
    )
    mirror_scored = [
        apply_mirror_symmetry(candidate, edge_image, roi_profile)
        for candidate in mirror_pool
    ]
    mirror_scored = sort_candidates(mirror_scored, sort_key=candidate_ranking_key)
    valid_candidates = [
        candidate for candidate in mirror_scored if bool(candidate.get("validation_passed", False))
    ]
    rankable_candidates = valid_candidates or mirror_scored

    save_all = bool(context.STEP_CONFIG.get("save_all_final_candidates", False))
    max_saved = (
        len(rankable_candidates)
        if save_all
        else max(1, int(cfg("candidate_deduplication", "max_saved_candidates", default=10)))
    )
    ranked_candidates = deduplicate_candidates(
        rankable_candidates,
        roi_profile,
        max_candidates=max_saved,
        sort_key=candidate_ranking_key,
    )
    ranked_candidates = sort_candidates(ranked_candidates, sort_key=candidate_ranking_key)
    best_candidate = ranked_candidates[0] if ranked_candidates else rankable_candidates[0]
    source_rank = max(1, int(best_candidate.get("source_hypothesis_rank", 1)))
    best_hypothesis = evaluated_hypotheses[min(source_rank - 1, len(evaluated_hypotheses) - 1)]

    return {
        "coarse_candidates": coarse_candidates,
        "fine_candidates": detailed_hypotheses,
        "ranked_candidates": ranked_candidates,
        "ranked_candidate_total_count": len(final_candidate_pool),
        "best_hypothesis": best_hypothesis,
        "best_candidate": best_candidate,
        "raw_search_line_count": int(raw_line_count),
        "nms_line_count": int(len(lines)),
        "nms_removed_line_count": int(raw_line_count - len(lines)),
    }
