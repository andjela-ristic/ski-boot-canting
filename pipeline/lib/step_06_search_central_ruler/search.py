from __future__ import annotations

import heapq
import math

import numpy as np

from . import context
from .context import cfg
from .geometry import blend_axis_toward_reference, line_from_angle_and_anchor, line_x_at_y
from .calculations import (
    build_line_selection_cache,
    build_support_analysis,
    extend_support_upward,
    fit_axis_from_support,
    merge_support_items,
    rescue_endpoint_support_fragments,
    select_support_fragments,
)
from .metrics import evaluate_candidate, summarize_candidate_from_support
from .sorting import (
    annotate_candidate_selection,
    candidate_ranking_key,
    filter_candidates_by_score_ratio,
    sort_candidates,
    unique_candidates_by_axis,
)


def search_central_ruler(lines: list[dict], roi_profile: dict) -> dict:
    return search_best_candidate(lines, roi_profile)


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

    probe_count = 7
    probe_rows = np.linspace(trimmed_y_min, trimmed_y_max, num=probe_count, dtype=np.float64)
    probe_row_indices = np.clip(np.round(probe_rows).astype(np.int32), 0, roi_profile["height"] - 1)
    probe_left = roi_profile["left_bounds"][probe_row_indices].astype(np.float64)
    probe_right = roi_profile["right_bounds"][probe_row_indices].astype(np.float64)
    probe_width = np.maximum(1.0, roi_profile["row_widths"][probe_row_indices].astype(np.float64))
    probe_center = np.asarray(
        [line_x_at_y(roi_profile["center_fit"], float(row_value)) for row_value in probe_rows],
        dtype=np.float64,
    )

    endpoint_bin_count = max(1, int(math.ceil(bin_count * float(cfg("endpoint_support", "band_ratio", default=0.15)))))

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
        "endpoint_bin_count": endpoint_bin_count,
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
    if angle_values.size == 0 or x_ref_values.size == 0:
        return []

    fragment_support_weight = float(cfg("scoring", "fragment_support_weight", default=0.34))
    vertical_coverage_weight = float(cfg("scoring", "vertical_coverage_weight", default=0.22))
    endpoint_anchor_weight = float(cfg("scoring", "endpoint_anchor_weight", default=0.10))
    gap_penalty_weight = float(cfg("scoring", "gap_penalty_weight", default=0.14))
    outside_mask_penalty_weight = float(cfg("scoring", "outside_mask_penalty_weight", default=0.16))
    symmetry_weight = float(cfg("scoring", "symmetry_weight", default=0.24))
    roi_center_weight = float(cfg("scoring", "roi_center_weight", default=0.08))
    center_proxy_weight = 0.55 * symmetry_weight + roi_center_weight
    low_support_penalty = float(cfg("scoring", "low_support_penalty", default=0.20))
    low_coverage_penalty = float(cfg("scoring", "low_coverage_penalty", default=0.20))
    min_support_fragments = int(cfg("search", "min_support_fragments", default=2))
    min_supported_bins = int(cfg("coverage", "min_supported_bins", default=4))

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
    endpoint_bin_count = int(fast_cache["endpoint_bin_count"])
    line_length_column = line_length[:, None]
    total_available_length_px = max(1.0, float(total_available_length_px))

    candidates: list[dict] = []
    for angle_deg in angle_values:
        angle_deg = float(angle_deg)
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

        angle_error = np.abs(line_tilt_deg - angle_deg)
        angle_error_matrix = angle_error[:, None]
        support_mask = (angle_error_matrix <= max_angle_error_deg) & (axis_distance <= band_half_width_px)
        distance_alignment = np.clip(1.0 - axis_distance / max(1e-6, band_half_width_px), 0.0, 1.0)
        angle_alignment = np.clip(1.0 - angle_error_matrix / max(1e-6, max_angle_error_deg), 0.0, 1.0)
        support_strength = line_length_column * (0.72 * distance_alignment + 0.28 * angle_alignment) * support_mask

        selected_fragment_count = np.sum(support_mask, axis=0)
        selected_total_length_px = np.sum(line_length_column * support_mask, axis=0)
        selected_total_support_strength = np.sum(support_strength, axis=0)
        fragment_support_score = np.clip(selected_total_support_strength / total_available_length_px, 0.0, 1.0)

        support_mask_u8 = support_mask.astype(np.uint8)
        covered_bin_counts = bin_coverage_t @ support_mask_u8
        covered_bins = covered_bin_counts > 0
        supported_bin_count = np.sum(covered_bins, axis=0)
        coverage_score = supported_bin_count.astype(np.float64) / max(1, bin_count)

        has_support = supported_bin_count > 0
        first_supported = np.where(has_support, np.argmax(covered_bins, axis=0), 0)
        last_supported = np.where(has_support, bin_count - 1 - np.argmax(covered_bins[::-1], axis=0), 0)
        support_span_bins = np.where(has_support, np.maximum(1, last_supported - first_supported + 1), 1)
        bin_continuity = supported_bin_count.astype(np.float64) / support_span_bins.astype(np.float64)
        gap_penalty = np.clip(1.0 - bin_continuity, 0.0, 1.0)

        top_endpoint_hit = np.any(covered_bins[:endpoint_bin_count], axis=0).astype(np.float64)
        bottom_endpoint_hit = np.any(covered_bins[-endpoint_bin_count:], axis=0).astype(np.float64)
        endpoint_score = 0.5 * (top_endpoint_hit + bottom_endpoint_hit)

        axis_probe_x = axis_a * roi_probe_rows[:, None] + axis_b_values[None, :]
        inside_mask = (axis_probe_x >= roi_probe_left[:, None]) & (axis_probe_x <= roi_probe_right[:, None])
        outside_mask_penalty = 1.0 - np.mean(inside_mask.astype(np.float64), axis=0)
        center_errors = np.abs(axis_probe_x - roi_probe_center[:, None]) / np.maximum(
            1.0,
            roi_probe_width[:, None] * 0.5,
        )
        center_score = 1.0 - np.clip(np.median(center_errors, axis=0), 0.0, 1.0)

        score = (
            fragment_support_weight * fragment_support_score
            + vertical_coverage_weight * coverage_score
            + endpoint_anchor_weight * endpoint_score
            + center_proxy_weight * center_score
            - gap_penalty_weight * gap_penalty
            - outside_mask_penalty_weight * outside_mask_penalty
        )
        score = np.where(selected_fragment_count < min_support_fragments, score - low_support_penalty, score)
        score = np.where(supported_bin_count < min_supported_bins, score - low_coverage_penalty, score)

        for candidate_index, x_ref in enumerate(x_ref_values):
            axis_b = float(axis_b_values[candidate_index])
            candidates.append(
                {
                    "a": float(axis_a),
                    "b": axis_b,
                    "tilt_deg": angle_deg,
                    "x_ref": float(x_ref),
                    "y_ref": float(y_ref),
                    "score": float(score[candidate_index]),
                    "selected_fragment_count": int(selected_fragment_count[candidate_index]),
                    "selected_total_length_px": float(selected_total_length_px[candidate_index]),
                    "selected_total_support_strength": float(selected_total_support_strength[candidate_index]),
                    "fragment_support_score": float(fragment_support_score[candidate_index]),
                    "supported_bin_count": int(supported_bin_count[candidate_index]),
                    "bin_count": int(bin_count),
                    "vertical_coverage_score": float(coverage_score[candidate_index]),
                    "gap_penalty": float(gap_penalty[candidate_index]),
                    "endpoint_anchor_score": float(endpoint_score[candidate_index]),
                    "outside_mask_penalty": float(outside_mask_penalty[candidate_index]),
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
    candidates = []
    for angle_deg in angle_values:
        for x_ref in x_ref_values:
            axis = line_from_angle_and_anchor(float(angle_deg), float(x_ref), float(y_ref))
            candidate = evaluate_candidate(
                axis=axis,
                lines=lines,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                band_half_width_px=band_half_width_px,
                max_angle_error_deg=max_angle_error_deg,
                allow_adjustment=allow_adjustment,
            )
            candidates.append(candidate)
    return candidates

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
    refreshed_support = select_support_fragments(
        lines=lines,
        axis=axis,
        band_half_width_px=band_half_width_px,
        max_angle_error_deg=max_angle_error_deg,
        allow_adjustment=allow_adjustment,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )
    if not seed_support:
        merged_support = refreshed_support
    elif not refreshed_support:
        merged_support = list(seed_support)
    else:
        merged_support = merge_support_items(seed_support, refreshed_support)
    return rescue_endpoint_support_fragments(
        lines=lines,
        axis=axis,
        selected_support=merged_support,
        max_angle_error_deg=max_angle_error_deg,
    )

def build_axis_harmonized_candidates(
    candidate: dict,
    hypothesis: dict,
    hypothesis_rank: int,
    lines: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
    final_band_half_width_px: float,
    final_max_angle_error_deg: float,
    y_ref: float,
    support_cache: dict | None = None,
    fit_cache: dict | None = None,
    row_metrics_cache: dict | None = None,
    candidate_summary_cache: dict | None = None,
    selection_cache: dict | None = None,
    line_selection_cache: dict | None = None,
) -> list[dict]:
    if not bool(cfg("best_fit_selection", "axis_harmonization", "enabled", default=True)):
        return []

    selected_support = list(candidate.get("selected_support", []))
    if not selected_support:
        return []

    current_axis = {
        "a": float(candidate["a"]),
        "b": float(candidate["b"]),
        "tilt_deg": float(candidate["tilt_deg"]),
        "x_ref": float(candidate["x_ref"]),
        "y_ref": float(candidate["y_ref"]),
    }
    min_x_ref_delta_px = float(cfg("best_fit_selection", "axis_harmonization", "min_x_ref_delta_px", default=0.75))
    min_tilt_delta_deg = float(
        cfg("best_fit_selection", "axis_harmonization", "min_tilt_delta_deg", default=0.05)
    )
    hypothesis_pull_ratio = float(
        cfg("best_fit_selection", "axis_harmonization", "hypothesis_pull_ratio", default=0.35)
    )
    refresh_with_adjustment = bool(int(candidate.get("adjusted_fragment_count", 0)) > 0)

    harmonized_candidates: list[dict] = []
    support_refit_axis = fit_axis_from_support(selected_support, y_ref=y_ref, fit_cache=fit_cache)
    if support_refit_axis is None:
        return harmonized_candidates

    force_support_refit = float(candidate.get("adjusted_fragment_ratio", 0.0)) > 0.0
    refit_x_ref_delta_px = abs(float(support_refit_axis["x_ref"]) - float(current_axis["x_ref"]))
    refit_tilt_delta_deg = abs(float(support_refit_axis["tilt_deg"]) - float(current_axis["tilt_deg"]))
    if not force_support_refit and refit_x_ref_delta_px < min_x_ref_delta_px and refit_tilt_delta_deg < min_tilt_delta_deg:
        return harmonized_candidates

    support_refit_support = refresh_support_for_axis(
        lines=lines,
        axis=support_refit_axis,
        seed_support=selected_support,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=refresh_with_adjustment,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )
    harmonized_candidates.append(
        annotate_candidate_selection(
                    candidate=summarize_candidate_from_support(
                        axis=support_refit_axis,
                        selected_support=support_refit_support,
                        roi_profile=roi_profile,
                        total_available_length_px=total_available_length_px,
                        support_cache=support_cache,
                        row_metrics_cache=row_metrics_cache,
                        candidate_summary_cache=candidate_summary_cache,
                    ),
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            stage_name=f"{candidate.get('search_stage', 'candidate')}_support_refit",
        )
    )

    if hypothesis_pull_ratio > 0.0:
        blended_axis = blend_axis_toward_reference(
            primary_axis=support_refit_axis,
            reference_axis=hypothesis,
            y_ref=y_ref,
            reference_pull_ratio=hypothesis_pull_ratio,
        )
        blend_x_ref_delta_px = abs(float(blended_axis["x_ref"]) - float(support_refit_axis["x_ref"]))
        blend_tilt_delta_deg = abs(float(blended_axis["tilt_deg"]) - float(support_refit_axis["tilt_deg"]))
        if blend_x_ref_delta_px >= min_x_ref_delta_px or blend_tilt_delta_deg >= min_tilt_delta_deg:
            blended_support = refresh_support_for_axis(
                lines=lines,
                axis=blended_axis,
                seed_support=support_refit_support,
                band_half_width_px=final_band_half_width_px,
                max_angle_error_deg=final_max_angle_error_deg,
                allow_adjustment=refresh_with_adjustment,
                selection_cache=selection_cache,
                line_selection_cache=line_selection_cache,
            )
            harmonized_candidates.append(
                annotate_candidate_selection(
                    candidate=summarize_candidate_from_support(
                        axis=blended_axis,
                        selected_support=blended_support,
                        roi_profile=roi_profile,
                        total_available_length_px=total_available_length_px,
                        support_cache=support_cache,
                        row_metrics_cache=row_metrics_cache,
                        candidate_summary_cache=candidate_summary_cache,
                    ),
                    hypothesis=hypothesis,
                    hypothesis_rank=hypothesis_rank,
                    stage_name=f"{candidate.get('search_stage', 'candidate')}_support_refit_hypothesis_blend",
                )
            )

    return harmonized_candidates

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
    stage_candidates = []
    resolved_support_cache = {} if support_cache is None else support_cache
    resolved_fit_cache = {} if fit_cache is None else fit_cache

    def refresh_stage_support(
        axis: dict[str, float],
        seed_support: list[dict],
        allow_adjustment: bool,
    ) -> list[dict]:
        return refresh_support_for_axis(
            lines=lines,
            axis=axis,
            seed_support=seed_support,
            band_half_width_px=final_band_half_width_px,
            max_angle_error_deg=final_max_angle_error_deg,
            allow_adjustment=allow_adjustment,
            selection_cache=selection_cache,
            line_selection_cache=line_selection_cache,
        )

    def append_chain_variant(
        support_items: list[dict],
        fallback_axis: dict[str, float],
        stage_name: str,
        allow_adjustment_for_refresh: bool,
    ) -> None:
        support_analysis = build_support_analysis(support_items, roi_profile, support_cache=resolved_support_cache)
        chain_support = support_analysis["chain_support"]
        if not chain_support:
            return
        if len(chain_support) == len(support_items):
            return
        chain_axis = fit_axis_from_support(chain_support, y_ref=y_ref, fit_cache=resolved_fit_cache)
        resolved_axis = fallback_axis if chain_axis is None else chain_axis
        resolved_support = chain_support
        if chain_axis is not None:
            resolved_support = refresh_stage_support(
                axis=resolved_axis,
                seed_support=chain_support,
                allow_adjustment=allow_adjustment_for_refresh,
            )
        stage_candidates.append(
            annotate_candidate_selection(
                candidate=summarize_candidate_from_support(
                    axis=resolved_axis,
                    selected_support=resolved_support,
                    roi_profile=roi_profile,
                    total_available_length_px=total_available_length_px,
                    support_cache=resolved_support_cache,
                    row_metrics_cache=row_metrics_cache,
                    candidate_summary_cache=candidate_summary_cache,
                ),
                hypothesis=hypothesis,
                hypothesis_rank=hypothesis_rank,
                stage_name=stage_name,
            )
        )

    hypothesis_support = select_support_fragments(
        lines=lines,
        axis=hypothesis,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=False,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )
    stage_candidates.append(
        annotate_candidate_selection(
            candidate=summarize_candidate_from_support(
                axis=hypothesis,
                selected_support=hypothesis_support,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                support_cache=resolved_support_cache,
                row_metrics_cache=row_metrics_cache,
                candidate_summary_cache=candidate_summary_cache,
            ),
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            stage_name="hypothesis_final_band",
        )
    )
    append_chain_variant(
        support_items=hypothesis_support,
        fallback_axis=hypothesis,
        stage_name="hypothesis_final_band_chain",
        allow_adjustment_for_refresh=False,
    )

    refined_support = hypothesis_support
    if use_support_adjustment:
        refined_support = select_support_fragments(
            lines=lines,
            axis=hypothesis,
            band_half_width_px=final_band_half_width_px,
            max_angle_error_deg=final_max_angle_error_deg,
            allow_adjustment=True,
            selection_cache=selection_cache,
            line_selection_cache=line_selection_cache,
        )
        stage_candidates.append(
            annotate_candidate_selection(
                candidate=summarize_candidate_from_support(
                    axis=hypothesis,
                    selected_support=refined_support,
                    roi_profile=roi_profile,
                    total_available_length_px=total_available_length_px,
                    support_cache=resolved_support_cache,
                    row_metrics_cache=row_metrics_cache,
                    candidate_summary_cache=candidate_summary_cache,
                ),
                hypothesis=hypothesis,
                hypothesis_rank=hypothesis_rank,
                stage_name="hypothesis_final_band_adjusted",
            )
        )
        append_chain_variant(
            support_items=refined_support,
            fallback_axis=hypothesis,
            stage_name="hypothesis_final_band_adjusted_chain",
            allow_adjustment_for_refresh=True,
        )

    fitted_axis = fit_axis_from_support(refined_support, y_ref=y_ref, fit_cache=resolved_fit_cache)
    base_axis = hypothesis if fitted_axis is None else fitted_axis
    base_support = select_support_fragments(
        lines=lines,
        axis=base_axis,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=use_support_adjustment,
        selection_cache=selection_cache,
        line_selection_cache=line_selection_cache,
    )
    stage_candidates.append(
        annotate_candidate_selection(
            candidate=summarize_candidate_from_support(
                axis=base_axis,
                selected_support=base_support,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                support_cache=resolved_support_cache,
                row_metrics_cache=row_metrics_cache,
                candidate_summary_cache=candidate_summary_cache,
            ),
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            stage_name="hypothesis_reselected_support" if fitted_axis is None else "final_fit_support",
        )
    )
    append_chain_variant(
        support_items=base_support,
        fallback_axis=base_axis,
        stage_name="hypothesis_reselected_chain" if fitted_axis is None else "final_fit_support_chain",
        allow_adjustment_for_refresh=use_support_adjustment,
    )

    extended_support = extend_support_upward(
        selected_support=base_support,
        lines=lines,
        axis=base_axis,
        roi_profile=roi_profile,
    )
    stage_candidates.append(
        annotate_candidate_selection(
            candidate=summarize_candidate_from_support(
                axis=base_axis,
                selected_support=extended_support,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                support_cache=resolved_support_cache,
                row_metrics_cache=row_metrics_cache,
                candidate_summary_cache=candidate_summary_cache,
            ),
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            stage_name="fine_hypothesis_extended" if fitted_axis is None else "final_fit_extended_support",
        )
    )
    append_chain_variant(
        support_items=extended_support,
        fallback_axis=base_axis,
        stage_name="fine_hypothesis_extended_chain" if fitted_axis is None else "final_fit_extended_chain",
        allow_adjustment_for_refresh=use_support_adjustment,
    )

    refit_axis = fit_axis_from_support(extended_support, y_ref=y_ref, fit_cache=resolved_fit_cache)
    if refit_axis is not None:
        final_support = refresh_stage_support(
            axis=refit_axis,
            seed_support=extended_support,
            allow_adjustment=use_support_adjustment,
        )
        final_refit_candidate = annotate_candidate_selection(
            candidate=summarize_candidate_from_support(
                axis=refit_axis,
                selected_support=final_support,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                support_cache=resolved_support_cache,
                row_metrics_cache=row_metrics_cache,
                candidate_summary_cache=candidate_summary_cache,
            ),
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            stage_name="final_fit_extended_refit",
        )
        stage_candidates.append(final_refit_candidate)
        append_chain_variant(
            support_items=final_support,
            fallback_axis=refit_axis,
            stage_name="final_fit_extended_refit_chain",
            allow_adjustment_for_refresh=use_support_adjustment,
        )
        recentered_axis = fit_axis_from_support(final_support, y_ref=y_ref, fit_cache=resolved_fit_cache)
        if recentered_axis is not None:
            recentered_support = refresh_stage_support(
                axis=recentered_axis,
                seed_support=final_support,
                allow_adjustment=use_support_adjustment,
            )
            stage_candidates.append(
                annotate_candidate_selection(
                    candidate=summarize_candidate_from_support(
                        axis=recentered_axis,
                        selected_support=recentered_support,
                        roi_profile=roi_profile,
                        total_available_length_px=total_available_length_px,
                        support_cache=resolved_support_cache,
                        row_metrics_cache=row_metrics_cache,
                        candidate_summary_cache=candidate_summary_cache,
                    ),
                    hypothesis=hypothesis,
                    hypothesis_rank=hypothesis_rank,
                    stage_name="final_fit_extended_recentered",
                )
            )
            append_chain_variant(
                support_items=recentered_support,
                fallback_axis=recentered_axis,
                stage_name="final_fit_extended_recentered_chain",
                allow_adjustment_for_refresh=use_support_adjustment,
            )
        joint_adjustment_top_hypotheses = int(
            cfg("support_adjustment", "joint_adjustment_top_hypotheses", default=8)
        )
        joint_adjustment_min_original_endpoint_ratio = float(
            cfg("support_adjustment", "joint_adjustment_min_original_endpoint_ratio", default=0.90)
        )
        joint_adjustment_min_original_endpoint_coverage = float(
            cfg("support_adjustment", "joint_adjustment_min_original_endpoint_coverage", default=0.24)
        )
        allow_joint_adjustment_for_hypothesis = (
            joint_adjustment_top_hypotheses <= 0
            or hypothesis_rank <= joint_adjustment_top_hypotheses
        )
        should_try_joint_adjustment = (
            use_support_adjustment
            and allow_joint_adjustment_for_hypothesis
            and (
                not bool(final_refit_candidate.get("has_top_bottom_original_anchor", False))
                or float(final_refit_candidate.get("top_original_endpoint_fragment_ratio", 0.0))
                < joint_adjustment_min_original_endpoint_ratio
                or float(final_refit_candidate.get("bottom_original_endpoint_fragment_ratio", 0.0))
                < joint_adjustment_min_original_endpoint_ratio
                or float(final_refit_candidate.get("top_original_endpoint_coverage", 0.0))
                < joint_adjustment_min_original_endpoint_coverage
                or float(final_refit_candidate.get("bottom_original_endpoint_coverage", 0.0))
                < joint_adjustment_min_original_endpoint_coverage
            )
        )
        if should_try_joint_adjustment:
            joint_adjusted_support = refresh_stage_support(
                axis=refit_axis,
                seed_support=final_support,
                allow_adjustment=True,
            )
            joint_adjusted_axis = fit_axis_from_support(
                joint_adjusted_support,
                y_ref=y_ref,
                fit_cache=resolved_fit_cache,
            )
            if joint_adjusted_axis is not None:
                joint_final_support = refresh_stage_support(
                    axis=joint_adjusted_axis,
                    seed_support=joint_adjusted_support,
                    allow_adjustment=True,
                )
                stage_candidates.append(
                    annotate_candidate_selection(
                        candidate=summarize_candidate_from_support(
                            axis=joint_adjusted_axis,
                            selected_support=joint_final_support,
                            roi_profile=roi_profile,
                            total_available_length_px=total_available_length_px,
                            support_cache=resolved_support_cache,
                            row_metrics_cache=row_metrics_cache,
                            candidate_summary_cache=candidate_summary_cache,
                        ),
                        hypothesis=hypothesis,
                        hypothesis_rank=hypothesis_rank,
                        stage_name="final_fit_joint_adjusted",
                    )
                )
                append_chain_variant(
                    support_items=joint_final_support,
                    fallback_axis=joint_adjusted_axis,
                    stage_name="final_fit_joint_adjusted_chain",
                    allow_adjustment_for_refresh=True,
                )
                joint_recentered_axis = fit_axis_from_support(
                    joint_final_support,
                    y_ref=y_ref,
                    fit_cache=resolved_fit_cache,
                )
                if joint_recentered_axis is not None:
                    joint_recentered_support = refresh_stage_support(
                        axis=joint_recentered_axis,
                        seed_support=joint_final_support,
                        allow_adjustment=True,
                    )
                    stage_candidates.append(
                        annotate_candidate_selection(
                            candidate=summarize_candidate_from_support(
                                axis=joint_recentered_axis,
                                selected_support=joint_recentered_support,
                                roi_profile=roi_profile,
                                total_available_length_px=total_available_length_px,
                                support_cache=resolved_support_cache,
                                row_metrics_cache=row_metrics_cache,
                                candidate_summary_cache=candidate_summary_cache,
                            ),
                            hypothesis=hypothesis,
                            hypothesis_rank=hypothesis_rank,
                            stage_name="final_fit_joint_recentered",
                        )
                    )
                    append_chain_variant(
                        support_items=joint_recentered_support,
                        fallback_axis=joint_recentered_axis,
                        stage_name="final_fit_joint_recentered_chain",
                        allow_adjustment_for_refresh=True,
                    )

    return stage_candidates

def search_best_candidate(lines: list[dict], roi_profile: dict) -> dict:
    if not lines:
        return {
            "coarse_candidates": [],
            "fine_candidates": [],
            "ranked_candidates": [],
            "best_hypothesis": None,
            "best_candidate": None,
        }

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
    fine_window_x_px = int(cfg("search", "fine_window_x_px", default=28))
    fine_window_angle_deg = float(cfg("search", "fine_window_angle_deg", default=1.6))
    coarse_score_floor_ratio = float(cfg("search", "coarse_score_floor_ratio", default=0.97))
    fine_score_floor_ratio = float(cfg("search", "fine_score_floor_ratio", default=0.98))

    trimmed_rows = roi_profile["trimmed_rows"]
    left_bounds = roi_profile["left_bounds"][trimmed_rows]
    right_bounds = roi_profile["right_bounds"][trimmed_rows]
    x_min = int(np.min(left_bounds))
    x_max = int(np.max(right_bounds))
    y_ref = float(roi_profile["y_ref"])
    fast_search_cache = build_fast_search_cache(lines, roi_profile)

    coarse_angles = np.arange(-max_candidate_tilt_deg, max_candidate_tilt_deg + 0.5 * coarse_angle_step_deg, coarse_angle_step_deg)
    coarse_x_values = np.arange(x_min, x_max + 1, max(1, coarse_x_step_px))

    coarse_candidates = make_candidate_grid_fast(
        angle_values=coarse_angles,
        x_ref_values=coarse_x_values,
        y_ref=y_ref,
        total_available_length_px=total_available_length_px,
        fast_cache=fast_search_cache,
        band_half_width_px=coarse_band_half_width_px,
        max_angle_error_deg=coarse_max_angle_error_deg,
    )
    coarse_candidates = sort_candidates(unique_candidates_by_axis(coarse_candidates))
    top_coarse = filter_candidates_by_score_ratio(coarse_candidates, coarse_score_floor_ratio)
    fine_candidates: list[dict] = []
    use_support_adjustment = bool(cfg("support_adjustment", "enabled", default=True))
    full_adjustment_top_hypotheses = int(cfg("support_adjustment", "full_adjustment_top_hypotheses", default=8))
    allow_support_adjustment_for_all_hypotheses = full_adjustment_top_hypotheses <= 0

    for coarse_candidate in top_coarse:
        angle_min = float(coarse_candidate["tilt_deg"]) - fine_window_angle_deg
        angle_max = float(coarse_candidate["tilt_deg"]) + fine_window_angle_deg
        x_ref_min = float(coarse_candidate["x_ref"]) - fine_window_x_px
        x_ref_max = float(coarse_candidate["x_ref"]) + fine_window_x_px

        fine_angles = np.arange(angle_min, angle_max + 0.5 * fine_angle_step_deg, fine_angle_step_deg)
        fine_x_values = np.arange(int(round(x_ref_min)), int(round(x_ref_max)) + 1, max(1, fine_x_step_px))

        fine_candidates.extend(
            make_candidate_grid_fast(
                angle_values=fine_angles,
                x_ref_values=fine_x_values,
                y_ref=y_ref,
                total_available_length_px=total_available_length_px,
                fast_cache=fast_search_cache,
                band_half_width_px=coarse_band_half_width_px,
                max_angle_error_deg=coarse_max_angle_error_deg,
            )
        )

    fine_candidates = sort_candidates(
        filter_candidates_by_score_ratio(
            unique_candidates_by_axis(fine_candidates or coarse_candidates),
            fine_score_floor_ratio,
        )
    )
    if not fine_candidates:
        return {
            "coarse_candidates": coarse_candidates,
            "fine_candidates": fine_candidates,
            "best_hypothesis": None,
            "best_candidate": None,
            "ranked_candidates": [],
        }

    fully_scored_hypotheses: list[dict] = []
    for screened_hypothesis in fine_candidates:
        detailed_hypothesis = evaluate_candidate(
            axis=screened_hypothesis,
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
        detailed_hypothesis["fast_screen_score"] = float(screened_hypothesis["score"])
        fully_scored_hypotheses.append(detailed_hypothesis)

    fine_candidates = sort_candidates(
        unique_candidates_by_axis(fully_scored_hypotheses),
        sort_key=lambda candidate: (
            float(candidate["score"]),
            float(candidate.get("fast_screen_score", candidate["score"])),
        ),
    )
    if not fine_candidates:
        return {
            "coarse_candidates": coarse_candidates,
            "fine_candidates": fine_candidates,
            "best_hypothesis": None,
            "best_candidate": None,
            "ranked_candidates": [],
        }

    evaluated_hypotheses = fine_candidates
    save_all_ranked_candidates = bool(context.STEP_CONFIG.get("save_all_final_candidates", False))
    ranked_candidate_limit = max(1, int(cfg("candidate_deduplication", "max_saved_candidates", default=8)))
    ranked_candidate_pool: list[dict] = [] if save_all_ranked_candidates else []
    ranked_candidate_heap: list[tuple[tuple[float, ...], int, dict]] = []
    ranked_candidate_total_count = 0
    fallback_finalists: list[dict] = []

    for hypothesis_rank, hypothesis in enumerate(evaluated_hypotheses, start=1):
        stage_candidates = evaluate_hypothesis_variants(
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            lines=lines,
            roi_profile=roi_profile,
            total_available_length_px=total_available_length_px,
            final_band_half_width_px=final_band_half_width_px,
            final_max_angle_error_deg=final_max_angle_error_deg,
            use_support_adjustment=(
                use_support_adjustment
                and (
                    allow_support_adjustment_for_all_hypotheses
                    or hypothesis_rank <= full_adjustment_top_hypotheses
                )
            ),
            y_ref=y_ref,
            support_cache=support_analysis_cache,
            fit_cache=fit_cache,
            row_metrics_cache=row_metrics_cache,
            candidate_summary_cache=candidate_summary_cache,
            selection_cache=support_selection_cache,
            line_selection_cache=line_selection_cache,
        )
        local_candidate_pool = list(stage_candidates)
        local_best = max(local_candidate_pool, key=candidate_ranking_key)
        harmonized_local_candidates = build_axis_harmonized_candidates(
            candidate=local_best,
            hypothesis=hypothesis,
            hypothesis_rank=hypothesis_rank,
            lines=lines,
            roi_profile=roi_profile,
            total_available_length_px=total_available_length_px,
            final_band_half_width_px=final_band_half_width_px,
            final_max_angle_error_deg=final_max_angle_error_deg,
            y_ref=y_ref,
            support_cache=support_analysis_cache,
            fit_cache=fit_cache,
            row_metrics_cache=row_metrics_cache,
            candidate_summary_cache=candidate_summary_cache,
            selection_cache=support_selection_cache,
            line_selection_cache=line_selection_cache,
        )
        if harmonized_local_candidates:
            local_candidate_pool.extend(harmonized_local_candidates)
            if float(local_best.get("adjusted_fragment_ratio", 0.0)) > 0.0:
                local_best = max(harmonized_local_candidates, key=candidate_ranking_key)
            else:
                local_best = max([local_best, *harmonized_local_candidates], key=candidate_ranking_key)
        for candidate in local_candidate_pool:
            ranked_candidate_total_count += 1
            if save_all_ranked_candidates:
                ranked_candidate_pool.append(candidate)
                continue

            ranking_key = candidate_ranking_key(candidate)
            heap_entry = (ranking_key, ranked_candidate_total_count, candidate)
            if len(ranked_candidate_heap) < ranked_candidate_limit:
                heapq.heappush(ranked_candidate_heap, heap_entry)
                continue
            if ranking_key > ranked_candidate_heap[0][0]:
                heapq.heapreplace(ranked_candidate_heap, heap_entry)
        fallback_finalists.append(local_best)

    if save_all_ranked_candidates:
        ranked_candidates = sort_candidates(
            ranked_candidate_pool or fallback_finalists,
            sort_key=candidate_ranking_key,
        )
    else:
        ranked_candidates = [
            entry[2]
            for entry in sorted(
                ranked_candidate_heap,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
        ]
    best_candidate = ranked_candidates[0] if ranked_candidates else max(fallback_finalists, key=candidate_ranking_key)
    best_hypothesis = evaluated_hypotheses[max(0, int(best_candidate.get("source_hypothesis_rank", 1)) - 1)]

    return {
        "coarse_candidates": coarse_candidates,
        "fine_candidates": fine_candidates,
        "ranked_candidates": ranked_candidates,
        "ranked_candidate_total_count": ranked_candidate_total_count if ranked_candidate_total_count > 0 else len(ranked_candidates),
        "best_hypothesis": best_hypothesis,
        "best_candidate": best_candidate,
    }
