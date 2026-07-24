from __future__ import annotations

import math

import numpy as np

from . import context
from .context import cfg, clip01
from .geometry import line_from_angle_and_anchor, line_geometry_key, line_x_at_y
from .calculations import (
    build_line_selection_cache,
    fit_axis_from_support,
    fragment_quality_sort_key,
    select_support_fragments,
    suppress_redundant_fragments,
)
from .metrics import (
    apply_mirror_symmetry,
    evaluate_candidate,
    prepare_mirror_symmetry_context,
    summarize_candidate_from_support,
)
from .sorting import (
    annotate_candidate_selection,
    candidate_ranking_key,
    deduplicate_candidates,
    mean_axis_distance_px,
    select_diverse_candidates,
    select_ranked_candidate_portfolio,
    sort_candidates,
    unique_candidates_by_axis,
)

def search_central_ruler(lines: list[dict],roi_profile: dict,edge_image: np.ndarray | None = None,) -> dict:
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
    probe_row_indices = np.clip(np.round(probe_rows).astype(np.int32), 0, roi_profile["height"] - 1)
    probe_left = roi_profile["left_bounds"][probe_row_indices].astype(np.float64)
    probe_right = roi_profile["right_bounds"][probe_row_indices].astype(np.float64)
    probe_width = np.maximum(1.0, roi_profile["row_widths"][probe_row_indices].astype(np.float64))
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

def make_candidate_grid_fast( angle_values: np.ndarray, x_ref_values: np.ndarray, y_ref: float, total_available_length_px: float, fast_cache: dict,band_half_width_px: float, max_angle_error_deg: float,) -> list[dict]:
    """Cheap geometry-only screening. Endpoint bonuses are deliberately absent."""
    if angle_values.size == 0 or x_ref_values.size == 0: return []

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
        support_mask = (angle_error <= max_angle_error_deg) & (axis_distance <= band_half_width_px)
        distance_alignment = np.clip(1.0 - axis_distance / max(1e-6, band_half_width_px), 0.0, 1.0)
        angle_alignment = np.clip(1.0 - angle_error / max(1e-6, max_angle_error_deg), 0.0, 1.0)
        support_strength = (line_length_column * (0.72 * distance_alignment + 0.28 * angle_alignment) * support_mask)

        selected_fragment_count = np.sum(support_mask, axis=0)
        selected_total_support_strength = np.sum(support_strength, axis=0)
        fragment_support_score = np.clip(selected_total_support_strength / total_available_length_px, 0.0, 1.0)
        covered_bins = (bin_coverage_t @ support_mask.astype(np.uint8)) > 0
        supported_bin_count = np.sum(covered_bins, axis=0)
        coverage_score = supported_bin_count.astype(np.float64) / max(1, bin_count)
        has_support = supported_bin_count > 0
        first_supported = np.where(has_support, np.argmax(covered_bins, axis=0), 0)
        last_supported = np.where(has_support,bin_count - 1 - np.argmax(covered_bins[::-1], axis=0),0,)
        support_span_bins = np.where(has_support, np.maximum(1, last_supported - first_supported + 1), 1)
        continuity_score = supported_bin_count.astype(np.float64) / support_span_bins

        axis_probe_x = axis_a * roi_probe_rows[:, None] + axis_b_values[None, :]
        inside_mask = (axis_probe_x >= roi_probe_left[:, None]) & (axis_probe_x <= roi_probe_right[:, None])
        inside_ratio = np.mean(inside_mask.astype(np.float64), axis=0)
        center_errors = np.abs(axis_probe_x - roi_probe_center[:, None]) / np.maximum(1.0, roi_probe_width[:, None] * 0.5)
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

def make_candidate_grid(angle_values: np.ndarray,x_ref_values: np.ndarray,y_ref: float,lines: list[dict],roi_profile: dict,total_available_length_px: float,band_half_width_px: float,max_angle_error_deg: float,allow_adjustment: bool = False,) -> list[dict]:
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

def _axis_inside_roi_ratio(axis: dict, roi_profile: dict, sample_count: int = 11) -> float:
    rows = np.linspace(float(roi_profile["trimmed_y_min"]),float(roi_profile["trimmed_y_max"]),max(3, int(sample_count)),dtype=np.float64,)
    row_indices = np.clip(np.round(rows).astype(np.int32), 0, int(roi_profile["height"]) - 1)
    x_values = float(axis["a"]) * rows + float(axis["b"])
    left = roi_profile["left_bounds"][row_indices].astype(np.float64)
    right = roi_profile["right_bounds"][row_indices].astype(np.float64)
    return float(np.mean((x_values >= left) & (x_values <= right)))

def _structural_line_quality(line: dict, roi_profile: dict) -> float:
    y_mid = int(np.clip(round(float(line["y_mid"])),0,int(roi_profile["height"]) - 1,))
    row_width = max(1.0, float(roi_profile["row_widths"][y_mid]))
    center_x = float(line_x_at_y(roi_profile["center_fit"], float(y_mid)))
    normalized_center_error = abs(float(line["x_mid"]) - center_x) / max(1.0, 0.5 * row_width)
    center_score = clip01(1.0 - normalized_center_error / 0.72)
    mask_support = clip01(float(line.get("mask_support_ratio", 1.0)))
    inside_strength = clip01(float(line.get("points_inside_mask", line.get("length", 0.0)))/ max(1.0, float(line.get("length", 1.0))))
    return float(
        max(1.0, float(line["length"]))
        * (0.55 + 0.25 * mask_support + 0.20 * inside_strength)
        * (0.30 + 0.70 * center_score)
    )

def _make_seed_axis(a_value: float,b_value: float,y_ref: float,source: str,seed_score: float,**metadata,) -> dict:
    tilt_deg = float(math.degrees(math.atan(float(a_value))))
    return {
        "a": float(a_value),
        "b": float(b_value),
        "tilt_deg": tilt_deg,
        "x_ref": float(a_value * y_ref + b_value),
        "y_ref": float(y_ref),
        "hypothesis_source": source,
        "structural_seed_score": float(seed_score),
        **metadata,
    }

def build_structural_seed_axes(lines: list[dict], roi_profile: dict) -> list[dict]:
    """Create high-recall axes from real fragments and a weak ROI prior.

    The normal grid remains authoritative. These seeds specifically recover axes
    that lie between two strong local maxima or that are represented only by
    separated upper/lower fragments.
    """
    if not bool(cfg("structural_hypotheses", "enabled", default=True)):return []
    if not lines: return []

    y_min = float(roi_profile["trimmed_y_min"])
    y_max = float(roi_profile["trimmed_y_max"])
    y_ref = float(roi_profile["y_ref"])
    roi_span = max(1.0, y_max - y_min)
    max_tilt_deg = float(cfg("search", "max_candidate_tilt_deg", default=12.0))
    min_inside_ratio = float(cfg("structural_hypotheses","pair_seed_min_axis_inside_ratio",default=0.72,))

    quality_by_index = {
        int(line["line_index"]): _structural_line_quality(line, roi_profile)
        for line in lines
    }
    max_quality = max(quality_by_index.values(), default=1.0)
    seed_axes: list[dict] = []

    if bool(cfg("structural_hypotheses", "line_seed_enabled", default=True)):
        min_length = float(cfg("structural_hypotheses","min_line_seed_length_px",default=70.0,))
        max_count = max(0,int(cfg("structural_hypotheses","max_line_seed_count",default=48,)),)
        line_candidates = [
            line
            for line in lines
            if float(line["length"]) >= min_length and abs(float(line["signed_tilt_deg"])) <= max_tilt_deg
        ]
        line_candidates.sort(
            key=lambda line: (
                quality_by_index[int(line["line_index"])],
                *fragment_quality_sort_key(line),
            ),
            reverse=True,
        )
        for line in line_candidates[:max_count]:
            axis = _make_seed_axis(
                float(line["a"]),
                float(line["b"]),
                y_ref,
                "fragment_axis",
                quality_by_index[int(line["line_index"])] / max_quality,
                seed_line_indices=[int(line["line_index"])],
            )
            if _axis_inside_roi_ratio(axis, roi_profile) >= min_inside_ratio:
                seed_axes.append(axis)

    if bool(
        cfg("structural_hypotheses", "pair_seed_enabled", default=True)
    ):
        zone_count = max(
            2,
            int(
                cfg(
                    "structural_hypotheses", "vertical_zone_count", default=6
                )
            ),
        )
        per_zone = max(
            1,
            int(
                cfg(
                    "structural_hypotheses",
                    "max_fragments_per_zone",
                    default=12,
                )
            ),
        )
        max_sources = max(
            2,
            int(
                cfg(
                    "structural_hypotheses",
                    "max_pair_source_fragments",
                    default=64,
                )
            ),
        )
        zones: list[list[dict]] = [[] for _ in range(zone_count)]
        for line in lines:
            ratio = clip01((float(line["y_mid"]) - y_min) / roi_span)
            zone_index = min(zone_count - 1, int(ratio * zone_count))
            zones[zone_index].append(line)
        source_lines: list[dict] = []
        for zone in zones:
            zone.sort(
                key=lambda line: (
                    quality_by_index[int(line["line_index"])],
                    *fragment_quality_sort_key(line),
                ),
                reverse=True,
            )
            source_lines.extend(zone[:per_zone])
        source_lines = list(
            {
                int(line["line_index"]): line for line in source_lines
            }.values()
        )
        source_lines.sort(
            key=lambda line: (
                quality_by_index[int(line["line_index"])],
                *fragment_quality_sort_key(line),
            ),
            reverse=True,
        )
        source_lines = source_lines[:max_sources]
        source_lines.sort(key=line_geometry_key)

        min_separation = roi_span * float(
            cfg(
                "structural_hypotheses",
                "min_pair_vertical_separation_ratio",
                default=0.18,
            )
        )
        max_separation = roi_span * float(
            cfg(
                "structural_hypotheses",
                "max_pair_vertical_separation_ratio",
                default=0.96,
            )
        )
        max_fragment_angle_error = float(
            cfg(
                "structural_hypotheses",
                "max_pair_fragment_angle_error_deg",
                default=6.5,
            )
        )
        pair_candidates: list[dict] = []
        for upper_index, upper in enumerate(source_lines):
            y_upper = float(upper["y_mid"])
            x_upper = float(line_x_at_y(upper, y_upper))
            for lower in source_lines[upper_index + 1 :]:
                y_lower = float(lower["y_mid"])
                separation = y_lower - y_upper
                if separation < min_separation:
                    continue
                if separation > max_separation:
                    break
                x_lower = float(line_x_at_y(lower, y_lower))
                a_value = (x_lower - x_upper) / max(1e-6, separation)
                tilt_deg = float(math.degrees(math.atan(a_value)))
                if abs(tilt_deg) > max_tilt_deg:
                    continue
                if max(
                    abs(float(upper["signed_tilt_deg"]) - tilt_deg),
                    abs(float(lower["signed_tilt_deg"]) - tilt_deg),
                ) > max_fragment_angle_error:
                    continue
                b_value = x_upper - a_value * y_upper
                separation_ratio = separation / roi_span
                line_quality = math.sqrt(
                    quality_by_index[int(upper["line_index"])]
                    * quality_by_index[int(lower["line_index"])]
                ) / max_quality
                angle_alignment = 1.0 - clip01(
                    (
                        abs(float(upper["signed_tilt_deg"]) - tilt_deg)
                        + abs(float(lower["signed_tilt_deg"]) - tilt_deg)
                    )
                    / max(1e-6, 2.0 * max_fragment_angle_error)
                )
                axis = _make_seed_axis(
                    a_value,
                    b_value,
                    y_ref,
                    "fragment_pair",
                    0.46 * clip01(separation_ratio)
                    + 0.34 * clip01(line_quality)
                    + 0.20 * clip01(angle_alignment),
                    seed_line_indices=[
                        int(upper["line_index"]),
                        int(lower["line_index"]),
                    ],
                    seed_vertical_separation_ratio=float(separation_ratio),
                )
                inside_ratio = _axis_inside_roi_ratio(axis, roi_profile)
                if inside_ratio < min_inside_ratio:
                    continue
                axis["structural_seed_inside_roi_ratio"] = inside_ratio
                pair_candidates.append(axis)

        pair_candidates.sort(
            key=lambda axis: (
                float(axis["structural_seed_score"]),
                *candidate_ranking_key(axis),
            ),
            reverse=True,
        )
        pair_limit = max(
            0,
            int(
                cfg(
                    "structural_hypotheses",
                    "max_pair_seed_count",
                    default=320,
                )
            ),
        )
        seed_axes.extend(pair_candidates[:pair_limit])

    if bool(
        cfg("structural_hypotheses", "roi_prior_seed_enabled", default=True)
    ):
        center_fit = roi_profile["center_fit"]
        center_x_ref = float(line_x_at_y(center_fit, y_ref))
        center_tilt = float(center_fit.get("tilt_deg", 0.0))
        reference_width = max(1.0, float(roi_profile["reference_width_px"]))
        x_offset_ratios = cfg(
            "structural_hypotheses",
            "center_x_offset_ratios",
            default=[-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10],
        )
        angle_offsets = cfg(
            "structural_hypotheses",
            "center_angle_offsets_deg",
            default=[-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
        )
        for x_ratio in x_offset_ratios:
            for angle_offset in angle_offsets:
                tilt_deg = center_tilt + float(angle_offset)
                if abs(tilt_deg) > max_tilt_deg:
                    continue
                x_ref = center_x_ref + float(x_ratio) * reference_width
                axis = line_from_angle_and_anchor(tilt_deg, x_ref, y_ref)
                axis.update(
                    {
                        "hypothesis_source": "roi_prior",
                        "structural_seed_score": float(
                            1.0
                            - 0.55 * min(1.0, abs(float(x_ratio)) / 0.10)
                            - 0.45 * min(1.0, abs(float(angle_offset)) / 4.0)
                        ),
                        "roi_prior_x_offset_ratio": float(x_ratio),
                        "roi_prior_angle_offset_deg": float(angle_offset),
                    }
                )
                if _axis_inside_roi_ratio(axis, roi_profile) >= min_inside_ratio:
                    seed_axes.append(axis)

    # First remove exact axes, then keep a source-balanced subset. Pair seeds
    # receive the largest share because they are the important sparse fallback.
    unique_axes = unique_candidates_by_axis(seed_axes)
    max_total = max(
        1,
        int(
            cfg(
                "structural_hypotheses",
                "max_structural_detailed_candidates",
                default=96,
            )
        ),
    )
    source_limits = {
        "fragment_pair": max(1, int(round(max_total * 0.58))),
        "fragment_axis": max(1, int(round(max_total * 0.17))),
        "roi_prior": max(1, int(round(max_total * 0.25))),
    }
    selected: list[dict] = []
    for source in ("fragment_pair", "fragment_axis", "roi_prior"):
        group = [
            axis for axis in unique_axes if axis.get("hypothesis_source") == source
        ]
        group.sort(
            key=lambda axis: (
                float(axis.get("structural_seed_score", 0.0)),
                *candidate_ranking_key(axis),
            ),
            reverse=True,
        )
        selected.extend(group[: source_limits[source]])
    selected.sort(
        key=lambda axis: (
            float(axis.get("structural_seed_score", 0.0)),
            *candidate_ranking_key(axis),
        ),
        reverse=True,
    )
    return selected[:max_total]

def select_hypothesis_portfolio(
    candidates: list[dict],
    roi_profile: dict,
    max_candidates: int,
) -> list[dict]:
    """Reserve some final-fit slots for structurally different seed families."""
    if max_candidates <= 0 or not candidates:
        return []
    ordered = sort_candidates(
        unique_candidates_by_axis(candidates), sort_key=candidate_ranking_key
    )
    source_quotas = {
        "fragment_pair": max(
            0,
            int(
                cfg(
                    "structural_hypotheses",
                    "reserved_pair_hypotheses",
                    default=12,
                )
            ),
        ),
        "fragment_axis": max(
            0,
            int(
                cfg(
                    "structural_hypotheses",
                    "reserved_line_hypotheses",
                    default=4,
                )
            ),
        ),
        "roi_prior": max(
            0,
            int(
                cfg(
                    "structural_hypotheses",
                    "reserved_roi_prior_hypotheses",
                    default=6,
                )
            ),
        ),
    }
    # Never let reservations consume every slot; the original grid keeps at
    # least one third of the portfolio.
    max_reserved = max(0, max_candidates - max(4, max_candidates // 3))
    quota_sum = sum(source_quotas.values())
    if quota_sum > max_reserved and quota_sum > 0:
        scale = max_reserved / quota_sum
        source_quotas = {
            source: int(math.floor(value * scale))
            for source, value in source_quotas.items()
        }

    probe_rows = np.linspace(
        float(roi_profile["trimmed_y_min"]),
        float(roi_profile["trimmed_y_max"]),
        8,
        dtype=np.float64,
    )
    duplicate_distance = float(
        cfg(
            "candidate_deduplication",
            "max_mean_axis_distance_px",
            default=5.0,
        )
    )
    duplicate_angle = float(
        cfg(
            "candidate_deduplication",
            "max_angle_difference_deg",
            default=0.25,
        )
    )
    selected: list[dict] = []

    def append_if_distinct(candidate: dict) -> bool:
        for existing in selected:
            if (
                abs(
                    float(candidate["tilt_deg"])
                    - float(existing["tilt_deg"])
                )
                <= duplicate_angle
                and mean_axis_distance_px(
                    candidate, existing, probe_rows=probe_rows
                )
                <= duplicate_distance
            ):
                return False
        selected.append(candidate)
        return True

    for source in ("fragment_pair", "fragment_axis", "roi_prior"):
        quota = source_quotas[source]
        if quota <= 0:
            continue
        added = 0
        for candidate in ordered:
            if candidate.get("hypothesis_source", "grid") != source:
                continue
            if append_if_distinct(candidate):
                added += 1
            if added >= quota or len(selected) >= max_candidates:
                break

    for candidate in ordered:
        if len(selected) >= max_candidates:
            break
        append_if_distinct(candidate)

    return sort_candidates(selected, sort_key=candidate_ranking_key)

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

    detailed_hypotheses: list[dict] = []
    for screened in fine_screened:
        screened_axis = dict(screened)
        screened_axis.setdefault("hypothesis_source", "grid")
        candidate = evaluate_candidate(
            axis=screened_axis,
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
        candidate["hypothesis_source"] = "grid"
        detailed_hypotheses.append(candidate)

    structural_axes = build_structural_seed_axes(lines, roi_profile)
    structural_band_half_width_px = float(
        cfg(
            "structural_hypotheses",
            "pair_seed_band_half_width_px",
            default=21.0,
        )
    )
    structural_max_angle_error_deg = float(
        cfg(
            "structural_hypotheses",
            "pair_seed_max_angle_error_deg",
            default=7.0,
        )
    )
    for structural_axis in structural_axes:
        candidate = evaluate_candidate(
            axis=structural_axis,
            lines=lines,
            roi_profile=roi_profile,
            total_available_length_px=total_available_length_px,
            band_half_width_px=structural_band_half_width_px,
            max_angle_error_deg=structural_max_angle_error_deg,
            allow_adjustment=False,
            support_cache=support_analysis_cache,
            row_metrics_cache=row_metrics_cache,
            candidate_summary_cache=candidate_summary_cache,
            selection_cache=support_selection_cache,
            line_selection_cache=line_selection_cache,
        )
        for metadata_key in (
            "hypothesis_source",
            "structural_seed_score",
            "seed_line_indices",
            "seed_vertical_separation_ratio",
            "structural_seed_inside_roi_ratio",
            "roi_prior_x_offset_ratio",
            "roi_prior_angle_offset_deg",
        ):
            if metadata_key in structural_axis:
                candidate[metadata_key] = structural_axis[metadata_key]
        candidate["hypothesis_band_half_width_px"] = float(
            structural_band_half_width_px
        )
        candidate["hypothesis_max_angle_error_deg"] = float(
            structural_max_angle_error_deg
        )
        detailed_hypotheses.append(candidate)

    detailed_hypotheses = sort_candidates(
        unique_candidates_by_axis(detailed_hypotheses),
        sort_key=candidate_ranking_key,
    )
    if not detailed_hypotheses:
        return _empty_result(raw_line_count=raw_line_count, nms_line_count=len(lines))

    top_hypothesis_count = max(
        1, int(cfg("best_fit_selection", "top_hypothesis_count", default=24))
    )
    evaluated_hypotheses = select_hypothesis_portfolio(
        detailed_hypotheses,
        roi_profile,
        max_candidates=top_hypothesis_count,
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
    mirror_context = prepare_mirror_symmetry_context(edge_image, roi_profile)
    mirror_scored = [
        apply_mirror_symmetry(
            candidate,
            edge_image,
            roi_profile,
            mirror_context=mirror_context,
        )
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
    ranked_candidates = select_ranked_candidate_portfolio(
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
        "structural_seed_count": int(len(structural_axes)),
        "evaluated_hypothesis_count": int(len(evaluated_hypotheses)),
        "nms_removed_line_count": int(raw_line_count - len(lines)),
    }
