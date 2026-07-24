from __future__ import annotations

import math

import numpy as np

from .context import cfg, clip01, safe_linear_polyfit
from .geometry import (
    line_geometry_key,
    line_x_at_y,
    make_axis_signature,
    segment_axis_distance_px,
    segment_axis_intersection_y,
    segment_axis_min_distance_px,
)

def support_item_sort_key(item: dict) -> tuple[float, ...]:
    line = item.get("effective_line", item["line"])
    return (
        float(item.get("support_strength", 0.0)),
        float(item.get("distance_alignment", 0.0)),
        float(item.get("angle_alignment", 0.0)),
        float(item.get("line", {}).get("length", 0.0)),
        *line_geometry_key(line),
    )

def fragment_quality_sort_key(line: dict) -> tuple[float, ...]:
    return (
        float(line.get("length", 0.0)),
        float(line.get("mask_support_ratio", 0.0)),
        float(line.get("points_inside_mask", 0.0)),
        -float(line.get("vertical_deviation_degrees", abs(line.get("signed_tilt_deg", 0.0)))),
        *line_geometry_key(line),
    )

def calculate_angle(line: dict[str, float | int | bool] | tuple[tuple[float, float], tuple[float, float]]) -> float:
    if isinstance(line, dict):
        if "signed_tilt_deg" in line:
            return float(line["signed_tilt_deg"])
        if "tilt_deg" in line:
            return float(line["tilt_deg"])
        if "a" in line:
            return float(math.degrees(math.atan(float(line["a"]))))
        if {"x1", "y1", "x2", "y2"}.issubset(line.keys()):
            return float(
                math.degrees(
                    math.atan2(
                        float(line["x2"]) - float(line["x1"]),
                        float(line["y2"]) - float(line["y1"]),
                    )
                )
            )
        raise KeyError("Line mapping must contain signed_tilt_deg, tilt_deg, a, or endpoint coordinates.")

    (x1, y1), (x2, y2) = line
    return float(math.degrees(math.atan2(float(x2) - float(x1), float(y2) - float(y1))))

def make_zero_adjustment(line: dict, axis: dict[str, float], axis_distance_px: float, angle_error_deg: float) -> dict[str, float | bool]:
    return {
        "is_adjusted": False,
        "midpoint_shift_px": 0.0,
        "abs_midpoint_shift_px": 0.0,
        "tilt_delta_deg": 0.0,
        "abs_tilt_delta_deg": 0.0,
        "mean_abs_shift_px": 0.0,
        "max_abs_shift_px": 0.0,
        "original_axis_distance_px": float(axis_distance_px),
        "original_angle_error_deg": float(angle_error_deg),
        "effective_axis_distance_px": float(axis_distance_px),
        "effective_angle_error_deg": float(angle_error_deg),
        "distance_gain_px": 0.0,
        "angle_gain_deg": 0.0,
    }

def make_zero_adjustment_from_metrics(axis_distance_px: float, angle_error_deg: float) -> dict[str, float | bool]:
    return {
        "is_adjusted": False,
        "midpoint_shift_px": 0.0,
        "abs_midpoint_shift_px": 0.0,
        "tilt_delta_deg": 0.0,
        "abs_tilt_delta_deg": 0.0,
        "mean_abs_shift_px": 0.0,
        "max_abs_shift_px": 0.0,
        "original_axis_distance_px": float(axis_distance_px),
        "original_angle_error_deg": float(angle_error_deg),
        "effective_axis_distance_px": float(axis_distance_px),
        "effective_angle_error_deg": float(angle_error_deg),
        "distance_gain_px": 0.0,
        "angle_gain_deg": 0.0,
    }

def build_line_selection_cache(lines: list[dict]) -> dict:
    return {
        "lines": tuple(lines),
        "line_count": int(len(lines)),
        "line_a": np.asarray([float(line["a"]) for line in lines], dtype=np.float64),
        "line_b": np.asarray([float(line["b"]) for line in lines], dtype=np.float64),
        "line_tilt_deg": np.asarray([float(line["signed_tilt_deg"]) for line in lines], dtype=np.float64),
        "line_length": np.asarray([float(line["length"]) for line in lines], dtype=np.float64),
        "line_y_min": np.asarray([float(line["y_min"]) for line in lines], dtype=np.float64),
        "line_y_mid": np.asarray([float(line["y_mid"]) for line in lines], dtype=np.float64),
        "line_y_max": np.asarray([float(line["y_max"]) for line in lines], dtype=np.float64),
    }

def build_support_item_from_metrics(
    line: dict,
    line_length_px: float,
    axis_distance_px: float,
    angle_error_deg: float,
    band_half_width_px: float,
    max_angle_error_deg: float,
) -> dict:
    distance_alignment = clip01(1.0 - axis_distance_px / max(1e-6, band_half_width_px))
    angle_alignment = clip01(1.0 - angle_error_deg / max(1e-6, max_angle_error_deg))
    support_strength = float(line_length_px) * (0.72 * distance_alignment + 0.28 * angle_alignment)
    return {
        "line": line,
        "effective_line": line,
        "axis_distance_px": float(axis_distance_px),
        "angle_error_deg": float(angle_error_deg),
        "distance_alignment": float(distance_alignment),
        "angle_alignment": float(angle_alignment),
        "support_strength": float(support_strength),
        "adjustment": make_zero_adjustment_from_metrics(axis_distance_px, angle_error_deg),
    }

def build_adjusted_line_variant(line: dict, axis: dict[str, float]) -> tuple[dict, dict] | tuple[None, None]:
    if not bool(cfg("support_adjustment", "enabled", default=True)):
        return None, None

    max_midpoint_shift_px = float(cfg("support_adjustment", "max_midpoint_shift_px", default=10.0))
    max_tilt_delta_deg = float(cfg("support_adjustment", "max_tilt_delta_deg", default=2.0))
    max_mean_shift_px = float(cfg("support_adjustment", "max_mean_shift_px", default=10.0))
    max_endpoint_shift_px = float(cfg("support_adjustment", "max_endpoint_shift_px", default=18.0))
    allow_tilt_without_intersection_axis_distance_px = float(
        cfg("support_adjustment", "allow_tilt_without_intersection_axis_distance_px", default=8.0)
    )
    require_axis_intersection_for_tilt_adjustment = bool(
        cfg("support_adjustment", "require_axis_intersection_for_tilt_adjustment", default=True)
    )

    original_tilt_deg = float(line["signed_tilt_deg"])
    target_tilt_deg = float(axis["tilt_deg"])
    original_axis_distance_px = float(segment_axis_distance_px(line, axis))
    applied_tilt_delta_deg = float(
        max(-max_tilt_delta_deg, min(max_tilt_delta_deg, target_tilt_deg - original_tilt_deg))
    )
    original_segment_intersection_y = segment_axis_intersection_y(line, axis)
    tilt_adjustment_blocked = False
    if (
        require_axis_intersection_for_tilt_adjustment
        and abs(applied_tilt_delta_deg) >= 0.05
        and original_segment_intersection_y is None
        and original_axis_distance_px > allow_tilt_without_intersection_axis_distance_px
    ):
        applied_tilt_delta_deg = 0.0
        tilt_adjustment_blocked = True
    adjusted_tilt_deg = original_tilt_deg + applied_tilt_delta_deg
    adjusted_a = math.tan(math.radians(adjusted_tilt_deg))

    y_mid = float(line["y_mid"])
    target_x_mid = float(line_x_at_y(axis, y_mid))
    applied_midpoint_shift_px = float(
        max(-max_midpoint_shift_px, min(max_midpoint_shift_px, target_x_mid - float(line["x_mid"])))
    )
    adjusted_b = float(line["x_mid"] + applied_midpoint_shift_px - adjusted_a * y_mid)

    adjusted_line = dict(line)
    adjusted_line["a"] = float(adjusted_a)
    adjusted_line["b"] = float(adjusted_b)
    adjusted_line["signed_tilt_deg"] = float(adjusted_tilt_deg)
    adjusted_line["x1"] = float(line_x_at_y(adjusted_line, float(line["y1"])))
    adjusted_line["x2"] = float(line_x_at_y(adjusted_line, float(line["y2"])))
    adjusted_line["x_mid"] = float(line_x_at_y(adjusted_line, y_mid))
    adjusted_line["length"] = float(
        math.hypot(
            float(adjusted_line["x2"]) - float(adjusted_line["x1"]),
            float(adjusted_line["y2"]) - float(adjusted_line["y1"]),
        )
    )

    delta_a = float(adjusted_line["a"]) - float(line["a"])
    delta_b = float(adjusted_line["b"]) - float(line["b"])
    shift_min = delta_a * float(line["y_min"]) + delta_b
    shift_mid = delta_a * y_mid + delta_b
    shift_max = delta_a * float(line["y_max"]) + delta_b
    abs_shift_min = abs(shift_min)
    abs_shift_mid = abs(shift_mid)
    abs_shift_max = abs(shift_max)
    mean_abs_shift_px = float((abs_shift_min + abs_shift_mid + abs_shift_max) / 3.0)
    max_abs_shift_px = float(max(abs_shift_min, abs_shift_mid, abs_shift_max))

    if mean_abs_shift_px > max_mean_shift_px or max_abs_shift_px > max_endpoint_shift_px:
        return None, None

    if abs(applied_midpoint_shift_px) < 0.05 and abs(applied_tilt_delta_deg) < 0.05:
        return None, None

    original_angle_error_deg = abs(original_tilt_deg - target_tilt_deg)
    adjusted_axis_distance_px = float(segment_axis_distance_px(adjusted_line, axis))
    adjusted_angle_error_deg = abs(adjusted_tilt_deg - target_tilt_deg)

    adjustment = {
        "is_adjusted": True,
        "midpoint_shift_px": float(applied_midpoint_shift_px),
        "abs_midpoint_shift_px": abs(float(applied_midpoint_shift_px)),
        "tilt_delta_deg": float(applied_tilt_delta_deg),
        "abs_tilt_delta_deg": abs(float(applied_tilt_delta_deg)),
        "mean_abs_shift_px": float(mean_abs_shift_px),
        "max_abs_shift_px": float(max_abs_shift_px),
        "original_axis_distance_px": float(original_axis_distance_px),
        "original_angle_error_deg": float(original_angle_error_deg),
        "effective_axis_distance_px": float(adjusted_axis_distance_px),
        "effective_angle_error_deg": float(adjusted_angle_error_deg),
        "distance_gain_px": float(original_axis_distance_px - adjusted_axis_distance_px),
        "angle_gain_deg": float(original_angle_error_deg - adjusted_angle_error_deg),
        "original_segment_intersection_y": original_segment_intersection_y,
        "tilt_adjustment_blocked_no_axis_intersection": bool(tilt_adjustment_blocked),
    }
    return adjusted_line, adjustment

def build_support_item(
    line: dict,
    effective_line: dict,
    axis: dict[str, float],
    band_half_width_px: float,
    max_angle_error_deg: float,
    adjustment: dict | None = None,
) -> dict | None:
    angle_error = abs(float(effective_line["signed_tilt_deg"]) - float(axis["tilt_deg"]))
    if angle_error > max_angle_error_deg:
        return None

    axis_distance_px = segment_axis_distance_px(effective_line, axis)
    if axis_distance_px > band_half_width_px:
        return None

    distance_alignment = clip01(1.0 - axis_distance_px / max(1e-6, band_half_width_px))
    angle_alignment = clip01(1.0 - angle_error / max(1e-6, max_angle_error_deg))
    support_strength = float(line["length"]) * (0.72 * distance_alignment + 0.28 * angle_alignment)

    resolved_adjustment = adjustment
    if resolved_adjustment is None:
        resolved_adjustment = make_zero_adjustment(line, axis, axis_distance_px, angle_error)
    elif bool(resolved_adjustment.get("is_adjusted", False)):
        min_support_strength_scale = float(
            cfg("support_adjustment", "min_support_strength_scale", default=0.62)
        )
        shift_reference_px = float(cfg("support_adjustment", "max_mean_shift_px", default=10.0))
        tilt_reference_deg = float(cfg("support_adjustment", "max_tilt_delta_deg", default=2.0))
        normalized_shift = clip01(
            float(resolved_adjustment["mean_abs_shift_px"]) / max(1e-6, shift_reference_px)
        )
        normalized_tilt = clip01(
            float(resolved_adjustment["abs_tilt_delta_deg"]) / max(1e-6, tilt_reference_deg)
        )
        normalized_cost = 0.5 * normalized_shift + 0.5 * normalized_tilt
        adjustment_scale = min_support_strength_scale + (1.0 - min_support_strength_scale) * (1.0 - normalized_cost)
        support_strength *= adjustment_scale
        resolved_adjustment = {
            **resolved_adjustment,
            "effective_axis_distance_px": float(axis_distance_px),
            "effective_angle_error_deg": float(angle_error),
        }

    return {
        "line": line,
        "effective_line": effective_line,
        "axis_distance_px": float(axis_distance_px),
        "angle_error_deg": float(angle_error),
        "distance_alignment": float(distance_alignment),
        "angle_alignment": float(angle_alignment),
        "support_strength": float(support_strength),
        "adjustment": resolved_adjustment,
    }

def build_best_support_item(
    line: dict,
    axis: dict[str, float],
    band_half_width_px: float,
    max_angle_error_deg: float,
    allow_adjustment: bool,
) -> dict | None:
    best_item = build_support_item(
        line=line,
        effective_line=line,
        axis=axis,
        band_half_width_px=band_half_width_px,
        max_angle_error_deg=max_angle_error_deg,
    )

    if allow_adjustment and bool(cfg("support_adjustment", "enabled", default=True)):
        if best_item is not None:
            if (
                float(best_item["axis_distance_px"]) <= band_half_width_px * 0.20
                and float(best_item["angle_error_deg"]) <= max_angle_error_deg * 0.20
            ):
                return best_item
        adjusted_line, adjustment = build_adjusted_line_variant(line, axis)
        if adjusted_line is not None and adjustment is not None:
            adjusted_item = build_support_item(
                line=line,
                effective_line=adjusted_line,
                axis=axis,
                band_half_width_px=band_half_width_px,
                max_angle_error_deg=max_angle_error_deg,
                adjustment=adjustment,
            )
            if adjusted_item is not None and (
                best_item is None
                or float(adjusted_item["support_strength"]) > float(best_item["support_strength"])
            ):
                best_item = adjusted_item

    return best_item

def select_support_fragments(
    lines: list[dict],
    axis: dict[str, float],
    band_half_width_px: float,
    max_angle_error_deg: float,
    allow_adjustment: bool = False,
    selection_cache: dict | None = None,
    line_selection_cache: dict | None = None,
) -> list[dict]:
    cache_key = None
    if selection_cache is not None:
        cache_key = (
            make_axis_signature(axis),
            float(band_half_width_px),
            float(max_angle_error_deg),
            bool(allow_adjustment),
        )
        cached_selection = selection_cache.get(cache_key)
        if cached_selection is not None:
            return list(cached_selection)

    use_fast_cache = (
        line_selection_cache is not None
        and int(line_selection_cache.get("line_count", -1)) == len(lines)
    )
    support_adjustment_enabled = allow_adjustment and bool(cfg("support_adjustment", "enabled", default=True))

    if use_fast_cache:
        axis_a = float(axis["a"])
        axis_b = float(axis["b"])
        axis_tilt_deg = float(axis["tilt_deg"])

        line_a = line_selection_cache["line_a"]
        line_b = line_selection_cache["line_b"]
        line_tilt_deg = line_selection_cache["line_tilt_deg"]
        line_length = line_selection_cache["line_length"]
        line_y_min = line_selection_cache["line_y_min"]
        line_y_mid = line_selection_cache["line_y_mid"]
        line_y_max = line_selection_cache["line_y_max"]

        angle_error = np.abs(line_tilt_deg - axis_tilt_deg)
        delta_a = line_a - axis_a
        axis_distance = (
            np.abs(delta_a * line_y_min + line_b - axis_b)
            + np.abs(delta_a * line_y_mid + line_b - axis_b)
            + np.abs(delta_a * line_y_max + line_b - axis_b)
        ) / 3.0
        valid_indices = np.flatnonzero(
            (angle_error <= max_angle_error_deg) & (axis_distance <= band_half_width_px)
        )

        base_items_by_index: dict[int, dict] = {}
        cached_lines = line_selection_cache["lines"]
        for raw_index in valid_indices.tolist():
            line = cached_lines[raw_index]
            base_items_by_index[raw_index] = build_support_item_from_metrics(
                line=line,
                line_length_px=float(line_length[raw_index]),
                axis_distance_px=float(axis_distance[raw_index]),
                angle_error_deg=float(angle_error[raw_index]),
                band_half_width_px=band_half_width_px,
                max_angle_error_deg=max_angle_error_deg,
            )

        if not support_adjustment_enabled:
            selected = list(base_items_by_index.values())
            selected.sort(key=support_item_sort_key, reverse=True)
            if selection_cache is not None and cache_key is not None:
                selection_cache[cache_key] = tuple(selected)
            return selected

        near_distance_limit = band_half_width_px * 0.20
        near_angle_limit = max_angle_error_deg * 0.20
        max_tilt_delta_deg = float(cfg("support_adjustment", "max_tilt_delta_deg", default=2.0))
        selected = []
        for line_index, line in enumerate(cached_lines):
            best_item = base_items_by_index.get(line_index)
            if (
                best_item is not None
                and float(best_item["axis_distance_px"]) <= near_distance_limit
                and float(best_item["angle_error_deg"]) <= near_angle_limit
            ):
                selected.append(best_item)
                continue
            if best_item is None and float(angle_error[line_index]) > max_angle_error_deg + max_tilt_delta_deg:
                continue

            adjusted_line, adjustment = build_adjusted_line_variant(line, axis)
            if adjusted_line is not None and adjustment is not None:
                adjusted_item = build_support_item(
                    line=line,
                    effective_line=adjusted_line,
                    axis=axis,
                    band_half_width_px=band_half_width_px,
                    max_angle_error_deg=max_angle_error_deg,
                    adjustment=adjustment,
                )
                if adjusted_item is not None and (
                    best_item is None
                    or float(adjusted_item["support_strength"]) > float(best_item["support_strength"])
                ):
                    best_item = adjusted_item
            if best_item is not None:
                selected.append(best_item)

        selected.sort(key=support_item_sort_key, reverse=True)
        if selection_cache is not None and cache_key is not None:
            selection_cache[cache_key] = tuple(selected)
        return selected

    selected = []
    for line in lines:
        item = build_best_support_item(
            line=line,
            axis=axis,
            band_half_width_px=band_half_width_px,
            max_angle_error_deg=max_angle_error_deg,
            allow_adjustment=allow_adjustment,
        )
        if item is not None:
            selected.append(item)

    selected.sort(key=support_item_sort_key, reverse=True)
    if selection_cache is not None and cache_key is not None:
        selection_cache[cache_key] = tuple(selected)
    return selected

def merge_support_intervals(selected_support: list[dict]) -> list[tuple[float, float]]:
    intervals = [
        (float(item["line"]["y_min"]), float(item["line"]["y_max"]))
        for item in selected_support
    ]
    return merge_numeric_intervals(intervals)

def merge_numeric_intervals(
    intervals: list[tuple[float, float]],
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for start, end in intervals:
        start_value = float(min(start, end))
        end_value = float(max(start, end))
        if clip_min is not None:
            start_value = max(start_value, float(clip_min))
        if clip_max is not None:
            end_value = min(end_value, float(clip_max))
        if end_value > start_value:
            normalized.append((start_value, end_value))
    normalized.sort()
    if not normalized:
        return []

    merged: list[list[float]] = [[normalized[0][0], normalized[0][1]]]
    for start, end in normalized[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]

def interval_union_length(intervals: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, end - start) for start, end in merge_numeric_intervals(intervals)))

def line_y_overlap_ratio(line_a: dict, line_b: dict) -> float:
    overlap = max(
        0.0,
        min(float(line_a["y_max"]), float(line_b["y_max"]))
        - max(float(line_a["y_min"]), float(line_b["y_min"])),
    )
    min_span = max(
        1.0,
        min(
            float(line_a["y_max"]) - float(line_a["y_min"]),
            float(line_b["y_max"]) - float(line_b["y_min"]),
        ),
    )
    return float(overlap / min_span)

def mean_line_distance_on_overlap(line_a: dict, line_b: dict) -> float:
    y_start = max(float(line_a["y_min"]), float(line_b["y_min"]))
    y_end = min(float(line_a["y_max"]), float(line_b["y_max"]))
    if y_end <= y_start:
        y_start = min(float(line_a["y_mid"]), float(line_b["y_mid"]))
        y_end = max(float(line_a["y_mid"]), float(line_b["y_mid"]))
    rows = np.linspace(y_start, y_end, 3, dtype=np.float64)
    distances = [abs(line_x_at_y(line_a, float(y)) - line_x_at_y(line_b, float(y))) for y in rows]
    return float(np.mean(distances))

def suppress_redundant_fragments(lines: list[dict]) -> list[dict]:
    """Greedy NMS for duplicate Hough segments representing the same physical edge."""
    if not bool(cfg("fragment_nms", "enabled", default=True)) or len(lines) <= 1:
        return list(lines)

    max_angle_difference_deg = float(
        cfg("fragment_nms", "max_angle_difference_deg", default=1.5)
    )
    max_mean_axis_distance_px = float(
        cfg("fragment_nms", "max_mean_axis_distance_px", default=8.0)
    )
    min_y_overlap_ratio = float(
        cfg("fragment_nms", "min_y_overlap_ratio", default=0.55)
    )

    kept: list[dict] = []
    for raw_line in sorted(lines, key=fragment_quality_sort_key, reverse=True):
        line = dict(raw_line)
        line.setdefault("source_line_indices", [int(line["line_index"])])
        duplicate_of: dict | None = None
        for existing in kept:
            if (
                abs(float(line["signed_tilt_deg"]) - float(existing["signed_tilt_deg"]))
                > max_angle_difference_deg
            ):
                continue
            if line_y_overlap_ratio(line, existing) < min_y_overlap_ratio:
                continue
            if mean_line_distance_on_overlap(line, existing) > max_mean_axis_distance_px:
                continue
            duplicate_of = existing
            break

        if duplicate_of is None:
            kept.append(line)
            continue

        merged_sources = set(int(value) for value in duplicate_of.get("source_line_indices", []))
        merged_sources.update(int(value) for value in line.get("source_line_indices", []))
        duplicate_of["source_line_indices"] = sorted(merged_sources)
        duplicate_of["nms_duplicate_count"] = max(0, len(merged_sources) - 1)

    kept.sort(key=line_geometry_key)
    return kept

def is_adjusted_support_item(item: dict) -> bool:
    return bool(item.get("adjustment", {}).get("is_adjusted", False))

def has_endpoint_anchor(endpoint_metrics: dict[str, float], side: str, require_original: bool = False) -> bool:
    min_anchor_band_coverage = float(cfg("best_fit_selection", "min_anchor_band_coverage", default=0.18))
    min_anchor_overlap_px = float(cfg("best_fit_selection", "min_anchor_overlap_px", default=24.0))
    min_anchor_fragment_ratio = float(cfg("best_fit_selection", "min_anchor_fragment_ratio", default=0.78))

    if require_original:
        coverage = float(endpoint_metrics[f"{side}_original_endpoint_coverage"])
        overlap_px = float(endpoint_metrics[f"{side}_original_endpoint_best_fragment_overlap_px"])
        fragment_ratio = float(endpoint_metrics[f"{side}_original_endpoint_fragment_ratio"])
    else:
        coverage = float(endpoint_metrics[f"{side}_endpoint_coverage"])
        overlap_px = float(endpoint_metrics[f"{side}_endpoint_best_fragment_overlap_px"])
        fragment_ratio = float(endpoint_metrics[f"{side}_endpoint_best_fragment_ratio"])

    return (
        coverage >= min_anchor_band_coverage
        and overlap_px >= min_anchor_overlap_px
        and fragment_ratio >= min_anchor_fragment_ratio
    )

def candidate_endpoint_strengths(candidate: dict) -> dict[str, float]:
    top_anchor_strength = min(
        float(candidate.get("top_endpoint_coverage", 0.0)),
        float(candidate.get("top_endpoint_best_fragment_ratio", 0.0)),
    )
    bottom_anchor_strength = min(
        float(candidate.get("bottom_endpoint_coverage", 0.0)),
        float(candidate.get("bottom_endpoint_best_fragment_ratio", 0.0)),
    )
    top_original_anchor_strength = min(
        float(candidate.get("top_original_endpoint_coverage", 0.0)),
        float(candidate.get("top_original_endpoint_fragment_ratio", 0.0)),
    )
    bottom_original_anchor_strength = min(
        float(candidate.get("bottom_original_endpoint_coverage", 0.0)),
        float(candidate.get("bottom_original_endpoint_fragment_ratio", 0.0)),
    )
    return {
        "top_anchor_strength": float(top_anchor_strength),
        "bottom_anchor_strength": float(bottom_anchor_strength),
        "top_original_anchor_strength": float(top_original_anchor_strength),
        "bottom_original_anchor_strength": float(bottom_original_anchor_strength),
        "paired_anchor_strength": float(min(top_anchor_strength, bottom_anchor_strength)),
        "paired_original_anchor_strength": float(
            min(top_original_anchor_strength, bottom_original_anchor_strength)
        ),
        "paired_endpoint_coverage": float(
            min(
                float(candidate.get("top_endpoint_coverage", 0.0)),
                float(candidate.get("bottom_endpoint_coverage", 0.0)),
            )
        ),
        "paired_original_endpoint_coverage": float(
            min(
                float(candidate.get("top_original_endpoint_coverage", 0.0)),
                float(candidate.get("bottom_original_endpoint_coverage", 0.0)),
            )
        ),
    }

def compute_gap_penalty(
    selected_support: list[dict],
    y_min: float,
    y_max: float,
) -> dict[str, float]:
    total_span = max(1.0, float(y_max) - float(y_min))
    merged = merge_support_intervals(selected_support)
    if not merged:
        return {
            "gap_penalty": 1.0,
            "largest_gap_px": total_span,
        }

    soft_gap_px = float(cfg("gaps", "soft_gap_px", default=80))
    hard_gap_px = float(cfg("gaps", "hard_gap_px", default=260))

    gap_penalty = 0.0
    largest_gap_px = 0.0
    previous_end = merged[0][1]

    for start, end in merged[1:]:
        gap = max(0.0, start - previous_end)
        largest_gap_px = max(largest_gap_px, gap)
        if gap > soft_gap_px:
            scaled = min(1.0, (gap - soft_gap_px) / max(1.0, hard_gap_px - soft_gap_px))
            gap_penalty += scaled * (gap / total_span)
        previous_end = end

    return {
        "gap_penalty": clip01(gap_penalty * 2.2),
        "largest_gap_px": float(largest_gap_px),
    }

def compute_endpoint_metrics(
    selected_support: list[dict],
    y_min: float,
    y_max: float,
) -> dict[str, float]:
    roi_y_min = float(min(y_min, y_max))
    roi_y_max = float(max(y_min, y_max))
    roi_span_px = max(1.0, roi_y_max - roi_y_min)

    empty_result = {
        "support_y_min": 0.0,
        "support_y_max": 0.0,
        "support_span_px": 0.0,
        "endpoint_band_px": 0.0,
        "top_endpoint_coverage": 0.0,
        "bottom_endpoint_coverage": 0.0,
        "top_endpoint_alignment_score": 0.0,
        "bottom_endpoint_alignment_score": 0.0,
        "top_endpoint_best_fragment_overlap_px": 0.0,
        "bottom_endpoint_best_fragment_overlap_px": 0.0,
        "top_endpoint_best_fragment_ratio": 0.0,
        "bottom_endpoint_best_fragment_ratio": 0.0,
        "top_endpoint_fragment_count": 0,
        "bottom_endpoint_fragment_count": 0,
        "top_original_endpoint_coverage": 0.0,
        "bottom_original_endpoint_coverage": 0.0,
        "top_original_endpoint_best_fragment_overlap_px": 0.0,
        "bottom_original_endpoint_best_fragment_overlap_px": 0.0,
        "top_original_endpoint_fragment_ratio": 0.0,
        "bottom_original_endpoint_fragment_ratio": 0.0,
        "top_original_endpoint_fragment_count": 0,
        "bottom_original_endpoint_fragment_count": 0,
        "endpoint_anchor_score": 0.0,
        "top_reach_gap_px": roi_span_px,
        "bottom_reach_gap_px": roi_span_px,
    }

    merged = merge_support_intervals(selected_support)
    if not merged:
        return empty_result

    support_y_min = float(merged[0][0])
    support_y_max = float(merged[-1][1])
    support_span_px = max(1.0, support_y_max - support_y_min)

    band_ratio = float(cfg("endpoint_support", "band_ratio", default=0.15))
    min_band_px = float(cfg("endpoint_support", "min_band_px", default=80))
    max_band_px = float(cfg("endpoint_support", "max_band_px", default=160))
    endpoint_band_px = float(np.clip(roi_span_px * band_ratio, min_band_px, max_band_px))
    endpoint_band_px = min(endpoint_band_px, roi_span_px * 0.5)

    top_band_start = roi_y_min
    top_band_end = min(roi_y_max, roi_y_min + endpoint_band_px)
    bottom_band_start = max(roi_y_min, roi_y_max - endpoint_band_px)
    bottom_band_end = roi_y_max
    top_band_size_px = max(1.0, top_band_end - top_band_start)
    bottom_band_size_px = max(1.0, bottom_band_end - bottom_band_start)

    def band_metrics(
        band_start: float,
        band_end: float,
        band_size_px: float,
        original_only: bool = False,
    ) -> dict[str, float]:
        overlaps: list[tuple[float, float]] = []
        weighted_alignment_sum = 0.0
        overlap_weight_sum = 0.0
        best_fragment_overlap_px = 0.0
        overlapping_fragment_count = 0

        for item in selected_support:
            if original_only and is_adjusted_support_item(item):
                continue
            line_start = float(item["line"]["y_min"])
            line_end = float(item["line"]["y_max"])
            overlap_start = max(line_start, band_start)
            overlap_end = min(line_end, band_end)
            overlap_px = max(0.0, overlap_end - overlap_start)
            if overlap_px <= 0.0:
                continue
            overlapping_fragment_count += 1
            overlaps.append((overlap_start, overlap_end))
            best_fragment_overlap_px = max(best_fragment_overlap_px, overlap_px)
            alignment_score = 0.7 * float(item["distance_alignment"]) + 0.3 * float(item["angle_alignment"])
            weighted_alignment_sum += alignment_score * overlap_px
            overlap_weight_sum += overlap_px

        unique_overlap_px = interval_union_length(overlaps)
        alignment_score = (
            clip01(weighted_alignment_sum / overlap_weight_sum)
            if overlap_weight_sum > 0.0
            else 0.0
        )
        return {
            "coverage": float(clip01(unique_overlap_px / band_size_px)),
            "alignment_score": float(alignment_score),
            "best_fragment_overlap_px": float(best_fragment_overlap_px),
            "best_fragment_ratio": float(clip01(best_fragment_overlap_px / band_size_px)),
            "overlapping_fragment_count": int(overlapping_fragment_count),
            "unique_overlap_px": float(unique_overlap_px),
        }

    top_metrics = band_metrics(top_band_start, top_band_end, top_band_size_px)
    bottom_metrics = band_metrics(bottom_band_start, bottom_band_end, bottom_band_size_px)
    top_original_metrics = band_metrics(top_band_start, top_band_end, top_band_size_px, original_only=True)
    bottom_original_metrics = band_metrics(bottom_band_start, bottom_band_end, bottom_band_size_px, original_only=True)

    top_reach_gap_px = max(0.0, support_y_min - roi_y_min)
    bottom_reach_gap_px = max(0.0, roi_y_max - support_y_max)
    paired_coverage = min(float(top_metrics["coverage"]), float(bottom_metrics["coverage"]))
    paired_alignment = min(float(top_metrics["alignment_score"]), float(bottom_metrics["alignment_score"]))
    paired_original_coverage = min(
        float(top_original_metrics["coverage"]),
        float(bottom_original_metrics["coverage"]),
    )
    endpoint_anchor_score = clip01(
        0.50 * paired_coverage
        + 0.25 * paired_alignment
        + 0.25 * paired_original_coverage
    )

    return {
        "support_y_min": support_y_min,
        "support_y_max": support_y_max,
        "support_span_px": support_span_px,
        "endpoint_band_px": endpoint_band_px,
        "top_endpoint_coverage": float(top_metrics["coverage"]),
        "bottom_endpoint_coverage": float(bottom_metrics["coverage"]),
        "top_endpoint_alignment_score": float(top_metrics["alignment_score"]),
        "bottom_endpoint_alignment_score": float(bottom_metrics["alignment_score"]),
        "top_endpoint_best_fragment_overlap_px": float(top_metrics["best_fragment_overlap_px"]),
        "bottom_endpoint_best_fragment_overlap_px": float(bottom_metrics["best_fragment_overlap_px"]),
        "top_endpoint_best_fragment_ratio": float(top_metrics["best_fragment_ratio"]),
        "bottom_endpoint_best_fragment_ratio": float(bottom_metrics["best_fragment_ratio"]),
        "top_endpoint_fragment_count": int(top_metrics["overlapping_fragment_count"]),
        "bottom_endpoint_fragment_count": int(bottom_metrics["overlapping_fragment_count"]),
        "top_original_endpoint_coverage": float(top_original_metrics["coverage"]),
        "bottom_original_endpoint_coverage": float(bottom_original_metrics["coverage"]),
        "top_original_endpoint_best_fragment_overlap_px": float(top_original_metrics["best_fragment_overlap_px"]),
        "bottom_original_endpoint_best_fragment_overlap_px": float(bottom_original_metrics["best_fragment_overlap_px"]),
        "top_original_endpoint_fragment_ratio": float(top_original_metrics["best_fragment_ratio"]),
        "bottom_original_endpoint_fragment_ratio": float(bottom_original_metrics["best_fragment_ratio"]),
        "top_original_endpoint_fragment_count": int(top_original_metrics["overlapping_fragment_count"]),
        "bottom_original_endpoint_fragment_count": int(bottom_original_metrics["overlapping_fragment_count"]),
        "endpoint_anchor_score": float(endpoint_anchor_score),
        "top_reach_gap_px": float(top_reach_gap_px),
        "bottom_reach_gap_px": float(bottom_reach_gap_px),
    }

def compute_chain_metrics(selected_support: list[dict]) -> dict[str, float | int]:
    merged = merge_support_intervals(selected_support)
    if not merged:
        return {
            "merged_interval_count": 0,
            "total_merged_length_px": 0.0,
            "longest_merged_interval_px": 0.0,
            "chain_total_gap_px": 0.0,
            "chain_continuity_ratio": 0.0,
        }

    interval_lengths = [max(0.0, float(end) - float(start)) for start, end in merged]
    total_merged_length_px = float(sum(interval_lengths))
    longest_merged_interval_px = float(max(interval_lengths, default=0.0))
    support_y_min = float(merged[0][0])
    support_y_max = float(merged[-1][1])
    support_span_px = max(1.0, support_y_max - support_y_min)
    chain_total_gap_px = float(max(0.0, support_span_px - total_merged_length_px))
    chain_continuity_ratio = clip01(total_merged_length_px / max(1.0, support_span_px))

    return {
        "merged_interval_count": int(len(merged)),
        "total_merged_length_px": float(total_merged_length_px),
        "longest_merged_interval_px": float(longest_merged_interval_px),
        "chain_total_gap_px": float(chain_total_gap_px),
        "chain_continuity_ratio": float(chain_continuity_ratio),
    }

def compute_support_connection(
    upper_item: dict,
    lower_item: dict,
) -> tuple[float, float, float]:
    upper_line = upper_item.get("effective_line", upper_item["line"])
    lower_line = lower_item.get("effective_line", lower_item["line"])

    upper_y_max = float(upper_line["y_max"])
    lower_y_min = float(lower_line["y_min"])
    vertical_gap_px = max(0.0, lower_y_min - upper_y_max)

    if vertical_gap_px <= 0.0:
        overlap_start = max(float(upper_line["y_min"]), float(lower_line["y_min"]))
        overlap_end = min(float(upper_line["y_max"]), float(lower_line["y_max"]))
        connection_y = 0.5 * (overlap_start + overlap_end)
    else:
        connection_y = 0.5 * (upper_y_max + lower_y_min)

    upper_x = float(line_x_at_y(upper_line, connection_y))
    lower_x = float(line_x_at_y(lower_line, connection_y))
    connection_dx_px = abs(lower_x - upper_x)
    angle_difference_deg = abs(float(upper_line["signed_tilt_deg"]) - float(lower_line["signed_tilt_deg"]))
    return float(vertical_gap_px), float(connection_dx_px), float(angle_difference_deg)

def support_component_key(component: list[dict], roi_profile: dict) -> tuple[float, ...]:
    endpoint_metrics = compute_endpoint_metrics(
        component,
        y_min=float(roi_profile["trimmed_y_min"]),
        y_max=float(roi_profile["trimmed_y_max"]),
    )
    chain_metrics = compute_chain_metrics(component)
    adjustment_metrics = summarize_support_adjustments(component)
    has_top_anchor = has_endpoint_anchor(endpoint_metrics, "top", require_original=False)
    has_bottom_anchor = has_endpoint_anchor(endpoint_metrics, "bottom", require_original=False)
    has_top_original_anchor = has_endpoint_anchor(endpoint_metrics, "top", require_original=True)
    has_bottom_original_anchor = has_endpoint_anchor(endpoint_metrics, "bottom", require_original=True)

    return (
        1 if (has_top_original_anchor and has_bottom_original_anchor) else 0,
        1 if (has_top_anchor and has_bottom_anchor) else 0,
        (1 if has_top_original_anchor else 0) + (1 if has_bottom_original_anchor else 0),
        (1 if has_top_anchor else 0) + (1 if has_bottom_anchor else 0),
        -float(endpoint_metrics["top_reach_gap_px"]),
        -float(endpoint_metrics["bottom_reach_gap_px"]),
        float(
            min(
                float(endpoint_metrics["top_endpoint_coverage"]),
                float(endpoint_metrics["bottom_endpoint_coverage"]),
            )
        ),
        float(
            min(
                float(endpoint_metrics["top_original_endpoint_coverage"]),
                float(endpoint_metrics["bottom_original_endpoint_coverage"]),
            )
        ),
        float(chain_metrics["chain_continuity_ratio"]),
        -float(chain_metrics["chain_total_gap_px"]),
        -float(adjustment_metrics["adjustment_penalty"]),
        -float(adjustment_metrics["length_weighted_mean_abs_shift_px"]),
        float(chain_metrics["longest_merged_interval_px"]),
        float(chain_metrics["total_merged_length_px"]),
    )

def prune_support_to_dominant_chain(selected_support: list[dict], roi_profile: dict) -> list[dict]:
    """Select one coherent directed ruler path.

    The ordinary edges keep the strict local-chain behaviour introduced by the
    first refactor. A path may additionally use a very small number of long,
    strongly collinear bridges. This matters when a real center ruler is hidden
    by a buckle, logo, or an untextured plastic region: the correct evidence is
    then split into two sparse groups and must not lose to one dense local group.
    """
    if not bool(cfg("support_chain", "enabled", default=True)):
        return list(selected_support)
    if len(selected_support) <= 1:
        return list(selected_support)

    items = sorted(
        selected_support,
        key=lambda item: (
            float(item.get("effective_line", item["line"])["y_mid"]),
            float(item.get("effective_line", item["line"])["x_mid"]),
        ),
    )
    roi_y_min = float(roi_profile["trimmed_y_min"])
    roi_y_max = float(roi_profile["trimmed_y_max"])
    roi_span = max(1.0, roi_y_max - roi_y_min)
    total_selected_length_px = float(
        sum(float(item["line"]["length"]) for item in selected_support)
    )
    total_selected_support_strength = float(
        sum(float(item.get("support_strength", 0.0)) for item in selected_support)
    )
    total_selected_fragment_count = max(1, len(selected_support))

    normal_max_gap_px = min(
        float(cfg("support_chain", "max_connection_gap_px", default=220.0)),
        roi_span * float(cfg("support_chain", "max_gap_ratio", default=0.10)),
    )
    base_max_dx_px = float(cfg("support_chain", "base_max_dx_px", default=10.0))
    dx_per_gap_ratio = float(cfg("support_chain", "dx_per_gap_ratio", default=0.03))
    absolute_max_dx_px = float(cfg("support_chain", "max_connection_dx_px", default=34.0))
    max_angle_difference_deg = float(
        cfg("support_chain", "max_angle_difference_deg", default=2.0)
    )
    min_vertical_advance_px = float(
        cfg("support_chain", "min_vertical_advance_px", default=8.0)
    )
    max_path_fit_rmse_px = float(
        cfg("support_chain", "max_path_fit_rmse_px", default=12.0)
    )
    beam_width = max(1, int(cfg("support_chain", "beam_width", default=10)))
    top_path_count = max(1, int(cfg("support_chain", "top_path_count", default=80)))

    allow_long_bridges = bool(
        cfg("support_chain", "allow_long_bridges", default=True)
    )
    max_long_bridges = max(
        0, int(cfg("support_chain", "max_long_bridges", default=1))
    )
    long_bridge_max_gap_px = min(
        float(cfg("support_chain", "max_long_bridge_gap_px", default=720.0)),
        roi_span
        * float(cfg("support_chain", "max_long_bridge_gap_ratio", default=0.38)),
    )
    long_bridge_base_max_dx_px = float(
        cfg("support_chain", "long_bridge_base_max_dx_px", default=13.0)
    )
    long_bridge_dx_per_gap_ratio = float(
        cfg("support_chain", "long_bridge_dx_per_gap_ratio", default=0.012)
    )
    long_bridge_absolute_max_dx_px = float(
        cfg("support_chain", "long_bridge_max_dx_px", default=27.0)
    )
    long_bridge_max_angle_difference_deg = float(
        cfg(
            "support_chain",
            "long_bridge_max_angle_difference_deg",
            default=4.5,
        )
    )
    long_bridge_quality_scale = clip01(
        float(cfg("support_chain", "long_bridge_quality_scale", default=0.52))
    )
    min_bridge_distance_alignment = float(
        cfg("support_chain", "min_bridge_distance_alignment", default=0.12)
    )
    min_bridge_angle_alignment = float(
        cfg("support_chain", "min_bridge_angle_alignment", default=0.10)
    )

    def edge_metrics(upper: dict, lower: dict) -> tuple[bool, float, bool]:
        upper_line = upper.get("effective_line", upper["line"])
        lower_line = lower.get("effective_line", lower["line"])
        vertical_advance = float(lower_line["y_max"]) - float(upper_line["y_max"])
        if vertical_advance < min_vertical_advance_px:
            return False, 0.0, False

        gap, dx, angle_delta = compute_support_connection(upper, lower)
        if gap <= normal_max_gap_px:
            adaptive_max_dx = min(
                absolute_max_dx_px,
                base_max_dx_px + dx_per_gap_ratio * gap,
            )
            if dx > adaptive_max_dx or angle_delta > max_angle_difference_deg:
                return False, 0.0, False
            edge_quality = (
                1.0
                - 0.45 * (dx / max(1e-6, adaptive_max_dx)) ** 2
                - 0.35 * (gap / max(1e-6, normal_max_gap_px)) ** 2
                - 0.20
                * (angle_delta / max(1e-6, max_angle_difference_deg)) ** 2
            )
            return True, float(max(0.0, edge_quality)), False

        if (
            not allow_long_bridges
            or max_long_bridges <= 0
            or gap > long_bridge_max_gap_px
        ):
            return False, 0.0, False
        if min(
            float(upper.get("distance_alignment", 0.0)),
            float(lower.get("distance_alignment", 0.0)),
        ) < min_bridge_distance_alignment:
            return False, 0.0, False
        if min(
            float(upper.get("angle_alignment", 0.0)),
            float(lower.get("angle_alignment", 0.0)),
        ) < min_bridge_angle_alignment:
            return False, 0.0, False

        bridge_max_dx = min(
            long_bridge_absolute_max_dx_px,
            long_bridge_base_max_dx_px + long_bridge_dx_per_gap_ratio * gap,
        )
        if (
            dx > bridge_max_dx
            or angle_delta > long_bridge_max_angle_difference_deg
        ):
            return False, 0.0, False

        gap_fraction = (gap - normal_max_gap_px) / max(
            1e-6, long_bridge_max_gap_px - normal_max_gap_px
        )
        edge_quality = long_bridge_quality_scale * (
            1.0
            - 0.45 * (dx / max(1e-6, bridge_max_dx)) ** 2
            - 0.30 * clip01(gap_fraction) ** 2
            - 0.25
            * (
                angle_delta
                / max(1e-6, long_bridge_max_angle_difference_deg)
            )
            ** 2
        )
        return True, float(max(0.0, edge_quality)), True

    path_metrics_cache: dict[tuple[int, ...], dict[str, float]] = {}

    def path_metrics(path_indices: tuple[int, ...]) -> dict[str, float]:
        cached_metrics = path_metrics_cache.get(path_indices)
        if cached_metrics is not None:
            return cached_metrics

        support = [items[index] for index in path_indices]
        merged = merge_support_intervals(support)
        if not merged:
            empty_metrics = {
                "unique_ratio": 0.0,
                "span_ratio": 0.0,
                "continuity": 0.0,
                "alignment": 0.0,
                "extent_ratio": 0.0,
                "support_length_ratio": 0.0,
                "support_strength_ratio": 0.0,
                "fragment_ratio": 0.0,
            }
            path_metrics_cache[path_indices] = empty_metrics
            return empty_metrics
        unique_length = interval_union_length(merged)
        span = max(1.0, merged[-1][1] - merged[0][0])
        alignment_weights = [
            max(1.0, float(item["line"]["length"])) for item in support
        ]
        alignment_values = [
            0.7 * float(item.get("distance_alignment", 0.0))
            + 0.3 * float(item.get("angle_alignment", 0.0))
            for item in support
        ]
        alignment = float(np.average(alignment_values, weights=alignment_weights))
        support_length = float(sum(float(item["line"]["length"]) for item in support))
        support_strength = float(
            sum(float(item.get("support_strength", 0.0)) for item in support)
        )
        top_gap = max(0.0, merged[0][0] - roi_y_min)
        bottom_gap = max(0.0, roi_y_max - merged[-1][1])
        extent_ratio = clip01(1.0 - (top_gap + bottom_gap) / roi_span)
        metrics = {
            "unique_ratio": clip01(unique_length / roi_span),
            "span_ratio": clip01(span / roi_span),
            "continuity": clip01(unique_length / span),
            "alignment": clip01(alignment),
            "extent_ratio": float(extent_ratio),
            "support_length_ratio": clip01(
                support_length / max(1.0, total_selected_length_px)
            ),
            "support_strength_ratio": clip01(
                support_strength / max(1.0, total_selected_support_strength)
            ),
            "fragment_ratio": clip01(
                len(support) / max(1, total_selected_fragment_count)
            ),
        }
        path_metrics_cache[path_indices] = metrics
        return metrics

    def fast_path_score(
        path_indices: tuple[int, ...],
        edge_quality_sum: float,
        bridge_count: int,
    ) -> float:
        metrics = path_metrics(path_indices)
        edge_mean = edge_quality_sum / max(1, len(path_indices) - 1)
        return float(
            0.20 * metrics["unique_ratio"]
            + 0.22 * metrics["span_ratio"]
            + 0.08 * metrics["continuity"]
            + 0.12 * metrics["alignment"]
            + 0.08 * metrics["extent_ratio"]
            + 0.13 * metrics["support_length_ratio"]
            + 0.17 * metrics["support_strength_ratio"]
            + 0.05 * metrics["fragment_ratio"]
            + 0.08
            * clip01(edge_mean if len(path_indices) > 1 else metrics["alignment"])
            - 0.03 * bridge_count
        )

    # State: (fast score, path indices, cumulative edge quality, bridge count).
    states_by_end: list[list[tuple[float, tuple[int, ...], float, int]]] = []
    all_states: list[tuple[float, tuple[int, ...], float, int]] = []
    for lower_index, lower_item in enumerate(items):
        states: list[tuple[float, tuple[int, ...], float, int]] = [
            (fast_path_score((lower_index,), 0.0, 0), (lower_index,), 0.0, 0)
        ]
        for upper_index in range(lower_index):
            compatible, edge_quality, is_long_bridge = edge_metrics(
                items[upper_index], lower_item
            )
            if not compatible:
                continue
            for _, path_indices, edge_quality_sum, bridge_count in states_by_end[
                upper_index
            ]:
                new_bridge_count = bridge_count + int(is_long_bridge)
                if new_bridge_count > max_long_bridges:
                    continue
                extended = (*path_indices, lower_index)
                new_edge_quality_sum = edge_quality_sum + edge_quality
                states.append(
                    (
                        fast_path_score(
                            extended, new_edge_quality_sum, new_bridge_count
                        ),
                        extended,
                        new_edge_quality_sum,
                        new_bridge_count,
                    )
                )

        states.sort(
            key=lambda state: (
                state[0],
                path_metrics(state[1])["span_ratio"],
                len(state[1]),
                state[2],
                -state[3],
                state[1],
            ),
            reverse=True,
        )
        deduplicated_states: list[tuple[float, tuple[int, ...], float, int]] = []
        seen_paths: set[tuple[int, ...]] = set()
        for state in states:
            if state[1] in seen_paths:
                continue
            seen_paths.add(state[1])
            deduplicated_states.append(state)

        # A pure score beam can delete the only long-span state before it
        # reaches the bottom of the ROI. Reserve one slot for the widest path
        # and one for the best bridged path whenever possible.
        unique_states: list[tuple[float, tuple[int, ...], float, int]] = []
        reserved_states: list[tuple[float, tuple[int, ...], float, int]] = []
        if deduplicated_states:
            reserved_states.append(
                max(
                    deduplicated_states,
                    key=lambda state: (
                        path_metrics(state[1])["span_ratio"],
                        state[0],
                        len(state[1]),
                        state[2],
                        -state[3],
                        state[1],
                    ),
                )
            )
        bridged_states = [state for state in deduplicated_states if state[3] > 0]
        if bridged_states:
            reserved_states.append(
                max(
                    bridged_states,
                    key=lambda state: (
                        state[0],
                        path_metrics(state[1])["span_ratio"],
                        len(state[1]),
                        state[2],
                        -state[3],
                        state[1],
                    ),
                )
            )

        primary_limit = max(1, beam_width - len(reserved_states))
        for state in deduplicated_states[:primary_limit]:
            if state not in unique_states:
                unique_states.append(state)
        for state in reserved_states:
            if state not in unique_states:
                unique_states.append(state)
        for state in deduplicated_states:
            if len(unique_states) >= beam_width:
                break
            if state not in unique_states:
                unique_states.append(state)
        unique_states = unique_states[:beam_width]

        states_by_end.append(unique_states)
        all_states.extend(unique_states)

    all_states.sort(
        key=lambda state: (
            state[0],
            path_metrics(state[1])["span_ratio"],
            len(state[1]),
            state[2],
            -state[3],
            state[1],
        ),
        reverse=True,
    )
    final_states = list(all_states[:top_path_count])
    if all_states:
        widest_states = sorted(
            all_states,
            key=lambda state: (
                path_metrics(state[1])["span_ratio"],
                state[0],
                len(state[1]),
                state[2],
                -state[3],
                state[1],
            ),
            reverse=True,
        )[: max(2, min(12, top_path_count // 4))]
        for state in widest_states:
            if state not in final_states:
                final_states.append(state)
        bridged_states = [state for state in all_states if state[3] > 0]
        bridged_states.sort(
            key=lambda state: (
                state[0],
                path_metrics(state[1])["span_ratio"],
                len(state[1]),
                state[2],
                -state[3],
                state[1],
            ),
            reverse=True,
        )
        for state in bridged_states[: max(2, min(12, top_path_count // 4))]:
            if state not in final_states:
                final_states.append(state)

    best_support: list[dict] | None = None
    best_key: tuple[float, ...] | None = None
    for fast_score, path_indices, _, bridge_count in final_states:
        support = [items[index] for index in path_indices]
        chain_metrics = compute_chain_metrics(support)
        metrics = path_metrics(path_indices)

        y_values: list[float] = []
        x_values: list[float] = []
        weights: list[float] = []
        for item in support:
            line = item.get("effective_line", item["line"])
            for y_value in (
                float(line["y_min"]),
                float(line["y_mid"]),
                float(line["y_max"]),
            ):
                y_values.append(y_value)
                x_values.append(float(line_x_at_y(line, y_value)))
                weights.append(max(1.0, float(line["length"])) / 3.0)
        fit = safe_linear_polyfit(y_values, x_values, weights)
        if fit is None:
            fit_rmse = float("inf")
        else:
            residuals = np.asarray(x_values) - (
                float(fit[0]) * np.asarray(y_values) + float(fit[1])
            )
            fit_rmse = float(
                np.sqrt(np.average(residuals**2, weights=np.asarray(weights)))
            )
        fit_score = (
            math.exp(-fit_rmse / max(1.0, max_path_fit_rmse_px))
            if math.isfinite(fit_rmse)
            else 0.0
        )
        quality = (
            0.18 * metrics["unique_ratio"]
            + 0.20 * metrics["span_ratio"]
            + 0.07 * metrics["continuity"]
            + 0.10 * metrics["alignment"]
            + 0.08 * metrics["extent_ratio"]
            + 0.14 * metrics["support_length_ratio"]
            + 0.19 * metrics["support_strength_ratio"]
            + 0.06 * metrics["fragment_ratio"]
            + 0.12 * clip01(fit_score)
            + 0.06 * clip01(fast_score)
            - 0.03 * bridge_count
        )
        if fit_rmse > max_path_fit_rmse_px * 1.8:
            quality -= 0.25
        key = (
            float(quality),
            float(metrics["support_strength_ratio"]),
            float(metrics["support_length_ratio"]),
            float(metrics["span_ratio"]),
            float(metrics["unique_ratio"]),
            float(metrics["fragment_ratio"]),
            float(metrics["extent_ratio"]),
            float(chain_metrics["chain_continuity_ratio"]),
            -float(fit_rmse),
            -float(chain_metrics["chain_total_gap_px"]),
            -float(bridge_count),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_support = support

    if not best_support:
        return [
            max(
                items,
                key=support_item_sort_key,
            )
        ]
    return sorted(best_support, key=lambda item: line_geometry_key(item["line"]))

def merge_support_items(primary_support: list[dict], secondary_support: list[dict]) -> list[dict]:
    merged_by_line_index: dict[int, dict] = {}
    for item in primary_support + secondary_support:
        line_index = int(item["line"]["line_index"])
        existing = merged_by_line_index.get(line_index)
        if existing is None or float(item["support_strength"]) > float(existing["support_strength"]):
            merged_by_line_index[line_index] = item

    merged = list(merged_by_line_index.values())
    merged.sort(key=support_item_sort_key, reverse=True)
    return merged

def rescue_endpoint_support_fragments(
    lines: list[dict],
    axis: dict[str, float],
    selected_support: list[dict],
    max_angle_error_deg: float,
) -> list[dict]:
    if not bool(cfg("endpoint_rescue", "enabled", default=True)):
        return selected_support
    if not selected_support:
        return selected_support

    margin_px = float(cfg("endpoint_rescue", "margin_px", default=180.0))
    max_fragments_per_side = int(cfg("endpoint_rescue", "max_fragments_per_side", default=2))
    max_mean_axis_distance_px = float(cfg("endpoint_rescue", "max_mean_axis_distance_px", default=16.0))
    max_min_axis_distance_px = float(cfg("endpoint_rescue", "max_min_axis_distance_px", default=2.5))
    angle_slack_deg = float(cfg("endpoint_rescue", "angle_slack_deg", default=0.75))
    support_strength_scale = float(cfg("endpoint_rescue", "support_strength_scale", default=0.84))
    resolved_max_angle_error_deg = float(max_angle_error_deg) + max(0.0, angle_slack_deg)

    merged = merge_support_intervals(selected_support)
    if not merged:
        return selected_support

    support_y_min = float(merged[0][0])
    support_y_max = float(merged[-1][1])
    selected_line_indices = {int(item["line"]["line_index"]) for item in selected_support}
    top_candidates: list[dict] = []
    bottom_candidates: list[dict] = []

    for line in lines:
        line_index = int(line["line_index"])
        if line_index in selected_line_indices:
            continue

        line_y_min = float(line["y_min"])
        line_y_max = float(line["y_max"])
        near_top = line_y_max >= support_y_min - margin_px and line_y_min <= support_y_min + margin_px
        near_bottom = line_y_max >= support_y_max - margin_px and line_y_min <= support_y_max + margin_px
        if not near_top and not near_bottom:
            continue

        angle_error_deg = abs(float(line["signed_tilt_deg"]) - float(axis["tilt_deg"]))
        if angle_error_deg > resolved_max_angle_error_deg:
            continue

        mean_axis_distance_px = float(segment_axis_distance_px(line, axis))
        if mean_axis_distance_px > max_mean_axis_distance_px:
            continue

        min_axis_distance_px = float(segment_axis_min_distance_px(line, axis))
        intersection_y = segment_axis_intersection_y(line, axis)
        if intersection_y is None and min_axis_distance_px > max_min_axis_distance_px:
            continue

        distance_alignment = clip01(1.0 - min_axis_distance_px / max(1e-6, max_min_axis_distance_px))
        angle_alignment = clip01(1.0 - angle_error_deg / max(1e-6, resolved_max_angle_error_deg))
        support_strength = float(line["length"]) * (
            0.62 * distance_alignment + 0.38 * angle_alignment
        ) * support_strength_scale
        support_item = {
            "line": line,
            "effective_line": line,
            "axis_distance_px": float(mean_axis_distance_px),
            "angle_error_deg": float(angle_error_deg),
            "distance_alignment": float(distance_alignment),
            "angle_alignment": float(angle_alignment),
            "support_strength": float(support_strength),
            "adjustment": make_zero_adjustment(line, axis, mean_axis_distance_px, angle_error_deg),
            "endpoint_rescue": {
                "is_endpoint_rescue": True,
                "min_axis_distance_px": float(min_axis_distance_px),
                "intersection_y": None if intersection_y is None else float(intersection_y),
            },
        }
        if near_top:
            top_candidates.append(support_item)
        if near_bottom:
            bottom_candidates.append(support_item)

    if not top_candidates and not bottom_candidates:
        return selected_support

    top_candidates.sort(key=support_item_sort_key, reverse=True)
    bottom_candidates.sort(key=support_item_sort_key, reverse=True)
    rescued_support = [
        *top_candidates[:max(0, max_fragments_per_side)],
        *bottom_candidates[:max(0, max_fragments_per_side)],
    ]
    return merge_support_items(selected_support, rescued_support)

def extend_support_upward(
    selected_support: list[dict],
    lines: list[dict],
    axis: dict[str, float],
    roi_profile: dict,
) -> list[dict]:
    if not bool(cfg("support_extension", "enabled", default=True)):
        return selected_support
    if not selected_support:
        return selected_support

    top_item = min(selected_support, key=lambda item: float(item["line"]["y_min"]))
    current_anchor_y = float(top_item["line"]["y_min"])
    trigger_gap_px = float(cfg("support_extension", "trigger_gap_px", default=180))
    if current_anchor_y - float(roi_profile["trimmed_y_min"]) < trigger_gap_px:
        return selected_support

    current_anchor_x = float(line_x_at_y(top_item.get("effective_line", top_item["line"]), current_anchor_y))
    selected_line_indices = {int(item["line"]["line_index"]) for item in selected_support}
    min_vertical_advance_px = float(cfg("support_extension", "min_vertical_advance_px", default=18))
    max_connection_dx_px = float(cfg("support_extension", "max_connection_dx_px", default=34.0))
    max_connection_gap_px = float(cfg("support_extension", "max_connection_gap_px", default=360.0))
    max_axis_distance_px = float(cfg("support_extension", "max_axis_distance_px", default=55.0))
    max_center_distance_px = float(cfg("support_extension", "max_center_distance_px", default=70.0))
    max_vertical_deviation_deg = float(cfg("support_extension", "max_vertical_deviation_deg", default=10.0))
    max_added_fragments = int(cfg("support_extension", "max_added_fragments", default=6))

    extended_support = list(selected_support)

    for _ in range(max_added_fragments):
        best_extension_item = None
        best_extension_score = -float("inf")

        for line in lines:
            line_index = int(line["line_index"])
            if line_index in selected_line_indices:
                continue
            if float(line["y_min"]) >= current_anchor_y - min_vertical_advance_px:
                continue

            support_item = build_best_support_item(
                line=line,
                axis=axis,
                band_half_width_px=max_axis_distance_px,
                max_angle_error_deg=max_vertical_deviation_deg,
                allow_adjustment=True,
            )
            if support_item is None:
                continue

            effective_line = support_item.get("effective_line", line)
            connection_y = min(current_anchor_y, float(effective_line["y_max"]))
            connection_gap_px = max(0.0, current_anchor_y - float(line["y_max"]))
            connection_x = float(line_x_at_y(effective_line, connection_y))
            connection_dx_px = abs(connection_x - current_anchor_x)
            axis_distance_px = float(support_item["axis_distance_px"])
            center_x = float(line_x_at_y(roi_profile["center_fit"], float(effective_line["y_mid"])))
            center_distance_px = abs(float(effective_line["x_mid"]) - center_x)
            verticality_score = clip01(
                1.0 - abs(float(effective_line["signed_tilt_deg"])) / max(1e-6, max_vertical_deviation_deg)
            )

            if connection_dx_px > max_connection_dx_px:
                continue
            if connection_gap_px > max_connection_gap_px:
                continue
            if axis_distance_px > max_axis_distance_px:
                continue
            if center_distance_px > max_center_distance_px:
                continue
            if verticality_score <= 0.0:
                continue

            continuity_dx_score = clip01(1.0 - connection_dx_px / max(1e-6, max_connection_dx_px))
            continuity_gap_score = clip01(1.0 - connection_gap_px / max(1e-6, max_connection_gap_px))
            continuity_score = 0.75 * continuity_dx_score + 0.25 * continuity_gap_score
            center_score = clip01(1.0 - center_distance_px / max(1e-6, max_center_distance_px))
            axis_score = clip01(1.0 - axis_distance_px / max(1e-6, max_axis_distance_px))
            length_score = clip01(float(line["length"]) / 220.0)
            advance_score = clip01((current_anchor_y - float(line["y_min"])) / max(1e-6, max_connection_gap_px))

            extension_score = (
                0.28 * continuity_score
                + 0.24 * verticality_score
                + 0.18 * center_score
                + 0.16 * axis_score
                + 0.10 * advance_score
                + 0.04 * length_score
            )

            if extension_score <= best_extension_score:
                continue

            distance_alignment = float(support_item["distance_alignment"])
            angle_alignment = float(support_item["angle_alignment"])
            support_strength = float(line["length"]) * (
                0.30 * continuity_score
                + 0.22 * verticality_score
                + 0.18 * center_score
                + 0.14 * distance_alignment
                + 0.08 * angle_alignment
                + 0.08 * advance_score
            )
            if bool(support_item.get("adjustment", {}).get("is_adjusted", False)):
                support_strength *= 0.90

            best_extension_item = {
                **support_item,
                "axis_distance_px": float(axis_distance_px),
                "distance_alignment": float(distance_alignment),
                "angle_alignment": float(angle_alignment),
                "support_strength": float(support_strength),
                "extension_score": float(extension_score),
                "connection_gap_px": float(connection_gap_px),
                "connection_dx_px": float(connection_dx_px),
                "center_distance_px": float(center_distance_px),
            }
            best_extension_score = float(extension_score)

        if best_extension_item is None:
            break

        extended_support.append(best_extension_item)
        selected_line_indices.add(int(best_extension_item["line"]["line_index"]))
        current_anchor_y = float(best_extension_item["line"]["y_min"])
        current_anchor_x = float(
            line_x_at_y(best_extension_item.get("effective_line", best_extension_item["line"]), current_anchor_y)
        )

        if current_anchor_y <= float(roi_profile["trimmed_y_min"]) + min_vertical_advance_px:
            break

    extended_support.sort(key=support_item_sort_key, reverse=True)
    return extended_support

def summarize_support_adjustments(selected_support: list[dict]) -> dict[str, float | int]:
    if not selected_support:
        return {
            "adjusted_fragment_count": 0,
            "adjusted_fragment_ratio": 0.0,
            "mean_abs_shift_px": 0.0,
            "length_weighted_mean_abs_shift_px": 0.0,
            "max_abs_shift_px": 0.0,
            "mean_abs_tilt_delta_deg": 0.0,
            "max_abs_tilt_delta_deg": 0.0,
            "adjustment_penalty": 0.0,
        }

    mean_abs_shifts = np.asarray(
        [float(item.get("adjustment", {}).get("mean_abs_shift_px", 0.0)) for item in selected_support],
        dtype=np.float64,
    )
    abs_tilt_deltas = np.asarray(
        [float(item.get("adjustment", {}).get("abs_tilt_delta_deg", 0.0)) for item in selected_support],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [max(1.0, float(item["line"]["length"])) for item in selected_support],
        dtype=np.float64,
    )
    adjusted_fragment_count = sum(
        1 for item in selected_support if bool(item.get("adjustment", {}).get("is_adjusted", False))
    )

    shift_reference_px = float(cfg("support_adjustment", "max_mean_shift_px", default=10.0))
    tilt_reference_deg = float(cfg("support_adjustment", "max_tilt_delta_deg", default=2.0))
    normalized_shift = clip01(
        float(np.average(mean_abs_shifts, weights=lengths)) / max(1e-6, shift_reference_px)
    )
    normalized_tilt = clip01(float(np.mean(abs_tilt_deltas)) / max(1e-6, tilt_reference_deg))

    return {
        "adjusted_fragment_count": int(adjusted_fragment_count),
        "adjusted_fragment_ratio": float(adjusted_fragment_count / max(1, len(selected_support))),
        "mean_abs_shift_px": float(np.mean(mean_abs_shifts)),
        "length_weighted_mean_abs_shift_px": float(np.average(mean_abs_shifts, weights=lengths)),
        "max_abs_shift_px": float(np.max(mean_abs_shifts)),
        "mean_abs_tilt_delta_deg": float(np.mean(abs_tilt_deltas)),
        "max_abs_tilt_delta_deg": float(np.max(abs_tilt_deltas)),
        "adjustment_penalty": float(0.5 * normalized_shift + 0.5 * normalized_tilt),
    }

def make_support_signature(selected_support: list[dict]) -> tuple[tuple[float | int, ...], ...]:
    signature_items = []
    for item in selected_support:
        effective_line = item.get("effective_line", item["line"])
        adjustment = item.get("adjustment", {})
        signature_items.append(
            (
                int(item["line"]["line_index"]),
                1 if bool(adjustment.get("is_adjusted", False)) else 0,
                round(float(effective_line["a"]), 6),
                round(float(effective_line["b"]), 3),
                round(float(item.get("support_strength", 0.0)), 3),
            )
        )
    signature_items.sort()
    return tuple(signature_items)

def build_point_cloud(selected_support: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample fragments without giving long fragments a quadratic influence."""
    step_px = float(cfg("final_fit", "sample_step_px", default=22))
    enable_endpoint_boost = bool(
        cfg("final_fit", "enable_endpoint_weight_boost", default=False)
    )
    endpoint_sample_weight_boost = float(
        cfg("final_fit", "endpoint_sample_weight_boost", default=1.0)
    )
    original_endpoint_sample_weight_boost = float(
        cfg("final_fit", "original_endpoint_sample_weight_boost", default=1.0)
    )

    y_values: list[float] = []
    x_values: list[float] = []
    weights: list[float] = []
    merged = merge_support_intervals(selected_support)
    top_band_start = top_band_end = bottom_band_start = bottom_band_end = 0.0
    top_band_size_px = bottom_band_size_px = 1.0
    if enable_endpoint_boost and merged:
        support_y_min = float(merged[0][0])
        support_y_max = float(merged[-1][1])
        support_span_px = max(1.0, support_y_max - support_y_min)
        endpoint_band_px = float(
            np.clip(
                support_span_px * float(cfg("endpoint_support", "band_ratio", default=0.15)),
                float(cfg("endpoint_support", "min_band_px", default=80)),
                float(cfg("endpoint_support", "max_band_px", default=160)),
            )
        )
        endpoint_band_px = min(endpoint_band_px, support_span_px * 0.5)
        top_band_start = support_y_min
        top_band_end = min(support_y_max, support_y_min + endpoint_band_px)
        bottom_band_start = max(support_y_min, support_y_max - endpoint_band_px)
        bottom_band_end = support_y_max
        top_band_size_px = max(1.0, top_band_end - top_band_start)
        bottom_band_size_px = max(1.0, bottom_band_end - bottom_band_start)

    for item in selected_support:
        line = item.get("effective_line", item["line"])
        length = max(1.0, float(line["length"]))
        sample_count = max(2, int(math.ceil(length / max(1.0, step_px))) + 1)
        alignment_quality = clip01(
            0.7 * float(item.get("distance_alignment", 0.0))
            + 0.3 * float(item.get("angle_alignment", 0.0))
        )
        # Linear total influence: length * quality, divided over all samples.
        fragment_total_weight = max(1.0, length * (0.35 + 0.65 * alignment_quality))
        item_weight_scale = 1.0
        if enable_endpoint_boost and merged:
            line_y_min = float(line["y_min"])
            line_y_max = float(line["y_max"])
            top_overlap_px = max(0.0, min(line_y_max, top_band_end) - max(line_y_min, top_band_start))
            bottom_overlap_px = max(0.0, min(line_y_max, bottom_band_end) - max(line_y_min, bottom_band_start))
            endpoint_overlap_ratio = clip01(
                max(
                    top_overlap_px / top_band_size_px,
                    bottom_overlap_px / bottom_band_size_px,
                )
            )
            if endpoint_overlap_ratio > 0.0:
                target_boost = (
                    endpoint_sample_weight_boost
                    if is_adjusted_support_item(item)
                    else original_endpoint_sample_weight_boost
                )
                item_weight_scale = 1.0 + (target_boost - 1.0) * endpoint_overlap_ratio
        sample_weight = fragment_total_weight * item_weight_scale / sample_count
        for t_value in np.linspace(0.0, 1.0, sample_count):
            y_coord = float(line["y1"] + t_value * (line["y2"] - line["y1"]))
            x_coord = float(line["x1"] + t_value * (line["x2"] - line["x1"]))
            y_values.append(y_coord)
            x_values.append(x_coord)
            weights.append(sample_weight)

    return (
        np.asarray(y_values, dtype=np.float64),
        np.asarray(x_values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )

def compute_axis_fit_metrics(axis: dict[str, float], selected_support: list[dict]) -> dict[str, float]:
    if not selected_support:
        return {
            "fit_rmse_px": float("inf"),
            "fit_median_abs_residual_px": float("inf"),
            "fit_p90_abs_residual_px": float("inf"),
        }
    y_values, x_values, weights = build_point_cloud(selected_support)
    if y_values.size == 0:
        return {
            "fit_rmse_px": float("inf"),
            "fit_median_abs_residual_px": float("inf"),
            "fit_p90_abs_residual_px": float("inf"),
        }
    residuals = np.abs(x_values - (float(axis["a"]) * y_values + float(axis["b"])))
    normalized_weights = weights / max(1e-9, float(np.sum(weights)))
    rmse = float(np.sqrt(np.sum(normalized_weights * residuals**2)))
    return {
        "fit_rmse_px": rmse,
        "fit_median_abs_residual_px": float(np.median(residuals)),
        "fit_p90_abs_residual_px": float(np.quantile(residuals, 0.90)),
    }

def fit_axis_from_support(
    selected_support: list[dict],
    y_ref: float,
    fit_cache: dict | None = None,
) -> dict[str, float] | None:
    if not selected_support:
        return None
    cache_key = None
    if fit_cache is not None:
        cache_key = (make_support_signature(selected_support), round(float(y_ref), 3))
        cached_fit = fit_cache.get(cache_key)
        if cached_fit is not None:
            return None if cached_fit is False else dict(cached_fit)

    y_values, x_values, base_weights = build_point_cloud(selected_support)
    if len(y_values) < 2:
        if fit_cache is not None and cache_key is not None:
            fit_cache[cache_key] = False
        return None

    fit = safe_linear_polyfit(y_values, x_values, base_weights)
    if fit is None:
        if fit_cache is not None and cache_key is not None:
            fit_cache[cache_key] = False
        return None

    a_value, b_value = fit
    huber_delta_px = float(cfg("final_fit", "huber_delta_px", default=10.0))
    huber_iterations = int(cfg("final_fit", "huber_iterations", default=4))
    for _ in range(max(0, huber_iterations)):
        residuals = np.abs(x_values - (a_value * y_values + b_value))
        huber_weights = np.ones_like(residuals)
        large_residual_mask = residuals > huber_delta_px
        huber_weights[large_residual_mask] = huber_delta_px / np.maximum(
            residuals[large_residual_mask], 1e-6
        )
        updated_fit = safe_linear_polyfit(y_values, x_values, base_weights * huber_weights)
        if updated_fit is None:
            break
        new_a, new_b = updated_fit
        if abs(new_a - a_value) < 1e-8 and abs(new_b - b_value) < 1e-4:
            a_value, b_value = new_a, new_b
            break
        a_value, b_value = new_a, new_b

    tilt_deg = float(math.degrees(math.atan(float(a_value))))
    max_fit_tilt_deg = float(cfg("final_fit", "max_fit_tilt_deg", default=12.0))
    if abs(tilt_deg) > max_fit_tilt_deg:
        if fit_cache is not None and cache_key is not None:
            fit_cache[cache_key] = False
        return None

    residuals = np.abs(x_values - (a_value * y_values + b_value))
    normalized_weights = base_weights / max(1e-9, float(np.sum(base_weights)))
    result = {
        "a": float(a_value),
        "b": float(b_value),
        "tilt_deg": float(tilt_deg),
        "x_ref": float(line_x_at_y({"a": float(a_value), "b": float(b_value)}, y_ref)),
        "y_ref": float(y_ref),
        "fit_rmse_px": float(np.sqrt(np.sum(normalized_weights * residuals**2))),
        "fit_median_abs_residual_px": float(np.median(residuals)),
        "fit_p90_abs_residual_px": float(np.quantile(residuals, 0.90)),
    }
    if fit_cache is not None and cache_key is not None:
        fit_cache[cache_key] = dict(result)
    return result

def build_support_analysis(
    selected_support: list[dict],
    roi_profile: dict,
    support_cache: dict | None = None,
) -> dict:
    cache_key = None
    if support_cache is not None:
        cache_key = make_support_signature(selected_support)
        cached_value = support_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

    selected_total_length_px = float(sum(float(item["line"]["length"]) for item in selected_support))
    selected_total_support_strength = float(sum(float(item["support_strength"]) for item in selected_support))
    chain_support = prune_support_to_dominant_chain(selected_support, roi_profile)
    if not chain_support:
        chain_support = selected_support
    chain_total_length_px = float(sum(float(item["line"]["length"]) for item in chain_support))
    outside_chain_length_ratio = 1.0 - (
        chain_total_length_px / max(1.0, selected_total_length_px)
        if selected_total_length_px > 0.0
        else 0.0
    )
    outside_chain_fragment_ratio = 1.0 - (
        len(chain_support) / max(1, len(selected_support))
        if selected_support
        else 0.0
    )

    result = {
        "selected_total_length_px": float(selected_total_length_px),
        "selected_total_support_strength": float(selected_total_support_strength),
        "chain_support": chain_support,
        "chain_total_length_px": float(chain_total_length_px),
        "outside_chain_length_ratio": float(clip01(outside_chain_length_ratio)),
        "outside_chain_fragment_ratio": float(clip01(outside_chain_fragment_ratio)),
        "chain_metrics": compute_chain_metrics(chain_support),
        "endpoint_metrics": compute_endpoint_metrics(
            selected_support=chain_support,
            y_min=float(roi_profile["trimmed_y_min"]),
            y_max=float(roi_profile["trimmed_y_max"]),
        ),
        "adjustment_metrics": summarize_support_adjustments(selected_support),
    }
    if support_cache is not None and cache_key is not None:
        support_cache[cache_key] = result
    return result
