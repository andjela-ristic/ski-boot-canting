from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from .context import cfg, safe_linear_polyfit


Point = tuple[float, float]
Line = dict[str, float | int | bool]


def line_x_at_y(model: dict[str, float], y_value: float) -> float:
    return float(model["a"] * float(y_value) + model["b"])


def ordered_line_endpoints(line: dict[str, float | int | bool]) -> tuple[float, float, float, float]:
    first = (float(line["y1"]), float(line["x1"]))
    second = (float(line["y2"]), float(line["x2"]))
    if first <= second:
        return (
            float(line["x1"]),
            float(line["y1"]),
            float(line["x2"]),
            float(line["y2"]),
        )
    return (
        float(line["x2"]),
        float(line["y2"]),
        float(line["x1"]),
        float(line["y1"]),
    )


def line_geometry_key(line: dict[str, float | int | bool]) -> tuple[float, ...]:
    start_x, start_y, end_x, end_y = ordered_line_endpoints(line)
    return (
        float(line["y_mid"]),
        float(line["x_mid"]),
        float(line["signed_tilt_deg"]),
        float(line["length"]),
        start_y,
        start_x,
        end_y,
        end_x,
        float(line.get("mask_support_ratio", 0.0)),
        float(line.get("points_inside_mask", 0.0)),
        float(line.get("sampled_points", 0.0)),
        float(line.get("vertical_deviation_degrees", 0.0)),
        float(line.get("angle_degrees", 0.0)),
        float(line.get("a", 0.0)),
        float(line.get("b", 0.0)),
        float(line.get("source_line_index", line.get("line_index", 0))),
    )


def canonicalize_lines(lines: list[dict[str, float | int | bool]]) -> list[dict[str, float | int | bool]]:
    ordered = sorted(lines, key=line_geometry_key)
    canonical_lines: list[dict[str, float | int | bool]] = []
    for canonical_index, line in enumerate(ordered, start=1):
        canonical_line = deepcopy(line)
        canonical_line["canonical_index"] = int(canonical_index)
        canonical_lines.append(canonical_line)
    return canonical_lines

def make_axis_signature(axis: dict[str, float]) -> tuple[float, float]:
    return (
        float(axis["a"]),
        float(axis["b"]),
    )

def line_from_angle_and_anchor(angle_deg: float, x_ref: float, y_ref: float) -> dict[str, float]:
    a = math.tan(math.radians(float(angle_deg)))
    b = float(x_ref) - a * float(y_ref)
    return {
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(angle_deg),
        "x_ref": float(x_ref),
        "y_ref": float(y_ref),
    }

def blend_axis_toward_reference(
    primary_axis: dict[str, float],
    reference_axis: dict[str, float],
    y_ref: float,
    reference_pull_ratio: float,
) -> dict[str, float]:
    reference_pull_ratio = float(np.clip(reference_pull_ratio, 0.0, 1.0))
    primary_weight = 1.0 - reference_pull_ratio
    blended_x_ref = primary_weight * float(primary_axis["x_ref"]) + reference_pull_ratio * float(reference_axis["x_ref"])
    blended_tilt_deg = (
        primary_weight * float(primary_axis["tilt_deg"]) + reference_pull_ratio * float(reference_axis["tilt_deg"])
    )
    return line_from_angle_and_anchor(blended_tilt_deg, blended_x_ref, y_ref)

def build_row_profile(mask: np.ndarray) -> dict | None:
    if mask is None:
        return None

    height, width = mask.shape[:2]
    left_bounds = np.full(height, width, dtype=np.int32)
    right_bounds = np.full(height, -1, dtype=np.int32)
    row_widths = np.zeros(height, dtype=np.int32)
    row_centers = np.full(height, np.nan, dtype=np.float64)

    mask_bool = mask > 0
    valid_row_mask = np.count_nonzero(mask_bool, axis=1) >= 2
    valid_row_indices = np.flatnonzero(valid_row_mask)
    if valid_row_indices.size:
        all_left = np.argmax(mask_bool, axis=1).astype(np.int32, copy=False)
        all_right = (
            width
            - 1
            - np.argmax(mask_bool[:, ::-1], axis=1).astype(np.int32, copy=False)
        )
        left = all_left[valid_row_indices]
        right = all_right[valid_row_indices]
        left_bounds[valid_row_indices] = left
        right_bounds[valid_row_indices] = right
        row_widths[valid_row_indices] = right - left + 1
        row_centers[valid_row_indices] = (left + right) / 2.0

    valid_rows = np.flatnonzero(row_widths > 0)
    if valid_rows.size == 0:
        return None

    y_min = int(valid_rows[0])
    y_max = int(valid_rows[-1])
    span = max(1, y_max - y_min)
    trim_top_ratio = float(cfg("roi_profile", "trim_top_ratio", default=0.06))
    trim_bottom_ratio = float(cfg("roi_profile", "trim_bottom_ratio", default=0.06))
    trimmed_y_min = int(round(y_min + span * trim_top_ratio))
    trimmed_y_max = int(round(y_max - span * trim_bottom_ratio))
    min_row_width_px = int(cfg("roi_profile", "min_row_width_px", default=140))

    trimmed_mask = (
        (np.arange(height) >= trimmed_y_min)
        & (np.arange(height) <= trimmed_y_max)
        & (row_widths >= min_row_width_px)
    )
    trimmed_rows = np.flatnonzero(trimmed_mask)
    if trimmed_rows.size < 10:
        trimmed_rows = valid_rows

    fit = safe_linear_polyfit(
        trimmed_rows.astype(np.float64),
        row_centers[trimmed_rows].astype(np.float64),
    )
    if fit is None:
        return None

    width_quantile = float(cfg("roi_profile", "center_width_quantile", default=0.35))
    reference_width = float(np.quantile(row_widths[trimmed_rows].astype(np.float64), width_quantile))
    center_fit = {
        "a": float(fit[0]),
        "b": float(fit[1]),
        "tilt_deg": float(math.degrees(math.atan(float(fit[0])))),
    }

    return {
        "height": height,
        "width": width,
        "mask": mask,
        "left_bounds": left_bounds,
        "right_bounds": right_bounds,
        "row_widths": row_widths,
        "row_centers": row_centers,
        "valid_rows": valid_rows,
        "trimmed_rows": trimmed_rows,
        "y_min": y_min,
        "y_max": y_max,
        "trimmed_y_min": int(trimmed_rows[0]),
        "trimmed_y_max": int(trimmed_rows[-1]),
        "y_ref": float((trimmed_rows[0] + trimmed_rows[-1]) / 2.0),
        "reference_width_px": reference_width,
        "median_center_x": float(np.nanmedian(row_centers[trimmed_rows])),
        "center_fit": center_fit,
    }

def normalize_line(raw_line: dict, fallback_index: int) -> dict[str, float | int | bool]:
    x1 = float(raw_line["x1"])
    y1 = float(raw_line["y1"])
    x2 = float(raw_line["x2"])
    y2 = float(raw_line["y2"])
    dx = x2 - x1
    dy = y2 - y1
    length = float(raw_line.get("length", math.hypot(dx, dy)))

    if abs(dy) > 1e-6:
        a = dx / dy
        b = x1 - a * y1
        signed_tilt_deg = math.degrees(math.atan(a))
    else:
        a = 999.0
        b = float((x1 + x2) / 2.0)
        signed_tilt_deg = 90.0 if dx >= 0 else -90.0

    return {
        "line_index": int(raw_line.get("line_index", raw_line.get("id", fallback_index))),
        "source_line_index": int(raw_line.get("line_index", raw_line.get("id", fallback_index))),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length": length,
        "angle_degrees": float(raw_line.get("angle_degrees", 0.0)),
        "mask_support_ratio": float(raw_line.get("mask_support_ratio", 1.0)),
        "sampled_points": int(raw_line.get("sampled_points", 0)),
        "points_inside_mask": int(raw_line.get("points_inside_mask", 0)),
        "vertical_deviation_degrees": float(raw_line.get("vertical_deviation_degrees", abs(signed_tilt_deg))),
        "signed_tilt_deg": float(signed_tilt_deg),
        "a": float(a),
        "b": float(b),
        "x_mid": float((x1 + x2) / 2.0),
        "y_mid": float((y1 + y2) / 2.0),
        "y_min": float(min(y1, y2)),
        "y_max": float(max(y1, y2)),
        "is_valid": bool(raw_line.get("is_valid", True)),
    }

def filter_fragments(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    max_vertical_deviation = float(cfg("fragment_filter", "max_vertical_deviation_deg", default=18.0))
    min_mask_support_ratio = float(cfg("fragment_filter", "min_mask_support_ratio", default=0.94))
    min_points_inside_mask = int(cfg("fragment_filter", "min_points_inside_mask", default=60))

    for line in lines:
        reasons = []
        if not line["is_valid"]:
            reasons.append("not_valid")
        if abs(float(line["signed_tilt_deg"])) > max_vertical_deviation:
            reasons.append("too_tilted")
        if float(line["mask_support_ratio"]) < min_mask_support_ratio:
            reasons.append("low_mask_support")
        if int(line["points_inside_mask"]) and int(line["points_inside_mask"]) < min_points_inside_mask:
            reasons.append("too_few_points_inside_mask")

        if reasons:
            rejected.append({**line, "reject_reasons": reasons})
        else:
            accepted.append(line)

    return accepted, rejected

def segment_axis_distance_px(line: dict, axis: dict[str, float]) -> float:
    delta_a = float(line["a"]) - float(axis["a"])
    delta_b = float(line["b"]) - float(axis["b"])
    distance_min = abs(delta_a * float(line["y_min"]) + delta_b)
    distance_mid = abs(delta_a * float(line["y_mid"]) + delta_b)
    distance_max = abs(delta_a * float(line["y_max"]) + delta_b)
    return float((distance_min + distance_mid + distance_max) / 3.0)

def segment_axis_min_distance_px(line: dict, axis: dict[str, float], epsilon: float = 1e-8) -> float:
    line_y_min = float(line["y_min"])
    line_y_max = float(line["y_max"])
    delta_a = float(line["a"]) - float(axis["a"])
    delta_b = float(line["b"]) - float(axis["b"])

    if abs(delta_a) > epsilon:
        intersection_y = -delta_b / delta_a
        if line_y_min - epsilon <= intersection_y <= line_y_max + epsilon:
            return 0.0

    distance_min = abs(delta_a * line_y_min + delta_b)
    distance_max = abs(delta_a * line_y_max + delta_b)
    return float(min(distance_min, distance_max))

def segment_axis_intersection_y(line: dict, axis: dict[str, float], epsilon: float = 1e-8) -> float | None:
    line_a = float(line["a"])
    axis_a = float(axis["a"])
    slope_delta = line_a - axis_a
    if abs(slope_delta) <= epsilon:
        return None

    intersection_y = (float(axis["b"]) - float(line["b"])) / slope_delta
    if intersection_y < float(line["y_min"]) - epsilon or intersection_y > float(line["y_max"]) + epsilon:
        return None
    return float(intersection_y)
