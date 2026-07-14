from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG.get("display", {})
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
STEP_05_CONFIG = CONFIG.get("step_05_valid_hough_lines_in_roi", {})

STEP_CONFIG_RAW = CONFIG.get("step_06_search_central_ruler", {})

DEFAULT_STEP_CONFIG = {
    "enabled": True,
    "inherit_step_05_output": True,
    "input_subdir": "05_valid_hough_lines_in_roi",
    "input_json_subdir": "valid_lines_json",
    "input_visual_subdir": "05_valid_hough_lines_in_roi/valid_lines_overlay",
    "input_overlay_subdir": "valid_lines_overlay",
    "output_subdir": "06_search_central_ruler",
    "cleanup_output_on_start": True,
    "fragment_filter": {
        "min_length_px": 90,
        "max_vertical_deviation_deg": 18.0,
        "min_mask_support_ratio": 0.94,
        "min_points_inside_mask": 60,
    },
    "roi_profile": {
        "trim_top_ratio": 0.06,
        "trim_bottom_ratio": 0.06,
        "min_row_width_px": 140,
        "sample_step_px": 10,
        "center_width_quantile": 0.35,
    },
    "search": {
        "max_candidate_tilt_deg": 12.0,
        "coarse_angle_step_deg": 1.0,
        "fine_angle_step_deg": 0.2,
        "coarse_x_step_px": 14,
        "fine_x_step_px": 2,
        "coarse_band_half_width_px": 16.0,
        "final_band_half_width_px": 9.0,
        "coarse_max_angle_error_deg": 6.0,
        "final_max_angle_error_deg": 4.0,
        "coarse_candidate_pool_limit": 40,
        "fine_candidate_pool_limit": 32,
        "coarse_angle_bucket_deg": 1.0,
        "max_coarse_candidates_per_angle_bucket": 2,
        "top_coarse_candidates": 10,
        "fine_window_x_px": 28,
        "fine_window_angle_deg": 1.6,
        "min_support_fragments": 2,
    },
    "coverage": {
        "bin_count": 12,
        "min_supported_bins": 4,
    },
    "gaps": {
        "soft_gap_px": 80,
        "hard_gap_px": 260,
    },
    "endpoint_support": {
        "band_ratio": 0.15,
        "min_band_px": 80,
        "max_band_px": 160,
    },
    "final_fit": {
        "sample_step_px": 22,
        "huber_delta_px": 10.0,
        "huber_iterations": 4,
        "max_fit_tilt_deg": 12.0,
    },
    "support_adjustment": {
        "enabled": True,
        "max_midpoint_shift_px": 6.0,
        "max_tilt_delta_deg": 2.0,
        "max_mean_shift_px": 6.0,
        "max_endpoint_shift_px": 9.0,
        "require_axis_intersection_for_tilt_adjustment": True,
        "min_support_strength_scale": 0.62,
    },
    "support_extension": {
        "enabled": True,
        "trigger_gap_px": 180,
        "max_added_fragments": 6,
        "min_vertical_advance_px": 18,
        "max_connection_dx_px": 34.0,
        "max_connection_gap_px": 360.0,
        "max_axis_distance_px": 55.0,
        "max_center_distance_px": 70.0,
        "max_vertical_deviation_deg": 10.0,
    },
    "candidate_deduplication": {
        "max_mean_axis_distance_px": 8.0,
        "max_angle_difference_deg": 0.45,
        "max_saved_candidates": 8,
    },
    "scoring": {
        "fragment_support_weight": 0.34,
        "vertical_coverage_weight": 0.22,
        "symmetry_weight": 0.24,
        "roi_center_weight": 0.08,
        "gap_penalty_weight": 0.14,
        "endpoint_anchor_weight": 0.10,
        "outside_mask_penalty_weight": 0.16,
        "support_adjustment_penalty_weight": 0.08,
        "low_support_penalty": 0.20,
        "low_coverage_penalty": 0.20,
    },
    "drawing": {
        "background_alpha": 0.78,
        "all_fragment_thickness": 2,
        "candidate_thickness": 2,
        "selected_fragment_thickness": 3,
        "final_line_thickness": 4,
        "show_candidate_lines": True,
        "candidate_count_to_draw": 3,
        "label_candidates": True,
        "font_scale": 0.64,
    },
    "test_presets": [
        {"name": "default", "override": {}},
        {
            "name": "wider_band",
            "override": {
                "search": {
                    "coarse_band_half_width_px": 20.0,
                    "final_band_half_width_px": 11.0,
                    "coarse_max_angle_error_deg": 7.0,
                    "final_max_angle_error_deg": 5.0,
                }
            },
        },
        {
            "name": "stricter_symmetry",
            "override": {
                "scoring": {
                    "symmetry_weight": 0.30,
                    "gap_penalty_weight": 0.16,
                    "outside_mask_penalty_weight": 0.18,
                }
            },
        },
        {
            "name": "wider_search",
            "override": {
                "search": {
                    "max_candidate_tilt_deg": 15.0,
                    "coarse_x_step_px": 12,
                    "fine_window_x_px": 34,
                }
            },
        },
    ],
}


COLOR_ALL_FRAGMENTS = (0, 255, 0)
COLOR_SELECTED_FRAGMENTS = (255, 0, 255)
COLOR_FINAL_AXIS = (255, 0, 0)
COLOR_CANDIDATE = (0, 165, 255)
COLOR_TEXT = (235, 235, 235)


def deep_merge(base: dict, override: dict | None) -> dict:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def apply_preset(config: dict, preset_name: str | None) -> dict:
    if not preset_name:
        return config

    for preset in config.get("test_presets", []):
        if str(preset.get("name")) == str(preset_name):
            return deep_merge(config, preset.get("override", {}))

    available = [preset.get("name") for preset in config.get("test_presets", [])]
    raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")


STEP_CONFIG = deep_merge(DEFAULT_STEP_CONFIG, STEP_CONFIG_RAW)


def cfg(*keys, default=None):
    current = STEP_CONFIG
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def get_step_dirs() -> dict[str, Path]:
    step_05_output_subdir = str(STEP_05_CONFIG.get("output_subdir", "05_valid_hough_lines_in_roi"))
    if bool(STEP_CONFIG.get("inherit_step_05_output", True)):
        input_subdir = step_05_output_subdir
    else:
        input_subdir = str(STEP_CONFIG.get("input_subdir", step_05_output_subdir))
    output_subdir = STEP_CONFIG.get("output_subdir", "06_search_central_ruler")
    input_dir = PROCESSED_DIR / input_subdir
    output_dir = PROCESSED_DIR / output_subdir
    return {
        "input_dir": input_dir,
        "input_json_dir": input_dir / STEP_CONFIG.get("input_json_subdir", "valid_lines_json"),
        "input_overlay_dir": input_dir / STEP_CONFIG.get("input_overlay_subdir", "valid_lines_overlay"),
        "output_dir": output_dir,
        "output_overlay_dir": output_dir / "overlay",
        "output_comparison_dir": output_dir / "comparison",
        "output_metadata_dir": output_dir / "metadata",
    }


def ensure_dirs(cleanup: bool = False) -> None:
    dirs = get_step_dirs()
    output_dir = dirs["output_dir"]
    if cleanup and output_dir.exists():
        shutil.rmtree(output_dir)
    dirs["output_overlay_dir"].mkdir(parents=True, exist_ok=True)
    dirs["output_comparison_dir"].mkdir(parents=True, exist_ok=True)
    dirs["output_metadata_dir"].mkdir(parents=True, exist_ok=True)


def resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def safe_linear_polyfit(
    y_values: np.ndarray | list[float],
    x_values: np.ndarray | list[float],
    weights: np.ndarray | list[float] | None = None,
) -> tuple[float, float] | None:
    if len(y_values) < 2:
        return None

    y_array = np.asarray(y_values, dtype=np.float64)
    x_array = np.asarray(x_values, dtype=np.float64)
    if np.ptp(y_array) <= 1e-6:
        return 0.0, float(np.median(x_array))

    fit_weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coefficients = np.polyfit(y_array, x_array, 1, w=fit_weights)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, Warning):
        return 0.0, float(np.median(x_array))

    return float(coefficients[0]), float(coefficients[1])


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def load_roi_mask(mask_path: Path | None) -> np.ndarray | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return ensure_binary_mask(mask)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def put_text(image: np.ndarray, text: str, x: int, y: int, color=COLOR_TEXT, scale: float | None = None) -> None:
    font_scale = float(cfg("drawing", "font_scale", default=0.64) if scale is None else scale)
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )


def line_x_at_y(model: dict[str, float], y_value: float) -> float:
    return float(model["a"] * float(y_value) + model["b"])


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


def build_row_profile(mask: np.ndarray) -> dict | None:
    if mask is None:
        return None

    height, width = mask.shape[:2]
    left_bounds = np.full(height, width, dtype=np.int32)
    right_bounds = np.full(height, -1, dtype=np.int32)
    row_widths = np.zeros(height, dtype=np.int32)
    row_centers = np.full(height, np.nan, dtype=np.float64)

    for y_index in range(height):
        row_x = np.flatnonzero(mask[y_index] > 0)
        if row_x.size < 2:
            continue
        left_bounds[y_index] = int(row_x[0])
        right_bounds[y_index] = int(row_x[-1])
        row_widths[y_index] = int(row_x[-1] - row_x[0] + 1)
        row_centers[y_index] = float((row_x[0] + row_x[-1]) / 2.0)

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
    min_length = float(cfg("fragment_filter", "min_length_px", default=90))
    max_vertical_deviation = float(cfg("fragment_filter", "max_vertical_deviation_deg", default=18.0))
    min_mask_support_ratio = float(cfg("fragment_filter", "min_mask_support_ratio", default=0.94))
    min_points_inside_mask = int(cfg("fragment_filter", "min_points_inside_mask", default=60))

    for line in lines:
        reasons = []
        if not line["is_valid"]:
            reasons.append("not_valid")
        if float(line["length"]) < min_length:
            reasons.append("too_short")
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
    probe_ys = [float(line["y_min"]), float(line["y_mid"]), float(line["y_max"])]
    distances = []
    for probe_y in probe_ys:
        line_x = line_x_at_y(line, probe_y)
        axis_x = line_x_at_y(axis, probe_y)
        distances.append(abs(line_x - axis_x))
    return float(np.mean(distances))


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


def build_adjusted_line_variant(line: dict, axis: dict[str, float]) -> tuple[dict, dict] | tuple[None, None]:
    if not bool(cfg("support_adjustment", "enabled", default=True)):
        return None, None

    max_midpoint_shift_px = float(cfg("support_adjustment", "max_midpoint_shift_px", default=10.0))
    max_tilt_delta_deg = float(cfg("support_adjustment", "max_tilt_delta_deg", default=2.0))
    max_mean_shift_px = float(cfg("support_adjustment", "max_mean_shift_px", default=10.0))
    max_endpoint_shift_px = float(cfg("support_adjustment", "max_endpoint_shift_px", default=18.0))
    require_axis_intersection_for_tilt_adjustment = bool(
        cfg("support_adjustment", "require_axis_intersection_for_tilt_adjustment", default=True)
    )

    original_tilt_deg = float(line["signed_tilt_deg"])
    target_tilt_deg = float(axis["tilt_deg"])
    applied_tilt_delta_deg = float(np.clip(target_tilt_deg - original_tilt_deg, -max_tilt_delta_deg, max_tilt_delta_deg))
    original_segment_intersection_y = segment_axis_intersection_y(line, axis)
    tilt_adjustment_blocked = False
    if (
        require_axis_intersection_for_tilt_adjustment
        and abs(applied_tilt_delta_deg) >= 0.05
        and original_segment_intersection_y is None
    ):
        applied_tilt_delta_deg = 0.0
        tilt_adjustment_blocked = True
    adjusted_tilt_deg = original_tilt_deg + applied_tilt_delta_deg
    adjusted_a = math.tan(math.radians(adjusted_tilt_deg))

    y_mid = float(line["y_mid"])
    target_x_mid = float(line_x_at_y(axis, y_mid))
    applied_midpoint_shift_px = float(
        np.clip(target_x_mid - float(line["x_mid"]), -max_midpoint_shift_px, max_midpoint_shift_px)
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

    probe_ys = [float(line["y_min"]), y_mid, float(line["y_max"])]
    signed_probe_shifts = np.asarray(
        [line_x_at_y(adjusted_line, probe_y) - line_x_at_y(line, probe_y) for probe_y in probe_ys],
        dtype=np.float64,
    )
    abs_probe_shifts = np.abs(signed_probe_shifts)
    mean_abs_shift_px = float(np.mean(abs_probe_shifts))
    max_abs_shift_px = float(np.max(abs_probe_shifts))

    if mean_abs_shift_px > max_mean_shift_px or max_abs_shift_px > max_endpoint_shift_px:
        return None, None

    if abs(applied_midpoint_shift_px) < 0.05 and abs(applied_tilt_delta_deg) < 0.05:
        return None, None

    original_axis_distance_px = float(segment_axis_distance_px(line, axis))
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
) -> list[dict]:
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

    selected.sort(key=lambda item: (item["support_strength"], item["line"]["length"]), reverse=True)
    return selected


def compute_vertical_coverage(
    selected_support: list[dict],
    y_min: float,
    y_max: float,
) -> dict[str, float | int]:
    bin_count = int(cfg("coverage", "bin_count", default=12))
    total_span = max(1.0, float(y_max) - float(y_min))
    supported_bins: set[int] = set()

    for item in selected_support:
        line = item["line"]
        start_bin = int(np.floor((float(line["y_min"]) - float(y_min)) / total_span * bin_count))
        end_bin = int(np.floor((float(line["y_max"]) - float(y_min)) / total_span * bin_count))
        start_bin = max(0, min(bin_count - 1, start_bin))
        end_bin = max(0, min(bin_count - 1, end_bin))
        for bin_index in range(start_bin, end_bin + 1):
            supported_bins.add(bin_index)

    supported_bin_count = len(supported_bins)
    return {
        "bin_count": bin_count,
        "supported_bin_count": supported_bin_count,
        "coverage_score": clip01(supported_bin_count / max(1, bin_count)),
    }


def merge_support_intervals(selected_support: list[dict]) -> list[tuple[float, float]]:
    intervals = sorted(
        (float(item["line"]["y_min"]), float(item["line"]["y_max"]))
        for item in selected_support
    )
    if not intervals:
        return []

    merged = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


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
    del y_min, y_max

    merged = merge_support_intervals(selected_support)
    if not merged:
        return {
            "support_y_min": 0.0,
            "support_y_max": 0.0,
            "support_span_px": 0.0,
            "endpoint_band_px": 0.0,
            "top_endpoint_coverage": 0.0,
            "bottom_endpoint_coverage": 0.0,
            "top_endpoint_alignment_score": 0.0,
            "bottom_endpoint_alignment_score": 0.0,
            "endpoint_anchor_score": 0.0,
        }

    support_y_min = float(merged[0][0])
    support_y_max = float(merged[-1][1])
    support_span_px = max(1.0, support_y_max - support_y_min)

    band_ratio = float(cfg("endpoint_support", "band_ratio", default=0.15))
    min_band_px = float(cfg("endpoint_support", "min_band_px", default=80))
    max_band_px = float(cfg("endpoint_support", "max_band_px", default=160))
    endpoint_band_px = float(np.clip(support_span_px * band_ratio, min_band_px, max_band_px))
    endpoint_band_px = min(endpoint_band_px, support_span_px * 0.5)

    top_band_end = float(min(support_y_max, support_y_min + endpoint_band_px))
    bottom_band_start = float(max(support_y_min, support_y_max - endpoint_band_px))
    top_band_size_px = max(1.0, top_band_end - support_y_min)
    bottom_band_size_px = max(1.0, support_y_max - bottom_band_start)

    def band_overlap_coverage(band_start: float, band_end: float, band_size_px: float) -> float:
        overlap_px = 0.0
        for start, end in merged:
            overlap_px += max(0.0, min(end, band_end) - max(start, band_start))
        return clip01(overlap_px / max(1.0, band_size_px))

    def band_alignment_score(band_start: float, band_end: float) -> float:
        weighted_alignment_sum = 0.0
        overlap_weight_sum = 0.0
        for item in selected_support:
            line_start = float(item["line"]["y_min"])
            line_end = float(item["line"]["y_max"])
            overlap_px = max(0.0, min(line_end, band_end) - max(line_start, band_start))
            if overlap_px <= 0.0:
                continue
            alignment_score = 0.7 * float(item["distance_alignment"]) + 0.3 * float(item["angle_alignment"])
            weighted_alignment_sum += alignment_score * overlap_px
            overlap_weight_sum += overlap_px
        if overlap_weight_sum <= 0.0:
            return 0.0
        return clip01(weighted_alignment_sum / overlap_weight_sum)

    top_endpoint_coverage = band_overlap_coverage(support_y_min, top_band_end, top_band_size_px)
    bottom_endpoint_coverage = band_overlap_coverage(bottom_band_start, support_y_max, bottom_band_size_px)
    top_endpoint_alignment_score = band_alignment_score(support_y_min, top_band_end)
    bottom_endpoint_alignment_score = band_alignment_score(bottom_band_start, support_y_max)

    endpoint_anchor_score = 0.25 * (
        top_endpoint_coverage
        + bottom_endpoint_coverage
        + top_endpoint_alignment_score
        + bottom_endpoint_alignment_score
    )

    return {
        "support_y_min": float(support_y_min),
        "support_y_max": float(support_y_max),
        "support_span_px": float(support_span_px),
        "endpoint_band_px": float(endpoint_band_px),
        "top_endpoint_coverage": float(top_endpoint_coverage),
        "bottom_endpoint_coverage": float(bottom_endpoint_coverage),
        "top_endpoint_alignment_score": float(top_endpoint_alignment_score),
        "bottom_endpoint_alignment_score": float(bottom_endpoint_alignment_score),
        "endpoint_anchor_score": float(endpoint_anchor_score),
    }


def compute_row_balance_metrics(axis: dict[str, float], roi_profile: dict) -> dict[str, float]:
    sample_step_px = int(cfg("roi_profile", "sample_step_px", default=10))
    trimmed_rows = roi_profile["trimmed_rows"]
    sampled_rows = trimmed_rows[::max(1, sample_step_px)]
    if sampled_rows.size == 0:
        sampled_rows = trimmed_rows

    symmetry_errors = []
    center_errors = []
    inside_rows = 0

    left_bounds = roi_profile["left_bounds"]
    right_bounds = roi_profile["right_bounds"]
    row_widths = roi_profile["row_widths"]
    center_fit = roi_profile["center_fit"]

    for row_index in sampled_rows:
        width = float(row_widths[row_index])
        if width <= 0:
            continue

        axis_x = line_x_at_y(axis, float(row_index))
        left = float(left_bounds[row_index])
        right = float(right_bounds[row_index])
        if axis_x < left or axis_x > right:
            continue

        inside_rows += 1
        left_width = axis_x - left
        right_width = right - axis_x
        symmetry_errors.append(abs(left_width - right_width) / max(1.0, width))

        fitted_center = line_x_at_y(center_fit, float(row_index))
        center_errors.append(abs(axis_x - fitted_center) / max(1.0, width * 0.5))

    total_rows = len(sampled_rows)
    outside_mask_penalty = clip01(1.0 - inside_rows / max(1, total_rows))
    symmetry_score = 0.0 if not symmetry_errors else clip01(1.0 - float(np.median(symmetry_errors)))
    center_score = 0.0 if not center_errors else clip01(1.0 - float(np.median(center_errors)))

    return {
        "sampled_row_count": int(total_rows),
        "rows_inside_mask_count": int(inside_rows),
        "outside_mask_penalty": float(outside_mask_penalty),
        "symmetry_score": float(symmetry_score),
        "roi_center_score": float(center_score),
    }


def summarize_candidate_from_support(
    axis: dict[str, float],
    selected_support: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
) -> dict:
    selected_total_length_px = float(sum(float(item["line"]["length"]) for item in selected_support))
    selected_total_support_strength = float(sum(float(item["support_strength"]) for item in selected_support))
    fragment_support_score = clip01(selected_total_support_strength / max(1.0, total_available_length_px))

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
    endpoint_metrics = compute_endpoint_metrics(
        selected_support=selected_support,
        y_min=float(roi_profile["trimmed_y_min"]),
        y_max=float(roi_profile["trimmed_y_max"]),
    )
    row_metrics = compute_row_balance_metrics(axis=axis, roi_profile=roi_profile)
    adjustment_metrics = summarize_support_adjustments(selected_support)

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

    return {
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
        "endpoint_anchor_score": float(endpoint_metrics["endpoint_anchor_score"]),
        "outside_mask_penalty": float(row_metrics["outside_mask_penalty"]),
        "symmetry_score": float(row_metrics["symmetry_score"]),
        "roi_center_score": float(row_metrics["roi_center_score"]),
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


def evaluate_candidate(
    axis: dict[str, float],
    lines: list[dict],
    roi_profile: dict,
    total_available_length_px: float,
    band_half_width_px: float,
    max_angle_error_deg: float,
    allow_adjustment: bool = False,
) -> dict:
    selected_support = select_support_fragments(
        lines=lines,
        axis=axis,
        band_half_width_px=band_half_width_px,
        max_angle_error_deg=max_angle_error_deg,
        allow_adjustment=allow_adjustment,
    )
    return summarize_candidate_from_support(axis, selected_support, roi_profile, total_available_length_px)


def merge_support_items(primary_support: list[dict], secondary_support: list[dict]) -> list[dict]:
    merged_by_line_index: dict[int, dict] = {}
    for item in primary_support + secondary_support:
        line_index = int(item["line"]["line_index"])
        existing = merged_by_line_index.get(line_index)
        if existing is None or float(item["support_strength"]) > float(existing["support_strength"]):
            merged_by_line_index[line_index] = item

    merged = list(merged_by_line_index.values())
    merged.sort(key=lambda item: (item["support_strength"], item["line"]["length"]), reverse=True)
    return merged


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

            extension_score = (
                0.42 * continuity_score
                + 0.26 * verticality_score
                + 0.18 * center_score
                + 0.10 * axis_score
                + 0.04 * length_score
            )

            if extension_score <= best_extension_score:
                continue

            distance_alignment = float(support_item["distance_alignment"])
            angle_alignment = float(support_item["angle_alignment"])
            support_strength = float(line["length"]) * (
                0.45 * continuity_score
                + 0.25 * verticality_score
                + 0.15 * center_score
                + 0.10 * distance_alignment
                + 0.05 * angle_alignment
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

    extended_support.sort(key=lambda item: (item["support_strength"], item["line"]["length"]), reverse=True)
    return extended_support


def mean_axis_distance_px(candidate_a: dict, candidate_b: dict, roi_profile: dict) -> float:
    ys = np.linspace(
        float(roi_profile["trimmed_y_min"]),
        float(roi_profile["trimmed_y_max"]),
        8,
    )
    return float(
        np.mean(
            [
                abs(line_x_at_y(candidate_a, float(y_value)) - line_x_at_y(candidate_b, float(y_value)))
                for y_value in ys
            ]
        )
    )


def deduplicate_candidates(
    candidates: list[dict],
    roi_profile: dict,
    max_candidates: int | None = None,
) -> list[dict]:
    kept: list[dict] = []
    max_mean_axis_distance_px = float(cfg("candidate_deduplication", "max_mean_axis_distance_px", default=8.0))
    max_angle_difference_deg = float(cfg("candidate_deduplication", "max_angle_difference_deg", default=0.45))

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        is_duplicate = False
        for existing in kept:
            if abs(float(candidate["tilt_deg"]) - float(existing["tilt_deg"])) <= max_angle_difference_deg:
                if mean_axis_distance_px(candidate, existing, roi_profile) <= max_mean_axis_distance_px:
                    is_duplicate = True
                    break
        if not is_duplicate:
            kept.append(candidate)

    if max_candidates is None:
        max_candidates = int(cfg("candidate_deduplication", "max_saved_candidates", default=8))

    return kept[: max(0, int(max_candidates))]


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


def build_point_cloud(selected_support: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_px = float(cfg("final_fit", "sample_step_px", default=22))
    y_values = []
    x_values = []
    weights = []

    for item in selected_support:
        line = item.get("effective_line", item["line"])
        length = max(1.0, float(line["length"]))
        sample_count = max(2, int(math.ceil(length / max(1.0, step_px))) + 1)
        ts = np.linspace(0.0, 1.0, sample_count)
        for t_value in ts:
            y_coord = float(line["y1"] + t_value * (line["y2"] - line["y1"]))
            x_coord = float(line["x1"] + t_value * (line["x2"] - line["x1"]))
            y_values.append(y_coord)
            x_values.append(x_coord)
            weights.append(max(1.0, float(item["support_strength"])))

    return (
        np.asarray(y_values, dtype=np.float64),
        np.asarray(x_values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def fit_axis_from_support(selected_support: list[dict], y_ref: float) -> dict[str, float] | None:
    if not selected_support:
        return None

    y_values, x_values, base_weights = build_point_cloud(selected_support)
    if len(y_values) < 2:
        return None

    fit = safe_linear_polyfit(y_values, x_values, base_weights)
    if fit is None:
        return None

    a_value, b_value = fit
    huber_delta_px = float(cfg("final_fit", "huber_delta_px", default=10.0))
    huber_iterations = int(cfg("final_fit", "huber_iterations", default=4))

    current_weights = base_weights.copy()
    for _ in range(max(0, huber_iterations)):
        residuals = np.abs(x_values - (a_value * y_values + b_value))
        huber_weights = np.ones_like(residuals)
        large_residual_mask = residuals > huber_delta_px
        huber_weights[large_residual_mask] = huber_delta_px / np.maximum(residuals[large_residual_mask], 1e-6)
        fit = safe_linear_polyfit(y_values, x_values, current_weights * huber_weights)
        if fit is None:
            break
        a_value, b_value = fit

    tilt_deg = float(math.degrees(math.atan(float(a_value))))
    max_fit_tilt_deg = float(cfg("final_fit", "max_fit_tilt_deg", default=12.0))
    if abs(tilt_deg) > max_fit_tilt_deg:
        return None

    return {
        "a": float(a_value),
        "b": float(b_value),
        "tilt_deg": float(tilt_deg),
        "x_ref": float(line_x_at_y({"a": float(a_value), "b": float(b_value)}, y_ref)),
        "y_ref": float(y_ref),
    }


def search_best_candidate(lines: list[dict], roi_profile: dict) -> dict:
    if not lines:
        return {
            "coarse_candidates": [],
            "fine_candidates": [],
            "best_hypothesis": None,
            "best_candidate": None,
        }

    total_available_length_px = float(sum(float(line["length"]) for line in lines))
    max_candidate_tilt_deg = float(cfg("search", "max_candidate_tilt_deg", default=12.0))
    coarse_angle_step_deg = float(cfg("search", "coarse_angle_step_deg", default=1.0))
    fine_angle_step_deg = float(cfg("search", "fine_angle_step_deg", default=0.2))
    coarse_x_step_px = int(cfg("search", "coarse_x_step_px", default=14))
    fine_x_step_px = int(cfg("search", "fine_x_step_px", default=2))
    coarse_band_half_width_px = float(cfg("search", "coarse_band_half_width_px", default=16.0))
    final_band_half_width_px = float(cfg("search", "final_band_half_width_px", default=9.0))
    coarse_max_angle_error_deg = float(cfg("search", "coarse_max_angle_error_deg", default=6.0))
    final_max_angle_error_deg = float(cfg("search", "final_max_angle_error_deg", default=4.0))
    coarse_candidate_pool_limit = int(cfg("search", "coarse_candidate_pool_limit", default=40))
    fine_candidate_pool_limit = int(cfg("search", "fine_candidate_pool_limit", default=32))
    coarse_angle_bucket_deg = float(cfg("search", "coarse_angle_bucket_deg", default=coarse_angle_step_deg))
    max_coarse_candidates_per_angle_bucket = int(
        cfg("search", "max_coarse_candidates_per_angle_bucket", default=2)
    )
    fine_window_x_px = int(cfg("search", "fine_window_x_px", default=28))
    fine_window_angle_deg = float(cfg("search", "fine_window_angle_deg", default=1.6))
    top_coarse_candidate_count = int(cfg("search", "top_coarse_candidates", default=10))

    trimmed_rows = roi_profile["trimmed_rows"]
    left_bounds = roi_profile["left_bounds"][trimmed_rows]
    right_bounds = roi_profile["right_bounds"][trimmed_rows]
    x_min = int(np.min(left_bounds))
    x_max = int(np.max(right_bounds))
    y_ref = float(roi_profile["y_ref"])

    coarse_angles = np.arange(-max_candidate_tilt_deg, max_candidate_tilt_deg + 0.5 * coarse_angle_step_deg, coarse_angle_step_deg)
    coarse_x_values = np.arange(x_min, x_max + 1, max(1, coarse_x_step_px))

    coarse_candidates = make_candidate_grid(
        angle_values=coarse_angles,
        x_ref_values=coarse_x_values,
        y_ref=y_ref,
        lines=lines,
        roi_profile=roi_profile,
        total_available_length_px=total_available_length_px,
        band_half_width_px=coarse_band_half_width_px,
        max_angle_error_deg=coarse_max_angle_error_deg,
        allow_adjustment=False,
    )
    coarse_candidates = deduplicate_candidates(
        coarse_candidates,
        roi_profile,
        max_candidates=max(coarse_candidate_pool_limit, top_coarse_candidate_count),
    )

    top_coarse = select_diverse_candidates_by_angle(
        coarse_candidates,
        max_candidates=top_coarse_candidate_count,
        angle_bucket_deg=coarse_angle_bucket_deg,
        max_per_bucket=max_coarse_candidates_per_angle_bucket,
    )
    fine_candidates: list[dict] = []
    use_support_adjustment = bool(cfg("support_adjustment", "enabled", default=True))

    for coarse_candidate in top_coarse:
        angle_min = float(coarse_candidate["tilt_deg"]) - fine_window_angle_deg
        angle_max = float(coarse_candidate["tilt_deg"]) + fine_window_angle_deg
        x_ref_min = float(coarse_candidate["x_ref"]) - fine_window_x_px
        x_ref_max = float(coarse_candidate["x_ref"]) + fine_window_x_px

        fine_angles = np.arange(angle_min, angle_max + 0.5 * fine_angle_step_deg, fine_angle_step_deg)
        fine_x_values = np.arange(int(round(x_ref_min)), int(round(x_ref_max)) + 1, max(1, fine_x_step_px))

        fine_candidates.extend(
            make_candidate_grid(
                angle_values=fine_angles,
                x_ref_values=fine_x_values,
                y_ref=y_ref,
                lines=lines,
                roi_profile=roi_profile,
                total_available_length_px=total_available_length_px,
                band_half_width_px=coarse_band_half_width_px,
                max_angle_error_deg=coarse_max_angle_error_deg,
                allow_adjustment=False,
            )
        )

    fine_candidates = deduplicate_candidates(
        fine_candidates or coarse_candidates,
        roi_profile,
        max_candidates=max(
            fine_candidate_pool_limit,
            int(cfg("candidate_deduplication", "max_saved_candidates", default=8)),
        ),
    )
    best_hypothesis = fine_candidates[0] if fine_candidates else None

    if best_hypothesis is None:
        return {
            "coarse_candidates": coarse_candidates,
            "fine_candidates": fine_candidates,
            "best_hypothesis": None,
            "best_candidate": None,
        }

    refined_support = select_support_fragments(
        lines=lines,
        axis=best_hypothesis,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=use_support_adjustment,
    )
    fitted_axis = fit_axis_from_support(refined_support, y_ref=y_ref)

    base_axis = best_hypothesis if fitted_axis is None else fitted_axis
    base_support = select_support_fragments(
        lines=lines,
        axis=base_axis,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=use_support_adjustment,
    )
    extended_support = extend_support_upward(
        selected_support=base_support,
        lines=lines,
        axis=base_axis,
        roi_profile=roi_profile,
    )

    refit_axis = fit_axis_from_support(extended_support, y_ref=y_ref)
    final_axis = base_axis if refit_axis is None else refit_axis
    reselection_support = select_support_fragments(
        lines=lines,
        axis=final_axis,
        band_half_width_px=final_band_half_width_px,
        max_angle_error_deg=final_max_angle_error_deg,
        allow_adjustment=use_support_adjustment,
    )
    final_support = merge_support_items(extended_support, reselection_support)
    best_candidate = summarize_candidate_from_support(
        axis=final_axis,
        selected_support=final_support,
        roi_profile=roi_profile,
        total_available_length_px=total_available_length_px,
    )
    if fitted_axis is None:
        best_candidate["search_stage"] = "fine_hypothesis_extended"
    elif refit_axis is None:
        best_candidate["search_stage"] = "final_fit_extended_support"
    else:
        best_candidate["search_stage"] = "final_fit_extended_refit"

    best_candidate["hypothesis_x_ref"] = float(best_hypothesis["x_ref"])
    best_candidate["hypothesis_tilt_deg"] = float(best_hypothesis["tilt_deg"])
    best_candidate["hypothesis_score"] = float(best_hypothesis["score"])

    return {
        "coarse_candidates": coarse_candidates,
        "fine_candidates": fine_candidates,
        "best_hypothesis": best_hypothesis,
        "best_candidate": best_candidate,
    }


def load_base_edge_image(data: dict) -> tuple[np.ndarray, str | None]:
    source_path = resolve_project_path(data.get("source_file"))
    if source_path is not None and source_path.exists():
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image, str(source_path.relative_to(PROJECT_ROOT))

    image_name = data.get("image_name", "")
    for path in [
        WORKING_PNG_DIR / image_name,
        PROCESSED_DIR / "03_edges" / "cleaned" / image_name,
    ]:
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image, str(path.relative_to(PROJECT_ROOT))

    height = int(data.get("height", 4032))
    width = int(data.get("width", 3024))
    return np.zeros((height, width), dtype=np.uint8), None


def load_step05_overlay(image_name: str) -> np.ndarray | None:
    overlay_path = get_step_dirs()["input_overlay_dir"] / image_name
    if not overlay_path.exists():
        return None
    return cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)


def draw_fragment(image: np.ndarray, line: dict, color: tuple[int, int, int], thickness: int) -> None:
    p1 = (int(round(float(line["x1"]))), int(round(float(line["y1"]))))
    p2 = (int(round(float(line["x2"]))), int(round(float(line["y2"]))))
    cv2.line(image, p1, p2, color, int(thickness), cv2.LINE_AA)


def draw_axis(image: np.ndarray, axis: dict[str, float], roi_profile: dict, color: tuple[int, int, int], thickness: int) -> None:
    y_start = int(roi_profile["trimmed_y_min"])
    y_end = int(roi_profile["trimmed_y_max"])
    p1 = (int(round(line_x_at_y(axis, y_start))), y_start)
    p2 = (int(round(line_x_at_y(axis, y_end))), y_end)
    cv2.line(image, p1, p2, color, int(thickness), cv2.LINE_AA)


def draw_overlay(
    base_edge_image: np.ndarray,
    filtered_lines: list[dict],
    best_candidate: dict | None,
    fine_candidates: list[dict],
    roi_profile: dict,
    image_name: str,
) -> np.ndarray:
    overlay = to_bgr(base_edge_image)
    alpha = float(cfg("drawing", "background_alpha", default=0.78))
    overlay = cv2.addWeighted(overlay, alpha, np.zeros_like(overlay), 1.0 - alpha, 0)

    for line in filtered_lines:
        draw_fragment(
            overlay,
            line,
            COLOR_ALL_FRAGMENTS,
            int(cfg("drawing", "all_fragment_thickness", default=2)),
        )

    if bool(cfg("drawing", "show_candidate_lines", default=True)):
        for candidate in fine_candidates[: int(cfg("drawing", "candidate_count_to_draw", default=3))]:
            draw_axis(
                overlay,
                candidate,
                roi_profile,
                COLOR_CANDIDATE,
                int(cfg("drawing", "candidate_thickness", default=2)),
            )

    if best_candidate is not None:
        for item in best_candidate.get("selected_support", []):
            draw_fragment(
                overlay,
                item.get("effective_line", item["line"]),
                COLOR_SELECTED_FRAGMENTS,
                int(cfg("drawing", "selected_fragment_thickness", default=3)),
            )
        draw_axis(
            overlay,
            best_candidate,
            roi_profile,
            COLOR_FINAL_AXIS,
            int(cfg("drawing", "final_line_thickness", default=4)),
        )

    put_text(overlay, image_name, 26, 34, COLOR_TEXT, scale=0.82)
    put_text(overlay, f"filtered fragments={len(filtered_lines)}", 26, 62)
    if best_candidate is not None:
        put_text(
            overlay,
            (
                f"selected={best_candidate['selected_fragment_count']} "
                f"bins={best_candidate['supported_bin_count']}/{best_candidate['bin_count']} "
                f"score={best_candidate['score']:.3f}"
            ),
            26,
            90,
        )
        put_text(
            overlay,
            (
                f"tilt={best_candidate['tilt_deg']:.2f}deg "
                f"sym={best_candidate['symmetry_score']:.3f} "
                f"end={best_candidate['endpoint_anchor_score']:.3f}"
            ),
            26,
            118,
        )
        put_text(
            overlay,
            (
                f"gap={best_candidate['gap_penalty']:.3f} "
                f"ealign={best_candidate['top_endpoint_alignment_score']:.3f}/{best_candidate['bottom_endpoint_alignment_score']:.3f} "
                f"cov={best_candidate['top_endpoint_coverage']:.2f}/{best_candidate['bottom_endpoint_coverage']:.2f}"
            ),
            26,
            146,
        )
        put_text(
            overlay,
            (
                f"adj={best_candidate['adjusted_fragment_count']} "
                f"shift={best_candidate['length_weighted_mean_abs_support_shift_px']:.1f}px "
                f"dtilt={best_candidate['mean_abs_support_tilt_delta_deg']:.2f}deg"
            ),
            26,
            174,
        )

    if bool(cfg("drawing", "label_candidates", default=True)):
        for index, candidate in enumerate(fine_candidates[: int(cfg("drawing", "candidate_count_to_draw", default=3))], start=1):
            label_y = int(roi_profile["trimmed_y_min"]) + 22 + (index - 1) * 18
            label_x = int(round(line_x_at_y(candidate, label_y))) + 8
            put_text(overlay, f"C{index}:{candidate['score']:.2f}", label_x, label_y, COLOR_CANDIDATE, scale=0.54)

    return overlay


def create_comparison(step05_overlay: np.ndarray | None, step07_overlay: np.ndarray) -> np.ndarray:
    left_image = step05_overlay if step05_overlay is not None else step07_overlay
    left_image = to_bgr(left_image)
    right_image = to_bgr(step07_overlay)

    height, width = left_image.shape[:2]
    max_width = 1300
    if width > max_width:
        scale = max_width / max(1, width)
        size = (int(width * scale), int(height * scale))
        left_image = cv2.resize(left_image, size, interpolation=cv2.INTER_AREA)
        right_image = cv2.resize(right_image, size, interpolation=cv2.INTER_AREA)

    separator = np.full((left_image.shape[0], 10, 3), 255, dtype=np.uint8)
    return np.hstack([left_image, separator, right_image])


def sanitize_candidate(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None

    return {
        "x_ref": float(candidate["x_ref"]),
        "y_ref": float(candidate["y_ref"]),
        "a": float(candidate["a"]),
        "b": float(candidate["b"]),
        "tilt_deg": float(candidate["tilt_deg"]),
        "score": float(candidate["score"]),
        "selected_fragment_count": int(candidate["selected_fragment_count"]),
        "selected_fragment_line_indices": [int(value) for value in candidate["selected_fragment_line_indices"]],
        "selected_total_length_px": float(candidate["selected_total_length_px"]),
        "selected_total_support_strength": float(candidate["selected_total_support_strength"]),
        "fragment_support_score": float(candidate["fragment_support_score"]),
        "vertical_coverage_score": float(candidate["vertical_coverage_score"]),
        "supported_bin_count": int(candidate["supported_bin_count"]),
        "bin_count": int(candidate["bin_count"]),
        "gap_penalty": float(candidate["gap_penalty"]),
        "largest_gap_px": float(candidate["largest_gap_px"]),
        "support_y_min": float(candidate["support_y_min"]),
        "support_y_max": float(candidate["support_y_max"]),
        "support_span_px": float(candidate["support_span_px"]),
        "endpoint_band_px": float(candidate["endpoint_band_px"]),
        "top_endpoint_coverage": float(candidate["top_endpoint_coverage"]),
        "bottom_endpoint_coverage": float(candidate["bottom_endpoint_coverage"]),
        "top_endpoint_alignment_score": float(candidate["top_endpoint_alignment_score"]),
        "bottom_endpoint_alignment_score": float(candidate["bottom_endpoint_alignment_score"]),
        "endpoint_anchor_score": float(candidate["endpoint_anchor_score"]),
        "outside_mask_penalty": float(candidate["outside_mask_penalty"]),
        "symmetry_score": float(candidate["symmetry_score"]),
        "roi_center_score": float(candidate["roi_center_score"]),
        "rows_inside_mask_count": int(candidate["rows_inside_mask_count"]),
        "sampled_row_count": int(candidate["sampled_row_count"]),
        "adjusted_fragment_count": int(candidate["adjusted_fragment_count"]),
        "adjusted_fragment_ratio": float(candidate["adjusted_fragment_ratio"]),
        "mean_abs_support_shift_px": float(candidate["mean_abs_support_shift_px"]),
        "length_weighted_mean_abs_support_shift_px": float(candidate["length_weighted_mean_abs_support_shift_px"]),
        "max_abs_support_shift_px": float(candidate["max_abs_support_shift_px"]),
        "mean_abs_support_tilt_delta_deg": float(candidate["mean_abs_support_tilt_delta_deg"]),
        "max_abs_support_tilt_delta_deg": float(candidate["max_abs_support_tilt_delta_deg"]),
        "support_adjustment_penalty": float(candidate["support_adjustment_penalty"]),
        "search_stage": candidate.get("search_stage"),
        "hypothesis_x_ref": candidate.get("hypothesis_x_ref"),
        "hypothesis_tilt_deg": candidate.get("hypothesis_tilt_deg"),
        "hypothesis_score": candidate.get("hypothesis_score"),
        "selected_support": [
            {
                "line_index": int(item["line"]["line_index"]),
                "length": float(item["line"]["length"]),
                "axis_distance_px": float(item["axis_distance_px"]),
                "angle_error_deg": float(item["angle_error_deg"]),
                "support_strength": float(item["support_strength"]),
                "effective_tilt_deg": float(item.get("effective_line", item["line"])["signed_tilt_deg"]),
                "is_adjusted": bool(item.get("adjustment", {}).get("is_adjusted", False)),
                "midpoint_shift_px": float(item.get("adjustment", {}).get("midpoint_shift_px", 0.0)),
                "mean_abs_shift_px": float(item.get("adjustment", {}).get("mean_abs_shift_px", 0.0)),
                "max_abs_shift_px": float(item.get("adjustment", {}).get("max_abs_shift_px", 0.0)),
                "tilt_delta_deg": float(item.get("adjustment", {}).get("tilt_delta_deg", 0.0)),
            }
            for item in candidate.get("selected_support", [])
        ],
    }


def build_analysis(json_path: Path) -> dict:
    data = load_json(json_path)
    image_name = data.get("image_name", json_path.stem + ".png")
    width = int(data.get("width", 0))
    height = int(data.get("height", 0)) or 4032

    raw_lines = data.get("valid_lines", [])
    lines = [normalize_line(line, index) for index, line in enumerate(raw_lines, start=1)]
    filtered_lines, rejected_lines = filter_fragments(lines)

    roi_mask_path = resolve_project_path(data.get("roi_mask_file"))
    roi_mask = load_roi_mask(roi_mask_path)
    roi_profile = build_row_profile(roi_mask)
    if roi_profile is None:
        raise RuntimeError(f"Could not build ROI profile for {image_name}")

    search_result = search_best_candidate(filtered_lines, roi_profile)
    best_candidate = search_result["best_candidate"]
    fine_candidates = search_result["fine_candidates"]

    base_edge_image, base_edge_path = load_base_edge_image(data)
    step05_overlay = load_step05_overlay(image_name)
    overlay = draw_overlay(
        base_edge_image=base_edge_image,
        filtered_lines=filtered_lines,
        best_candidate=best_candidate,
        fine_candidates=fine_candidates,
        roi_profile=roi_profile,
        image_name=image_name,
    )
    comparison = create_comparison(step05_overlay, overlay)

    metadata = {
        "image_name": image_name,
        "processing_step": "06_search_central_ruler",
        "source_step": data.get("processing_step", "05_valid_hough_lines_in_roi"),
        "width": width,
        "height": height,
        "input_json_file": str(json_path.relative_to(PROJECT_ROOT)),
        "base_edge_file": base_edge_path,
        "source_file": data.get("source_file"),
        "roi_mask_file": data.get("roi_mask_file"),
        "resolved_input_dir": str(get_step_dirs()["input_dir"].relative_to(PROJECT_ROOT)),
        "input_line_count": len(lines),
        "filtered_line_count": len(filtered_lines),
        "rejected_fragment_count": len(rejected_lines),
        "coarse_candidate_count": len(search_result["coarse_candidates"]),
        "fine_candidate_count": len(fine_candidates),
        "best_candidate": sanitize_candidate(best_candidate),
        "top_candidates": [
            sanitize_candidate(candidate)
            for candidate in fine_candidates[: int(cfg("candidate_deduplication", "max_saved_candidates", default=8))]
        ],
        "roi_profile": {
            "y_min": int(roi_profile["y_min"]),
            "y_max": int(roi_profile["y_max"]),
            "trimmed_y_min": int(roi_profile["trimmed_y_min"]),
            "trimmed_y_max": int(roi_profile["trimmed_y_max"]),
            "y_ref": float(roi_profile["y_ref"]),
            "reference_width_px": float(roi_profile["reference_width_px"]),
            "median_center_x": float(roi_profile["median_center_x"]),
            "center_fit": {
                "a": float(roi_profile["center_fit"]["a"]),
                "b": float(roi_profile["center_fit"]["b"]),
                "tilt_deg": float(roi_profile["center_fit"]["tilt_deg"]),
            },
        },
        "parameters": STEP_CONFIG,
    }

    return {
        "image_name": image_name,
        "metadata": metadata,
        "overlay": overlay,
        "comparison": comparison,
        "best_candidate": best_candidate,
        "fine_candidates": fine_candidates,
        "filtered_lines": filtered_lines,
    }


def process_json_file(json_path: Path) -> dict:
    analysis = build_analysis(json_path)
    dirs = get_step_dirs()
    image_name = analysis["image_name"]

    overlay_path = dirs["output_overlay_dir"] / image_name
    comparison_path = dirs["output_comparison_dir"] / image_name
    metadata_path = dirs["output_metadata_dir"] / f"{Path(image_name).stem}_central_ruler.json"

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(overlay_path), analysis["overlay"]):
        raise RuntimeError(f"Could not save overlay: {overlay_path}")
    if not cv2.imwrite(str(comparison_path), analysis["comparison"]):
        raise RuntimeError(f"Could not save comparison: {comparison_path}")

    metadata = deepcopy(analysis["metadata"])
    metadata["output_overlay_file"] = str(overlay_path.relative_to(PROJECT_ROOT))
    metadata["output_comparison_file"] = str(comparison_path.relative_to(PROJECT_ROOT))
    save_json(metadata_path, metadata)

    best_candidate = analysis["best_candidate"]
    return {
        "image_name": image_name,
        "filtered_line_count": len(analysis["filtered_lines"]),
        "candidate_count": len(analysis["fine_candidates"]),
        "selected_fragment_count": best_candidate["selected_fragment_count"] if best_candidate else 0,
        "best_score": best_candidate["score"] if best_candidate else None,
        "best_tilt_deg": best_candidate["tilt_deg"] if best_candidate else None,
        "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        "comparison_path": str(comparison_path.relative_to(PROJECT_ROOT)),
    }


def collect_json_files(image_filter: str | None = None, limit: int | None = None) -> list[Path]:
    input_json_dir = get_step_dirs()["input_json_dir"]
    if not input_json_dir.exists():
        raise FileNotFoundError(f"Input JSON dir does not exist: {input_json_dir}")

    files = sorted(input_json_dir.glob("*.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [path for path in files if path.stem.lower() == wanted or wanted in path.stem.lower()]
    if limit is not None:
        files = files[:limit]
    return files


def show_image(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return

    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / max(1, height)
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow(path.name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 06: search a central ruler hypothesis across ROI and fit the final center axis.")
    parser.add_argument("--image", type=str, default=None, help="Optional image name filter, for example IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    global STEP_CONFIG

    args = parse_args()
    STEP_CONFIG = apply_preset(STEP_CONFIG, args.preset)
    ensure_dirs(cleanup=bool(STEP_CONFIG.get("cleanup_output_on_start", True)) and args.image is None)

    json_files = collect_json_files(args.image, args.limit)
    if not json_files:
        print("No JSON files found.")
        print(f"Input dir: {get_step_dirs()['input_json_dir']}")
        return

    print(f"Step 06 input dir: {get_step_dirs()['input_json_dir']}")
    print(f"Step 06 output dir: {get_step_dirs()['output_dir']}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found JSON files: {len(json_files)}")

    summary = []
    for json_path in json_files:
        print(f"\nProcessing: {json_path.name}")
        try:
            result = process_json_file(json_path)
            summary.append(result)
            score_text = "none" if result["best_score"] is None else f"{result['best_score']:.3f}"
            tilt_text = "none" if result["best_tilt_deg"] is None else f"{result['best_tilt_deg']:.2f}"
            print(
                f"  filtered={result['filtered_line_count']} candidates={result['candidate_count']} "
                f"selected={result['selected_fragment_count']} score={score_text} tilt={tilt_text}"
            )
            print(f"  overlay: {result['overlay_path']}")
            print(f"  metadata: {result['metadata_path']}")
            if args.show:
                show_image(PROJECT_ROOT / result["comparison_path"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = get_step_dirs()["output_dir"] / "step_06_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
