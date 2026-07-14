from pathlib import Path
import sys
import json
import math
import argparse
import os
import shutil
import traceback
import warnings
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "pipeline"))

from config_loader import load_config


CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG.get("display", {})
STEP_05_CONFIG = CONFIG.get("step_05_valid_hough_lines_in_roi", {})

WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]

STEP_CONFIG_RAW = CONFIG.get("step_06_fit_central_axis_from_fragments")
if STEP_CONFIG_RAW is None:
    STEP_CONFIG_RAW = CONFIG.get("step_06_axis_fragment_chains", {})


DEFAULT_STEP_CONFIG = {
    "enabled": True,
    "inherit_step_05_output": True,
    "input_subdir": "05_valid_hough_lines_in_roi",
    "input_json_subdir": "valid_lines_json",
    "input_overlay_subdir": "valid_lines_overlay",
    "output_subdir": "06_fit_central_axis_from_fragments",
    "cleanup_output_on_start": True,
    "fragment_filter": {
        "min_length_px": 70,
        "max_vertical_deviation_deg": 14.0,
        "min_mask_support_ratio": 0.94,
        "min_points_inside_mask": 60,
    },
    "fragment_seed_selection": {
        "max_ranked_lines": 140,
        "seed_pool_size": 28,
        "min_consensus_score": 180.0,
        "min_neighbor_overlap_px": 28,
        "max_neighbor_axis_distance_px": 30,
        "max_neighbor_vertical_gap_px": 170,
        "max_neighbor_tilt_difference_deg": 8.0,
    },
    "candidate_generation": {
        "sample_step_px": 14,
        "max_seed_tilt_deg": 18.0,
        "seed_offset_px_values": [-36, -18, 0, 18, 36],
        "seed_tilt_delta_deg_values": [-4.0, 0.0, 4.0],
        "use_pair_seeds": True,
        "max_pair_seed_lines": 18,
        "max_pair_neighbors": 6,
        "max_pair_vertical_gap_px": 320,
        "max_pair_x_offset_px": 48,
        "use_midpoint_pair_seeds": True,
        "midpoint_pair_min_axis_distance_px": 36,
        "midpoint_pair_max_axis_distance_px": 150,
        "midpoint_pair_min_overlap_px": 70,
        "midpoint_pair_max_vertical_gap_px": 90,
    },
    "axis_fit": {
        "iterations": 5,
        "huber_delta_px": 12.0,
        "use_opencv_fitline": True,
        "fitline_dist_type": "DIST_WELSCH",
        "fitline_param": 0.0,
        "fitline_reps": 0.01,
        "fitline_aeps": 0.01,
        "enforce_seed_locality": True,
        "max_support_distance_to_seed_px": 42,
        "max_support_distance_px": 38,
        "max_support_distance_after_refit_px": 30,
        "max_tilt_difference_deg": 9.0,
        "max_candidate_tilt_deg": 7.0,
    },
    "candidate_acceptance": {
        "min_support_fragments": 2,
        "min_support_total_length_px": 180,
        "min_observed_y_coverage_px": 220,
        "max_mean_residual_px": 18.0,
        "max_median_residual_px": 14.0,
        "max_length_weighted_mean_abs_shift_px": 22.0,
        "max_mean_abs_tilt_delta_deg": 6.8,
        "max_candidates_per_image": 14,
        "always_keep_best_fallback": True,
    },
    "scoring": {
        "support_length_weight": 0.24,
        "observed_coverage_weight": 0.26,
        "residual_weight": 0.14,
        "adjustment_weight": 0.30,
        "balance_weight": 0.06,
        "adjustment_reference_px": 22.0,
        "tilt_adjustment_reference_deg": 6.0,
    },
    "deduplication": {
        "enabled": True,
        "max_mean_axis_distance_px": 22,
        "min_shared_support_ratio": 0.5,
    },
    "drawing": {
        "background_alpha_fallback": 0.72,
        "draw_filtered_fragments_when_overlay_missing": True,
        "project_axis_through_roi": True,
        "axis_thickness": 4,
        "secondary_axis_thickness": 3,
        "font_scale": 0.66,
        "label_candidates": True,
    },
    "test_presets": [
        {"name": "default", "override": {}},
        {
            "name": "fragment_tight",
            "override": {
                "fragment_seed_selection": {
                    "seed_pool_size": 24,
                    "max_neighbor_axis_distance_px": 26,
                    "max_neighbor_vertical_gap_px": 145,
                    "max_neighbor_tilt_difference_deg": 7.0,
                },
                "candidate_generation": {
                    "seed_offset_px_values": [-24, 0, 24],
                    "seed_tilt_delta_deg_values": [-3.0, 0.0, 3.0],
                    "midpoint_pair_max_axis_distance_px": 130,
                },
                "axis_fit": {
                    "max_support_distance_to_seed_px": 34,
                    "max_support_distance_px": 32,
                    "max_support_distance_after_refit_px": 24,
                    "max_candidate_tilt_deg": 5.6,
                },
                "candidate_acceptance": {
                    "max_mean_residual_px": 16.0,
                    "max_median_residual_px": 12.0,
                    "max_length_weighted_mean_abs_shift_px": 19.0,
                    "max_mean_abs_tilt_delta_deg": 5.8,
                },
            },
        },
        {
            "name": "fragment_wide_search",
            "override": {
                "fragment_filter": {
                    "min_length_px": 55,
                    "max_vertical_deviation_deg": 17.0,
                    "min_mask_support_ratio": 0.9,
                    "min_points_inside_mask": 45,
                },
                "fragment_seed_selection": {
                    "seed_pool_size": 34,
                    "max_neighbor_axis_distance_px": 36,
                    "max_neighbor_vertical_gap_px": 240,
                    "max_neighbor_tilt_difference_deg": 10.0,
                },
                "candidate_generation": {
                    "seed_offset_px_values": [-54, -27, 0, 27, 54],
                    "seed_tilt_delta_deg_values": [-5.5, 0.0, 5.5],
                    "max_pair_seed_lines": 22,
                    "max_pair_neighbors": 7,
                    "max_pair_vertical_gap_px": 380,
                    "max_pair_x_offset_px": 60,
                    "midpoint_pair_max_axis_distance_px": 180,
                    "midpoint_pair_max_vertical_gap_px": 120,
                },
                "axis_fit": {
                    "max_support_distance_to_seed_px": 52,
                    "max_support_distance_px": 44,
                    "max_support_distance_after_refit_px": 34,
                    "max_tilt_difference_deg": 11.0,
                    "max_candidate_tilt_deg": 8.5,
                },
                "candidate_acceptance": {
                    "min_support_total_length_px": 150,
                    "min_observed_y_coverage_px": 180,
                    "max_mean_residual_px": 22.0,
                    "max_median_residual_px": 16.0,
                    "max_length_weighted_mean_abs_shift_px": 28.0,
                    "max_mean_abs_tilt_delta_deg": 8.5,
                },
            },
        },
        {
            "name": "fragment_tilted",
            "override": {
                "candidate_generation": {
                    "seed_offset_px_values": [-30, -15, 0, 15, 30],
                    "seed_tilt_delta_deg_values": [-7.0, -3.5, 0.0, 3.5, 7.0],
                    "midpoint_pair_max_axis_distance_px": 170,
                },
                "axis_fit": {
                    "max_support_distance_to_seed_px": 48,
                    "max_support_distance_px": 40,
                    "max_support_distance_after_refit_px": 31,
                    "max_tilt_difference_deg": 12.0,
                    "max_candidate_tilt_deg": 10.0,
                },
                "candidate_acceptance": {
                    "max_mean_abs_tilt_delta_deg": 9.0,
                    "max_candidates_per_image": 16,
                },
            },
        },
    ],
}


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def apply_preset(config, preset_name):
    if not preset_name:
        return config
    for preset in config.get("test_presets", []):
        if preset.get("name") == preset_name:
            return deep_merge(config, preset.get("override", {}))
    available = [preset.get("name") for preset in config.get("test_presets", [])]
    raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")


STEP_CONFIG = deep_merge(DEFAULT_STEP_CONFIG, STEP_CONFIG_RAW)

step_05_output_subdir = str(STEP_05_CONFIG.get("output_subdir", "05_valid_hough_lines_in_roi"))
if bool(STEP_CONFIG.get("inherit_step_05_output", True)):
    INPUT_SUBDIR = step_05_output_subdir
else:
    INPUT_SUBDIR = str(STEP_CONFIG.get("input_subdir", step_05_output_subdir))
OUTPUT_SUBDIR = STEP_CONFIG.get("output_subdir", "06_fit_central_axis_from_fragments")
INPUT_DIR = PROCESSED_DIR / INPUT_SUBDIR
INPUT_JSON_DIR = INPUT_DIR / STEP_CONFIG.get("input_json_subdir", "valid_lines_json")
INPUT_OVERLAY_DIR = INPUT_DIR / STEP_CONFIG.get("input_overlay_subdir", "valid_lines_overlay")
OUTPUT_DIR = PROCESSED_DIR / OUTPUT_SUBDIR
OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "overlay"
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
OUTPUT_COMPARISON_DIR = OUTPUT_DIR / "comparison"


COLOR_VALID = (0, 255, 0)
COLOR_TEXT = (240, 240, 240)
COLOR_CANDIDATE_PALETTE = [
    (255, 0, 0),
    (255, 70, 0),
    (255, 135, 0),
    (255, 180, 20),
    (255, 220, 60),
]


def cfg(*keys, default=None):
    current = STEP_CONFIG
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def resolve_project_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configure_runtime(step_config):
    global STEP_CONFIG
    global INPUT_SUBDIR, OUTPUT_SUBDIR
    global INPUT_DIR, INPUT_JSON_DIR, INPUT_OVERLAY_DIR
    global OUTPUT_DIR, OUTPUT_OVERLAY_DIR, OUTPUT_METADATA_DIR, OUTPUT_COMPARISON_DIR

    STEP_CONFIG = deepcopy(step_config)
    step_05_output_subdir = str(STEP_05_CONFIG.get("output_subdir", "05_valid_hough_lines_in_roi"))
    if bool(STEP_CONFIG.get("inherit_step_05_output", True)):
        INPUT_SUBDIR = step_05_output_subdir
    else:
        INPUT_SUBDIR = str(STEP_CONFIG.get("input_subdir", step_05_output_subdir))
    OUTPUT_SUBDIR = STEP_CONFIG.get("output_subdir", "06_fit_central_axis_from_fragments")
    INPUT_DIR = PROCESSED_DIR / INPUT_SUBDIR
    INPUT_JSON_DIR = INPUT_DIR / STEP_CONFIG.get("input_json_subdir", "valid_lines_json")
    INPUT_OVERLAY_DIR = INPUT_DIR / STEP_CONFIG.get("input_overlay_subdir", "valid_lines_overlay")
    OUTPUT_DIR = PROCESSED_DIR / OUTPUT_SUBDIR
    OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "overlay"
    OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
    OUTPUT_COMPARISON_DIR = OUTPUT_DIR / "comparison"


def ensure_dirs(cleanup=False):
    if cleanup and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def safe_linear_polyfit(y, x, weights=None):
    if len(y) < 2:
        return None

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(y) <= 1e-6:
        return 0.0, float(np.median(x))

    fit_weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.RankWarning)
            return np.polyfit(y, x, 1, w=fit_weights)
    except (np.exceptions.RankWarning, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 0.0, float(np.median(x))


def opencv_fit_line(points):
    if len(points) < 2 or not bool(cfg("axis_fit", "use_opencv_fitline", default=True)):
        return None
    if not hasattr(cv2, "fitLine"):
        return None

    xy_points = np.asarray(
        [[float(point[1]), float(point[0])] for point in points],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    dist_type_name = str(cfg("axis_fit", "fitline_dist_type", default="DIST_WELSCH"))
    dist_type = getattr(cv2, dist_type_name, getattr(cv2, "DIST_L2", 2))
    param = float(cfg("axis_fit", "fitline_param", default=0.0))
    reps = float(cfg("axis_fit", "fitline_reps", default=0.01))
    aeps = float(cfg("axis_fit", "fitline_aeps", default=0.01))

    try:
        fit = cv2.fitLine(xy_points, dist_type, param, reps, aeps)
    except Exception:
        return None

    vx, vy, x0, y0 = [float(np.asarray(value).reshape(-1)[0]) for value in fit]
    if abs(vy) <= 1e-6:
        return None

    a = vx / vy
    b = x0 - a * y0
    y = np.asarray([point[0] for point in points], dtype=np.float64)
    x = np.asarray([point[1] for point in points], dtype=np.float64)
    residuals = np.abs((a * y + b) - x)

    return {
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(math.degrees(math.atan(a))),
        "mean_residual_px": float(np.mean(residuals)),
        "median_residual_px": float(np.median(residuals)),
        "max_residual_px": float(np.max(residuals)),
    }


def normalize_line(raw_line, fallback_index):
    x1 = float(raw_line["x1"])
    y1 = float(raw_line["y1"])
    x2 = float(raw_line["x2"])
    y2 = float(raw_line["y2"])
    dx = x2 - x1
    dy = y2 - y1
    length = float(raw_line.get("length", math.hypot(dx, dy)))

    a = dx / dy if abs(dy) > 1e-6 else 999.0
    b = x1 - a * y1 if abs(dy) > 1e-6 else (x1 + x2) / 2.0
    signed_tilt_deg = math.degrees(math.atan(a)) if abs(a) < 999 else 90.0

    return {
        "line_index": int(raw_line.get("line_index", raw_line.get("id", fallback_index))),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length": length,
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


def line_x_at_y(model, y):
    return float(model["a"] * float(y) + model["b"])


def sample_line_points(line, step_px=None):
    step_px = step_px or cfg("candidate_generation", "sample_step_px", default=14)
    length = max(1.0, float(line["length"]))
    count = max(2, int(math.ceil(length / max(1.0, step_px))) + 1)
    ts = np.linspace(0.0, 1.0, count)
    points = []
    for t in ts:
        points.append(
            (
                float(line["y1"] + t * (line["y2"] - line["y1"])),
                float(line["x1"] + t * (line["x2"] - line["x1"])),
                line["line_index"],
            )
        )
    return points


def load_roi_mask(mask_path):
    if mask_path is None or not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def build_roi_profile(mask):
    if mask is None or not cfg("roi_axis_prior", "enabled", default=True):
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None

    all_y_min = int(np.min(ys))
    all_y_max = int(np.max(ys))
    height, _ = mask.shape[:2]
    trim_top = float(cfg("roi_axis_prior", "trim_top_ratio", default=0.08))
    trim_bottom = float(cfg("roi_axis_prior", "trim_bottom_ratio", default=0.08))
    y0 = int(all_y_min + (all_y_max - all_y_min) * trim_top)
    y1 = int(all_y_max - (all_y_max - all_y_min) * trim_bottom)
    sample_count = int(cfg("roi_axis_prior", "sample_count", default=100))
    min_row_width = int(cfg("roi_axis_prior", "min_row_width_px", default=160))

    rows = []
    widths = []
    for y in np.linspace(y0, y1, sample_count):
        yi = int(round(np.clip(y, 0, height - 1)))
        row_xs = np.where(mask[yi, :] > 0)[0]
        if len(row_xs) < 2:
            continue
        left = int(row_xs[0])
        right = int(row_xs[-1])
        width = right - left
        if width < min_row_width:
            continue
        center_x = float((left + right) / 2.0)
        rows.append((float(yi), center_x, float(width)))
        widths.append(width)

    if len(rows) < 8:
        return None

    y = np.asarray([row[0] for row in rows], dtype=np.float64)
    x = np.asarray([row[1] for row in rows], dtype=np.float64)
    fit = safe_linear_polyfit(y, x)
    if fit is None:
        return None

    a, b = fit
    max_tilt_deg = float(cfg("roi_axis_prior", "fit_max_tilt_deg", default=2.5))
    max_a = math.tan(math.radians(max_tilt_deg))
    a = float(np.clip(a, -max_a, max_a))
    b = float(np.median(x - a * y))

    width_reference_quantile = float(cfg("roi_axis_prior", "width_reference_quantile", default=0.3))
    width_reference_px = float(np.quantile(np.asarray(widths, dtype=np.float64), width_reference_quantile))

    half_width_ratio = float(cfg("center_corridor", "half_width_ratio_of_mask", default=0.16))
    corridor_half_width_px = width_reference_px * half_width_ratio
    corridor_half_width_px = float(
        np.clip(
            corridor_half_width_px,
            float(cfg("center_corridor", "min_half_width_px", default=55)),
            float(cfg("center_corridor", "max_half_width_px", default=150)),
        )
    )

    return {
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(math.degrees(math.atan(a))),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "width_reference_px": width_reference_px,
        "corridor_half_width_px": corridor_half_width_px,
    }


def mask_vertical_extent(mask):
    if mask is None:
        return None
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return None
    return int(np.min(ys)), int(np.max(ys))


def filter_fragments(lines):
    accepted = []
    rejected = []
    min_length = float(cfg("fragment_filter", "min_length_px", default=70))
    max_vertical_deviation = float(cfg("fragment_filter", "max_vertical_deviation_deg", default=14.0))
    min_mask_support_ratio = float(cfg("fragment_filter", "min_mask_support_ratio", default=0.94))
    min_points_inside_mask = int(cfg("fragment_filter", "min_points_inside_mask", default=60))

    for line in lines:
        reasons = []
        if not line["is_valid"]:
            reasons.append("not_valid")
        if line["length"] < min_length:
            reasons.append("too_short")
        if abs(line["signed_tilt_deg"]) > max_vertical_deviation:
            reasons.append("too_far_from_vertical")
        if line["mask_support_ratio"] < min_mask_support_ratio:
            reasons.append("low_mask_support")
        if line["points_inside_mask"] and line["points_inside_mask"] < min_points_inside_mask:
            reasons.append("too_few_points_inside_mask")

        if reasons:
            rejected.append({**line, "reject_reasons": reasons})
        else:
            accepted.append(line)

    return accepted, rejected


def fragment_pair_geometry(line_a, line_b):
    overlap_y0 = max(float(line_a["y_min"]), float(line_b["y_min"]))
    overlap_y1 = min(float(line_a["y_max"]), float(line_b["y_max"]))

    if overlap_y1 > overlap_y0:
        probe_y = (overlap_y0 + overlap_y1) / 2.0
        overlap_px = overlap_y1 - overlap_y0
        vertical_gap_px = 0.0
    else:
        probe_y = (float(line_a["y_mid"]) + float(line_b["y_mid"])) / 2.0
        overlap_px = 0.0
        vertical_gap_px = max(
            0.0,
            max(float(line_a["y_min"]) - float(line_b["y_max"]), float(line_b["y_min"]) - float(line_a["y_max"])),
        )

    axis_distance_px = abs(line_x_at_y(line_a, probe_y) - line_x_at_y(line_b, probe_y))
    tilt_difference_deg = abs(float(line_a["signed_tilt_deg"]) - float(line_b["signed_tilt_deg"]))

    return {
        "probe_y": float(probe_y),
        "axis_distance_px": float(axis_distance_px),
        "overlap_px": float(overlap_px),
        "vertical_gap_px": float(vertical_gap_px),
        "tilt_difference_deg": float(tilt_difference_deg),
    }


def annotate_fragment_consensus(lines):
    if not lines:
        return []

    max_ranked_lines = int(cfg("fragment_seed_selection", "max_ranked_lines", default=140))
    min_neighbor_overlap_px = float(cfg("fragment_seed_selection", "min_neighbor_overlap_px", default=28))
    max_neighbor_axis_distance_px = float(cfg("fragment_seed_selection", "max_neighbor_axis_distance_px", default=30))
    max_neighbor_vertical_gap_px = float(cfg("fragment_seed_selection", "max_neighbor_vertical_gap_px", default=170))
    max_neighbor_tilt_difference_deg = float(cfg("fragment_seed_selection", "max_neighbor_tilt_difference_deg", default=8.0))

    comparison_pool = sorted(lines, key=lambda line: line["length"], reverse=True)[:max_ranked_lines]
    annotated = []

    for line in lines:
        neighbor_count = 0
        consensus_score = float(line["length"])
        consensus_support_length_px = 0.0
        best_neighbor_axis_distance_px = None

        for neighbor in comparison_pool:
            if neighbor["line_index"] == line["line_index"]:
                continue

            relation = fragment_pair_geometry(line, neighbor)
            if relation["tilt_difference_deg"] > max_neighbor_tilt_difference_deg:
                continue
            if relation["axis_distance_px"] > max_neighbor_axis_distance_px:
                continue
            if relation["overlap_px"] < min_neighbor_overlap_px and relation["vertical_gap_px"] > max_neighbor_vertical_gap_px:
                continue

            proximity = 1.0 - min(1.0, relation["axis_distance_px"] / max(1.0, max_neighbor_axis_distance_px))
            continuity = (
                1.0
                if relation["overlap_px"] >= min_neighbor_overlap_px
                else 1.0 - min(1.0, relation["vertical_gap_px"] / max(1.0, max_neighbor_vertical_gap_px))
            )
            contribution = float(neighbor["length"]) * max(0.0, proximity) * max(0.0, continuity)
            if contribution <= 0.0:
                continue

            neighbor_count += 1
            consensus_score += contribution
            consensus_support_length_px += float(neighbor["length"])
            if best_neighbor_axis_distance_px is None or relation["axis_distance_px"] < best_neighbor_axis_distance_px:
                best_neighbor_axis_distance_px = relation["axis_distance_px"]

        annotated.append(
            {
                **line,
                "consensus_neighbor_count": int(neighbor_count),
                "consensus_score": float(consensus_score),
                "consensus_support_length_px": float(consensus_support_length_px),
                "best_neighbor_axis_distance_px": best_neighbor_axis_distance_px,
            }
        )

    annotated.sort(
        key=lambda line: (
            -float(line["consensus_score"]),
            -int(line["consensus_neighbor_count"]),
            -float(line["length"]),
        )
    )
    return annotated


def select_seed_pool(lines, roi_profile=None):
    annotated = annotate_fragment_consensus(lines)
    if not annotated:
        return [], []

    seed_pool_size = int(cfg("fragment_seed_selection", "seed_pool_size", default=28))
    min_consensus_score = float(cfg("fragment_seed_selection", "min_consensus_score", default=180.0))

    strong_pool = [line for line in annotated if float(line["consensus_score"]) >= min_consensus_score]
    if len(strong_pool) < min(seed_pool_size, len(annotated)):
        seed_pool = annotated[:seed_pool_size]
    else:
        seed_pool = strong_pool[:seed_pool_size]

    seed_ids = {line["line_index"] for line in seed_pool}
    remaining_pool = [line for line in annotated if line["line_index"] not in seed_ids]
    return seed_pool, remaining_pool


def weighted_huber_fit(points, base_weights=None, initial=None):
    if len(points) < 2:
        return None

    y = np.asarray([point[0] for point in points], dtype=np.float64)
    x = np.asarray([point[1] for point in points], dtype=np.float64)
    weights = np.ones_like(y) if base_weights is None else np.asarray(base_weights, dtype=np.float64)

    fitline_initial = opencv_fit_line(points)
    if fitline_initial is not None:
        a = float(fitline_initial["a"])
        b = float(fitline_initial["b"])
    elif initial is None:
        fit = safe_linear_polyfit(y, x, weights)
        if fit is None:
            return None
        a, b = fit
    else:
        a = float(initial["a"])
        b = float(initial["b"])

    delta = float(cfg("axis_fit", "huber_delta_px", default=12.0))
    iterations = int(cfg("axis_fit", "iterations", default=5))

    for _ in range(iterations):
        residuals = np.abs((a * y + b) - x)
        huber = np.ones_like(residuals)
        mask = residuals > delta
        huber[mask] = delta / np.maximum(residuals[mask], 1e-6)
        fit = safe_linear_polyfit(y, x, weights * huber)
        if fit is None:
            return None
        a, b = fit

    residuals = np.abs((a * y + b) - x)
    return {
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(math.degrees(math.atan(a))),
        "mean_residual_px": float(np.mean(residuals)),
        "median_residual_px": float(np.median(residuals)),
        "max_residual_px": float(np.max(residuals)),
    }


def fit_axis_from_lines(lines, initial_axis=None):
    if not lines:
        return None
    points = []
    weights = []
    for line in lines:
        line_weight = 1.0 + min(2.5, float(line["length"]) / 220.0)
        for point in sample_line_points(line):
            points.append(point)
            weights.append(line_weight)
    return weighted_huber_fit(points, base_weights=weights, initial=initial_axis)


def line_distance_to_axis(line, axis):
    distances = [abs(line_x_at_y(axis, y) - x) for y, x, _ in sample_line_points(line)]
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def build_fragment_adjustment(line, axis):
    sampled_points = sample_line_points(line)
    signed_sample_shifts = [float(line_x_at_y(axis, y) - x) for y, x, _ in sampled_points]
    abs_sample_shifts = np.abs(np.asarray(signed_sample_shifts, dtype=np.float64))

    candidate_x_at_midpoint = float(line_x_at_y(axis, line["y_mid"]))
    signed_midpoint_shift = candidate_x_at_midpoint - float(line["x_mid"])
    signed_start_shift = float(line_x_at_y(axis, line["y1"]) - line["x1"])
    signed_end_shift = float(line_x_at_y(axis, line["y2"]) - line["x2"])
    tilt_delta = float(axis["tilt_deg"] - line["signed_tilt_deg"])

    return {
        "line_index": int(line["line_index"]),
        "length_px": float(line["length"]),
        "original_tilt_deg": float(line["signed_tilt_deg"]),
        "candidate_tilt_deg": float(axis["tilt_deg"]),
        "tilt_delta_deg": tilt_delta,
        "abs_tilt_delta_deg": abs(tilt_delta),
        "original_midpoint_x_px": float(line["x_mid"]),
        "original_midpoint_y_px": float(line["y_mid"]),
        "candidate_x_at_midpoint_px": candidate_x_at_midpoint,
        "signed_midpoint_shift_px": float(signed_midpoint_shift),
        "abs_midpoint_shift_px": abs(float(signed_midpoint_shift)),
        "signed_start_shift_px": signed_start_shift,
        "signed_end_shift_px": signed_end_shift,
        "mean_signed_shift_px": float(np.mean(signed_sample_shifts)),
        "mean_abs_shift_px": float(np.mean(abs_sample_shifts)),
        "median_abs_shift_px": float(np.median(abs_sample_shifts)),
        "max_abs_shift_px": float(np.max(abs_sample_shifts)),
        "distance_to_axis_mean_px": float(np.mean(abs_sample_shifts)),
        "distance_to_axis_median_px": float(np.median(abs_sample_shifts)),
        "distance_to_axis_max_px": float(np.max(abs_sample_shifts)),
        "prior_median_distance_px": (
            float(line["prior_median_distance_px"])
            if "prior_median_distance_px" in line
            else None
        ),
    }


def summarize_fragment_adjustments(adjustments):
    if not adjustments:
        return {
            "fragment_count": 0,
            "mean_abs_tilt_delta_deg": None,
            "median_abs_tilt_delta_deg": None,
            "max_abs_tilt_delta_deg": None,
            "mean_abs_midpoint_shift_px": None,
            "median_abs_midpoint_shift_px": None,
            "max_abs_midpoint_shift_px": None,
            "mean_abs_shift_px": None,
            "median_abs_shift_px": None,
            "max_abs_shift_px": None,
            "length_weighted_mean_abs_shift_px": None,
            "mean_signed_shift_px": None,
            "positive_shift_count": 0,
            "negative_shift_count": 0,
            "near_zero_shift_count": 0,
            "signed_shift_balance_score": None,
            "worst_fragment_by_shift_line_index": None,
            "worst_fragment_by_tilt_line_index": None,
        }

    abs_tilt_deltas = np.asarray(
        [adjustment["abs_tilt_delta_deg"] for adjustment in adjustments],
        dtype=np.float64,
    )
    abs_midpoint_shifts = np.asarray(
        [adjustment["abs_midpoint_shift_px"] for adjustment in adjustments],
        dtype=np.float64,
    )
    abs_sample_shifts = np.asarray(
        [adjustment["mean_abs_shift_px"] for adjustment in adjustments],
        dtype=np.float64,
    )
    signed_sample_shifts = np.asarray(
        [adjustment["mean_signed_shift_px"] for adjustment in adjustments],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [max(1.0, adjustment["length_px"]) for adjustment in adjustments],
        dtype=np.float64,
    )
    positive_shift_count = sum(1 for adjustment in adjustments if adjustment["signed_midpoint_shift_px"] > 2.0)
    negative_shift_count = sum(1 for adjustment in adjustments if adjustment["signed_midpoint_shift_px"] < -2.0)
    near_zero_shift_count = len(adjustments) - positive_shift_count - negative_shift_count
    signed_shift_balance_score = 1.0 - (
        abs(positive_shift_count - negative_shift_count) / max(1, positive_shift_count + negative_shift_count)
    ) if (positive_shift_count + negative_shift_count) > 0 else 0.5

    worst_shift = max(adjustments, key=lambda adjustment: adjustment["max_abs_shift_px"])
    worst_tilt = max(adjustments, key=lambda adjustment: adjustment["abs_tilt_delta_deg"])

    return {
        "fragment_count": len(adjustments),
        "mean_abs_tilt_delta_deg": float(np.mean(abs_tilt_deltas)),
        "median_abs_tilt_delta_deg": float(np.median(abs_tilt_deltas)),
        "max_abs_tilt_delta_deg": float(np.max(abs_tilt_deltas)),
        "mean_abs_midpoint_shift_px": float(np.mean(abs_midpoint_shifts)),
        "median_abs_midpoint_shift_px": float(np.median(abs_midpoint_shifts)),
        "max_abs_midpoint_shift_px": float(np.max(abs_midpoint_shifts)),
        "mean_abs_shift_px": float(np.mean(abs_sample_shifts)),
        "median_abs_shift_px": float(np.median(abs_sample_shifts)),
        "max_abs_shift_px": float(np.max([adjustment["max_abs_shift_px"] for adjustment in adjustments])),
        "length_weighted_mean_abs_shift_px": float(np.average(abs_sample_shifts, weights=lengths)),
        "mean_signed_shift_px": float(np.mean(signed_sample_shifts)),
        "positive_shift_count": int(positive_shift_count),
        "negative_shift_count": int(negative_shift_count),
        "near_zero_shift_count": int(near_zero_shift_count),
        "signed_shift_balance_score": float(signed_shift_balance_score),
        "worst_fragment_by_shift_line_index": int(worst_shift["line_index"]),
        "worst_fragment_by_tilt_line_index": int(worst_tilt["line_index"]),
    }


def collect_support_lines(lines, axis, after_refit=False, seed_axis=None):
    threshold = float(
        cfg(
            "axis_fit",
            "max_support_distance_after_refit_px" if after_refit else "max_support_distance_px",
            default=26 if after_refit else 34,
        )
    )
    max_tilt_difference = float(cfg("axis_fit", "max_tilt_difference_deg", default=8.0))
    enforce_seed_locality = bool(cfg("axis_fit", "enforce_seed_locality", default=True)) and seed_axis is not None
    max_seed_distance = float(cfg("axis_fit", "max_support_distance_to_seed_px", default=42))

    support = []
    rejected = []
    for line in lines:
        distances = line_distance_to_axis(line, axis)
        tilt_difference = abs(line["signed_tilt_deg"] - axis["tilt_deg"])

        if tilt_difference > max_tilt_difference:
            rejected.append({**line, "reject_reason": "tilt_difference"})
            continue
        if distances["median"] > threshold:
            rejected.append({**line, "reject_reason": "median_distance"})
            continue
        if distances["max"] > threshold * 1.9:
            rejected.append({**line, "reject_reason": "max_distance"})
            continue
        if enforce_seed_locality:
            seed_distances = line_distance_to_axis(line, seed_axis)
            if seed_distances["median"] > max_seed_distance:
                rejected.append({**line, "reject_reason": "seed_distance"})
                continue

        support.append(
            {
                **line,
                "distance_to_axis_mean_px": distances["mean"],
                "distance_to_axis_median_px": distances["median"],
                "distance_to_axis_max_px": distances["max"],
            }
        )

    return support, rejected


def merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def observed_intervals_for_support(lines):
    return merge_intervals([(line["y_min"], line["y_max"]) for line in lines])


def score_candidate(candidate, image_height, roi_profile):
    weights = cfg("scoring", default={})
    support_length_weight = float(weights.get("support_length_weight", 0.24))
    observed_coverage_weight = float(weights.get("observed_coverage_weight", 0.26))
    residual_weight = float(weights.get("residual_weight", 0.14))
    adjustment_weight = float(weights.get("adjustment_weight", 0.30))
    balance_weight = float(weights.get("balance_weight", 0.06))

    adjustment_summary = candidate["support_fragment_adjustment_summary"]
    support_length_norm = min(1.0, candidate["support_total_length_px"] / max(1.0, image_height * 0.34))
    coverage_norm = min(1.0, candidate["observed_y_coverage_px"] / max(1.0, image_height * 0.28))
    residual_limit = float(cfg("candidate_acceptance", "max_mean_residual_px", default=18.0))
    residual_norm = 1.0 - min(1.0, candidate["axis_fit"]["mean_residual_px"] / max(1.0, residual_limit))
    adjustment_reference_px = float(weights.get("adjustment_reference_px", 22.0))
    tilt_adjustment_reference_deg = float(weights.get("tilt_adjustment_reference_deg", 6.0))
    shift_alignment_norm = 1.0 - min(
        1.0,
        float(adjustment_summary["length_weighted_mean_abs_shift_px"]) / max(1.0, adjustment_reference_px),
    )
    tilt_alignment_norm = 1.0 - min(
        1.0,
        float(adjustment_summary["mean_abs_tilt_delta_deg"]) / max(1.0, tilt_adjustment_reference_deg),
    )
    adjustment_norm = 0.72 * shift_alignment_norm + 0.28 * tilt_alignment_norm
    balance_norm = float(adjustment_summary.get("signed_shift_balance_score", 0.5))

    total_weight = support_length_weight + observed_coverage_weight + residual_weight + adjustment_weight + balance_weight
    score = (
        support_length_norm * support_length_weight
        + coverage_norm * observed_coverage_weight
        + residual_norm * residual_weight
        + adjustment_norm * adjustment_weight
        + balance_norm * balance_weight
    ) / max(1e-6, total_weight)

    candidate["score"] = float(score)
    candidate["score_parts"] = {
        "support_length_norm": float(support_length_norm),
        "observed_coverage_norm": float(coverage_norm),
        "residual_norm": float(residual_norm),
        "shift_alignment_norm": float(shift_alignment_norm),
        "tilt_alignment_norm": float(tilt_alignment_norm),
        "adjustment_norm": float(adjustment_norm),
        "balance_norm": float(balance_norm),
    }
    return candidate


def build_candidate(seed_axis, lines, image_height, roi_profile, roi_mask, seed_label):
    support, _ = collect_support_lines(lines, seed_axis, after_refit=False, seed_axis=seed_axis)
    if not support:
        return None

    fit = fit_axis_from_lines(support, initial_axis=seed_axis)
    if fit is None:
        return None

    for _ in range(int(cfg("axis_fit", "iterations", default=5))):
        support_next, _ = collect_support_lines(lines, fit, after_refit=True, seed_axis=seed_axis)
        if not support_next:
            break
        new_fit = fit_axis_from_lines(support_next, initial_axis=fit)
        if new_fit is None:
            break
        old_ids = {line["line_index"] for line in support}
        new_ids = {line["line_index"] for line in support_next}
        support = support_next
        fit = new_fit
        if old_ids == new_ids:
            break

    y_min = min(line["y_min"] for line in support)
    y_max = max(line["y_max"] for line in support)
    fit["y_min"] = float(y_min)
    fit["y_max"] = float(y_max)

    roi_extent = mask_vertical_extent(roi_mask)
    if roi_extent and bool(cfg("drawing", "project_axis_through_roi", default=True)):
        draw_y_min = float(roi_extent[0])
        draw_y_max = float(roi_extent[1])
    else:
        draw_y_min = float(y_min)
        draw_y_max = float(y_max)

    intervals = observed_intervals_for_support(support)
    observed_coverage_px = float(sum(end - start for start, end in intervals))
    support_total_length_px = float(sum(line["length"] for line in support))
    support_fragment_adjustments = [
        build_fragment_adjustment(line, fit)
        for line in support
    ]
    support_fragment_adjustment_summary = summarize_fragment_adjustments(support_fragment_adjustments)

    candidate = {
        "seed_label": seed_label,
        "support_fragment_line_indices": [line["line_index"] for line in support],
        "num_support_fragments": len(support),
        "support_total_length_px": support_total_length_px,
        "observed_y_coverage_px": observed_coverage_px,
        "y_min": float(y_min),
        "y_max": float(y_max),
        "draw_y_min": draw_y_min,
        "draw_y_max": draw_y_max,
        "observed_support_intervals": intervals,
        "axis_fit": fit,
        "support_fragment_adjustments": support_fragment_adjustments,
        "support_fragment_adjustment_summary": support_fragment_adjustment_summary,
    }
    return score_candidate(candidate, image_height, roi_profile)


def candidate_acceptance(candidate, roi_profile):
    reasons = []
    if candidate["num_support_fragments"] < int(cfg("candidate_acceptance", "min_support_fragments", default=2)):
        reasons.append("too_few_support_fragments")
    if candidate["support_total_length_px"] < float(cfg("candidate_acceptance", "min_support_total_length_px", default=180)):
        reasons.append("support_length_too_short")
    if candidate["observed_y_coverage_px"] < float(cfg("candidate_acceptance", "min_observed_y_coverage_px", default=220)):
        reasons.append("observed_coverage_too_low")
    if candidate["axis_fit"]["mean_residual_px"] > float(cfg("candidate_acceptance", "max_mean_residual_px", default=18.0)):
        reasons.append("mean_residual_too_high")
    if candidate["axis_fit"]["median_residual_px"] > float(cfg("candidate_acceptance", "max_median_residual_px", default=14.0)):
        reasons.append("median_residual_too_high")
    if abs(candidate["axis_fit"]["tilt_deg"]) > float(cfg("axis_fit", "max_candidate_tilt_deg", default=5.0)):
        reasons.append("axis_tilt_too_large")
    adjustment_summary = candidate["support_fragment_adjustment_summary"]
    if adjustment_summary["length_weighted_mean_abs_shift_px"] > float(
        cfg("candidate_acceptance", "max_length_weighted_mean_abs_shift_px", default=22.0)
    ):
        reasons.append("shift_adjustment_too_high")
    if adjustment_summary["mean_abs_tilt_delta_deg"] > float(
        cfg("candidate_acceptance", "max_mean_abs_tilt_delta_deg", default=6.8)
    ):
        reasons.append("tilt_adjustment_too_high")

    return len(reasons) == 0, reasons


def mean_axis_distance(candidate_a, candidate_b):
    y0 = max(candidate_a["y_min"], candidate_b["y_min"])
    y1 = min(candidate_a["y_max"], candidate_b["y_max"])
    if y1 <= y0:
        return float("inf")
    ys = np.linspace(y0, y1, 30)
    return float(
        np.mean(
            [
                abs(line_x_at_y(candidate_a["axis_fit"], y) - line_x_at_y(candidate_b["axis_fit"], y))
                for y in ys
            ]
        )
    )


def shared_support_ratio(candidate_a, candidate_b):
    support_a = set(candidate_a["support_fragment_line_indices"])
    support_b = set(candidate_b["support_fragment_line_indices"])
    if not support_a or not support_b:
        return 0.0
    return len(support_a & support_b) / max(1, min(len(support_a), len(support_b)))


def deduplicate_candidates(candidates):
    if not bool(cfg("deduplication", "enabled", default=True)):
        return candidates

    max_mean_axis_distance = float(cfg("deduplication", "max_mean_axis_distance_px", default=22))
    kept = []

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        duplicate = False
        for existing in kept:
            if mean_axis_distance(candidate, existing) <= max_mean_axis_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


def build_axis_seed(anchor_y, anchor_x, tilt_deg):
    a = math.tan(math.radians(float(tilt_deg)))
    b = float(anchor_x) - a * float(anchor_y)
    return {
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(tilt_deg),
    }


def create_seed_axes(lines, roi_profile):
    seeds = []
    max_seed_tilt = float(cfg("candidate_generation", "max_seed_tilt_deg", default=14.0))
    seed_offsets = list(cfg("candidate_generation", "seed_offset_px_values", default=[-24, 0, 24]))
    seed_tilt_deltas = list(cfg("candidate_generation", "seed_tilt_delta_deg_values", default=[-3.0, 0.0, 3.0]))
    offset_values = [0.0]
    offset_values.extend(float(offset) for offset in seed_offsets if float(offset) != 0.0)
    tilt_delta_values = [0.0]
    tilt_delta_values.extend(float(delta) for delta in seed_tilt_deltas if float(delta) != 0.0)
    variant_specs = []
    seen_variant_specs = set()
    for offset in offset_values:
        for tilt_delta in tilt_delta_values:
            key = (float(offset), float(tilt_delta))
            if key in seen_variant_specs:
                continue
            seen_variant_specs.add(key)
            variant_specs.append(key)
    seen_seed_keys = set()

    def append_seed(axis, label):
        key = (round(float(axis["a"]), 6), round(float(axis["b"]), 2))
        if key in seen_seed_keys:
            return
        seen_seed_keys.add(key)
        seeds.append((axis, label))

    seed_lines = [
        line for line in lines
        if abs(line["signed_tilt_deg"]) <= max_seed_tilt
    ]

    for line in seed_lines:
        for offset, tilt_delta in variant_specs:
            tilt_deg = float(line["signed_tilt_deg"]) + float(tilt_delta)
            if abs(tilt_deg) > max_seed_tilt:
                continue
            anchor_x = float(line["x_mid"]) + float(offset)
            append_seed(
                build_axis_seed(line["y_mid"], anchor_x, tilt_deg),
                f"line_{line['line_index']}_offset_{offset}_tilt_{tilt_deg:.2f}_seed",
            )

    if bool(cfg("candidate_generation", "use_pair_seeds", default=True)):
        pair_lines = seed_lines[: int(cfg("candidate_generation", "max_pair_seed_lines", default=20))]
        max_pair_neighbors = int(cfg("candidate_generation", "max_pair_neighbors", default=5))
        max_pair_vertical_gap_px = float(cfg("candidate_generation", "max_pair_vertical_gap_px", default=260))
        max_pair_x_offset_px = float(cfg("candidate_generation", "max_pair_x_offset_px", default=40))
        max_pair_tilt_difference_deg = float(cfg("axis_fit", "max_tilt_difference_deg", default=8.0))

        for index, line_a in enumerate(pair_lines):
            for line_b in pair_lines[index + 1:index + 1 + max_pair_neighbors]:
                relation = fragment_pair_geometry(line_a, line_b)
                if relation["tilt_difference_deg"] > max_pair_tilt_difference_deg:
                    continue
                if relation["vertical_gap_px"] > max_pair_vertical_gap_px:
                    continue
                if relation["axis_distance_px"] > max_pair_x_offset_px:
                    continue
                fit = weighted_huber_fit(
                    [
                        (line_a["y_mid"], line_a["x_mid"], line_a["line_index"]),
                        (line_b["y_mid"], line_b["x_mid"], line_b["line_index"]),
                    ]
                )
                if fit is None:
                    continue
                if abs(fit["tilt_deg"]) > max_seed_tilt:
                    continue
                for offset, tilt_delta in variant_specs:
                    tilt_deg = float(fit["tilt_deg"]) + float(tilt_delta)
                    if abs(tilt_deg) > max_seed_tilt:
                        continue
                    anchor_y = (line_a["y_mid"] + line_b["y_mid"]) / 2.0
                    anchor_x = line_x_at_y(fit, anchor_y) + float(offset)
                    append_seed(
                        build_axis_seed(anchor_y, anchor_x, tilt_deg),
                        f"pair_{line_a['line_index']}_{line_b['line_index']}_offset_{offset}_tilt_{tilt_deg:.2f}_seed",
                    )

    if bool(cfg("candidate_generation", "use_midpoint_pair_seeds", default=True)):
        midpoint_pair_lines = seed_lines[: int(cfg("candidate_generation", "max_pair_seed_lines", default=20))]
        midpoint_pair_min_axis_distance_px = float(
            cfg("candidate_generation", "midpoint_pair_min_axis_distance_px", default=36)
        )
        midpoint_pair_max_axis_distance_px = float(
            cfg("candidate_generation", "midpoint_pair_max_axis_distance_px", default=150)
        )
        midpoint_pair_min_overlap_px = float(
            cfg("candidate_generation", "midpoint_pair_min_overlap_px", default=70)
        )
        midpoint_pair_max_vertical_gap_px = float(
            cfg("candidate_generation", "midpoint_pair_max_vertical_gap_px", default=90)
        )
        max_pair_tilt_difference_deg = float(cfg("axis_fit", "max_tilt_difference_deg", default=8.0))

        for index, line_a in enumerate(midpoint_pair_lines):
            for line_b in midpoint_pair_lines[index + 1:]:
                relation = fragment_pair_geometry(line_a, line_b)
                if relation["tilt_difference_deg"] > max_pair_tilt_difference_deg:
                    continue
                if relation["axis_distance_px"] < midpoint_pair_min_axis_distance_px:
                    continue
                if relation["axis_distance_px"] > midpoint_pair_max_axis_distance_px:
                    continue
                if (
                    relation["overlap_px"] < midpoint_pair_min_overlap_px
                    and relation["vertical_gap_px"] > midpoint_pair_max_vertical_gap_px
                ):
                    continue

                anchor_y = float(relation["probe_y"])
                anchor_x = 0.5 * (
                    line_x_at_y(line_a, anchor_y) + line_x_at_y(line_b, anchor_y)
                )
                total_length = max(1.0, float(line_a["length"]) + float(line_b["length"]))
                avg_a = (
                    float(line_a["a"]) * float(line_a["length"])
                    + float(line_b["a"]) * float(line_b["length"])
                ) / total_length
                base_tilt_deg = float(math.degrees(math.atan(avg_a)))
                if abs(base_tilt_deg) > max_seed_tilt:
                    continue

                for offset, tilt_delta in variant_specs:
                    tilt_deg = float(base_tilt_deg) + float(tilt_delta)
                    if abs(tilt_deg) > max_seed_tilt:
                        continue
                    append_seed(
                        build_axis_seed(anchor_y, anchor_x + float(offset), tilt_deg),
                        (
                            f"midpair_{line_a['line_index']}_{line_b['line_index']}"
                            f"_dist_{relation['axis_distance_px']:.1f}"
                            f"_offset_{offset}_tilt_{tilt_deg:.2f}_seed"
                        ),
                    )

    return seeds


def fit_central_candidates(seed_lines, support_lines, image_height, roi_profile, roi_mask):
    seeds = create_seed_axes(seed_lines, roi_profile)
    accepted = []
    rejected = []

    for seed_axis, seed_label in seeds:
        candidate = build_candidate(seed_axis, support_lines, image_height, roi_profile, roi_mask, seed_label)
        if candidate is None:
            continue
        is_accepted, reject_reasons = candidate_acceptance(candidate, roi_profile)
        candidate["accepted"] = bool(is_accepted)
        candidate["reject_reasons"] = reject_reasons
        if is_accepted:
            accepted.append(candidate)
        else:
            rejected.append(candidate)

    accepted = deduplicate_candidates(accepted)
    accepted = sorted(accepted, key=lambda item: item["score"], reverse=True)[
        : int(cfg("candidate_acceptance", "max_candidates_per_image", default=8))
    ]

    if not accepted and rejected and bool(cfg("candidate_acceptance", "always_keep_best_fallback", default=True)):
        fallback = sorted(rejected, key=lambda item: item["score"], reverse=True)[0]
        fallback["fallback_used"] = True
        accepted = [fallback]

    for index, candidate in enumerate(accepted):
        candidate["candidate_id"] = index

    return {
        "seed_count": len(seeds),
        "candidates": accepted,
        "best_candidate": accepted[0] if accepted else None,
        "accepted_candidate_count": sum(1 for candidate in accepted if candidate.get("accepted")),
        "rejected_candidate_count": len(rejected),
    }


def load_base_image(data):
    image_name = data.get("image_name")
    overlay_path = INPUT_OVERLAY_DIR / image_name
    if overlay_path.exists():
        image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image, str(overlay_path.relative_to(PROJECT_ROOT)), True

    source = resolve_project_path(data.get("source_file"))
    if source is not None and source.exists():
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is not None:
            return image, str(source.relative_to(PROJECT_ROOT)), False

    for path in [
        WORKING_PNG_DIR / image_name,
        PROCESSED_DIR / "03_edges" / "cleaned" / image_name,
    ]:
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image, str(path.relative_to(PROJECT_ROOT)), False

    height = int(data.get("height", 4032))
    width = int(data.get("width", 3024))
    return np.zeros((height, width, 3), dtype=np.uint8), None, False


def draw_line(image, line, color, thickness):
    p1 = (int(round(line["x1"])), int(round(line["y1"])))
    p2 = (int(round(line["x2"])), int(round(line["y2"])))
    cv2.line(image, p1, p2, color, int(thickness), cv2.LINE_AA)


def draw_axis(image, candidate, color, thickness):
    axis = candidate["axis_fit"]
    y0 = int(round(candidate.get("draw_y_min", candidate["y_min"])))
    y1 = int(round(candidate.get("draw_y_max", candidate["y_max"])))
    p1 = (int(round(line_x_at_y(axis, y0))), y0)
    p2 = (int(round(line_x_at_y(axis, y1))), y1)
    cv2.line(image, p1, p2, color, int(thickness), cv2.LINE_AA)


def put_text(image, text, x, y, color=COLOR_TEXT, scale=None):
    font_scale = float(cfg("drawing", "font_scale", default=0.66) if scale is None else scale)
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


def draw_overlay(base_image, base_is_overlay, filtered_lines, candidates, image_name):
    image = base_image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if not base_is_overlay:
        background_alpha = float(cfg("drawing", "background_alpha_fallback", default=0.72))
        image = cv2.addWeighted(image, background_alpha, np.zeros_like(image), 1.0 - background_alpha, 0)
        if bool(cfg("drawing", "draw_filtered_fragments_when_overlay_missing", default=True)):
            for line in filtered_lines:
                draw_line(image, line, COLOR_VALID, 2)

    for index, candidate in enumerate(candidates):
        color = COLOR_CANDIDATE_PALETTE[index % len(COLOR_CANDIDATE_PALETTE)]
        thickness = (
            int(cfg("drawing", "axis_thickness", default=4))
            if index == 0
            else int(cfg("drawing", "secondary_axis_thickness", default=3))
        )
        draw_axis(image, candidate, color, thickness)

        if bool(cfg("drawing", "label_candidates", default=True)):
            label_y = int(round(candidate.get("draw_y_min", candidate["y_min"]))) + 18 + index * 16
            label_x = int(round(line_x_at_y(candidate["axis_fit"], label_y))) + 8
            put_text(image, f"C{index + 1}:{candidate['score']:.2f}", label_x, label_y, color=color, scale=0.56)

    put_text(image, image_name, 28, 36, COLOR_TEXT, scale=0.84)
    put_text(
        image,
        f"candidates={len(candidates)}",
        28,
        68,
        COLOR_CANDIDATE_PALETTE[0] if candidates else COLOR_TEXT,
        scale=0.68,
    )
    return image


def create_comparison(step5_overlay, step6_overlay):
    if len(step5_overlay.shape) == 2:
        step5_overlay = cv2.cvtColor(step5_overlay, cv2.COLOR_GRAY2BGR)
    if len(step6_overlay.shape) == 2:
        step6_overlay = cv2.cvtColor(step6_overlay, cv2.COLOR_GRAY2BGR)

    height, width = step5_overlay.shape[:2]
    max_width = 1300
    if width > max_width:
        scale = max_width / width
        size = (int(width * scale), int(height * scale))
        step5_overlay = cv2.resize(step5_overlay, size, interpolation=cv2.INTER_AREA)
        step6_overlay = cv2.resize(step6_overlay, size, interpolation=cv2.INTER_AREA)

    separator = np.full((step5_overlay.shape[0], 10, 3), 255, dtype=np.uint8)
    return np.hstack([step5_overlay, separator, step6_overlay])


def process_json_file(json_path):
    data = load_json(json_path)
    image_name = data.get("image_name", json_path.stem + ".png")
    width = int(data.get("width", 0))
    height = int(data.get("height", 0)) or 4032

    raw_lines = data.get("valid_lines", [])
    lines = [normalize_line(line, index) for index, line in enumerate(raw_lines)]
    filtered_lines, rejected_lines = filter_fragments(lines)

    roi_mask_path = resolve_project_path(data.get("roi_mask_file"))
    roi_mask = load_roi_mask(roi_mask_path)
    roi_profile = None

    seed_pool, remaining_pool = select_seed_pool(filtered_lines)
    fit_result = fit_central_candidates(
        seed_lines=seed_pool,
        support_lines=seed_pool + remaining_pool,
        image_height=height,
        roi_profile=roi_profile,
        roi_mask=roi_mask,
    )
    candidates = fit_result["candidates"]
    best_candidate = fit_result["best_candidate"]

    base_image, base_image_path, base_is_overlay = load_base_image(data)
    overlay = draw_overlay(base_image, base_is_overlay, filtered_lines, candidates, image_name)
    comparison = create_comparison(base_image, overlay)

    overlay_path = OUTPUT_OVERLAY_DIR / image_name
    comparison_path = OUTPUT_COMPARISON_DIR / image_name
    metadata_path = OUTPUT_METADATA_DIR / f"{Path(image_name).stem}_central_axis.json"

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(comparison_path), comparison)

    metadata = {
        "image_name": image_name,
        "processing_step": "06_fit_central_axis_from_fragments",
        "source_step": data.get("processing_step", "05_valid_hough_lines_in_roi"),
        "width": width,
        "height": height,
        "input_json_file": str(json_path.relative_to(PROJECT_ROOT)),
        "resolved_input_dir": str(INPUT_DIR.relative_to(PROJECT_ROOT)),
        "base_visual_file": base_image_path,
        "base_visual_is_step_05_overlay": bool(base_is_overlay),
        "source_file": data.get("source_file"),
        "roi_mask_file": data.get("roi_mask_file"),
        "input_line_count": len(lines),
        "filtered_line_count": len(filtered_lines),
        "seed_pool_count": len(seed_pool),
        "remaining_pool_count": len(remaining_pool),
        "rejected_fragment_count": len(rejected_lines),
        "seed_count": fit_result["seed_count"],
        "candidate_count": len(candidates),
        "accepted_candidate_count": fit_result["accepted_candidate_count"],
        "best_candidate": best_candidate,
        "candidates": candidates,
        "parameters": STEP_CONFIG,
        "output_overlay_file": str(overlay_path.relative_to(PROJECT_ROOT)),
        "output_comparison_file": str(comparison_path.relative_to(PROJECT_ROOT)),
    }
    save_json(metadata_path, metadata)

    return {
        "image_name": image_name,
        "input_line_count": len(lines),
        "filtered_line_count": len(filtered_lines),
        "seed_pool_count": len(seed_pool),
        "remaining_pool_count": len(remaining_pool),
        "candidate_count": len(candidates),
        "accepted_candidate_count": fit_result["accepted_candidate_count"],
        "best_score": best_candidate["score"] if best_candidate else None,
        "best_tilt_deg": best_candidate["axis_fit"]["tilt_deg"] if best_candidate else None,
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
        "comparison_path": str(comparison_path.relative_to(PROJECT_ROOT)),
    }


def process_json_file_worker(json_path_str, step_config):
    configure_runtime(step_config)

    try:
        result = process_json_file(Path(json_path_str))
        return {
            "ok": True,
            "json_path": json_path_str,
            "result": result,
        }
    except Exception:
        return {
            "ok": False,
            "json_path": json_path_str,
            "error": traceback.format_exc(),
        }


def collect_json_files(image_filter=None, limit=None):
    if not INPUT_JSON_DIR.exists():
        raise FileNotFoundError(f"Input JSON dir does not exist: {INPUT_JSON_DIR}")

    files = sorted(INPUT_JSON_DIR.glob("*.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [
            path
            for path in files
            if path.stem.lower() == wanted or wanted in path.stem.lower()
        ]
    if limit is not None:
        files = files[:limit]
    return files


def show_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return
    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / height
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(path.name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def worker_count_for_run(requested_workers, item_count):
    if item_count <= 1:
        return 1

    if requested_workers is not None:
        return max(1, min(int(requested_workers), item_count))

    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, item_count, 6))


def print_result_summary(result, show_image_flag=False):
    score_text = "none" if result["best_score"] is None else f"{result['best_score']:.3f}"
    tilt_text = "none" if result["best_tilt_deg"] is None else f"{result['best_tilt_deg']:.2f}"
    print(
        f"  input={result['input_line_count']} filtered={result['filtered_line_count']} "
        f"seed_pool={result['seed_pool_count']} remaining={result['remaining_pool_count']} "
        f"candidates={result['candidate_count']} accepted={result['accepted_candidate_count']} "
        f"score={score_text} tilt={tilt_text}"
    )
    print(f"  overlay: {result['overlay_path']}")
    print(f"  metadata: {result['metadata_path']}")
    if show_image_flag:
        show_image(PROJECT_ROOT / result["comparison_path"])


def main():
    global STEP_CONFIG

    parser = argparse.ArgumentParser(description="Step 06: fit fragment-derived central-axis candidates without ROI-center bias.")
    parser.add_argument("--image", type=str, default=None, help="Optional image name filter, e.g. IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count. Default: auto.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    STEP_CONFIG = apply_preset(STEP_CONFIG, args.preset)
    configure_runtime(STEP_CONFIG)

    ensure_dirs(cleanup=bool(STEP_CONFIG.get("cleanup_output_on_start", True)) and args.image is None)

    json_files = collect_json_files(args.image, args.limit)
    if not json_files:
        print("No JSON files found.")
        print(f"Input dir: {INPUT_JSON_DIR}")
        return

    print(f"Step 06 input dir: {INPUT_JSON_DIR}")
    print(f"Step 06 visual base dir: {INPUT_OVERLAY_DIR}")
    print(f"Step 06 output dir: {OUTPUT_DIR}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found JSON files: {len(json_files)}")
    worker_count = worker_count_for_run(args.workers, len(json_files))
    print(f"Using workers: {worker_count}")

    summary_by_path = {}
    if worker_count == 1:
        for json_path in json_files:
            print(f"\nProcessing: {json_path.name}")
            try:
                result = process_json_file(json_path)
                summary_by_path[str(json_path)] = result
                print_result_summary(result, show_image_flag=args.show)
            except Exception as exc:
                print(f"  ERROR: {exc}")
    else:
        if args.show:
            print("Comparison windows will open in completion order because workers > 1.")

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_json_file_worker, str(json_path), STEP_CONFIG): json_path
                for json_path in json_files
            }

            for future in as_completed(futures):
                json_path = futures[future]
                print(f"\nProcessing: {json_path.name}")
                payload = future.result()

                if not payload.get("ok"):
                    error_text = (payload.get("error") or "").strip()
                    print(f"  ERROR: {error_text or 'worker failed'}")
                    continue

                result = payload["result"]
                summary_by_path[str(json_path)] = result
                print_result_summary(result, show_image_flag=args.show)

    summary = [summary_by_path[str(json_path)] for json_path in json_files if str(json_path) in summary_by_path]
    summary_path = OUTPUT_DIR / "step_06_summary.json"
    save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
