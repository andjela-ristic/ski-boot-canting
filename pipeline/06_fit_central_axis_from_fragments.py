from pathlib import Path
import sys
import json
import math
import argparse
import shutil
import warnings
from copy import deepcopy

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "pipeline"))

from config_loader import load_config


CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG.get("display", {})

WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]

STEP_CONFIG_RAW = CONFIG.get("step_06_fit_central_axis_from_fragments")
if STEP_CONFIG_RAW is None:
    # Backward-compatible fallback if you keep the old key name in yaml.
    STEP_CONFIG_RAW = CONFIG.get("step_06_axis_fragment_chains", {})


DEFAULT_STEP_CONFIG = {
    "enabled": True,
    "input_subdir": "05_valid_hough_lines_in_roi",
    "input_json_subdir": "valid_lines_json",
    "output_subdir": "06_fit_central_axis_from_fragments",
    "cleanup_output_on_start": True,

    "fragment_filter": {
        "min_length_px": 75,
        "max_vertical_deviation_deg": 12.0,
        "min_mask_support_ratio": 0.94,
        "min_points_inside_mask": 60,
    },

    "roi_axis_prior": {
        "enabled": True,
        "sample_count": 90,
        "trim_top_ratio": 0.08,
        "trim_bottom_ratio": 0.08,
        "min_row_width_px": 180,
        "max_row_width_quantile": 0.88,
        "fit_max_tilt_deg": 2.0,
        "max_seed_distance_px": 120,
        "max_final_median_distance_px": 105,
        "weight": 0.52,
    },

    "axis_fit": {
        "sample_step_px": 14,
        "seed_every_fragment": True,
        "include_prior_seed": True,
        "max_seed_tilt_deg": 14.0,
        "max_seed_fragment_distance_px": 95,
        "max_support_distance_px": 24,
        "max_support_distance_after_refit_px": 18,
        "max_tilt_difference_deg": 6.0,
        "iterations": 5,
        "huber_delta_px": 12.0,
        "candidate_max_tilt_deg": 4.0,
        "max_line_seed_count": 36,
        "use_pair_seeds": True,
        "max_pair_seed_lines": 14,
        "max_pair_neighbors": 4,
    },

    "candidate_acceptance": {
        "min_support_fragments": 3,
        "min_support_total_length_px": 220,
        "min_observed_y_coverage_px": 300,
        "max_mean_residual_px": 14.0,
        "max_median_residual_px": 10.0,
        "max_candidates_per_image": 6,
        "always_keep_best_fallback": True,
    },

    "scoring": {
        "support_length_weight": 0.20,
        "observed_coverage_weight": 0.22,
        "residual_weight": 0.20,
        "centrality_weight": 0.38,
        "endpoint_alignment_weight": 0.0,
        "band_balance_weight": 0.0,
    },

    "endpoint_centering": {
        "enabled": True,
        "top_band": {
            "y_start_ratio": 0.12,
            "y_end_ratio": 0.30,
        },
        "middle_band": {
            "y_start_ratio": 0.38,
            "y_end_ratio": 0.62,
        },
        "bottom_band": {
            "y_start_ratio": 0.70,
            "y_end_ratio": 0.90,
        },
        "min_rows_per_band": 8,
        "max_band_center_error_px": 170,
        "max_endpoint_error_px": 190,
        "max_top_bottom_error_difference_px": 140,
        "require_top_bottom_reasonable": False,
    },

    "deduplication": {
        "enabled": True,
        "max_mean_axis_distance_px": 28,
        "min_shared_support_ratio": 0.55,
    },

    "drawing": {
        "background_alpha": 0.68,
        "draw_all_valid_fragments": True,
        "draw_rejected_fragments": True,
        "draw_support_fragments": True,
        "draw_final_axis": True,
        "draw_prior_guide": False,
        "draw_gap_intervals": True,
        "draw_endpoint_bands": False,
        "valid_fragment_thickness": 1,
        "support_fragment_thickness": 3,
        "rejected_fragment_thickness": 2,
        "axis_thickness": 2,
        "font_scale": 0.72,
    },

    "test_presets": [
        {"name": "default", "override": {}},
        {
            "name": "looser_support",
            "override": {
                "fragment_filter": {
                    "min_length_px": 45,
                    "max_vertical_deviation_deg": 25.0,
                    "min_mask_support_ratio": 0.82,
                    "min_points_inside_mask": 25,
                },
                "axis_fit": {
                    "max_seed_fragment_distance_px": 125,
                    "max_support_distance_px": 58,
                    "max_support_distance_after_refit_px": 46,
                    "max_tilt_difference_deg": 13.0,
                    "candidate_max_tilt_deg": 10.0,
                },
                "candidate_acceptance": {
                    "min_support_fragments": 2,
                    "min_support_total_length_px": 120,
                    "min_observed_y_coverage_px": 180,
                    "max_mean_residual_px": 32.0,
                    "max_median_residual_px": 26.0,
                },
            },
        },
        {
            "name": "stricter_center",
            "override": {
                "roi_axis_prior": {
                    "max_seed_distance_px": 170,
                    "max_final_median_distance_px": 150,
                    "weight": 0.34,
                },
                "axis_fit": {
                    "max_support_distance_px": 34,
                    "max_support_distance_after_refit_px": 28,
                    "candidate_max_tilt_deg": 6.0,
                },
            },
        },
        {
            "name": "endpoint_balanced",
            "override": {
                "output_subdir": "06_fit_central_axis_from_fragments/endpoint_balanced",
                "scoring": {
                    "support_length_weight": 0.18,
                    "observed_coverage_weight": 0.18,
                    "residual_weight": 0.20,
                    "centrality_weight": 0.18,
                    "endpoint_alignment_weight": 0.18,
                    "band_balance_weight": 0.08,
                },
                "endpoint_centering": {
                    "enabled": True,
                    "max_band_center_error_px": 170,
                    "max_endpoint_error_px": 190,
                    "max_top_bottom_error_difference_px": 140,
                    "require_top_bottom_reasonable": False,
                },
            },
        },
        {
            "name": "endpoint_strict",
            "override": {
                "output_subdir": "06_fit_central_axis_from_fragments/endpoint_strict",
                "scoring": {
                    "support_length_weight": 0.14,
                    "observed_coverage_weight": 0.16,
                    "residual_weight": 0.18,
                    "centrality_weight": 0.20,
                    "endpoint_alignment_weight": 0.24,
                    "band_balance_weight": 0.08,
                },
                "endpoint_centering": {
                    "enabled": True,
                    "max_band_center_error_px": 130,
                    "max_endpoint_error_px": 150,
                    "max_top_bottom_error_difference_px": 100,
                    "require_top_bottom_reasonable": False,
                },
            },
        },
        {
            "name": "endpoint_soft",
            "override": {
                "output_subdir": "06_fit_central_axis_from_fragments/endpoint_soft",
                "scoring": {
                    "support_length_weight": 0.22,
                    "observed_coverage_weight": 0.20,
                    "residual_weight": 0.20,
                    "centrality_weight": 0.16,
                    "endpoint_alignment_weight": 0.14,
                    "band_balance_weight": 0.08,
                },
                "endpoint_centering": {
                    "enabled": True,
                    "max_band_center_error_px": 220,
                    "max_endpoint_error_px": 240,
                    "max_top_bottom_error_difference_px": 180,
                    "require_top_bottom_reasonable": False,
                },
            },
        },
        {
            "name": "endpoint_strict_accept",
            "override": {
                "output_subdir": "06_fit_central_axis_from_fragments/endpoint_strict_accept",
                "scoring": {
                    "support_length_weight": 0.14,
                    "observed_coverage_weight": 0.16,
                    "residual_weight": 0.18,
                    "centrality_weight": 0.20,
                    "endpoint_alignment_weight": 0.24,
                    "band_balance_weight": 0.08,
                },
                "endpoint_centering": {
                    "enabled": True,
                    "max_band_center_error_px": 140,
                    "max_endpoint_error_px": 160,
                    "max_top_bottom_error_difference_px": 110,
                    "require_top_bottom_reasonable": True,
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
    available = [p.get("name") for p in config.get("test_presets", [])]
    raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")


STEP_CONFIG = deep_merge(DEFAULT_STEP_CONFIG, STEP_CONFIG_RAW)

INPUT_SUBDIR = STEP_CONFIG.get("input_subdir", "05_valid_hough_lines_in_roi")
OUTPUT_SUBDIR = STEP_CONFIG.get("output_subdir", "06_fit_central_axis_from_fragments")
INPUT_DIR = PROCESSED_DIR / INPUT_SUBDIR
INPUT_JSON_DIR = INPUT_DIR / STEP_CONFIG.get("input_json_subdir", "valid_lines_json")
OUTPUT_DIR = PROCESSED_DIR / OUTPUT_SUBDIR
OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "overlay"
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
OUTPUT_COMPARISON_DIR = OUTPUT_DIR / "comparison"


# BGR colors for OpenCV.
COLOR_VALID = (0, 150, 0)       # dark green: valid but not support
COLOR_SUPPORT = (0, 255, 0)     # bright green: fragments used by final axis
COLOR_REJECTED = (0, 0, 255)    # red
COLOR_AXIS = (0, 255, 255)      # yellow: fitted/joined axis through green supports
COLOR_PRIOR = (180, 180, 180)   # gray helper only, off by default
COLOR_ENDPOINT = (255, 255, 0)  # cyan: optional ROI band center debug markers
COLOR_TEXT = (240, 240, 240)
COLOR_CANDIDATE_PALETTE = [
    (0, 255, 255),
    (255, 200, 0),
    (255, 120, 120),
    (120, 255, 120),
    (120, 180, 255),
    (255, 120, 255),
]


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


def cfg(*keys, default=None):
    cur = STEP_CONFIG
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def resolve_project_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dirs(cleanup=False):
    if cleanup and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_line(raw_line, fallback_index):
    x1 = float(raw_line["x1"])
    y1 = float(raw_line["y1"])
    x2 = float(raw_line["x2"])
    y2 = float(raw_line["y2"])
    dx = x2 - x1
    dy = y2 - y1
    length = float(raw_line.get("length", math.hypot(dx, dy)))

    angle_deg = float(raw_line.get("angle_degrees", math.degrees(math.atan2(dy, dx))))
    # signed tilt for model x = a*y+b. positive means x grows with y.
    a = dx / dy if abs(dy) > 1e-6 else 999.0
    b = x1 - a * y1 if abs(dy) > 1e-6 else (x1 + x2) / 2.0
    signed_tilt_deg = math.degrees(math.atan(a)) if abs(a) < 999 else 90.0
    vertical_deviation = float(raw_line.get("vertical_deviation_degrees", abs(signed_tilt_deg)))

    sampled_points = int(raw_line.get("sampled_points", 0))
    points_inside_mask = int(raw_line.get("points_inside_mask", 0))

    return {
        "line_index": int(raw_line.get("line_index", raw_line.get("id", fallback_index))),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length": length,
        "angle_degrees": angle_deg,
        "vertical_deviation_degrees": vertical_deviation,
        "signed_tilt_deg": float(signed_tilt_deg),
        "mask_support_ratio": float(raw_line.get("mask_support_ratio", 1.0)),
        "sampled_points": sampled_points,
        "points_inside_mask": points_inside_mask,
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
    step_px = step_px or cfg("axis_fit", "sample_step_px", default=14)
    length = max(1.0, float(line["length"]))
    n = max(2, int(math.ceil(length / max(1.0, step_px))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    pts = []
    for t in ts:
        pts.append((float(y1 + t * (y2 - y1)), float(x1 + t * (x2 - x1)), line["line_index"]))
    return pts


def filter_fragments(lines):
    accepted = []
    rejected = []
    min_len = cfg("fragment_filter", "min_length_px", default=55)
    max_dev = cfg("fragment_filter", "max_vertical_deviation_deg", default=22.0)
    min_mask = cfg("fragment_filter", "min_mask_support_ratio", default=0.86)
    min_inside = cfg("fragment_filter", "min_points_inside_mask", default=35)

    for line in lines:
        reasons = []
        if not line["is_valid"]:
            reasons.append("not_valid")
        if line["length"] < min_len:
            reasons.append("too_short")
        if abs(line["signed_tilt_deg"]) > max_dev:
            reasons.append("too_far_from_vertical")
        if line["mask_support_ratio"] < min_mask:
            reasons.append("low_mask_support")
        if line["points_inside_mask"] and line["points_inside_mask"] < min_inside:
            reasons.append("too_few_points_inside_mask")

        if reasons:
            rejected.append({**line, "reject_reasons": reasons})
        else:
            accepted.append(line)
    return accepted, rejected


def load_roi_mask(mask_path):
    if mask_path is None or not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def robust_roi_axis_prior(mask):
    """Returns a straight weak prior x=a*y+b from the ROI mask.

    It is deliberately constrained and used only for centrality scoring/seed filtering.
    Final yellow axis is fitted from support Hough fragments, not from this prior.
    """
    if mask is None or not cfg("roi_axis_prior", "enabled", default=True):
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    h, w = mask.shape[:2]
    trim_top = cfg("roi_axis_prior", "trim_top_ratio", default=0.08)
    trim_bottom = cfg("roi_axis_prior", "trim_bottom_ratio", default=0.08)
    y0 = int(y_min + (y_max - y_min) * trim_top)
    y1 = int(y_max - (y_max - y_min) * trim_bottom)
    sample_count = cfg("roi_axis_prior", "sample_count", default=90)
    min_width = cfg("roi_axis_prior", "min_row_width_px", default=180)

    rows = []
    widths = []
    for y in np.linspace(y0, y1, sample_count):
        yi = int(round(np.clip(y, 0, h - 1)))
        row_xs = np.where(mask[yi, :] > 0)[0]
        if len(row_xs) < 2:
            continue
        left = int(row_xs[0])
        right = int(row_xs[-1])
        width = right - left
        if width < min_width:
            continue
        rows.append((float(yi), float((left + right) / 2.0), float(width)))
        widths.append(width)

    if len(rows) < 8:
        return None

    max_q = cfg("roi_axis_prior", "max_row_width_quantile", default=0.88)
    max_width = float(np.quantile(np.array(widths, dtype=np.float64), max_q))
    filtered = [(y, x, width) for y, x, width in rows if width <= max_width]
    if len(filtered) < 6:
        filtered = rows

    y = np.array([p[0] for p in filtered], dtype=np.float64)
    x = np.array([p[1] for p in filtered], dtype=np.float64)
    weights = np.ones_like(y)

    fit = safe_linear_polyfit(y, x, weights)
    if fit is None:
        return None
    a, b = fit
    max_tilt = cfg("roi_axis_prior", "fit_max_tilt_deg", default=2.0)
    max_a = math.tan(math.radians(max_tilt))
    a = float(np.clip(a, -max_a, max_a))
    # Recompute intercept after tilt clamp.
    b = float(np.median(x - a * y))

    return {
        "type": "weak_roi_axis_prior",
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(math.degrees(math.atan(a))),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "points": [{"y": float(yy), "x": float(xx)} for yy, xx, _ in filtered],
    }


def weighted_huber_fit(points, base_weights=None, initial=None):
    if len(points) < 2:
        return None
    y = np.array([p[0] for p in points], dtype=np.float64)
    x = np.array([p[1] for p in points], dtype=np.float64)
    if base_weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.array(base_weights, dtype=np.float64)

    if initial is None:
        fit = safe_linear_polyfit(y, x, weights)
        if fit is None:
            return None
        a, b = fit
    else:
        a, b = float(initial["a"]), float(initial["b"])

    delta = cfg("axis_fit", "huber_delta_px", default=18.0)
    for _ in range(int(cfg("axis_fit", "iterations", default=5))):
        residuals = np.abs((a * y + b) - x)
        huber = np.ones_like(residuals)
        mask = residuals > delta
        huber[mask] = delta / np.maximum(residuals[mask], 1e-6)
        final_weights = weights * huber
        try:
            fit = safe_linear_polyfit(y, x, final_weights)
            if fit is None:
                return None
            a, b = fit
        except Exception:
            return None

    residuals = np.abs((a * y + b) - x)
    return {
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(math.degrees(math.atan(a))),
        "mean_residual_px": float(np.mean(residuals)),
        "median_residual_px": float(np.median(residuals)),
        "max_residual_px": float(np.max(residuals)),
    }


def line_distance_to_axis(line, axis):
    pts = sample_line_points(line)
    distances = [abs(line_x_at_y(axis, y) - x) for y, x, _ in pts]
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
        "mid": abs(line_x_at_y(axis, line["y_mid"]) - line["x_mid"]),
    }


def collect_support_lines(lines, axis, after_refit=False):
    max_dist = cfg(
        "axis_fit",
        "max_support_distance_after_refit_px" if after_refit else "max_support_distance_px",
        default=34 if after_refit else 42,
    )
    max_tilt_diff = cfg("axis_fit", "max_tilt_difference_deg", default=10.0)
    support = []
    reject_info = []
    axis_tilt = float(axis["tilt_deg"])

    for line in lines:
        dist = line_distance_to_axis(line, axis)
        tilt_diff = abs(line["signed_tilt_deg"] - axis_tilt)
        if dist["median"] <= max_dist and dist["mid"] <= max_dist * 1.35 and tilt_diff <= max_tilt_diff:
            support.append(line)
        else:
            reason = []
            if dist["median"] > max_dist or dist["mid"] > max_dist * 1.35:
                reason.append("far_from_axis")
            if tilt_diff > max_tilt_diff:
                reason.append("tilt_mismatch")
            reject_info.append({"line_index": line["line_index"], "reason": reason, "distance": dist, "tilt_diff_deg": tilt_diff})
    return support, reject_info


def fit_from_lines(lines, initial_axis=None):
    if len(lines) < 1:
        return None
    points = []
    weights = []
    for line in lines:
        line_weight = math.sqrt(max(1.0, line["length"]))
        for y, x, line_id in sample_line_points(line):
            points.append((y, x, line_id))
            weights.append(line_weight)
    fit = weighted_huber_fit(points, weights, initial=initial_axis)
    if fit is None:
        return None
    residual_by_line = {}
    for line in lines:
        residual_by_line[line["line_index"]] = line_distance_to_axis(line, fit)
    fit["fit_source"] = "support_hough_fragments"
    fit["residual_by_line"] = residual_by_line
    return fit


def observed_intervals_for_support(support_lines, axis):
    if not support_lines:
        return [], []
    intervals = sorted([(line["y_min"], line["y_max"]) for line in support_lines])
    merged = []
    gap_join_px = cfg("axis_fit", "max_support_distance_px", default=42) * 1.5
    for start, end in intervals:
        if not merged or start > merged[-1][1] + gap_join_px:
            merged.append([float(start), float(end)])
        else:
            merged[-1][1] = float(max(merged[-1][1], end))
    gaps = []
    for a, b in zip(merged[:-1], merged[1:]):
        gaps.append([float(a[1]), float(b[0])])
    return merged, gaps


def axis_prior_distance(candidate_axis, prior):
    if prior is None:
        return 0.5, None
    y0 = max(float(candidate_axis.get("y_min", prior["y_min"])), float(prior["y_min"]))
    y1 = min(float(candidate_axis.get("y_max", prior["y_max"])), float(prior["y_max"]))
    if y1 <= y0:
        y0, y1 = prior["y_min"], prior["y_max"]
    ys = np.linspace(y0, y1, 30)
    distances = [abs(line_x_at_y(candidate_axis, y) - line_x_at_y(prior, y)) for y in ys]
    median_dist = float(np.median(distances))
    max_dist = cfg("roi_axis_prior", "max_final_median_distance_px", default=230)
    score = 1.0 - min(1.0, median_dist / max(1.0, max_dist))
    return float(score), median_dist


def _neutral_band_result(label, y_start=None, y_end=None):
    return {
        "available": False,
        "median_center_error_px": None,
        "mean_center_error_px": None,
        "sample_count": 0,
        "y_start": None if y_start is None else int(y_start),
        "y_end": None if y_end is None else int(y_end),
        "label": label,
    }


def compute_roi_band_center_errors(fit, roi_mask, endpoint_config):
    save_debug_rows = bool(cfg("drawing", "draw_endpoint_bands", default=False))
    enabled = bool((endpoint_config or {}).get("enabled", False))
    result = {
        "enabled": enabled,
        "available": False,
        "top": _neutral_band_result("top"),
        "middle": _neutral_band_result("middle"),
        "bottom": _neutral_band_result("bottom"),
        "endpoint_alignment_score": 0.5,
        "band_balance_score": 0.5,
        "top_bottom_error_difference_px": None,
        "max_band_error_px": None,
        "mean_band_error_px": None,
        "endpoint_error_px": None,
    }

    if not enabled or roi_mask is None:
        return result

    occupied_rows = np.flatnonzero(np.any(roi_mask > 0, axis=1))
    if len(occupied_rows) == 0:
        return result

    y_min = int(occupied_rows[0])
    y_max = int(occupied_rows[-1])
    if y_max <= y_min:
        return result

    span = float(y_max - y_min)
    min_rows = int((endpoint_config or {}).get("min_rows_per_band", 8))
    band_specs = {
        "top": endpoint_config.get("top_band", {}),
        "middle": endpoint_config.get("middle_band", {}),
        "bottom": endpoint_config.get("bottom_band", {}),
    }

    band_errors = {}
    band_available_errors = []

    for label, spec in band_specs.items():
        start_ratio = float(spec.get("y_start_ratio", 0.0))
        end_ratio = float(spec.get("y_end_ratio", 1.0))
        band_y_start = int(round(y_min + start_ratio * span))
        band_y_end = int(round(y_min + end_ratio * span))
        band_y_start = int(np.clip(band_y_start, y_min, y_max))
        band_y_end = int(np.clip(band_y_end, y_min, y_max))
        if band_y_end < band_y_start:
            band_y_start, band_y_end = band_y_end, band_y_start

        row_errors = []
        row_centers = []
        for y in range(band_y_start, band_y_end + 1):
            row_xs = np.where(roi_mask[y, :] > 0)[0]
            if len(row_xs) < 2:
                continue
            center_x = float((row_xs[0] + row_xs[-1]) / 2.0)
            axis_x = line_x_at_y(fit, y)
            error = abs(axis_x - center_x)
            row_errors.append(float(error))
            row_centers.append({
                "y": int(y),
                "center_x": float(center_x),
                "axis_x": float(axis_x),
                "error_px": float(error),
            })

        if len(row_errors) >= min_rows:
            median_error = float(np.median(np.array(row_errors, dtype=np.float64)))
            mean_error = float(np.mean(np.array(row_errors, dtype=np.float64)))
            result[label] = {
                "available": True,
                "median_center_error_px": median_error,
                "mean_center_error_px": mean_error,
                "sample_count": int(len(row_errors)),
                "y_start": int(band_y_start),
                "y_end": int(band_y_end),
                "label": label,
            }
            if save_debug_rows:
                result[label]["row_centers"] = row_centers
            band_errors[label] = median_error
            band_available_errors.append(median_error)
        else:
            result[label] = _neutral_band_result(label, band_y_start, band_y_end)

    result["available"] = bool(result["top"]["available"] or result["middle"]["available"] or result["bottom"]["available"])

    top_error = band_errors.get("top")
    middle_error = band_errors.get("middle")
    bottom_error = band_errors.get("bottom")

    max_endpoint_error = float(endpoint_config.get("max_endpoint_error_px", 190))
    max_band_error = float(endpoint_config.get("max_band_center_error_px", 170))
    max_diff = float(endpoint_config.get("max_top_bottom_error_difference_px", 140))

    endpoint_components = []
    if top_error is not None:
        endpoint_components.append(top_error)
    if bottom_error is not None:
        endpoint_components.append(bottom_error)

    if endpoint_components:
        endpoint_error = float(np.median(np.array(endpoint_components, dtype=np.float64)))
        endpoint_score = 1.0 - min(1.0, endpoint_error / max(1.0, max_endpoint_error))
    else:
        endpoint_error = None
        endpoint_score = 0.5

    if band_available_errors:
        max_band_error_seen = float(max(band_available_errors))
        mean_band_error = float(np.mean(np.array(band_available_errors, dtype=np.float64)))
        overall_band_score = 1.0 - min(1.0, max_band_error_seen / max(1.0, max_band_error))
    else:
        max_band_error_seen = None
        mean_band_error = None
        overall_band_score = 0.5

    if top_error is not None and bottom_error is not None:
        top_bottom_diff = float(abs(top_error - bottom_error))
        band_balance_score = 1.0 - min(1.0, top_bottom_diff / max(1.0, max_diff))
    else:
        top_bottom_diff = None
        band_balance_score = 0.5

    result["endpoint_alignment_score"] = float(0.65 * endpoint_score + 0.35 * overall_band_score)
    result["band_balance_score"] = float(band_balance_score)
    result["top_bottom_error_difference_px"] = top_bottom_diff
    result["max_band_error_px"] = max_band_error_seen
    result["mean_band_error_px"] = mean_band_error
    result["endpoint_error_px"] = endpoint_error
    return result


def score_candidate(candidate, image_height, prior, roi_mask=None):
    weights = cfg("scoring", default={})
    endpoint_info = compute_roi_band_center_errors(
        candidate["axis_fit"],
        roi_mask,
        cfg("endpoint_centering", default={}),
    )
    candidate["endpoint_centering"] = endpoint_info

    endpoint_enabled = bool(endpoint_info.get("enabled"))
    active_weights = {
        "support_length_weight": float(weights.get("support_length_weight", 0.28)),
        "observed_coverage_weight": float(weights.get("observed_coverage_weight", 0.26)),
        "residual_weight": float(weights.get("residual_weight", 0.24)),
        "centrality_weight": float(weights.get("centrality_weight", 0.22)),
        "endpoint_alignment_weight": float(weights.get("endpoint_alignment_weight", 0.0)) if endpoint_enabled else 0.0,
        "band_balance_weight": float(weights.get("band_balance_weight", 0.0)) if endpoint_enabled else 0.0,
    }
    total_w = sum(active_weights.values()) or 1.0
    length_norm = min(1.0, candidate["support_total_length_px"] / max(1.0, image_height * 0.32))
    coverage_norm = min(1.0, candidate["observed_y_coverage_px"] / max(1.0, image_height * 0.34))
    residual_limit = cfg("candidate_acceptance", "max_mean_residual_px", default=24.0)
    residual_norm = 1.0 - min(1.0, candidate["axis_fit"]["mean_residual_px"] / max(1.0, residual_limit))
    centrality_norm, median_prior_distance = axis_prior_distance(candidate["axis_fit"], prior)
    endpoint_alignment_score = float(endpoint_info.get("endpoint_alignment_score", 0.5))
    band_balance_score = float(endpoint_info.get("band_balance_score", 0.5))
    score = (
        length_norm * active_weights["support_length_weight"]
        + coverage_norm * active_weights["observed_coverage_weight"]
        + residual_norm * active_weights["residual_weight"]
        + centrality_norm * active_weights["centrality_weight"]
        + endpoint_alignment_score * active_weights["endpoint_alignment_weight"]
        + band_balance_score * active_weights["band_balance_weight"]
    ) / total_w
    candidate["score"] = float(score)
    candidate["score_parts"] = {
        "support_length_norm": float(length_norm),
        "observed_coverage_norm": float(coverage_norm),
        "residual_norm": float(residual_norm),
        "centrality_norm": float(centrality_norm),
        "endpoint_alignment_score": endpoint_alignment_score,
        "band_balance_score": band_balance_score,
        "median_prior_distance_px": median_prior_distance,
        "weights_used": active_weights,
    }
    return candidate


def build_candidate(seed_axis, lines, image_height, prior, seed_label, roi_mask=None):
    # First support pass around seed.
    support, _ = collect_support_lines(lines, seed_axis, after_refit=False)
    if not support:
        return None

    fit = fit_from_lines(support, initial_axis=seed_axis)
    if fit is None:
        return None

    # Iterative support/refit: this is where separate central fragments are effectively joined.
    for _ in range(int(cfg("axis_fit", "iterations", default=5))):
        support2, _ = collect_support_lines(lines, fit, after_refit=True)
        if not support2:
            break
        new_fit = fit_from_lines(support2, initial_axis=fit)
        if new_fit is None:
            break
        old_ids = {l["line_index"] for l in support}
        new_ids = {l["line_index"] for l in support2}
        support, fit = support2, new_fit
        if old_ids == new_ids:
            break

    y_min = min(line["y_min"] for line in support)
    y_max = max(line["y_max"] for line in support)
    fit["y_min"] = float(y_min)
    fit["y_max"] = float(y_max)

    intervals, gaps = observed_intervals_for_support(support, fit)
    observed_coverage = sum(end - start for start, end in intervals)
    total_length = sum(line["length"] for line in support)

    candidate = {
        "seed_label": seed_label,
        "support_fragment_line_indices": [line["line_index"] for line in support],
        "num_support_fragments": len(support),
        "support_total_length_px": float(total_length),
        "observed_y_coverage_px": float(observed_coverage),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "observed_support_intervals": intervals,
        "gap_intervals": gaps,
        "axis_fit": fit,
    }
    return score_candidate(candidate, image_height, prior, roi_mask=roi_mask)


def candidate_acceptance(candidate, prior):
    reasons = []
    if candidate["num_support_fragments"] < cfg("candidate_acceptance", "min_support_fragments", default=2):
        reasons.append("too_few_support_fragments")
    if candidate["support_total_length_px"] < cfg("candidate_acceptance", "min_support_total_length_px", default=160):
        reasons.append("support_length_too_short")
    if candidate["observed_y_coverage_px"] < cfg("candidate_acceptance", "min_observed_y_coverage_px", default=240):
        reasons.append("observed_coverage_too_low")
    if candidate["axis_fit"]["mean_residual_px"] > cfg("candidate_acceptance", "max_mean_residual_px", default=24.0):
        reasons.append("mean_residual_too_high")
    if candidate["axis_fit"]["median_residual_px"] > cfg("candidate_acceptance", "max_median_residual_px", default=18.0):
        reasons.append("median_residual_too_high")
    if abs(candidate["axis_fit"]["tilt_deg"]) > cfg("axis_fit", "candidate_max_tilt_deg", default=8.0):
        reasons.append("axis_tilt_too_large")
    if prior is not None:
        _, median_prior_dist = axis_prior_distance(candidate["axis_fit"], prior)
        max_dist = cfg("roi_axis_prior", "max_final_median_distance_px", default=230)
        if median_prior_dist is not None and median_prior_dist > max_dist:
            reasons.append("too_far_from_roi_prior")
    endpoint_cfg = cfg("endpoint_centering", default={})
    endpoint_info = candidate.get("endpoint_centering", {})
    endpoint_rejected = False
    if bool(endpoint_cfg.get("enabled", False)) and bool(endpoint_cfg.get("require_top_bottom_reasonable", False)):
        top = endpoint_info.get("top", {})
        bottom = endpoint_info.get("bottom", {})
        max_endpoint_error = float(endpoint_cfg.get("max_endpoint_error_px", 190))
        max_diff = float(endpoint_cfg.get("max_top_bottom_error_difference_px", 140))
        top_error = top.get("median_center_error_px")
        bottom_error = bottom.get("median_center_error_px")
        diff = endpoint_info.get("top_bottom_error_difference_px")
        if top_error is None or bottom_error is None:
            endpoint_rejected = True
            reasons.append("endpoint_top_bottom_missing")
        else:
            if top_error > max_endpoint_error:
                endpoint_rejected = True
                reasons.append("top_endpoint_error_too_high")
            if bottom_error > max_endpoint_error:
                endpoint_rejected = True
                reasons.append("bottom_endpoint_error_too_high")
            if diff is not None and diff > max_diff:
                endpoint_rejected = True
                reasons.append("top_bottom_endpoint_imbalance")
    candidate["endpoint_centering_rejected"] = endpoint_rejected
    return len(reasons) == 0, reasons


def mean_axis_distance(a, b):
    y0 = max(a["y_min"], b["y_min"])
    y1 = min(a["y_max"], b["y_max"])
    if y1 <= y0:
        return float("inf")
    ys = np.linspace(y0, y1, 30)
    return float(np.mean([abs(line_x_at_y(a["axis_fit"], y) - line_x_at_y(b["axis_fit"], y)) for y in ys]))


def shared_support_ratio(a, b):
    sa = set(a["support_fragment_line_indices"])
    sb = set(b["support_fragment_line_indices"])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def deduplicate_candidates(candidates):
    if not cfg("deduplication", "enabled", default=True):
        return candidates
    max_dist = cfg("deduplication", "max_mean_axis_distance_px", default=28)
    min_shared = cfg("deduplication", "min_shared_support_ratio", default=0.55)
    kept = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        duplicate = False
        for existing in kept:
            if mean_axis_distance(cand, existing) <= max_dist or shared_support_ratio(cand, existing) >= min_shared:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def create_seed_axes(lines, prior):
    seeds = []
    max_seed_tilt = cfg("axis_fit", "max_seed_tilt_deg", default=14.0)
    max_seed_distance = cfg("roi_axis_prior", "max_seed_distance_px", default=260)

    if prior is not None and cfg("axis_fit", "include_prior_seed", default=True):
        seeds.append((prior, "roi_prior_seed"))

    for line in lines:
        if abs(line["signed_tilt_deg"]) > max_seed_tilt:
            continue
        if prior is not None:
            d = abs(line_x_at_y(prior, line["y_mid"]) - line["x_mid"])
            if d > max_seed_distance:
                continue
        seeds.append(({"a": line["a"], "b": line["b"], "tilt_deg": line["signed_tilt_deg"]}, f"line_{line['line_index']}_seed"))

    # Speed guard: do not seed from every fragment if there are hundreds of valid Hough lines.
    # For this project the central axis is normally found by the most central/longest fragments.
    max_line_seed_count = int(cfg("axis_fit", "max_line_seed_count", default=90))

    if prior is not None:
        seed_lines = sorted(
            lines,
            key=lambda l: (
                abs(line_x_at_y(prior, l["y_mid"]) - l["x_mid"]),
                -l["length"],
            ),
        )
    else:
        seed_lines = sorted(lines, key=lambda l: l["length"], reverse=True)

    if not cfg("axis_fit", "seed_every_fragment", default=True):
        seed_lines = []
    else:
        seed_lines = seed_lines[:max_line_seed_count]

    # Rebuild line seeds with the speed-limited list. Keep the prior seed already appended above.
    seeds = seeds[:1] if (prior is not None and cfg("axis_fit", "include_prior_seed", default=True)) else []

    for line in seed_lines:
        if abs(line["signed_tilt_deg"]) > max_seed_tilt:
            continue
        if prior is not None:
            d = abs(line_x_at_y(prior, line["y_mid"]) - line["x_mid"])
            if d > max_seed_distance:
                continue
        seeds.append(({"a": line["a"], "b": line["b"], "tilt_deg": line["signed_tilt_deg"]}, f"line_{line['line_index']}_seed"))

    # Pair seeds from lines close in x but separated in y can catch interrupted central axis.
    # This was the slowest part, so it is now configurable.
    if cfg("axis_fit", "use_pair_seeds", default=True):
        max_pair_seed_lines = int(cfg("axis_fit", "max_pair_seed_lines", default=32))
        max_pair_neighbors = int(cfg("axis_fit", "max_pair_neighbors", default=8))

        if prior is not None:
            sorted_lines = sorted(
                lines,
                key=lambda l: (
                    abs(line_x_at_y(prior, l["y_mid"]) - l["x_mid"]),
                    -l["length"],
                ),
            )[:max_pair_seed_lines]
        else:
            sorted_lines = sorted(lines, key=lambda l: l["length"], reverse=True)[:max_pair_seed_lines]

        for i, line_a in enumerate(sorted_lines):
            for line_b in sorted_lines[i + 1:i + 1 + max_pair_neighbors]:
                if abs(line_a["signed_tilt_deg"] - line_b["signed_tilt_deg"]) > cfg("axis_fit", "max_tilt_difference_deg", default=10.0):
                    continue
                pts = [
                    (line_a["y_mid"], line_a["x_mid"], line_a["line_index"]),
                    (line_b["y_mid"], line_b["x_mid"], line_b["line_index"]),
                ]
                fit = weighted_huber_fit(pts)
                if fit is None:
                    continue
                if abs(fit["tilt_deg"]) <= max_seed_tilt:
                    if prior is not None:
                        mid_y = (line_a["y_mid"] + line_b["y_mid"]) / 2.0
                        if abs(line_x_at_y(prior, mid_y) - line_x_at_y(fit, mid_y)) > max_seed_distance:
                            continue
                    seeds.append((fit, f"pair_{line_a['line_index']}_{line_b['line_index']}_seed"))

    return seeds


def fit_central_axis(lines, image_height, prior, roi_mask=None):
    seeds = create_seed_axes(lines, prior)
    candidates = []
    rejected_candidates = []

    for seed_axis, seed_label in seeds:
        candidate = build_candidate(seed_axis, lines, image_height, prior, seed_label, roi_mask=roi_mask)
        if candidate is None:
            continue
        ok, reasons = candidate_acceptance(candidate, prior)
        candidate["accepted"] = bool(ok)
        candidate["reject_reasons"] = reasons
        if ok:
            candidates.append(candidate)
        else:
            rejected_candidates.append(candidate)

    candidates = deduplicate_candidates(candidates)
    max_candidates = cfg("candidate_acceptance", "max_candidates_per_image", default=6)
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)[:max_candidates]

    fallback_candidate = None
    if not candidates and rejected_candidates and cfg("candidate_acceptance", "always_keep_best_fallback", default=True):
        fallback_candidate = sorted(rejected_candidates, key=lambda c: c["score"], reverse=True)[0]
        fallback_candidate["fallback_used"] = True
        candidates = [fallback_candidate]

    for i, candidate in enumerate(candidates):
        candidate["candidate_id"] = i

    best = candidates[0] if candidates else None
    return {
        "candidates": candidates,
        "best_candidate": best,
        "accepted_candidate_count": sum(1 for c in candidates if c.get("accepted")),
        "rejected_candidate_count": len(rejected_candidates),
        "seed_count": len(seeds),
    }


def load_base_image(data):
    source = resolve_project_path(data.get("source_file"))
    if source is not None and source.exists():
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    image_name = data.get("image_name")
    for path in [
        WORKING_PNG_DIR / image_name,
        PROCESSED_DIR / "03_edges" / "cleaned" / image_name,
        PROCESSED_DIR / "02_grayscale_blur" / "gaussian_blur" / image_name,
        PROCESSED_DIR / "02_grayscale_blur" / "bilateral_filter" / image_name,
    ]:
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    h = int(data.get("height", 4032))
    w = int(data.get("width", 3024))
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_line(img, line, color, thickness):
    p1 = (int(round(line["x1"])), int(round(line["y1"])))
    p2 = (int(round(line["x2"])), int(round(line["y2"])))
    cv2.line(img, p1, p2, color, int(thickness), cv2.LINE_AA)


def draw_dashed_segment(img, p1, p2, color, thickness=1, dash=28, gap=16):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = int(math.hypot(dx, dy))
    if length <= 0:
        return
    vx, vy = dx / length, dy / length
    pos = 0
    while pos < length:
        end = min(pos + dash, length)
        s = (int(round(x1 + vx * pos)), int(round(y1 + vy * pos)))
        e = (int(round(x1 + vx * end)), int(round(y1 + vy * end)))
        cv2.line(img, s, e, color, thickness, cv2.LINE_AA)
        pos += dash + gap


def draw_axis_for_candidate(img, candidate, color=None, thickness=None):
    axis = candidate["axis_fit"]
    intervals = candidate.get("observed_support_intervals", [])
    gaps = candidate.get("gap_intervals", [])
    thickness = cfg("drawing", "axis_thickness", default=2) if thickness is None else int(thickness)
    color = COLOR_AXIS if color is None else color

    # Draw observed y intervals as solid yellow, gaps as dashed yellow.
    if intervals and cfg("drawing", "draw_gap_intervals", default=True):
        for start, end in intervals:
            y0, y1 = int(round(start)), int(round(end))
            p1 = (int(round(line_x_at_y(axis, y0))), y0)
            p2 = (int(round(line_x_at_y(axis, y1))), y1)
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        for start, end in gaps:
            y0, y1 = int(round(start)), int(round(end))
            p1 = (int(round(line_x_at_y(axis, y0))), y0)
            p2 = (int(round(line_x_at_y(axis, y1))), y1)
            draw_dashed_segment(img, p1, p2, color, thickness=max(1, thickness - 1))
    else:
        y0 = int(round(candidate["y_min"]))
        y1 = int(round(candidate["y_max"]))
        p1 = (int(round(line_x_at_y(axis, y0))), y0)
        p2 = (int(round(line_x_at_y(axis, y1))), y1)
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_candidate_axes(img, candidates):
    if not candidates or not cfg("drawing", "draw_final_axis", default=True):
        return

    base_thickness = cfg("drawing", "axis_thickness", default=2)
    for idx, candidate in enumerate(candidates):
        color = COLOR_CANDIDATE_PALETTE[idx % len(COLOR_CANDIDATE_PALETTE)]
        thickness = base_thickness + 1 if idx == 0 else base_thickness
        draw_axis_for_candidate(img, candidate, color=color, thickness=thickness)

        axis = candidate["axis_fit"]
        label_y = int(round(candidate["y_min"])) + 18
        label_x = int(round(line_x_at_y(axis, candidate["y_min"]))) + 8
        put_text(
            img,
            f"C{idx + 1}:{candidate['score']:.2f}",
            label_x,
            label_y,
            color=color,
            scale=0.55,
        )


def draw_endpoint_bands(img, candidate):
    if candidate is None or not cfg("drawing", "draw_endpoint_bands", default=False):
        return
    endpoint_info = candidate.get("endpoint_centering", {})
    if not endpoint_info.get("enabled"):
        return
    for band_name in ("top", "middle", "bottom"):
        band = endpoint_info.get(band_name, {})
        for row_center in band.get("row_centers", []):
            x = int(round(row_center["center_x"]))
            y = int(row_center["y"])
            cv2.line(img, (x - 5, y), (x + 5, y), COLOR_ENDPOINT, 1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 1, COLOR_ENDPOINT, -1, cv2.LINE_AA)


def draw_prior(img, prior):
    if prior is None or not cfg("drawing", "draw_prior_guide", default=False):
        return
    y0 = int(round(prior["y_min"]))
    y1 = int(round(prior["y_max"]))
    p1 = (int(round(line_x_at_y(prior, y0))), y0)
    p2 = (int(round(line_x_at_y(prior, y1))), y1)
    draw_dashed_segment(img, p1, p2, COLOR_PRIOR, thickness=1, dash=18, gap=16)


def put_text(img, text, x, y, color=COLOR_TEXT, scale=None):
    scale = cfg("drawing", "font_scale", default=0.72) if scale is None else scale
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, 1, cv2.LINE_AA)


def draw_overlay(base_img, filtered_lines, rejected_lines, candidates_or_best, best_candidate=None, prior=None, image_name=""):
    if image_name == "" and isinstance(prior, str):
        candidates = [candidates_or_best] if candidates_or_best is not None else []
        image_name = prior
        prior = best_candidate
        best_candidate = candidates_or_best
    elif best_candidate is None and (prior is None or isinstance(prior, str)):
        candidates = [candidates_or_best] if candidates_or_best is not None else []
        best_candidate = candidates_or_best
        if isinstance(prior, str):
            image_name = prior
            prior = None
    else:
        candidates = candidates_or_best or []

    img = base_img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = float(cfg("drawing", "background_alpha", default=0.68))
    img = cv2.addWeighted(img, alpha, np.zeros_like(img), 1.0 - alpha, 0)

    if cfg("drawing", "draw_all_valid_fragments", default=True):
        for line in filtered_lines:
            draw_line(img, line, COLOR_VALID, cfg("drawing", "valid_fragment_thickness", default=1))

    if cfg("drawing", "draw_rejected_fragments", default=True):
        for line in rejected_lines:
            draw_line(img, line, COLOR_REJECTED, cfg("drawing", "rejected_fragment_thickness", default=2))

    support_ids = set()
    if best_candidate is not None:
        support_ids = set(best_candidate["support_fragment_line_indices"])

    if best_candidate is not None and cfg("drawing", "draw_support_fragments", default=True):
        for line in filtered_lines:
            if line["line_index"] in support_ids:
                draw_line(img, line, COLOR_SUPPORT, cfg("drawing", "support_fragment_thickness", default=3))

    draw_prior(img, prior)

    draw_candidate_axes(img, candidates)
    draw_endpoint_bands(img, best_candidate)

    # Compact status text.
    put_text(img, image_name, 28, 38, COLOR_TEXT, scale=0.9)
    if best_candidate is None:
        put_text(img, "central axis candidates: 0", 28, 76, (0, 0, 255), scale=0.72)
    else:
        status = "ACCEPTED" if best_candidate.get("accepted") else "FALLBACK"
        put_text(
            img,
            f"best axis: {status}  score={best_candidate['score']:.3f}  "
            f"frags={best_candidate['num_support_fragments']}  "
            f"coverage={best_candidate['observed_y_coverage_px']:.0f}px  "
            f"res={best_candidate['axis_fit']['mean_residual_px']:.1f}px  "
            f"tilt={best_candidate['axis_fit']['tilt_deg']:.2f}deg",
            28,
            76,
            COLOR_AXIS if best_candidate.get("accepted") else (0, 180, 255),
            scale=0.72,
        )
        if not best_candidate.get("accepted"):
            put_text(img, "reject: " + ",".join(best_candidate.get("reject_reasons", [])), 28, 110, (0, 180, 255), scale=0.58)

    return img


def create_comparison(base_img, overlay_img):
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    h, w = base_img.shape[:2]
    max_w = 1300
    if w > max_w:
        scale = max_w / w
        new_size = (int(w * scale), int(h * scale))
        base_img = cv2.resize(base_img, new_size, interpolation=cv2.INTER_AREA)
        overlay_img = cv2.resize(overlay_img, new_size, interpolation=cv2.INTER_AREA)
    sep = np.full((base_img.shape[0], 10, 3), 255, dtype=np.uint8)
    return np.hstack([base_img, sep, overlay_img])


def process_json_file(json_path):
    data = load_json(json_path)
    image_name = data.get("image_name", json_path.stem + ".png")
    width = int(data.get("width", 0))
    height = int(data.get("height", 0)) or 4032

    raw_lines = data.get("valid_lines", [])
    lines = [normalize_line(line, i) for i, line in enumerate(raw_lines)]
    filtered_lines, rejected_lines = filter_fragments(lines)

    roi_mask_path = resolve_project_path(data.get("roi_mask_file"))
    roi_mask = load_roi_mask(roi_mask_path)
    prior = robust_roi_axis_prior(roi_mask)

    fit_result = fit_central_axis(filtered_lines, height, prior, roi_mask=roi_mask)
    best = fit_result["best_candidate"]
    candidates = fit_result["candidates"]

    base_img = load_base_image(data)
    overlay = draw_overlay(base_img, filtered_lines, rejected_lines, candidates, best, prior, image_name)

    overlay_path = OUTPUT_OVERLAY_DIR / image_name
    comparison_path = OUTPUT_COMPARISON_DIR / image_name
    metadata_path = OUTPUT_METADATA_DIR / f"{Path(image_name).stem}_central_axis.json"
    comparison_img = create_comparison(base_img, overlay)

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(comparison_path), comparison_img)

    metadata = {
        "image_name": image_name,
        "processing_step": "06_fit_central_axis_from_fragments",
        "source_step": data.get("processing_step", "05_valid_hough_lines_in_roi"),
        "width": width,
        "height": height,
        "input_json_file": str(json_path.relative_to(PROJECT_ROOT)),
        "source_file": data.get("source_file"),
        "roi_mask_file": data.get("roi_mask_file"),
        "input_line_count": len(lines),
        "filtered_line_count": len(filtered_lines),
        "rejected_fragment_count": len(rejected_lines),
        "seed_count": fit_result["seed_count"],
        "candidate_count": len(fit_result["candidates"]),
        "accepted_candidate_count": fit_result["accepted_candidate_count"],
        "best_candidate": best,
        "candidates": fit_result["candidates"],
        "roi_axis_prior": prior,
        "parameters": STEP_CONFIG,
        "output_overlay_file": str(overlay_path.relative_to(PROJECT_ROOT)),
        "output_comparison_file": str(comparison_path.relative_to(PROJECT_ROOT)),
    }
    save_json(metadata_path, metadata)

    endpoint_info = (best or {}).get("endpoint_centering", {})
    top = endpoint_info.get("top", {})
    middle = endpoint_info.get("middle", {})
    bottom = endpoint_info.get("bottom", {})

    return {
        "image_name": image_name,
        "input_line_count": len(lines),
        "filtered_line_count": len(filtered_lines),
        "rejected_fragment_count": len(rejected_lines),
        "candidate_count": len(fit_result["candidates"]),
        "accepted_candidate_count": fit_result["accepted_candidate_count"],
        "best_score": best["score"] if best else None,
        "best_tilt_deg": best["axis_fit"]["tilt_deg"] if best else None,
        "top_center_error_px": top.get("median_center_error_px"),
        "middle_center_error_px": middle.get("median_center_error_px"),
        "bottom_center_error_px": bottom.get("median_center_error_px"),
        "endpoint_alignment_score": endpoint_info.get("endpoint_alignment_score"),
        "band_balance_score": endpoint_info.get("band_balance_score"),
        "top_bottom_error_difference_px": endpoint_info.get("top_bottom_error_difference_px"),
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
        "comparison_path": str(comparison_path.relative_to(PROJECT_ROOT)),
    }


def collect_json_files(image_filter=None, limit=None):
    if not INPUT_JSON_DIR.exists():
        raise FileNotFoundError(f"Input JSON dir does not exist: {INPUT_JSON_DIR}")
    files = sorted(INPUT_JSON_DIR.glob("*.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [
            p for p in files
            if p.stem.lower() == wanted
            or p.stem.lower().replace("_lines", "") == wanted
            or wanted in p.stem.lower()
        ]
    if limit is not None:
        files = files[:limit]
    return files


def show_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return
    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(path.name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    global STEP_CONFIG, OUTPUT_SUBDIR, OUTPUT_DIR, OUTPUT_OVERLAY_DIR, OUTPUT_METADATA_DIR, OUTPUT_COMPARISON_DIR

    parser = argparse.ArgumentParser(description="Step 06: robustly fit central boot axis from Hough fragments.")
    parser.add_argument("--image", type=str, default=None, help="Optional image name filter, e.g. IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None, help="Preset from step_06_fit_central_axis_from_fragments.test_presets")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    STEP_CONFIG = apply_preset(STEP_CONFIG, args.preset)
    OUTPUT_SUBDIR = STEP_CONFIG.get("output_subdir", "06_fit_central_axis_from_fragments")
    OUTPUT_DIR = PROCESSED_DIR / OUTPUT_SUBDIR
    OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "overlay"
    OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
    OUTPUT_COMPARISON_DIR = OUTPUT_DIR / "comparison"

    ensure_dirs(cleanup=bool(STEP_CONFIG.get("cleanup_output_on_start", True)) and args.image is None)

    json_files = collect_json_files(args.image, args.limit)
    if not json_files:
        print("No JSON files found.")
        print(f"Input dir: {INPUT_JSON_DIR}")
        return

    print(f"Step 06 input dir: {INPUT_JSON_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found JSON files: {len(json_files)}")

    summary = []
    for json_path in json_files:
        print(f"\nProcessing: {json_path.name}")
        try:
            result = process_json_file(json_path)
            summary.append(result)
            score_txt = f"{result['best_score']:.3f}" if result["best_score"] is not None else "none"
            tilt_txt = f"{result['best_tilt_deg']:.2f}" if result["best_tilt_deg"] is not None else "none"
            print(
                f"  lines input={result['input_line_count']} filtered={result['filtered_line_count']} "
                f"rejected={result['rejected_fragment_count']} candidates={result['candidate_count']} "
                f"accepted={result['accepted_candidate_count']} score={score_txt} tilt={tilt_txt}"
            )
            print(f"  overlay: {result['overlay_path']}")
            print(f"  metadata: {result['metadata_path']}")
            if args.show:
                show_image(PROJECT_ROOT / result["comparison_path"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = OUTPUT_DIR / "step_06_summary.json"
    save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
