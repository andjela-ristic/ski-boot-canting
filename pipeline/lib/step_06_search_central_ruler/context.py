from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
import shutil
import time
import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

try:
    from config_loader import load_config
except ModuleNotFoundError:
    from ...config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
        "coarse_candidate_pool_limit": 120,
        "fine_candidate_pool_limit": 144,
        "coarse_angle_bucket_deg": 1.0,
        "max_coarse_candidates_per_angle_bucket": 3,
        "coarse_x_bucket_px": 18.0,
        "max_coarse_candidates_per_x_bucket": 3,
        "top_coarse_candidates": 24,
        "coarse_score_floor_ratio": 0.97,
        "fine_score_floor_ratio": 0.98,
        "fine_window_x_px": 40,
        "fine_window_angle_deg": 2.2,
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
    "endpoint_rescue": {
        "enabled": True,
        "margin_px": 180.0,
        "max_fragments_per_side": 2,
        "max_mean_axis_distance_px": 16.0,
        "max_min_axis_distance_px": 2.5,
        "angle_slack_deg": 0.75,
        "support_strength_scale": 0.84,
    },
    "final_fit": {
        "sample_step_px": 22,
        "huber_delta_px": 10.0,
        "huber_iterations": 4,
        "max_fit_tilt_deg": 12.0,
        "endpoint_sample_weight_boost": 1.7,
        "original_endpoint_sample_weight_boost": 2.2,
    },
    "best_fit_selection": {
        "top_hypothesis_count": 24,
        "min_anchor_band_coverage": 0.18,
        "min_anchor_overlap_px": 24.0,
        "min_anchor_fragment_ratio": 0.78,
        "minimal_symmetry": {
            "min_side_ratio": 0.035,
            "min_clearance_row_ratio": 0.65,
            "target_side_ratio": 0.12,
        },
        "formula_buckets": {
            "longest_interval_px": 20.0,
            "continuity_ratio_scale": 20.0,
        },
        "formula_weights": {
            "has_top_bottom_anchor": 1000.0,
            "has_top_anchor": 160.0,
            "has_bottom_anchor": 160.0,
            "paired_anchor_strength": 340.0,
            "paired_endpoint_coverage": 220.0,
            "longest_interval_bucket": 22.0,
            "continuity_bucket": 14.0,
            "longest_interval_px": 0.08,
            "chain_continuity_ratio": 14.0,
            "has_top_bottom_original_anchor": 520.0,
            "has_top_original_anchor": 110.0,
            "has_bottom_original_anchor": 110.0,
            "paired_original_anchor_strength": 260.0,
            "paired_original_endpoint_coverage": 180.0,
            "endpoint_anchor_score": 0.10,
            "has_min_side_clearance": 180.0,
            "side_clearance_row_ratio": 120.0,
            "side_clearance_score": 90.0,
            "outside_chain_length_ratio": 140.0,
            "outside_chain_fragment_ratio": 80.0,
            "total_reach_gap_px": 0.10,
            "chain_total_gap_px": 0.03,
            "gap_penalty": 0.12,
            "merged_interval_count": 4.0,
            "adjusted_fragment_ratio": 80.0,
            "support_adjustment_penalty": 0.18,
            "length_weighted_mean_abs_support_shift_px": 0.08,
            "max_abs_support_shift_px": 0.12,
            "outside_mask_penalty": 0.08,
            "hypothesis_x_ref_delta_px": 8.0,
            "hypothesis_tilt_delta_deg": 24.0,
            "score": 60.0,
        },
        "axis_harmonization": {
            "enabled": True,
            "min_x_ref_delta_px": 0.75,
            "min_tilt_delta_deg": 0.05,
            "hypothesis_pull_ratio": 0.35,
        },
    },
    "support_chain": {
        "enabled": True,
        "max_connection_gap_px": 220.0,
        "max_connection_dx_px": 22.0,
        "max_angle_difference_deg": 3.0,
    },
    "support_adjustment": {
        "enabled": True,
        "max_midpoint_shift_px": 6.0,
        "max_tilt_delta_deg": 1.5,
        "max_mean_shift_px": 6.0,
        "max_endpoint_shift_px": 9.0,
        "allow_tilt_without_intersection_axis_distance_px": 8.0,
        "require_axis_intersection_for_tilt_adjustment": True,
        "min_support_strength_scale": 0.68,
        "full_adjustment_top_hypotheses": 8,
        "joint_adjustment_top_hypotheses": 4,
        "joint_adjustment_min_original_endpoint_ratio": 0.90,
        "joint_adjustment_min_original_endpoint_coverage": 0.24,
    },
    "support_extension": {
        "enabled": True,
        "trigger_gap_px": 180,
        "max_added_fragments": 6,
        "min_vertical_advance_px": 18,
        "max_connection_dx_px": 26.0,
        "max_connection_gap_px": 360.0,
        "max_axis_distance_px": 55.0,
        "max_center_distance_px": 70.0,
        "max_vertical_deviation_deg": 10.0,
    },
    "candidate_deduplication": {
        "max_mean_axis_distance_px": 5.0,
        "max_angle_difference_deg": 0.25,
        "max_saved_candidates": 10,
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
        "output_candidate_snapshot_dir": output_dir / "candidate_snapshots",
    }


@dataclass(frozen=True)
class SearchContext:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    processed_dir: Path = field(default_factory=lambda: PROCESSED_DIR)
    working_png_dir: Path = field(default_factory=lambda: WORKING_PNG_DIR)
    display_config: dict = field(default_factory=lambda: deepcopy(DISPLAY_CONFIG))
    step_config: dict = field(default_factory=lambda: deepcopy(STEP_CONFIG))
    step_dirs: dict[str, Path] = field(default_factory=get_step_dirs)

def ensure_dirs(cleanup: bool = False) -> None:
    dirs = get_step_dirs()
    output_dir = dirs["output_dir"]
    if cleanup and output_dir.exists():
        shutil.rmtree(output_dir)
    dirs["output_overlay_dir"].mkdir(parents=True, exist_ok=True)
    dirs["output_comparison_dir"].mkdir(parents=True, exist_ok=True)
    dirs["output_metadata_dir"].mkdir(parents=True, exist_ok=True)
    dirs["output_candidate_snapshot_dir"].mkdir(parents=True, exist_ok=True)

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
    value = float(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value

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
