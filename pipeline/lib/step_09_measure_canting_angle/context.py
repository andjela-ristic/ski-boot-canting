from __future__ import annotations

from copy import deepcopy
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from config_loader import load_config
except ModuleNotFoundError:
    try:
        from ...config_loader import load_config
    except (ModuleNotFoundError, ImportError):
        def load_config() -> dict:
            return {
                "paths": {
                    "working_png_dir": "data/working_png",
                    "processed_dir": "data/processed",
                },
                "display": {"max_height": 900},
            }


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config()
PATHS_CONFIG = CONFIG.get("paths", {})
DISPLAY_CONFIG = CONFIG.get("display", {})
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG.get("processed_dir", "data/processed")
WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG.get("working_png_dir", "data/working_png")
STEP_08_CONFIG = CONFIG.get("step_08_multi_validate_central_ruler", {})
STEP_CONFIG_RAW = CONFIG.get("step_09_measure_canting_angle", {})


DEFAULT_STEP_CONFIG = {
    "enabled": True,
    "inherit_step_08_output": True,
    "input_subdir": "08_multi_validate_central_ruler",
    "input_metadata_subdir": "metadata",
    "cleaned_edge_subdir": "03_edges/cleaned",
    "output_subdir": "09_measure_canting_angle",
    "cleanup_output_on_start": True,
    "evaluation_mask": {
        "prefer_step_07_evaluation_mask": True,
        "require_step_07_evaluation_mask": False,
        "dilate_kernel_width_ratio": 0.030,
        "dilate_kernel_height_ratio": 0.010,
        "dilate_iterations": 1,
        "fallback_axis_half_width_ratio": 0.18,
        "fallback_vertical_margin_ratio": 0.04,
    },
    "table_search": {
        # The physical table edge normally crosses the lower part of the boot.
        # The interval is relative to the Step 08 boot/evaluation vertical range.
        "primary_vertical_range_ratio": [0.68, 0.96],
        "fallback_vertical_range_ratio": [0.58, 1.02],
        "max_abs_angle_deg": 5.0,
        "hough_rho_px": 1.0,
        "hough_theta_deg": 0.05,
        "hough_threshold": 25,
        "hough_detection_scale": 0.50,
        "min_line_length_ratio": 0.023,
        "max_line_gap_ratio": 0.012,
        "cluster_angle_tolerance_deg": 0.80,
        "cluster_y_tolerance_px": 11.0,
        "cluster_merge_gap_px": 7.0,
        "minimum_cluster_segments": 2,
        "max_segments": 700,
        "max_clusters_to_refine": 6,
        "minimum_total_coverage_ratio": 0.34,
        "minimum_left_coverage_ratio": 0.18,
        "minimum_right_coverage_ratio": 0.18,
        "minimum_bilateral_balance": 0.42,
        "location_center_ratio": 0.82,
        "location_sigma_ratio": 0.095,
    },
    "line_refinement": {
        "support_band_px": 5.0,
        "max_iterations": 5,
        "residual_keep_px": 7.0,
        "minimum_support_pixels": 180,
        "x_bin_count": 96,
        "minimum_occupied_bin_ratio": 0.30,
        "fit_rmse_scale_px": 4.5,
        "fit_p90_scale_px": 7.0,
    },
    "table_score": {
        "total_coverage_weight": 0.22,
        "bilateral_coverage_weight": 0.20,
        "left_right_balance_weight": 0.10,
        "edge_support_weight": 0.18,
        "fit_quality_weight": 0.14,
        "location_prior_weight": 0.11,
        "horizontal_prior_weight": 0.05,
    },
    "stability": {
        "enabled": True,
        "equivalent_angle_tolerance_deg": 0.22,
        "equivalent_y_tolerance_px": 14.0,
        "angle_std_scale_deg": 0.30,
        "y_std_scale_px": 18.0,
        "variants": [
            {
                "name": "default",
                "override": {},
            },
            {
                "name": "conservative",
                "override": {
                    "hough_threshold": 36,
                    "min_line_length_ratio": 0.032,
                    "max_line_gap_ratio": 0.009,
                },
            },
            {
                "name": "permissive",
                "override": {
                    "hough_threshold": 18,
                    "min_line_length_ratio": 0.017,
                    "max_line_gap_ratio": 0.016,
                },
            },
            {
                "name": "narrow_vertical_band",
                "override": {
                    "vertical_range_ratio": [0.72, 0.92],
                },
            },
            {
                "name": "wide_vertical_band",
                "override": {
                    "vertical_range_ratio": [0.62, 1.00],
                },
            },
        ],
    },
    "axis_quality": {
        # These weights combine distinct Step 08 diagnostics. Missing values are
        # omitted and the remaining weights are renormalized.
        "symmetry_weight": 0.20,
        "multi_validation_weight": 0.20,
        "step08_confidence_weight": 0.11,
        "winner_validator_evidence_weight": 0.17,
        "validator_agreement_weight": 0.11,
        "ensemble_stability_weight": 0.10,
        "validator_availability_weight": 0.05,
        "distinct_margin_weight": 0.06,
        "step07_disagreement_factor": 0.90,
        "decision_factors": {
            "accepted": 1.00,
            "accepted_low_confidence": 0.96,
            "manual_review": 0.90,
            "recapture_required": 0.82,
        },
    },
    "measurement_confidence": {
        "axis_quality_weight": 0.43,
        "table_quality_weight": 0.39,
        "joint_stability_weight": 0.12,
        "joint_margin_weight": 0.06,
        "accepted_min_percent": 82.0,
        "accepted_low_confidence_min_percent": 68.0,
        "manual_review_min_percent": 52.0,
        "reference_line_min_percent": 52.0,
        "max_accepted_uncertainty_deg": 0.55,
        "calibration_status": "uncalibrated_estimated_measurement_confidence",
    },
    "direction": {
        "neutral_threshold_deg": 0.05,
        "positive_label": "right",
        "negative_label": "left",
        "neutral_label": "neutral",
    },
    "drawing": {
        "axis_thickness": 4,
        "table_line_thickness": 4,
        "normal_thickness": 2,
        "support_point_radius": 1,
        "font_scale": 0.62,
        "comparison_panel_width": 680,
    },
    "test_presets": [
        {"name": "default", "override": {}},
        {
            "name": "strict_table",
            "override": {
                "table_search": {
                    "minimum_total_coverage_ratio": 0.42,
                    "minimum_left_coverage_ratio": 0.24,
                    "minimum_right_coverage_ratio": 0.24,
                },
                "measurement_confidence": {
                    "reference_line_min_percent": 65.0,
                },
            },
        },
    ],
}


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


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def get_step_dirs() -> dict[str, Path]:
    step08_output = str(STEP_08_CONFIG.get("output_subdir", "08_multi_validate_central_ruler"))
    input_subdir = (
        step08_output
        if bool(STEP_CONFIG.get("inherit_step_08_output", True))
        else str(STEP_CONFIG.get("input_subdir", step08_output))
    )
    output_subdir = str(STEP_CONFIG.get("output_subdir", "09_measure_canting_angle"))
    input_dir = PROCESSED_DIR / input_subdir
    output_dir = PROCESSED_DIR / output_subdir
    return {
        "input_dir": input_dir,
        "input_metadata_dir": input_dir / str(STEP_CONFIG.get("input_metadata_subdir", "metadata")),
        "cleaned_edge_dir": PROCESSED_DIR / str(STEP_CONFIG.get("cleaned_edge_subdir", "03_edges/cleaned")),
        "output_dir": output_dir,
        "output_overlay_dir": output_dir / "overlay",
        "output_comparison_dir": output_dir / "comparison",
        "output_metadata_dir": output_dir / "metadata",
        "output_diagnostics_dir": output_dir / "diagnostics",
    }


def ensure_dirs(cleanup: bool = False) -> None:
    dirs = get_step_dirs()
    if cleanup and dirs["output_dir"].exists():
        shutil.rmtree(dirs["output_dir"])
    for key, path in dirs.items():
        if key.startswith("output_"):
            path.mkdir(parents=True, exist_ok=True)


def normalize_path_value(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    text = text.replace("\\", os.sep).replace("/", os.sep)
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_grayscale(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def load_color(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)
