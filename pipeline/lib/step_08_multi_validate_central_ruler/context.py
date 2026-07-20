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
STEP_07_CONFIG = CONFIG.get("step_07_verify_central_ruler_symmetry", {})
STEP_CONFIG_RAW = CONFIG.get("step_08_multi_validate_central_ruler", {})


DEFAULT_STEP_CONFIG = {
    "enabled": True,
    "inherit_step_07_output": True,
    "input_subdir": "07_verify_central_ruler_symmetry",
    "input_metadata_subdir": "metadata",
    "output_subdir": "08_multi_validate_central_ruler",
    "cleanup_output_on_start": True,
    "candidate_limit": 10,
    "roi_preparation": {
        "keep_largest_component": True,
        "close_kernel_size": 7,
        "close_iterations": 1,
        "row_smoothing_window": 41,
        "row_width_cap_quantile": 0.72,
        "row_width_cap_scale": 1.08,
        "minimum_half_width_px": 55,
        "sample_row_step_px": 6,
    },
    "medial_axis": {
        "enabled": True,
        "minimum_peak_radius_px": 4.0,
        "huber_delta_px": 16.0,
        "huber_iterations": 6,
        "normalized_distance_sigma": 0.085,
        "tilt_difference_sigma_deg": 2.5,
        "distance_weight": 0.82,
        "tilt_weight": 0.18,
        "minimum_valid_row_ratio": 0.55,
    },
    "structural_anchors": {
        "enabled": True,
        "zones": [
            {"name": "top", "start_ratio": 0.08, "end_ratio": 0.30},
            {"name": "middle", "start_ratio": 0.36, "end_ratio": 0.66},
            {"name": "bottom", "start_ratio": 0.76, "end_ratio": 0.97},
        ],
        "normalized_distance_sigma": 0.11,
        "minimum_valid_rows_per_zone": 8,
    },
    "fragment_evidence": {
        "enabled": True,
        "span_saturation": 0.65,
        "coverage_saturation": 0.35,
        "minimum_available_metrics": 3,
    },
    "roi_balance": {
        "enabled": True,
        "minimum_axis_inside_ratio": 0.70,
        "valid_ratio_power": 0.35,
    },
    "fusion": {
        "validator_names": [
            "step_07_symmetry",
            "medial_axis",
            "structural_anchors",
            "fragment_evidence",
            "roi_balance",
        ],
        "minimum_available_validators": 4,
        "absolute_score_floor": 0.0001,
        "invalid_candidate_scale": 0.45,
        "require_step_07_valid": True,
    },
    "equivalence": {
        "max_mean_axis_distance_px": 5.0,
        "max_tilt_difference_deg": 0.25,
        "sample_count": 24,
    },
    "confidence": {
        "distinct_margin_saturation": 0.10,
        "accepted_min_percent": 82.0,
        "accepted_low_confidence_min_percent": 68.0,
        "manual_review_min_percent": 52.0,
    },
    "drawing": {
        "axis_thickness": 4,
        "other_axis_thickness": 1,
        "medial_axis_thickness": 2,
        "anchor_radius": 9,
        "font_scale": 0.60,
        "background_alpha": 0.84,
    },
    "test_presets": [
        {"name": "default", "override": {}},
        {
            "name": "no_fragment_evidence",
            "override": {
                "fragment_evidence": {"enabled": False},
                "fusion": {
                    "validator_names": [
                        "step_07_symmetry",
                        "medial_axis",
                        "structural_anchors",
                        "roi_balance",
                    ],
                    "minimum_available_validators": 3,
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
    step_07_output = str(STEP_07_CONFIG.get("output_subdir", "07_verify_central_ruler_symmetry"))
    input_subdir = (
        step_07_output
        if bool(STEP_CONFIG.get("inherit_step_07_output", True))
        else str(STEP_CONFIG.get("input_subdir", step_07_output))
    )
    output_subdir = str(STEP_CONFIG.get("output_subdir", "08_multi_validate_central_ruler"))
    input_dir = PROCESSED_DIR / input_subdir
    output_dir = PROCESSED_DIR / output_subdir
    return {
        "input_dir": input_dir,
        "input_metadata_dir": input_dir / str(STEP_CONFIG.get("input_metadata_subdir", "metadata")),
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
