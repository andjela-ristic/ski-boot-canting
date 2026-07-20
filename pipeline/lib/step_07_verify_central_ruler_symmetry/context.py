from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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
STEP_06_CONFIG = CONFIG.get("step_06_search_central_ruler", {})
STEP_CONFIG_RAW = CONFIG.get("step_07_verify_central_ruler_symmetry", {})


DEFAULT_STEP_CONFIG = {
    "enabled": True,
    "inherit_step_06_output": True,
    "input_subdir": "06_search_central_ruler",
    "input_metadata_subdir": "metadata",
    "output_subdir": "07_verify_central_ruler_symmetry",
    "cleanup_output_on_start": True,
    "candidate_limit": 10,
    "segment_count": 12,
    "vertical_range": {
        "use_step_06_trimmed_range": True,
        "trim_top_ratio": 0.02,
        "trim_bottom_ratio": 0.02,
    },
    "roi_preparation": {
        "keep_largest_component": True,
        "close_kernel_size": 7,
        "close_iterations": 1,
    },
    "consensus_corridor": {
        "candidate_score_power": 2.0,
        "reference_half_width_ratio": 0.42,
        "row_half_width_quantile": 0.72,
        "row_half_width_cap_scale": 1.05,
        "minimum_half_width_px": 80,
        "maximum_half_width_image_ratio": 0.30,
        "edge_margin_px": 8,
        "smooth_window": 51,
        "max_normalized_candidate_offset": 0.60,
        "min_axis_inside_corridor_ratio": 0.82,
        "absolute_max_tilt_delta_deg": 5.0,
        "tilt_mad_multiplier": 6.0,
        "tilt_mad_floor_deg": 0.75,
    },
    "edge_input": {
        "threshold": 24,
        "binary_nonzero_ratio_max": 0.20,
        "canny_low": 45,
        "canny_high": 135,
        "close_kernel_size": 3,
        "close_iterations": 0,
        "remove_border_components": True,
        "border_component_min_height_ratio": 0.80,
        "border_component_max_center_distance_ratio": 0.42,
    },
    "mirror": {
        "center_exclusion_px": 4,
        "max_chamfer_distance_px": 16.0,
        "distance_score_scale_px": 7.0,
        "min_edge_pixels_per_side": 14,
        "min_edge_rows_per_side_ratio": 0.16,
        "histogram_smoothing_sigma": 2.0,
        "band_boundaries": [0.0, 0.22, 0.52, 1.0],
        "band_weights": [0.46, 0.39, 0.15],
        "band_match_fractions": [0.90, 0.76, 0.58],
        "chamfer_weight": 0.64,
        "radial_histogram_weight": 0.24,
        "edge_count_balance_weight": 0.12,
        "coverage_reliability_power": 0.20,
    },
    "segment_aggregation": {
        "min_valid_segments": 8,
        "min_valid_segments_per_zone": 2,
        "global_weighted_mean_weight": 0.55,
        "global_weighted_median_weight": 0.45,
        "global_score_weight": 0.55,
        "zone_harmonic_weight": 0.45,
        "zone_weights": [0.30, 0.35, 0.35],
    },
    "final_scoring": {
        "mirror_symmetry_weight": 0.72,
        "bilateral_coverage_weight": 0.10,
        "consensus_centrality_weight": 0.08,
        "step_06_prior_weight": 0.10,
    },
    "confidence": {
        "high_min_verification_percent": 72.0,
        "high_min_margin_percent": 2.0,
        "medium_min_verification_percent": 60.0,
        "medium_min_margin_percent": 0.8,
    },
    "drawing": {
        "axis_thickness": 4,
        "other_axis_thickness": 1,
        "corridor_boundary_thickness": 2,
        "segment_line_thickness": 1,
        "font_scale": 0.62,
        "background_alpha": 0.82,
    },
    "test_presets": [
        {"name": "default", "override": {}},
        {
            "name": "edges_only_no_prior",
            "override": {
                "final_scoring": {
                    "mirror_symmetry_weight": 0.84,
                    "bilateral_coverage_weight": 0.10,
                    "consensus_centrality_weight": 0.06,
                    "step_06_prior_weight": 0.0,
                }
            },
        },
    ],
}

COLOR_WINNER = (255, 80, 30)
COLOR_OTHER = (100, 100, 100)
COLOR_CORRIDOR = (210, 210, 210)
COLOR_SEGMENT_GOOD = (70, 210, 70)
COLOR_SEGMENT_MEDIUM = (0, 210, 255)
COLOR_SEGMENT_BAD = (70, 70, 230)
COLOR_TEXT = (245, 245, 245)
COLOR_LEFT = (70, 220, 70)
COLOR_RIGHT = (220, 70, 220)
COLOR_OVERLAP = (245, 245, 245)


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
    step_06_output = str(STEP_06_CONFIG.get("output_subdir", "06_search_central_ruler"))
    input_subdir = (
        step_06_output
        if bool(STEP_CONFIG.get("inherit_step_06_output", True))
        else str(STEP_CONFIG.get("input_subdir", step_06_output))
    )
    output_subdir = str(STEP_CONFIG.get("output_subdir", "07_verify_central_ruler_symmetry"))
    input_dir = PROCESSED_DIR / input_subdir
    output_dir = PROCESSED_DIR / output_subdir
    return {
        "input_dir": input_dir,
        "input_metadata_dir": input_dir / str(STEP_CONFIG.get("input_metadata_subdir", "metadata")),
        "output_dir": output_dir,
        "output_overlay_dir": output_dir / "overlay",
        "output_comparison_dir": output_dir / "comparison",
        "output_metadata_dir": output_dir / "metadata",
        "output_candidate_snapshot_dir": output_dir / "candidate_snapshots",
        "output_rectified_dir": output_dir / "rectified",
    }


@dataclass(frozen=True)
class VerificationContext:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    processed_dir: Path = field(default_factory=lambda: PROCESSED_DIR)
    working_png_dir: Path = field(default_factory=lambda: WORKING_PNG_DIR)
    display_config: dict = field(default_factory=lambda: deepcopy(DISPLAY_CONFIG))
    step_config: dict = field(default_factory=lambda: deepcopy(STEP_CONFIG))
    step_dirs: dict[str, Path] = field(default_factory=get_step_dirs)


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


def ensure_binary(image: np.ndarray) -> np.ndarray:
    return np.where(image > 0, 255, 0).astype(np.uint8)


def load_grayscale(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def load_color(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)
