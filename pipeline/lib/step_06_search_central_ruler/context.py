from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
import os
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

DEFAULT_STEP_CONFIG = {}


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_json(data: object) -> str:
    return sha256_bytes(canonical_json_bytes(data))


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_project_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_pipeline_config_path() -> Path:
    config_path = os.environ.get("PIPELINE_CONFIG")
    if config_path:
        return resolve_project_path(config_path) or (PROJECT_ROOT / config_path)
    return PROJECT_ROOT / "config" / "pipeline_config.yaml"

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
