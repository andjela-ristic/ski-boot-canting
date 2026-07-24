from __future__ import annotations

import os
from pathlib import Path

import cv2

from ...config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]
STEP_CONFIG = CONFIG["step_04_boot_roi_from_edges"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

if bool(STEP_CONFIG.get("inherit_step_03_output", True)): INPUT_ROOT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]
else: INPUT_ROOT_DIR = PROCESSED_DIR / STEP_CONFIG["input_subdir"]

SELECTED_INPUT = STEP_CONFIG.get("selected_input")
INPUT_DIR = INPUT_ROOT_DIR / str(SELECTED_INPUT).strip() if SELECTED_INPUT is not None else INPUT_ROOT_DIR
OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

MASK_DIR = OUTPUT_DIR / "mask"
OVERLAY_DIR = OUTPUT_DIR / "overlay"
SELECTED_COMPONENT_DIR = OUTPUT_DIR / "selected_component"
COMPARISON_DIR = OUTPUT_DIR / "comparison"

CSV_PATH = METADATA_DIR / "processing_04_boot_roi_from_edges.csv"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CSV_FIELDNAMES = [
    "source_file",
    "mask_output_file",
    "overlay_output_file",
    "selected_component_output_file",
    "comparison_output_file",
    "width",
    "height",
    "processing_step",
    "input_from_step_03",
    "edge_threshold",
    "min_component_area",
    "density_kernel_size",
    "density_threshold_percentile",
    "density_min_threshold",
    "close_kernel_size",
    "dilate_kernel_size",
    "second_close_kernel_size",
    "smooth_kernel_size",
    "iterations_close",
    "iterations_dilate",
    "iterations_second_close",
    "hull_enabled",
    "hull_mode",
    "debug_enabled",
    "density_threshold",
    "roi_pixels",
    "read_time_ms",
    "processing_time_ms",
    "write_time_ms",
    "total_time_ms",
]

# too many OpenCV workers slow this step down
OPENCV_THREADS = max(1, min(4, os.cpu_count() or 1))
cv2.setUseOptimized(True)
cv2.setNumThreads(OPENCV_THREADS)

def refresh_context() -> None:
    global INPUT_ROOT_DIR, SELECTED_INPUT, INPUT_DIR, OUTPUT_DIR, MASK_DIR, OVERLAY_DIR, SELECTED_COMPONENT_DIR, COMPARISON_DIR
    if bool(STEP_CONFIG.get("inherit_step_03_output", True)): INPUT_ROOT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]
    else: INPUT_ROOT_DIR = PROCESSED_DIR / STEP_CONFIG["input_subdir"]

    SELECTED_INPUT = STEP_CONFIG.get("selected_input")
    INPUT_DIR = INPUT_ROOT_DIR / str(SELECTED_INPUT).strip() if SELECTED_INPUT is not None else INPUT_ROOT_DIR
    OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]
    MASK_DIR = OUTPUT_DIR / "mask"
    OVERLAY_DIR = OUTPUT_DIR / "overlay"
    SELECTED_COMPONENT_DIR = OUTPUT_DIR / "selected_component"
    COMPARISON_DIR = OUTPUT_DIR / "comparison"

def set_step_config(step_config: dict) -> None:
    global STEP_CONFIG
    from . import runtime
    STEP_CONFIG = step_config
    refresh_context()
    runtime.refresh_runtime()
