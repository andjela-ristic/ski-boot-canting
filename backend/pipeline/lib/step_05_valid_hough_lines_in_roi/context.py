from __future__ import annotations

from pathlib import Path

from ...config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]
STEP_04_CONFIG = CONFIG["step_04_boot_roi_from_edges"]
STEP_CONFIG = CONFIG["step_05_valid_hough_lines_in_roi"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

if bool(STEP_CONFIG.get("inherit_step_03_output", True)):
    edge_input_name = str(STEP_CONFIG.get("edge_input_name", STEP_03_CONFIG.get("selected_output", "cleaned"))).strip()
    EDGE_INPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"] / edge_input_name
else:
    EDGE_INPUT_DIR = PROCESSED_DIR / STEP_CONFIG["edge_input_subdir"]

if bool(STEP_CONFIG.get("inherit_step_04_output", True)):
    roi_mask_subdir_name = str(STEP_CONFIG.get("roi_mask_subdir_name", "mask")).strip()
    ROI_MASK_DIR = PROCESSED_DIR / STEP_04_CONFIG["output_subdir"] / roi_mask_subdir_name
else:
    ROI_MASK_DIR = PROCESSED_DIR / STEP_CONFIG["roi_mask_subdir"]

OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]
RAW_OVERLAY_DIR = OUTPUT_DIR / "raw_lines_overlay"
VALID_OVERLAY_DIR = OUTPUT_DIR / "valid_lines_overlay"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
VALID_LINES_JSON_DIR = OUTPUT_DIR / "valid_lines_json"
CSV_PATH = METADATA_DIR / "processing_05_valid_hough_lines_in_roi.csv"

CSV_FIELDNAMES = [
    "source_file",
    "edge_input_file",
    "roi_mask_file",
    "raw_overlay_output_file",
    "valid_overlay_output_file",
    "comparison_output_file",
    "width",
    "height",
    "processing_step",
    "raw_line_count",
    "valid_line_count",
    "roi_use_inner_mask",
    "roi_inner_erode_kernel_size",
    "roi_inner_erode_iterations",
    "hough_rho",
    "hough_theta_degrees",
    "hough_threshold",
    "hough_min_line_length",
    "hough_max_line_gap",
    "validation_min_mask_support_ratio",
    "validation_min_points_inside_mask",
    "validation_reference_mask_width_quantile",
    "validation_min_horizontal_clearance_ratio_of_mask_width",
    "validation_max_deviation_from_vertical_degrees",
    "read_time_ms",
    "processing_time_ms",
    "write_time_ms",
    "total_time_ms",
]


def refresh_context() -> None:
    global EDGE_INPUT_DIR, ROI_MASK_DIR, OUTPUT_DIR, RAW_OVERLAY_DIR, VALID_OVERLAY_DIR, COMPARISON_DIR, VALID_LINES_JSON_DIR
    if bool(STEP_CONFIG.get("inherit_step_03_output", True)):
        edge_input_name = str(STEP_CONFIG.get("edge_input_name", STEP_03_CONFIG.get("selected_output", "cleaned"))).strip()
        EDGE_INPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"] / edge_input_name
    else:
        EDGE_INPUT_DIR = PROCESSED_DIR / STEP_CONFIG["edge_input_subdir"]

    if bool(STEP_CONFIG.get("inherit_step_04_output", True)):
        roi_mask_subdir_name = str(STEP_CONFIG.get("roi_mask_subdir_name", "mask")).strip()
        ROI_MASK_DIR = PROCESSED_DIR / STEP_04_CONFIG["output_subdir"] / roi_mask_subdir_name
    else:
        ROI_MASK_DIR = PROCESSED_DIR / STEP_CONFIG["roi_mask_subdir"]

    OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]
    RAW_OVERLAY_DIR = OUTPUT_DIR / "raw_lines_overlay"
    VALID_OVERLAY_DIR = OUTPUT_DIR / "valid_lines_overlay"
    COMPARISON_DIR = OUTPUT_DIR / "comparison"
    VALID_LINES_JSON_DIR = OUTPUT_DIR / "valid_lines_json"


def set_step_config(step_config: dict) -> None:
    global STEP_CONFIG
    STEP_CONFIG = step_config
    refresh_context()
