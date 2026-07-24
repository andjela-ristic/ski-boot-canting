from __future__ import annotations

from pathlib import Path

from ...config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_02_CONFIG = CONFIG["step_02_grayscale_and_blur"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

STEP_02_OUTPUT_DIR = PROCESSED_DIR / STEP_02_CONFIG["output_subdir"]
SELECTED_STEP_02_OUTPUT = str(STEP_03_CONFIG.get("selected_input", STEP_02_CONFIG["selected_output"])).strip()
STEP_03_TEST_INPUT_NAME = str(STEP_02_CONFIG["selected_output"]).strip()

INPUT_DIR = STEP_02_OUTPUT_DIR / SELECTED_STEP_02_OUTPUT
TEST_INPUT_DIR = STEP_02_OUTPUT_DIR / STEP_03_TEST_INPUT_NAME
OUTPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]

CLEANED_DIR = OUTPUT_DIR / "cleaned"
CSV_PATH = METADATA_DIR / "processing_03_edge_detection.csv"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CSV_FIELDNAMES = [
    "source_file",
    "cleaned_output_file",
    "roi_output_file",
    "width",
    "height",
    "processing_step",
    "input_from_step_02",
    "canny_mode",
    "threshold_1",
    "threshold_2",
    "aperture_size",
    "use_l2_gradient",
    "auto_sigma",
    "preprocessing_enabled",
    "preprocessing_gaussian_kernel_size",
    "preprocessing_gaussian_sigma_x",
    "postprocessing_enabled",
    "dilate_iterations",
    "dilate_kernel_size",
    "close_kernel_size",
    "open_kernel_size",
    "erode_iterations",
    "erode_kernel_size",
    "read_time_ms",
    "processing_time_ms",
    "write_time_ms",
    "total_time_ms",
]

def refresh_context() -> None:
    global SELECTED_STEP_02_OUTPUT, STEP_03_TEST_INPUT_NAME, INPUT_DIR, TEST_INPUT_DIR, OUTPUT_DIR, CLEANED_DIR
    SELECTED_STEP_02_OUTPUT = str(STEP_03_CONFIG.get("selected_input", STEP_02_CONFIG["selected_output"])).strip()
    STEP_03_TEST_INPUT_NAME = str(STEP_02_CONFIG["selected_output"]).strip()
    INPUT_DIR = STEP_02_OUTPUT_DIR / SELECTED_STEP_02_OUTPUT
    TEST_INPUT_DIR = STEP_02_OUTPUT_DIR / STEP_03_TEST_INPUT_NAME
    OUTPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]
    CLEANED_DIR = OUTPUT_DIR / "cleaned"

def set_step_config(step_config: dict) -> None:
    global STEP_03_CONFIG
    STEP_03_CONFIG = step_config
    refresh_context()
