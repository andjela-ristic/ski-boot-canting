from __future__ import annotations

from pathlib import Path

import cv2

from backend.pipeline.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_01_CONFIG = CONFIG["step_01_illumination_normalization"]
STEP_CONFIG = CONFIG["step_02_grayscale_and_blur"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

if bool(STEP_CONFIG.get("inherit_step_01_output", True)):
    INPUT_DIR = PROCESSED_DIR / STEP_01_CONFIG["output_subdir"]
else:
    INPUT_DIR = PROCESSED_DIR / STEP_CONFIG["input_subdir"]

OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

# active outputs
# step 03 reads grayscale_lab_l
# the API pipeline runner validates that bilateral_filter exists
GRAYSCALE_LAB_L_DIR = OUTPUT_DIR / "grayscale_lab_l"
BILATERAL_DIR = OUTPUT_DIR / "bilateral_filter"

# disabled outputs
# GRAYSCALE_DIR = OUTPUT_DIR / "grayscale"
# GRAYSCALE_BGR2GRAY_DIR = OUTPUT_DIR / "grayscale_bgr2gray"
# GRAYSCALE_YCRCB_Y_DIR = OUTPUT_DIR / "grayscale_ycrcb_y"
# GAUSSIAN_DIR = OUTPUT_DIR / "gaussian_blur"

CSV_PATH = METADATA_DIR / "processing_02_grayscale_and_blur.csv"

# PNG stays lossless
PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]
