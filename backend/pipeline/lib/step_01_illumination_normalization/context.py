from __future__ import annotations

from pathlib import Path

import cv2

from backend.pipeline.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_CONFIG = CONFIG["step_01_illumination_normalization"]

INPUT_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
OUTPUT_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"] / STEP_CONFIG["output_subdir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]
CSV_PATH = METADATA_DIR / "processing_01_illumination_normalization.csv"

CLAHE_CONFIG = STEP_CONFIG["clahe"]
CLAHE_CLIP_LIMIT = float(CLAHE_CONFIG["clip_limit"])
CLAHE_TILE_GRID_SIZE = tuple(int(value) for value in CLAHE_CONFIG["tile_grid_size"])
CLAHE_TILE_GRID_SIZE_LABEL = f"{CLAHE_TILE_GRID_SIZE[0]}x{CLAHE_TILE_GRID_SIZE[1]}"

# 0 is fastest and produces larger PNG files
PNG_COMPRESSION = max(0, min(9, int(STEP_CONFIG.get("png_compression", 0))))

# explicit for OpenCV
cv2.setUseOptimized(True)

# one CLAHE instance is enough here
CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
