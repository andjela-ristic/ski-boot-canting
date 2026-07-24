from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .context import CSV_PATH, INPUT_DIR, METADATA_DIR, PNG_WRITE_PARAMS, PROJECT_ROOT

def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    if not INPUT_DIR.exists(): return []
    return sorted(path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions)

def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, PNG_WRITE_PARAMS): raise RuntimeError(f"Could not write image: {path}")

def save_metadata(rows: list[dict[str, object]]) -> None:
    if not rows: return
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    # keep the old CSV shape
    fieldnames = [
        "source_file",
        "grayscale_file",
        "grayscale_bgr2gray_file",
        "grayscale_lab_l_file",
        "grayscale_ycrcb_y_file",
        "gaussian_file",
        "bilateral_file",
        "width",
        "height",
        "processing_step",
        "grayscale_method",
        "gaussian_kernel_size",
        "gaussian_sigma_x",
        "bilateral_diameter",
        "bilateral_sigma_color",
        "bilateral_sigma_space",
        "read_time_ms",
        "processing_time_ms",
        "write_time_ms",
        "total_time_ms",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
