from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .context import CSV_PATH, INPUT_DIR, METADATA_DIR, PNG_COMPRESSION, PROJECT_ROOT


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions)


def save_processed_image(output_path: Path, image: np.ndarray) -> None:
    params: list[int] = []
    if output_path.suffix.lower() == ".png": params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
    saved = cv2.imwrite(str(output_path), image, params)
    if not saved: raise OSError(f"Could not write image: {output_path}")


def save_metadata(rows: list[dict]) -> None:
    if not rows: return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "output_file",
        "width",
        "height",
        "processing_step",
        "method",
        "clahe_clip_limit",
        "clahe_tile_grid_size",
        "read_time_ms",
        "processing_time_ms",
        "write_time_ms",
        "total_time_ms",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
