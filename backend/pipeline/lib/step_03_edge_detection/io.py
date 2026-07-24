from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from . import context

def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(context.PROJECT_ROOT))
    except ValueError:
        return str(path)

def collect_images() -> list[Path]:
    if not context.INPUT_DIR.exists(): return []
    return sorted(path for path in context.INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in context.ALLOWED_IMAGE_EXTENSIONS)

def load_grayscale_image(image_path: Path) -> np.ndarray:
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None: raise ValueError(f"Could not read image: {image_path}")
    return image_gray

def load_variant_input_image(image_name: str, variant: dict) -> np.ndarray:
    image_path = variant.get("input_dir", context.TEST_INPUT_DIR if variant.get("use_step_03_test_input", False) else context.INPUT_DIR) / image_name
    try:
        return load_grayscale_image(image_path)
    except ValueError as error:
        raise ValueError(f"Could not read variant input image: {image_path}") from error

def save_metadata(rows: list[dict]) -> None:
    if not rows: return
    context.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(context.CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=context.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
