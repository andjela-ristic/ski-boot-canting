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

def collect_images(selected_image: str | None = None) -> list[Path]:
    if not context.INPUT_DIR.exists(): return []

    image_paths = sorted(path for path in context.INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in context.ALLOWED_EXTENSIONS)
    if selected_image is None: return image_paths

    selected_image = selected_image.strip()
    candidates = {selected_image}
    if "." not in selected_image: candidates.update(f"{selected_image}{extension}" for extension in context.ALLOWED_EXTENSIONS)
    return [path for path in image_paths if path.name in candidates]

def load_edge_image(image_path: Path) -> np.ndarray:
    edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if edge_image is None: raise ValueError(f"Could not read edge image: {image_path}")
    return edge_image

def write_image(path: Path, image: np.ndarray) -> None:
    params = [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE] if path.suffix.lower() == ".png" else []
    if not cv2.imwrite(str(path), image, params): raise IOError(f"Could not write image: {path}")

def save_metadata(rows: list[dict]) -> None:
    if not rows: return
    context.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(context.CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=context.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
