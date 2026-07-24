from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import context


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(context.PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_images(selected_image: str | None = None) -> list[str]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    if not context.EDGE_INPUT_DIR.exists() or not context.ROI_MASK_DIR.exists(): return []

    edge_names = {path.name for path in context.EDGE_INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions}
    roi_names = {path.name for path in context.ROI_MASK_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions}
    image_names = sorted(edge_names & roi_names)
    if selected_image is None: return image_names

    selected_image = selected_image.strip()
    candidates = {selected_image}
    if "." not in selected_image: candidates.update(f"{selected_image}{extension}" for extension in allowed_extensions)
    return [image_name for image_name in image_names if image_name in candidates]


def load_grayscale_image(path: Path, *, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise ValueError(f"Could not read {label}: {path}")
    return image


def save_metadata(rows: list[dict[str, str | int | float]]) -> None:
    if not rows: return
    context.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(context.CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=context.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_valid_lines_json(
    output_path: Path,
    image_name: str,
    edge_path: Path,
    roi_path: Path,
    image_shape: tuple[int, int],
    raw_line_count: int,
    valid_lines: list[dict[str, float | int | bool]],
) -> None:
    height, width = image_shape
    roi_config = context.STEP_CONFIG["roi"]
    hough_config = context.STEP_CONFIG["hough_lines_p"]
    validation_config = context.STEP_CONFIG["validation"]
    payload = {
        "image_name": image_name,
        "source_file": relative_project_path(edge_path),
        "roi_mask_file": relative_project_path(roi_path),
        "processing_step": "05_valid_hough_lines_in_roi",
        "width": width,
        "height": height,
        "raw_line_count": raw_line_count,
        "valid_line_count": len(valid_lines),
        "parameters": {
            "roi": {
                "use_inner_mask": bool(roi_config.get("use_inner_mask", True)),
                "inner_erode_kernel_size": int(roi_config["inner_erode_kernel_size"]),
                "inner_erode_iterations": int(roi_config.get("inner_erode_iterations", 1)),
            },
            "hough_lines_p": {
                "rho": float(hough_config["rho"]),
                "theta_degrees": float(hough_config["theta_degrees"]),
                "threshold": int(hough_config["threshold"]),
                "min_line_length": int(hough_config["min_line_length"]),
                "max_line_gap": int(hough_config["max_line_gap"]),
            },
            "validation": {
                "min_mask_support_ratio": float(validation_config["min_mask_support_ratio"]),
                "min_points_inside_mask": int(validation_config["min_points_inside_mask"]),
                "reference_mask_width_quantile": float(validation_config.get("reference_mask_width_quantile", 0.25)),
                "min_horizontal_clearance_ratio_of_mask_width": float(validation_config.get("min_horizontal_clearance_ratio_of_mask_width", 0.0)),
                "max_deviation_from_vertical_degrees": float(validation_config["max_deviation_from_vertical_degrees"]),
            },
        },
        "valid_lines": valid_lines,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
