from __future__ import annotations

import cv2
import numpy as np

from . import context


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(context.DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]
    if height <= max_height: return image
    scale = max_height / height
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 45), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def draw_lines(base_image: np.ndarray, line_records: list[dict[str, float | int | bool]], valid_only: bool) -> np.ndarray:
    overlay = to_bgr(base_image).copy()
    drawing_config = context.STEP_CONFIG["drawing"]
    raw_thickness = int(drawing_config["raw_line_thickness"])
    valid_thickness = int(drawing_config["valid_line_thickness"])

    for record in line_records:
        if valid_only and not bool(record["is_valid"]): continue
        x1 = int(record["x1"])
        y1 = int(record["y1"])
        x2 = int(record["x2"])
        y2 = int(record["y2"])
        is_valid = bool(record["is_valid"])
        color = (0, 220, 0) if is_valid else (0, 0, 255)
        thickness = valid_thickness if is_valid else raw_thickness
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    return overlay


def make_comparison_view(edge_image: np.ndarray, raw_overlay: np.ndarray, valid_overlay: np.ndarray) -> np.ndarray:
    displays = [
        add_label(resize_for_display(to_bgr(edge_image)), "cleaned edges"),
        add_label(resize_for_display(raw_overlay), "raw hough lines"),
        add_label(resize_for_display(valid_overlay), "valid hough lines"),
    ]
    target_height = min(image.shape[0] for image in displays)
    resized = [cv2.resize(image, (int(image.shape[1] * target_height / image.shape[0]), target_height), interpolation=cv2.INTER_AREA) for image in displays]
    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    combined = resized[0]
    for image in resized[1:]:
        combined = np.hstack([combined, separator, image])
    return combined
