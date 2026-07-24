from __future__ import annotations

import cv2
import numpy as np

from . import context

def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(context.DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]
    if height <= max_height: return image
    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def to_bgr_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def _draw_label_in_place(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 45), (0, 0, 0), thickness=-1)
    cv2.putText(image, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = to_bgr_for_display(image).copy()
    _draw_label_in_place(labeled, label)
    return labeled

def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height: return image
    new_width = int(width * target_height / height)
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)

def make_comparison_view(input_image: np.ndarray, raw_edges: np.ndarray, cleaned_edges: np.ndarray) -> np.ndarray:
    # resize grayscale panels before BGR conversion
    resized_images = [resize_for_display(input_image), resize_for_display(raw_edges), resize_for_display(cleaned_edges)]
    target_height = min(image.shape[0] for image in resized_images)
    panels = [to_bgr_for_display(_resize_to_height(image, target_height)) for image in resized_images]
    labels = [f"input: {context.SELECTED_STEP_02_OUTPUT}", "raw canny", "cleaned edges"]
    separator_width = 10
    total_width = sum(panel.shape[1] for panel in panels) + separator_width * (len(panels) - 1)
    combined = np.full((target_height, total_width, 3), 255, dtype=np.uint8)

    x_offset = 0
    for panel, label in zip(panels, labels):
        panel_width = panel.shape[1]
        destination = combined[:, x_offset:x_offset + panel_width]
        destination[:] = panel
        _draw_label_in_place(destination, label)
        x_offset += panel_width + separator_width

    return combined
