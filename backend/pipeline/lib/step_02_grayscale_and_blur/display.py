from __future__ import annotations

import cv2
import numpy as np

from .context import DISPLAY_CONFIG

def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]
    if height <= max_height: return image
    scale = max_height / height
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def to_bgr_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 45), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled

def make_comparison_view(original_bgr: np.ndarray, grayscale_lab_l: np.ndarray, bilateral: np.ndarray) -> np.ndarray:
    images = [
        resize_for_display(original_bgr),
        resize_for_display(to_bgr_for_display(grayscale_lab_l)),
        resize_for_display(to_bgr_for_display(bilateral)),
    ]
    target_height = min(image.shape[0] for image in images)
    resized_images: list[np.ndarray] = []

    for image in images:
        height, width = image.shape[:2]
        if height == target_height:
            resized_images.append(image)
            continue
        new_width = int(width * target_height / height)
        resized_images.append(cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA))

    labeled_images = [
        add_label(resized_images[0], "01 normalized"),
        add_label(resized_images[1], "grayscale: lab_l"),
        add_label(resized_images[2], "bilateral filter"),
    ]
    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    return np.hstack([labeled_images[0], separator, labeled_images[1], separator, labeled_images[2]])