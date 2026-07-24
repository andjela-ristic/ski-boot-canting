from __future__ import annotations

import cv2
import numpy as np

from .context import DISPLAY_CONFIG


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]
    if height <= max_height: return image
    scale = max_height / height
    return cv2.resize(image, (round(width * scale), max_height), interpolation=cv2.INTER_AREA)


def make_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    # both views come from the same source
    original_display = resize_for_display(original)
    processed_display = resize_for_display(processed)
    separator = np.full((original_display.shape[0], 10, 3), 255, dtype=np.uint8)
    return np.concatenate((original_display, separator, processed_display), axis=1)
