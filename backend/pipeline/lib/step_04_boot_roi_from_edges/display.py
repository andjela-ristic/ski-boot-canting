from __future__ import annotations

import cv2
import numpy as np

from . import context
from . import runtime

def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(context.DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]
    if height <= max_height: return image
    scale = max_height / height
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 45), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled

def build_center_prior(height: int, width: int) -> np.ndarray:
    if not bool(runtime.CENTER_PRIOR_CONFIG.get("enabled", True)):
        prior = np.ones((height, width), dtype=np.float32)
        prior.setflags(write=False)
        return prior

    sigma_x_ratio = float(runtime.CENTER_PRIOR_CONFIG["sigma_x_ratio"])
    sigma_y_ratio = float(runtime.CENTER_PRIOR_CONFIG["sigma_y_ratio"])
    power = float(runtime.CENTER_PRIOR_CONFIG.get("power", 1.0))
    sigma_x = max(width * sigma_x_ratio, 1.0)
    sigma_y = max(height * sigma_y_ratio, 1.0)
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    x_exponent = ((x_coords - center_x) ** 2) / (2.0 * sigma_x * sigma_x)
    y_exponent = ((y_coords - center_y) ** 2) / (2.0 * sigma_y * sigma_y)
    exponent = y_exponent[:, None] + x_exponent[None, :]
    prior = np.exp(-exponent)
    if power != 1.0: prior = np.power(prior, power)
    prior = prior.astype(np.float32, copy=False)
    prior.setflags(write=False)
    return prior

def _prepare_debug_display(image: np.ndarray, label: str) -> np.ndarray:
    resized = resize_for_display(image)
    return add_label(to_bgr(resized), label)

def make_overlay(edge_image: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    alpha = float(runtime.OVERLAY_CONFIG["alpha"])
    contour_thickness = int(runtime.OVERLAY_CONFIG["contour_thickness"])
    edge_bgr = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    green_fill = np.zeros_like(edge_bgr)
    green_fill[:, :, 1] = final_mask
    overlay = cv2.addWeighted(edge_bgr, 1.0, green_fill, alpha, 0.0)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours: cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=contour_thickness, lineType=cv2.LINE_AA)
    return overlay

def make_comparison_view(edge_image: np.ndarray, density_image: np.ndarray, weighted_density_image: np.ndarray, final_overlay: np.ndarray) -> np.ndarray:
    displays = [
        _prepare_debug_display(edge_image, "input edges"),
        _prepare_debug_display(density_image, "density"),
        _prepare_debug_display(weighted_density_image, "weighted density"),
        _prepare_debug_display(final_overlay, "final roi"),
    ]
    target_height = min(image.shape[0] for image in displays)
    normalized_displays: list[np.ndarray] = []

    for image in displays:
        height, width = image.shape[:2]
        if height == target_height:
            normalized_displays.append(image)
        else:
            normalized_displays.append(cv2.resize(image, (int(width * target_height / height), target_height), interpolation=cv2.INTER_AREA))

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for index, image in enumerate(normalized_displays):
        if index: parts.append(separator)
        parts.append(image)
    return np.concatenate(parts, axis=1)
