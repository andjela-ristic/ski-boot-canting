from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from . import context

def ensure_odd_kernel_size(value: int, name: str) -> int:
    value = int(value)
    if value < 1: raise ValueError(f"{name} must be positive. Got: {value}")
    if value % 2 == 0: raise ValueError(f"{name} must be odd. Got: {value}")
    return value

@lru_cache(maxsize=32)
def make_ellipse_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = ensure_odd_kernel_size(kernel_size, "kernel_size")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel.setflags(write=False)
    return kernel

def scale_to_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1

def _kernel_radius(kernel: np.ndarray) -> int:
    return max(kernel.shape[0] // 2, kernel.shape[1] // 2)

def _compute_morphology_margin() -> int:
    margin = (
        _kernel_radius(DENSITY_SEED_KERNEL) * 0
        + _kernel_radius(CLOSE_KERNEL) * ITERATIONS_CLOSE
        + _kernel_radius(DILATE_KERNEL) * ITERATIONS_DILATE
        + _kernel_radius(SECOND_CLOSE_KERNEL) * ITERATIONS_SECOND_CLOSE
        + _kernel_radius(SMOOTH_KERNEL)
    )
    if HULL_ENABLED and HULL_MODE == "soft": margin += _kernel_radius(SMOOTH_KERNEL)
    return margin + 2

def refresh_runtime() -> None:
    global DENSITY_CONFIG, NOISE_CONFIG, MORPHOLOGY_CONFIG, SELECTION_CONFIG, HULL_CONFIG, OVERLAY_CONFIG, CENTER_PRIOR_CONFIG
    global EDGE_THRESHOLD, MIN_COMPONENT_AREA, DENSITY_KERNEL_SIZE, DENSITY_SEED_KERNEL
    global CLOSE_KERNEL, DILATE_KERNEL, SECOND_CLOSE_KERNEL, SMOOTH_KERNEL
    global ITERATIONS_CLOSE, ITERATIONS_DILATE, ITERATIONS_SECOND_CLOSE
    global HULL_ENABLED, HULL_MODE, HULL_EPSILON_RATIO, MORPHOLOGY_MARGIN

    DENSITY_CONFIG = context.STEP_CONFIG["density"]
    NOISE_CONFIG = context.STEP_CONFIG["noise"]
    MORPHOLOGY_CONFIG = context.STEP_CONFIG["morphology"]
    SELECTION_CONFIG = context.STEP_CONFIG["component_selection"]
    HULL_CONFIG = context.STEP_CONFIG["hull"]
    OVERLAY_CONFIG = context.STEP_CONFIG["overlay"]
    CENTER_PRIOR_CONFIG = context.STEP_CONFIG["center_prior"]

    EDGE_THRESHOLD = int(context.STEP_CONFIG["edge_threshold"])
    MIN_COMPONENT_AREA = int(NOISE_CONFIG["min_component_area"])
    DENSITY_KERNEL_SIZE = ensure_odd_kernel_size(int(DENSITY_CONFIG["kernel_size"]), "density.kernel_size")
    DENSITY_SEED_KERNEL = make_ellipse_kernel(scale_to_odd(DENSITY_KERNEL_SIZE // 3))
    CLOSE_KERNEL = make_ellipse_kernel(int(MORPHOLOGY_CONFIG["close_kernel_size"]))
    DILATE_KERNEL = make_ellipse_kernel(int(MORPHOLOGY_CONFIG["dilate_kernel_size"]))
    SECOND_CLOSE_KERNEL = make_ellipse_kernel(int(MORPHOLOGY_CONFIG["second_close_kernel_size"]))
    SMOOTH_KERNEL = make_ellipse_kernel(int(MORPHOLOGY_CONFIG["smooth_kernel_size"]))
    ITERATIONS_CLOSE = int(MORPHOLOGY_CONFIG["iterations_close"])
    ITERATIONS_DILATE = int(MORPHOLOGY_CONFIG["iterations_dilate"])
    ITERATIONS_SECOND_CLOSE = int(MORPHOLOGY_CONFIG["iterations_second_close"])
    HULL_ENABLED = bool(HULL_CONFIG.get("enabled", False))
    HULL_MODE = str(HULL_CONFIG.get("mode", "convex")).strip().lower()
    HULL_EPSILON_RATIO = float(HULL_CONFIG.get("approx_epsilon_ratio", 0.008))
    if HULL_MODE not in {"convex", "approx", "soft"}:
        raise ValueError(f"Unsupported hull mode: {HULL_MODE}. Supported: convex, approx, soft")
    MORPHOLOGY_MARGIN = _compute_morphology_margin()

refresh_runtime()
