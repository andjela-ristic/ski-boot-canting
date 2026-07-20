from __future__ import annotations

import math

import cv2
import numpy as np

from .context import cfg, clip01, ensure_binary


def axis_x_at_y(axis: dict, y: float | np.ndarray) -> float | np.ndarray:
    return float(axis.get("a", 0.0)) * y + float(axis.get("b", axis.get("x_ref", 0.0)))


def axis_tilt_deg(axis: dict) -> float:
    if "tilt_deg" in axis:
        return float(axis["tilt_deg"])
    return float(math.degrees(math.atan(float(axis.get("a", 0.0)))))


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = ensure_binary(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def prepare_roi_mask(mask: np.ndarray) -> np.ndarray:
    result = ensure_binary(mask)
    if bool(cfg("roi_core", "keep_largest_component", default=True)):
        result = largest_component(result)
    kernel_size = max(1, int(cfg("roi_core", "close_kernel_size", default=7)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iterations = max(0, int(cfg("roi_core", "close_iterations", default=1)))
    if iterations > 0 and kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations,
        )
    return ensure_binary(result)


def resolve_vertical_range(mask: np.ndarray, metadata: dict) -> tuple[int, int]:
    ys = np.flatnonzero(np.any(mask > 0, axis=1))
    if ys.size == 0:
        raise ValueError("ROI mask is empty")
    mask_y_min = int(ys[0])
    mask_y_max = int(ys[-1])

    roi_profile = metadata.get("roi_profile", {})
    if bool(cfg("vertical_range", "use_step_06_trimmed_range", default=True)):
        y_min = int(roi_profile.get("trimmed_y_min", mask_y_min))
        y_max = int(roi_profile.get("trimmed_y_max", mask_y_max))
    else:
        span = max(1, mask_y_max - mask_y_min)
        y_min = mask_y_min + int(
            round(span * float(cfg("vertical_range", "trim_top_ratio", default=0.02)))
        )
        y_max = mask_y_max - int(
            round(span * float(cfg("vertical_range", "trim_bottom_ratio", default=0.02)))
        )

    y_min = max(mask_y_min, min(mask_y_max - 1, y_min))
    y_max = min(mask_y_max, max(y_min + 1, y_max))
    return y_min, y_max


def _row_bounds(mask: np.ndarray, y_min: int, y_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = mask.shape[0]
    left = np.full(height, -1, dtype=np.int32)
    right = np.full(height, -1, dtype=np.int32)
    widths = np.zeros(height, dtype=np.float64)
    for y in range(y_min, y_max + 1):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size == 0:
            continue
        left[y] = int(xs[0])
        right[y] = int(xs[-1])
        widths[y] = float(xs[-1] - xs[0] + 1)
    return left, right, widths


def _fallback_center_x_by_row(
    left: np.ndarray,
    right: np.ndarray,
    y_min: int,
    y_max: int,
) -> np.ndarray:
    centers = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = (left >= 0) & (right >= left)
    centers[valid] = 0.5 * (left[valid] + right[valid])
    y_values = np.arange(left.shape[0], dtype=np.float64)
    known = np.flatnonzero(np.isfinite(centers))
    if known.size == 0:
        centers[:] = 0.0
        return centers
    centers = np.interp(y_values, known.astype(np.float64), centers[known])
    window = max(3, int(cfg("roi_core", "center_fit_fallback_smoothing", default=31)))
    if window % 2 == 0:
        window += 1
    if window > 3:
        centers = cv2.GaussianBlur(
            centers.reshape(-1, 1),
            (1, window),
            0,
        ).reshape(-1)
    return centers


def build_core_roi_mask(
    mask: np.ndarray,
    metadata: dict,
    y_min: int,
    y_max: int,
) -> tuple[np.ndarray, dict]:
    """Remove extreme lateral appendages without using any candidate axis.

    The row corridor is centered on the Step 06 ROI center fit (or a smoothed
    row midpoint fallback), making this preprocessing candidate-independent.
    """
    left, right, widths = _row_bounds(mask, y_min, y_max)
    valid_widths = widths[(widths > 0) & (np.arange(mask.shape[0]) >= y_min) & (np.arange(mask.shape[0]) <= y_max)]
    if valid_widths.size == 0:
        raise ValueError("ROI mask has no valid rows in the verification range")

    quantile = clip01(float(cfg("roi_core", "row_width_cap_quantile", default=0.80)))
    cap_scale = max(0.5, float(cfg("roi_core", "row_width_cap_scale", default=1.08)))
    width_cap = float(np.quantile(valid_widths, quantile) * cap_scale)
    image_width = mask.shape[1]
    max_half_ratio = float(cfg("roi_core", "max_evaluation_half_width_ratio", default=0.46))
    min_half_width = float(cfg("roi_core", "min_evaluation_half_width_px", default=70))
    half_width = int(
        round(
            np.clip(
                0.5 * width_cap,
                min_half_width,
                max(2.0, image_width * max_half_ratio),
            )
        )
    )

    center_fit = metadata.get("roi_profile", {}).get("center_fit", {})
    has_center_fit = "a" in center_fit and "b" in center_fit
    fallback_centers = None
    if not has_center_fit:
        fallback_centers = _fallback_center_x_by_row(left, right, y_min, y_max)

    core = np.zeros_like(mask, dtype=np.uint8)
    for y in range(y_min, y_max + 1):
        if widths[y] <= 0:
            continue
        center_x = (
            float(center_fit["a"]) * float(y) + float(center_fit["b"])
            if has_center_fit
            else float(fallback_centers[y])
        )
        x0 = max(0, int(math.floor(center_x - half_width)))
        x1 = min(mask.shape[1] - 1, int(math.ceil(center_x + half_width)))
        if x1 >= x0:
            core[y, x0 : x1 + 1] = mask[y, x0 : x1 + 1]

    core = ensure_binary(core)
    return core, {
        "evaluation_half_width_px": int(half_width),
        "row_width_cap_px": float(width_cap),
        "row_width_quantile": float(quantile),
        "used_step_06_center_fit": bool(has_center_fit),
    }


def prepare_edge_mask(edge_image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    if edge_image.ndim == 3:
        gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_image.astype(np.uint8, copy=False)

    roi_pixels = gray[roi_mask > 0]
    if roi_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    threshold = int(cfg("edge_input", "threshold", default=24))
    nonzero_ratio = float(np.mean(roi_pixels > threshold))
    binary_ratio_max = float(
        cfg("edge_input", "binary_nonzero_ratio_max", default=0.20)
    )
    sampled = roi_pixels[:: max(1, roi_pixels.size // 50000)]
    unique_count = int(np.unique(sampled).size)
    if nonzero_ratio > binary_ratio_max or unique_count > 32:
        edge = cv2.Canny(
            gray,
            int(cfg("edge_input", "canny_low", default=45)),
            int(cfg("edge_input", "canny_high", default=135)),
        )
    else:
        edge = np.where(gray > threshold, 255, 0).astype(np.uint8)

    close_iterations = max(0, int(cfg("edge_input", "close_iterations", default=0)))
    kernel_size = max(1, int(cfg("edge_input", "close_kernel_size", default=3)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if close_iterations > 0 and kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        edge = cv2.morphologyEx(
            edge,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )
    return np.where(roi_mask > 0, edge, 0).astype(np.uint8)


def rectify_about_axis(
    image: np.ndarray,
    axis: dict,
    y_min: int,
    y_max: int,
    half_width: int,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    rows = np.arange(y_min, y_max + 1, dtype=np.float32)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
    map_y = np.repeat(rows[:, None], offsets.size, axis=1)
    axis_x = (
        float(axis.get("a", 0.0)) * rows
        + float(axis.get("b", axis.get("x_ref", 0.0)))
    )
    map_x = axis_x[:, None] + offsets[None, :]
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def split_mirrored_sides(
    rectified: np.ndarray,
    center_exclusion_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = rectified.shape[1] // 2
    exclusion = max(0, int(center_exclusion_px))
    left_end = max(0, center - exclusion)
    right_start = min(rectified.shape[1], center + exclusion + 1)
    side_width = min(left_end, rectified.shape[1] - right_start)
    if side_width <= 0:
        empty = rectified[:, :0]
        return empty, empty
    left = rectified[:, left_end - side_width : left_end]
    right = rectified[:, right_start : right_start + side_width]
    return np.fliplr(left), right


def segment_ranges(height: int, segment_count: int) -> list[tuple[int, int]]:
    count = max(1, min(int(segment_count), max(1, height)))
    boundaries = np.rint(np.linspace(0, height, count + 1)).astype(np.int32)
    ranges: list[tuple[int, int]] = []
    for index in range(count):
        start = int(boundaries[index])
        end = int(boundaries[index + 1])
        if end <= start:
            end = min(height, start + 1)
        ranges.append((start, end))
    return ranges


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = ensure_binary(mask)
    eroded = cv2.erode(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return np.where((binary > 0) & (eroded == 0), 255, 0).astype(np.uint8)
