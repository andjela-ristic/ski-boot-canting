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
    if bool(cfg("roi_preparation", "keep_largest_component", default=True)):
        result = largest_component(result)
    kernel_size = max(1, int(cfg("roi_preparation", "close_kernel_size", default=7)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iterations = max(0, int(cfg("roi_preparation", "close_iterations", default=1)))
    if iterations > 0 and kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return ensure_binary(result)


def resolve_vertical_range(mask: np.ndarray, metadata: dict) -> tuple[int, int]:
    ys = np.flatnonzero(np.any(mask > 0, axis=1))
    if ys.size == 0:
        raise ValueError("ROI mask is empty")
    mask_y_min, mask_y_max = int(ys[0]), int(ys[-1])
    roi_profile = metadata.get("roi_profile", {})
    if bool(cfg("vertical_range", "use_step_06_trimmed_range", default=True)):
        y_min = int(roi_profile.get("trimmed_y_min", mask_y_min))
        y_max = int(roi_profile.get("trimmed_y_max", mask_y_max))
    else:
        span = max(1, mask_y_max - mask_y_min)
        y_min = mask_y_min + int(round(span * float(cfg("vertical_range", "trim_top_ratio", default=0.02))))
        y_max = mask_y_max - int(round(span * float(cfg("vertical_range", "trim_bottom_ratio", default=0.02))))
    y_min = max(mask_y_min, min(mask_y_max - 1, y_min))
    y_max = min(mask_y_max, max(y_min + 1, y_max))
    return y_min, y_max


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    if float(np.sum(weights)) <= 1e-12:
        return float(np.median(values))
    cutoff = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def build_candidate_consensus(candidates: list[dict], y_min: int, y_max: int) -> dict:
    if not candidates:
        raise ValueError("Cannot build consensus without candidates")
    y_values = np.linspace(float(y_min), float(y_max), 96, dtype=np.float64)
    power = max(0.0, float(cfg("consensus_corridor", "candidate_score_power", default=2.0)))
    raw_weights = np.asarray([
        max(1e-6, float(c.get("final_score", c.get("score", 0.0)))) ** power
        for c in candidates
    ], dtype=np.float64)
    x_matrix = np.asarray([
        np.asarray(axis_x_at_y(c, y_values), dtype=np.float64)
        for c in candidates
    ])
    median_x = np.asarray([
        _weighted_median(x_matrix[:, i], raw_weights)
        for i in range(x_matrix.shape[1])
    ], dtype=np.float64)
    fit_a, fit_b = np.polyfit(y_values, median_x, 1)
    tilts = np.asarray([axis_tilt_deg(c) for c in candidates], dtype=np.float64)
    tilt_median = _weighted_median(tilts, raw_weights)
    tilt_mad = _weighted_median(np.abs(tilts - tilt_median), raw_weights)
    return {
        "a": float(fit_a),
        "b": float(fit_b),
        "tilt_deg": float(math.degrees(math.atan(float(fit_a)))),
        "candidate_tilt_median_deg": float(tilt_median),
        "candidate_tilt_mad_deg": float(tilt_mad),
        "candidate_count": int(len(candidates)),
    }


def _row_bounds(mask: np.ndarray, y_min: int, y_max: int) -> tuple[np.ndarray, np.ndarray]:
    height = mask.shape[0]
    left = np.full(height, -1, dtype=np.int32)
    right = np.full(height, -1, dtype=np.int32)
    for y in range(y_min, y_max + 1):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size:
            left[y], right[y] = int(xs[0]), int(xs[-1])
    return left, right


def _smooth_series(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    indexes = np.arange(values.size, dtype=np.float64)
    known = np.flatnonzero(valid)
    if known.size == 0:
        return np.zeros_like(result)
    result = np.interp(indexes, known.astype(np.float64), result[known])
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > 3:
        result = cv2.GaussianBlur(result.reshape(-1, 1), (1, window), 0).reshape(-1)
    return result


def build_symmetric_evaluation_corridor(
    roi_mask: np.ndarray,
    metadata: dict,
    candidates: list[dict],
    y_min: int,
    y_max: int,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Build a candidate-independent, symmetric corridor around candidate consensus.

    Step 04 ROI is used only as a coarse containment mask. Its asymmetric outline
    is never scored. At every row we keep only the radius available on both sides
    of the robust consensus axis, then cap it by the Step 06 reference width.
    """
    consensus = build_candidate_consensus(candidates, y_min, y_max)
    left, right = _row_bounds(roi_mask, y_min, y_max)
    height, width = roi_mask.shape
    rows = np.arange(height, dtype=np.float64)
    center_x = np.asarray(axis_x_at_y(consensus, rows), dtype=np.float64)
    left_distance = center_x - left.astype(np.float64)
    right_distance = right.astype(np.float64) - center_x
    common = np.minimum(left_distance, right_distance)
    valid = (
        (left >= 0)
        & (right >= left)
        & (common > 1.0)
        & (np.arange(height) >= y_min)
        & (np.arange(height) <= y_max)
    )
    reference_width = float(metadata.get("roi_profile", {}).get("reference_width_px", 0.0))
    if reference_width <= 0:
        widths = (right - left + 1).astype(np.float64)
        reference_width = float(np.median(widths[valid])) if np.any(valid) else width * 0.35
    max_by_reference = reference_width * float(
        cfg("consensus_corridor", "reference_half_width_ratio", default=0.42)
    )
    max_by_image = width * float(
        cfg("consensus_corridor", "maximum_half_width_image_ratio", default=0.30)
    )
    valid_common = common[valid]
    quantile = clip01(float(cfg("consensus_corridor", "row_half_width_quantile", default=0.72)))
    quantile_cap = (
        float(np.quantile(valid_common, quantile))
        * float(cfg("consensus_corridor", "row_half_width_cap_scale", default=1.05))
        if valid_common.size
        else max_by_reference
    )
    cap = max(2.0, min(max_by_reference, max_by_image, quantile_cap))
    margin = max(0.0, float(cfg("consensus_corridor", "edge_margin_px", default=8)))
    half_widths = np.maximum(0.0, np.minimum(common, cap) - margin)
    half_widths = _smooth_series(
        half_widths,
        valid & (half_widths > 0),
        int(cfg("consensus_corridor", "smooth_window", default=51)),
    )
    minimum = float(cfg("consensus_corridor", "minimum_half_width_px", default=80))
    half_widths = np.where(valid, np.clip(half_widths, minimum, cap), 0.0)

    corridor = np.zeros_like(roi_mask, dtype=np.uint8)
    for y in range(y_min, y_max + 1):
        radius = int(math.floor(float(half_widths[y])))
        if radius < 2:
            continue
        cx = float(center_x[y])
        x0 = max(0, int(math.ceil(cx - radius)))
        x1 = min(width - 1, int(math.floor(cx + radius)))
        if x1 >= x0:
            corridor[y, x0 : x1 + 1] = 255
    corridor = ensure_binary(corridor)
    info = {
        "consensus_axis": consensus,
        "reference_width_px": float(reference_width),
        "corridor_half_width_cap_px": float(cap),
        "corridor_median_half_width_px": float(np.median(half_widths[half_widths > 0])) if np.any(half_widths > 0) else 0.0,
        "corridor_min_half_width_px": float(np.min(half_widths[half_widths > 0])) if np.any(half_widths > 0) else 0.0,
        "corridor_max_half_width_px": float(np.max(half_widths)) if half_widths.size else 0.0,
        "row_half_width_quantile": float(quantile),
    }
    return corridor, info, half_widths


def prepare_edge_mask(edge_image: np.ndarray, corridor_mask: np.ndarray, consensus: dict, y_min: int, y_max: int) -> np.ndarray:
    if edge_image.ndim == 3:
        gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_image.astype(np.uint8, copy=False)
    pixels = gray[corridor_mask > 0]
    if pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    threshold = int(cfg("edge_input", "threshold", default=24))
    nonzero_ratio = float(np.mean(pixels > threshold))
    unique_count = int(np.unique(pixels[:: max(1, pixels.size // 50000)]).size)
    if nonzero_ratio > float(cfg("edge_input", "binary_nonzero_ratio_max", default=0.20)) or unique_count > 32:
        edge = cv2.Canny(
            gray,
            int(cfg("edge_input", "canny_low", default=45)),
            int(cfg("edge_input", "canny_high", default=135)),
        )
    else:
        edge = np.where(gray > threshold, 255, 0).astype(np.uint8)
    iterations = max(0, int(cfg("edge_input", "close_iterations", default=0)))
    kernel_size = max(1, int(cfg("edge_input", "close_kernel_size", default=3)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if iterations > 0 and kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    edge = np.where(corridor_mask > 0, edge, 0).astype(np.uint8)

    # Remove very long border-connected components only when they are far from
    # the consensus axis. This suppresses curtain/table lines without deleting
    # central boot structures.
    if bool(cfg("edge_input", "remove_border_components", default=True)):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(edge, connectivity=8)
        keep = np.ones(count, dtype=bool)
        span = max(1, y_max - y_min + 1)
        reference = max(1.0, float(np.max(np.sum(corridor_mask > 0, axis=1))))
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            touches_border = x <= 1 or x + w >= edge.shape[1] - 1 or y <= y_min + 1 or y + h >= y_max - 1
            long_vertical = h >= span * float(cfg("edge_input", "border_component_min_height_ratio", default=0.80))
            cy = float(centroids[label][1])
            cx = float(centroids[label][0])
            axis_x = float(axis_x_at_y(consensus, cy))
            far = abs(cx - axis_x) >= reference * float(cfg("edge_input", "border_component_max_center_distance_ratio", default=0.42))
            if touches_border and long_vertical and far:
                keep[label] = False
        edge = np.where(keep[labels], edge, 0).astype(np.uint8)
    return ensure_binary(edge)


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
    axis_x = float(axis.get("a", 0.0)) * rows + float(axis.get("b", axis.get("x_ref", 0.0)))
    map_x = axis_x[:, None] + offsets[None, :]
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def split_mirrored_sides(rectified: np.ndarray, center_exclusion_px: int) -> tuple[np.ndarray, np.ndarray]:
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
    # Both outputs are near-axis -> far-axis.
    return np.fliplr(left), right


def segment_ranges(height: int, segment_count: int) -> list[tuple[int, int]]:
    count = max(1, min(int(segment_count), max(1, height)))
    boundaries = np.rint(np.linspace(0, height, count + 1)).astype(np.int32)
    return [(int(boundaries[i]), max(int(boundaries[i]) + 1, int(boundaries[i + 1]))) for i in range(count)]


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = ensure_binary(mask)
    eroded = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return np.where((binary > 0) & (eroded == 0), 255, 0).astype(np.uint8)
