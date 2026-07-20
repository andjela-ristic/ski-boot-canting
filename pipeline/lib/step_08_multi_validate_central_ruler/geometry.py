from __future__ import annotations

import math

import cv2
import numpy as np

from .context import cfg, clip01


def axis_x_at_y(axis: dict, y: float) -> float:
    if "a" in axis and "b" in axis:
        return float(axis["a"]) * float(y) + float(axis["b"])
    return float(axis.get("x_ref", 0.0)) + float(axis.get("a", 0.0)) * (
        float(y) - float(axis.get("y_ref", 0.0))
    )


def axis_tilt_deg(axis: dict) -> float:
    if "tilt_deg" in axis:
        return float(axis["tilt_deg"])
    return float(math.degrees(math.atan(float(axis.get("a", 0.0)))))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _nanmedian_smooth(values: np.ndarray, window: int) -> np.ndarray:
    result = values.astype(np.float64).copy()
    valid = np.isfinite(result)
    if not np.any(valid):
        return np.zeros_like(result)
    indices = np.arange(result.size)
    result[~valid] = np.interp(indices[~valid], indices[valid], result[valid])
    kernel = _odd(window)
    if kernel > 1:
        radius = kernel // 2
        padded = np.pad(result, (radius, radius), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
        result = np.median(windows, axis=1).astype(np.float64)
    return result


def prepare_roi_core(mask: np.ndarray, y_min: int, y_max: int) -> tuple[np.ndarray, dict]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if bool(cfg("roi_preparation", "keep_largest_component", default=True)):
        binary = keep_largest_component(binary)
    kernel_size = _odd(int(cfg("roi_preparation", "close_kernel_size", default=7)))
    iterations = max(0, int(cfg("roi_preparation", "close_iterations", default=1)))
    if iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    height, width = binary.shape
    y_min = max(0, int(y_min))
    y_max = min(height - 1, int(y_max))
    row_left = np.full(height, np.nan, dtype=np.float64)
    row_right = np.full(height, np.nan, dtype=np.float64)
    row_center = np.full(height, np.nan, dtype=np.float64)
    row_width = np.full(height, np.nan, dtype=np.float64)
    for y in range(y_min, y_max + 1):
        xs = np.flatnonzero(binary[y] > 0)
        if xs.size < 2:
            continue
        row_left[y], row_right[y] = float(xs[0]), float(xs[-1])
        row_center[y] = 0.5 * float(xs[0] + xs[-1])
        row_width[y] = float(xs[-1] - xs[0] + 1)

    smooth_window = int(cfg("roi_preparation", "row_smoothing_window", default=41))
    smooth_center = _nanmedian_smooth(row_center, smooth_window)
    valid_widths = row_width[y_min : y_max + 1]
    valid_widths = valid_widths[np.isfinite(valid_widths)]
    if valid_widths.size == 0:
        raise RuntimeError("ROI mask contains no valid rows in the requested vertical range")
    quantile = float(cfg("roi_preparation", "row_width_cap_quantile", default=0.72))
    cap_scale = float(cfg("roi_preparation", "row_width_cap_scale", default=1.08))
    width_cap = float(np.quantile(valid_widths, np.clip(quantile, 0.05, 0.98)) * cap_scale)
    min_half = float(cfg("roi_preparation", "minimum_half_width_px", default=55))

    core = np.zeros_like(binary)
    core_left = np.full(height, np.nan, dtype=np.float64)
    core_right = np.full(height, np.nan, dtype=np.float64)
    half_widths = np.zeros(height, dtype=np.float64)
    for y in range(y_min, y_max + 1):
        if not np.isfinite(row_left[y]) or not np.isfinite(row_right[y]):
            continue
        center = float(smooth_center[y])
        available_left = max(0.0, center - row_left[y])
        available_right = max(0.0, row_right[y] - center)
        half = min(available_left, available_right, 0.5 * width_cap)
        if half < min_half:
            half = min(available_left, available_right)
        if half < 2.0:
            continue
        left = max(0, int(round(center - half)))
        right = min(width - 1, int(round(center + half)))
        core[y, left : right + 1] = binary[y, left : right + 1]
        core_left[y], core_right[y] = float(left), float(right)
        half_widths[y] = 0.5 * float(right - left + 1)

    info = {
        "y_min": int(y_min),
        "y_max": int(y_max),
        "row_center": smooth_center,
        "row_left": core_left,
        "row_right": core_right,
        "row_half_width": half_widths,
        "row_width_cap_px": float(width_cap),
    }
    return core, info


def robust_line_fit(y: np.ndarray, x: np.ndarray, base_weights: np.ndarray) -> tuple[float, float, np.ndarray]:
    if y.size < 2:
        raise ValueError("At least two points are required for a line fit")
    design = np.column_stack([y.astype(np.float64), np.ones_like(y, dtype=np.float64)])
    weights = np.maximum(base_weights.astype(np.float64), 1e-6)
    beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]), x * np.sqrt(weights), rcond=None)[0]
    delta = float(cfg("medial_axis", "huber_delta_px", default=16.0))
    for _ in range(max(1, int(cfg("medial_axis", "huber_iterations", default=6)))):
        residual = x - design @ beta
        robust = np.ones_like(residual)
        large = np.abs(residual) > delta
        robust[large] = delta / np.maximum(np.abs(residual[large]), 1e-9)
        w = weights * robust
        beta = np.linalg.lstsq(design * np.sqrt(w[:, None]), x * np.sqrt(w), rcond=None)[0]
    residual = x - design @ beta
    return float(beta[0]), float(beta[1]), residual


def compute_medial_reference(core_mask: np.ndarray, core_info: dict) -> dict:
    distance = cv2.distanceTransform(np.where(core_mask > 0, 255, 0).astype(np.uint8), cv2.DIST_L2, 5)
    y_min, y_max = int(core_info["y_min"]), int(core_info["y_max"])
    step = max(1, int(cfg("roi_preparation", "sample_row_step_px", default=6)))
    min_radius = float(cfg("medial_axis", "minimum_peak_radius_px", default=4.0))
    ys, xs, radii, widths = [], [], [], []
    for y in range(y_min, y_max + 1, step):
        valid_x = np.flatnonzero(core_mask[y] > 0)
        if valid_x.size == 0:
            continue
        values = distance[y, valid_x]
        peak = float(np.max(values))
        if peak < min_radius:
            continue
        peak_xs = valid_x[np.flatnonzero(values >= peak - 1e-5)]
        center = float(core_info["row_center"][y])
        x = float(peak_xs[int(np.argmin(np.abs(peak_xs.astype(np.float64) - center)))])
        ys.append(float(y))
        xs.append(x)
        radii.append(peak)
        widths.append(max(1.0, float(core_info["row_half_width"][y])))
    if len(ys) < 3:
        return {"available": False, "reason": "insufficient_medial_rows", "points": []}
    y_arr = np.asarray(ys, dtype=np.float64)
    x_arr = np.asarray(xs, dtype=np.float64)
    radius_arr = np.asarray(radii, dtype=np.float64)
    width_arr = np.asarray(widths, dtype=np.float64)
    weights = radius_arr / max(float(np.median(radius_arr)), 1e-6)
    a, b, residual = robust_line_fit(y_arr, x_arr, weights)
    return {
        "available": True,
        "a": float(a),
        "b": float(b),
        "tilt_deg": float(math.degrees(math.atan(a))),
        "valid_row_count": int(y_arr.size),
        "points": [
            {"y": float(y), "x": float(x), "radius_px": float(r), "half_width_px": float(w)}
            for y, x, r, w in zip(y_arr, x_arr, radius_arr, width_arr)
        ],
        "fit_median_abs_residual_px": float(np.median(np.abs(residual))),
        "y": y_arr,
        "x": x_arr,
        "radius": radius_arr,
        "half_width": width_arr,
    }


def candidate_medial_score(candidate: dict, medial: dict) -> dict:
    if not medial.get("available"):
        return {"available": False, "score": None, "reason": medial.get("reason", "medial_unavailable")}
    y = medial["y"]
    candidate_x = np.asarray([axis_x_at_y(candidate, yi) for yi in y], dtype=np.float64)
    normalized = np.abs(candidate_x - medial["x"]) / np.maximum(medial["half_width"], 1.0)
    median_normalized = float(np.median(normalized))
    sigma = max(1e-6, float(cfg("medial_axis", "normalized_distance_sigma", default=0.085)))
    distance_score = float(math.exp(-((median_normalized / sigma) ** 2)))
    tilt_delta = abs(axis_tilt_deg(candidate) - float(medial["tilt_deg"]))
    tilt_sigma = max(1e-6, float(cfg("medial_axis", "tilt_difference_sigma_deg", default=2.5)))
    tilt_score = float(math.exp(-((tilt_delta / tilt_sigma) ** 2)))
    dw = float(cfg("medial_axis", "distance_weight", default=0.82))
    tw = float(cfg("medial_axis", "tilt_weight", default=0.18))
    total = max(dw + tw, 1e-9)
    score = (dw * distance_score + tw * tilt_score) / total
    return {
        "available": True,
        "score": clip01(score),
        "median_normalized_distance": median_normalized,
        "median_distance_px": float(np.median(np.abs(candidate_x - medial["x"]))),
        "distance_score": clip01(distance_score),
        "tilt_difference_deg": float(tilt_delta),
        "tilt_score": clip01(tilt_score),
    }


def compute_structural_anchors(core_info: dict) -> dict:
    y_min, y_max = int(core_info["y_min"]), int(core_info["y_max"])
    height = max(1, y_max - y_min + 1)
    anchors = []
    minimum_rows = int(cfg("structural_anchors", "minimum_valid_rows_per_zone", default=8))
    for zone in cfg("structural_anchors", "zones", default=[]):
        start = y_min + int(round(float(zone["start_ratio"]) * (height - 1)))
        end = y_min + int(round(float(zone["end_ratio"]) * (height - 1)))
        ys = np.arange(max(y_min, start), min(y_max, end) + 1)
        centers = core_info["row_center"][ys]
        widths = core_info["row_half_width"][ys]
        valid = np.isfinite(centers) & (widths > 1.0)
        if int(np.sum(valid)) < minimum_rows:
            anchors.append({"name": str(zone["name"]), "available": False})
            continue
        anchors.append({
            "name": str(zone["name"]),
            "available": True,
            "y": float(np.median(ys[valid])),
            "x": float(np.median(centers[valid])),
            "half_width_px": float(np.median(widths[valid])),
            "valid_row_count": int(np.sum(valid)),
        })
    return {"available": sum(bool(a.get("available")) for a in anchors) >= 2, "anchors": anchors}


def candidate_anchor_score(candidate: dict, anchor_reference: dict) -> dict:
    valid_anchors = [a for a in anchor_reference.get("anchors", []) if a.get("available")]
    if len(valid_anchors) < 2:
        return {"available": False, "score": None, "zones": valid_anchors, "reason": "insufficient_anchor_zones"}
    sigma = max(1e-6, float(cfg("structural_anchors", "normalized_distance_sigma", default=0.11)))
    zones = []
    scores = []
    for anchor in valid_anchors:
        distance = abs(axis_x_at_y(candidate, anchor["y"]) - anchor["x"])
        normalized = distance / max(1.0, anchor["half_width_px"])
        score = float(math.exp(-((normalized / sigma) ** 2)))
        zones.append({**anchor, "distance_px": float(distance), "normalized_distance": float(normalized), "score": clip01(score)})
        scores.append(max(score, 1e-6))
    overall = float(math.exp(float(np.mean(np.log(np.asarray(scores, dtype=np.float64))))))
    return {"available": True, "score": clip01(overall), "zones": zones, "worst_zone_score": float(min(scores))}


def candidate_roi_balance_score(candidate: dict, core_info: dict) -> dict:
    y_min, y_max = int(core_info["y_min"]), int(core_info["y_max"])
    step = max(1, int(cfg("roi_preparation", "sample_row_step_px", default=6)))
    scores = []
    inside = 0
    total = 0
    for y in range(y_min, y_max + 1, step):
        left, right = core_info["row_left"][y], core_info["row_right"][y]
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            continue
        total += 1
        x = axis_x_at_y(candidate, y)
        if x <= left or x >= right:
            scores.append(0.0)
            continue
        inside += 1
        dl, dr = x - left, right - x
        scores.append(clip01(1.0 - abs(dl - dr) / max(dl + dr, 1e-6)))
    if total == 0:
        return {"available": False, "score": None, "reason": "no_valid_roi_rows"}
    inside_ratio = inside / total
    raw = float(np.median(scores)) if scores else 0.0
    power = float(cfg("roi_balance", "valid_ratio_power", default=0.35))
    score = raw * (inside_ratio ** power)
    return {
        "available": True,
        "score": clip01(score),
        "median_balance_score": clip01(raw),
        "axis_inside_ratio": float(inside_ratio),
    }


def mean_axis_distance(first: dict, second: dict, y_min: int, y_max: int, sample_count: int) -> float:
    ys = np.linspace(float(y_min), float(y_max), max(2, int(sample_count)))
    distances = [abs(axis_x_at_y(first, y) - axis_x_at_y(second, y)) for y in ys]
    return float(np.mean(distances))
