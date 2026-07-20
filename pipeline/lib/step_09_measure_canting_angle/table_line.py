from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable

import cv2
import numpy as np

from . import context
from .geometry import axis_x, union_length


def build_exclusion_mask(
    shape: tuple[int, int],
    evaluation_mask: np.ndarray | None,
    axis_candidate: dict,
    y_min: int,
    y_max: int,
) -> tuple[np.ndarray, dict]:
    height, width = shape
    cfg = context.STEP_CONFIG.get("evaluation_mask", {})
    if evaluation_mask is not None and evaluation_mask.shape[:2] == shape:
        mask = (evaluation_mask > 0).astype(np.uint8) * 255
        kernel_w = max(3, int(round(width * float(cfg.get("dilate_kernel_width_ratio", 0.030)))))
        kernel_h = max(3, int(round(height * float(cfg.get("dilate_kernel_height_ratio", 0.010)))))
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
        mask = cv2.dilate(mask, kernel, iterations=max(1, int(cfg.get("dilate_iterations", 1))))
        return mask, {
            "source": "step_07_evaluation_mask",
            "kernel_size": [kernel_w, kernel_h],
            "pixel_count": int(np.count_nonzero(mask)),
        }

    mask = np.zeros(shape, dtype=np.uint8)
    half_width = max(50, int(round(width * float(cfg.get("fallback_axis_half_width_ratio", 0.18)))))
    margin = int(round(max(1, y_max - y_min) * float(cfg.get("fallback_vertical_margin_ratio", 0.04))))
    for y in range(max(0, y_min - margin), min(height, y_max + margin + 1)):
        center = int(round(axis_x(axis_candidate, y)))
        cv2.line(mask, (max(0, center - half_width), y), (min(width - 1, center + half_width), y), 255, 1)
    return mask, {
        "source": "axis_centered_fallback",
        "half_width_px": int(half_width),
        "pixel_count": int(np.count_nonzero(mask)),
    }


def _search_bounds(height: int, y_min: int, y_max: int, ratios: Iterable[float]) -> tuple[int, int]:
    start_ratio, end_ratio = [float(value) for value in ratios]
    span = max(1, y_max - y_min)
    start = max(0, int(round(y_min + start_ratio * span)))
    end = min(height - 1, int(round(y_min + end_ratio * span)))
    if end <= start:
        end = min(height - 1, start + max(10, span // 5))
    return start, end


def _line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1)))
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def extract_horizontal_segments(
    edge: np.ndarray,
    exclusion_mask: np.ndarray,
    y_min: int,
    y_max: int,
    axis_candidate: dict,
    search_cfg: dict,
) -> tuple[list[dict], dict, np.ndarray]:
    height, width = edge.shape
    ratios = search_cfg.get("vertical_range_ratio", search_cfg.get("primary_vertical_range_ratio", [0.68, 0.96]))
    search_y_min, search_y_max = _search_bounds(height, y_min, y_max, ratios)
    search = np.zeros_like(edge)
    search[search_y_min : search_y_max + 1] = edge[search_y_min : search_y_max + 1]
    search[exclusion_mask > 0] = 0

    rho = float(search_cfg.get("hough_rho_px", 1.0))
    theta = math.radians(float(search_cfg.get("hough_theta_deg", 0.05)))
    threshold = max(8, int(search_cfg.get("hough_threshold", 25)))
    min_length = max(30, int(round(width * float(search_cfg.get("min_line_length_ratio", 0.023)))))
    max_gap = max(5, int(round(width * float(search_cfg.get("max_line_gap_ratio", 0.012)))))
    # Hough only needs the narrow vertical search crop. The proposal stage is
    # additionally downscaled; final fitting still uses full-resolution pixels.
    search_crop = search[search_y_min : search_y_max + 1]
    detection_scale = float(np.clip(search_cfg.get("hough_detection_scale", 0.50), 0.25, 1.0))
    if detection_scale < 0.999:
        search_hough = cv2.resize(
            search_crop,
            (max(1, int(round(search_crop.shape[1] * detection_scale))),
             max(1, int(round(search_crop.shape[0] * detection_scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        search_hough = search_crop
    raw = cv2.HoughLinesP(
        search_hough,
        rho=max(0.5, rho * detection_scale),
        theta=theta,
        threshold=max(8, int(round(threshold * detection_scale))),
        minLineLength=max(15, int(round(min_length * detection_scale))),
        maxLineGap=max(3, int(round(max_gap * detection_scale))),
    )

    center_x = width / 2.0
    axis_y = (search_y_min + search_y_max) / 2.0
    axis_center_x = axis_x(axis_candidate, axis_y)
    max_angle = float(search_cfg.get("max_abs_angle_deg", 5.0))
    segments: list[dict] = []
    if raw is not None:
        for raw_line in raw[:, 0]:
            x1, y1, x2, y2 = [float(value) / detection_scale for value in raw_line]
            y1 += float(search_y_min)
            y2 += float(search_y_min)
            if x2 < x1:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            dx = x2 - x1
            if dx <= 1e-6:
                continue
            angle = _line_angle_deg(x1, y1, x2, y2)
            if abs(angle) > max_angle:
                continue
            slope = (y2 - y1) / dx
            y_at_center = y1 + slope * (center_x - x1)
            y_at_axis = y1 + slope * (axis_center_x - x1)
            segments.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length_px": float(math.hypot(dx, y2 - y1)),
                "angle_deg": angle,
                "slope": float(slope),
                "y_at_image_center": float(y_at_center),
                "y_at_axis": float(y_at_axis),
            })

    segments.sort(
        key=lambda item: (
            -float(item["length_px"]),
            abs(float(item["angle_deg"])),
            float(item["y_at_image_center"]),
            float(item["x1"]),
            float(item["x2"]),
        )
    )
    segments = segments[: max(1, int(search_cfg.get("max_segments", 700)))]
    return segments, {
        "search_y_min": int(search_y_min),
        "search_y_max": int(search_y_max),
        "hough_threshold": int(threshold),
        "min_line_length_px": int(min_length),
        "max_line_gap_px": int(max_gap),
        "raw_hough_count": 0 if raw is None else int(len(raw)),
        "horizontal_segment_count": int(len(segments)),
        "hough_detection_scale": float(detection_scale),
    }, search


def cluster_segments(segments: list[dict], search_cfg: dict) -> list[list[dict]]:
    angle_tolerance = float(search_cfg.get("cluster_angle_tolerance_deg", 0.80))
    y_tolerance = float(search_cfg.get("cluster_y_tolerance_px", 11.0))
    clusters: list[list[dict]] = []
    for segment in segments:
        best_cluster: list[dict] | None = None
        best_distance = float("inf")
        for cluster in clusters:
            angle_center = float(np.median([item["angle_deg"] for item in cluster]))
            y_center = float(np.median([item["y_at_image_center"] for item in cluster]))
            angle_distance = abs(float(segment["angle_deg"]) - angle_center)
            y_distance = abs(float(segment["y_at_image_center"]) - y_center)
            if angle_distance <= angle_tolerance and y_distance <= y_tolerance:
                normalized_distance = angle_distance / max(angle_tolerance, 1e-9) + y_distance / max(y_tolerance, 1e-9)
                if normalized_distance < best_distance:
                    best_distance = normalized_distance
                    best_cluster = cluster
        if best_cluster is None:
            clusters.append([segment])
        else:
            best_cluster.append(segment)
    return clusters


def _fit_line_from_points(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 2:
        raise RuntimeError("At least two points are required to fit a line")
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(float(vx)) <= 1e-8:
        raise RuntimeError("Degenerate near-vertical table-line fit")
    slope = float(vy / vx)
    intercept = float(y0 - slope * x0)
    return slope, intercept


def _line_residuals(points: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    denominator = math.sqrt(1.0 + slope * slope)
    return np.abs(points[:, 1] - (slope * points[:, 0] + intercept)) / denominator


def _side_boundaries(exclusion_mask: np.ndarray, y: float, fallback_axis_x: float) -> tuple[float, float]:
    height, width = exclusion_mask.shape
    row_index = int(np.clip(round(y), 0, height - 1))
    xs = np.flatnonzero(exclusion_mask[row_index] > 0)
    if xs.size:
        return float(xs.min()), float(xs.max())
    half = 0.18 * width
    return max(0.0, fallback_axis_x - half), min(float(width - 1), fallback_axis_x + half)


def _interval_coverage(cluster: list[dict], left_boundary: float, right_boundary: float, width: int, merge_gap: float) -> dict:
    left_intervals = []
    right_intervals = []
    all_intervals = []
    for segment in cluster:
        interval = (float(segment["x1"]), float(segment["x2"]))
        all_intervals.append(interval)
        if interval[0] < left_boundary:
            left_intervals.append((interval[0], min(interval[1], left_boundary)))
        if interval[1] > right_boundary:
            right_intervals.append((max(interval[0], right_boundary), interval[1]))
    left_length = union_length(left_intervals, merge_gap)
    right_length = union_length(right_intervals, merge_gap)
    total_length = union_length(all_intervals, merge_gap)
    left_available = max(1.0, left_boundary)
    right_available = max(1.0, float(width - 1) - right_boundary)
    left_ratio = float(np.clip(left_length / left_available, 0.0, 1.0))
    right_ratio = float(np.clip(right_length / right_available, 0.0, 1.0))
    total_ratio = float(np.clip(total_length / max(1.0, width - (right_boundary - left_boundary)), 0.0, 1.0))
    bilateral_ratio = min(left_ratio, right_ratio)
    balance = float(2.0 * min(left_ratio, right_ratio) / max(1e-9, left_ratio + right_ratio))
    return {
        "left_union_length_px": float(left_length),
        "right_union_length_px": float(right_length),
        "total_union_length_px": float(total_length),
        "left_coverage_ratio": left_ratio,
        "right_coverage_ratio": right_ratio,
        "total_coverage_ratio": total_ratio,
        "bilateral_coverage_ratio": bilateral_ratio,
        "left_right_balance": balance,
    }


def _collect_support_pixels(
    search_points: np.ndarray,
    slope: float,
    intercept: float,
    support_band_px: float,
) -> np.ndarray:
    if search_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    xs = search_points[:, 0]
    ys = search_points[:, 1]
    residual = np.abs(ys - (slope * xs + intercept)) / math.sqrt(1.0 + slope * slope)
    keep = residual <= float(support_band_px)
    return search_points[keep]


def refine_table_line(
    search_points: np.ndarray,
    initial_slope: float,
    initial_intercept: float,
    search_info: dict,
    refine_cfg: dict,
) -> dict:
    slope, intercept = float(initial_slope), float(initial_intercept)
    support_band = float(refine_cfg.get("support_band_px", 5.0))
    residual_keep = float(refine_cfg.get("residual_keep_px", 7.0))
    points = _collect_support_pixels(
        search_points,
        slope,
        intercept,
        support_band,
    )
    for _ in range(max(1, int(refine_cfg.get("max_iterations", 5)))):
        if points.shape[0] < 2:
            break
        slope, intercept = _fit_line_from_points(points)
        residuals = _line_residuals(points, slope, intercept)
        robust_limit = min(
            residual_keep,
            max(2.0, float(np.median(residuals)) + 2.8 * float(np.median(np.abs(residuals - np.median(residuals))))),
        )
        next_points = points[residuals <= robust_limit]
        if next_points.shape[0] == points.shape[0]:
            break
        points = next_points
    if points.shape[0] >= 2:
        slope, intercept = _fit_line_from_points(points)
        residuals = _line_residuals(points, slope, intercept)
    else:
        residuals = np.asarray([], dtype=np.float64)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "angle_deg": float(math.degrees(math.atan(slope))),
        "support_points": points,
        "support_pixel_count": int(points.shape[0]),
        "fit_rmse_px": None if residuals.size == 0 else float(np.sqrt(np.mean(residuals ** 2))),
        "fit_median_abs_residual_px": None if residuals.size == 0 else float(np.median(residuals)),
        "fit_p90_abs_residual_px": None if residuals.size == 0 else float(np.quantile(residuals, 0.90)),
    }


def _support_bin_metrics(
    points: np.ndarray,
    width: int,
    left_boundary: float,
    right_boundary: float,
    bin_count: int,
) -> dict:
    if points.shape[0] == 0:
        return {
            "occupied_bin_ratio": 0.0,
            "left_occupied_bin_ratio": 0.0,
            "right_occupied_bin_ratio": 0.0,
            "occupied_bin_count": 0,
        }
    bins = np.clip((points[:, 0] / max(1.0, width) * bin_count).astype(int), 0, bin_count - 1)
    occupied = set(int(value) for value in bins)
    left_max_bin = int(np.clip(left_boundary / max(1.0, width) * bin_count, 0, bin_count))
    right_min_bin = int(np.clip(right_boundary / max(1.0, width) * bin_count, 0, bin_count))
    left_bins = set(range(0, max(0, left_max_bin)))
    right_bins = set(range(min(bin_count, right_min_bin), bin_count))
    available_bins = left_bins | right_bins
    occupied_available = occupied & available_bins
    return {
        "occupied_bin_ratio": float(len(occupied_available) / max(1, len(available_bins))),
        "left_occupied_bin_ratio": float(len(occupied & left_bins) / max(1, len(left_bins))),
        "right_occupied_bin_ratio": float(len(occupied & right_bins) / max(1, len(right_bins))),
        "occupied_bin_count": int(len(occupied_available)),
    }


def _pre_score_cluster(
    cluster: list[dict],
    exclusion_mask: np.ndarray,
    y_min: int,
    y_max: int,
    axis_candidate: dict,
    search_cfg: dict,
) -> float:
    if len(cluster) < max(1, int(search_cfg.get("minimum_cluster_segments", 2))):
        return -1.0
    endpoints = np.asarray(
        [[segment["x1"], segment["y1"]] for segment in cluster]
        + [[segment["x2"], segment["y2"]] for segment in cluster],
        dtype=np.float64,
    )
    try:
        slope, intercept = _fit_line_from_points(endpoints)
    except RuntimeError:
        return -1.0
    width = exclusion_mask.shape[1]
    axis_mid_y = (y_min + y_max) / 2.0
    y_axis = float(slope * axis_x(axis_candidate, axis_mid_y) + intercept)
    fallback_axis_x = axis_x(axis_candidate, y_axis)
    left_boundary, right_boundary = _side_boundaries(exclusion_mask, y_axis, fallback_axis_x)
    coverage = _interval_coverage(
        cluster,
        left_boundary,
        right_boundary,
        width,
        float(search_cfg.get("cluster_merge_gap_px", 7.0)),
    )
    normalized_y = float((y_axis - y_min) / max(1.0, y_max - y_min))
    location_center = float(search_cfg.get("location_center_ratio", 0.82))
    location_sigma = max(1e-6, float(search_cfg.get("location_sigma_ratio", 0.095)))
    location_prior = math.exp(-0.5 * ((normalized_y - location_center) / location_sigma) ** 2)
    angle = math.degrees(math.atan(slope))
    horizontal_prior = math.exp(-0.5 * (abs(angle) / 2.5) ** 2)
    bilateral = min(1.0, coverage["bilateral_coverage_ratio"] / 0.35)
    total = min(1.0, coverage["total_coverage_ratio"] / 0.60)
    return float(
        0.34 * bilateral
        + 0.30 * total
        + 0.16 * coverage["left_right_balance"]
        + 0.14 * location_prior
        + 0.06 * horizontal_prior
    )


def score_cluster(
    cluster: list[dict],
    edge: np.ndarray,
    search_points: np.ndarray,
    exclusion_mask: np.ndarray,
    y_min: int,
    y_max: int,
    axis_candidate: dict,
    search_info: dict,
    search_cfg: dict,
    refine_cfg: dict,
    score_cfg: dict,
) -> dict | None:
    if len(cluster) < max(1, int(search_cfg.get("minimum_cluster_segments", 2))):
        return None
    endpoints = np.asarray(
        [[segment["x1"], segment["y1"]] for segment in cluster]
        + [[segment["x2"], segment["y2"]] for segment in cluster],
        dtype=np.float64,
    )
    initial_slope, initial_intercept = _fit_line_from_points(endpoints)
    refined = refine_table_line(search_points, initial_slope, initial_intercept, search_info, refine_cfg)
    slope = float(refined["slope"])
    intercept = float(refined["intercept"])
    height, width = edge.shape
    y_axis = float(slope * axis_x(axis_candidate, (y_min + y_max) / 2.0) + intercept)
    fallback_axis_x = axis_x(axis_candidate, y_axis)
    left_boundary, right_boundary = _side_boundaries(exclusion_mask, y_axis, fallback_axis_x)
    coverage = _interval_coverage(
        cluster,
        left_boundary,
        right_boundary,
        width,
        float(search_cfg.get("cluster_merge_gap_px", 7.0)),
    )
    bins = _support_bin_metrics(
        refined["support_points"],
        width,
        left_boundary,
        right_boundary,
        max(16, int(refine_cfg.get("x_bin_count", 96))),
    )
    # Hough fragments are only proposals. Full-resolution supporting edge pixels
    # are stronger evidence that the fitted line exists on both sides. This also
    # prevents a downscaled Hough pass from under-reporting one side.
    hough_coverage = dict(coverage)
    effective_left = max(float(coverage["left_coverage_ratio"]), float(bins["left_occupied_bin_ratio"]))
    effective_right = max(float(coverage["right_coverage_ratio"]), float(bins["right_occupied_bin_ratio"]))
    effective_total = max(float(coverage["total_coverage_ratio"]), float(bins["occupied_bin_ratio"]))
    effective_bilateral = min(effective_left, effective_right)
    effective_balance = float(2.0 * min(effective_left, effective_right) / max(1e-9, effective_left + effective_right))
    coverage.update({
        "hough_left_coverage_ratio": hough_coverage["left_coverage_ratio"],
        "hough_right_coverage_ratio": hough_coverage["right_coverage_ratio"],
        "hough_total_coverage_ratio": hough_coverage["total_coverage_ratio"],
        "hough_bilateral_coverage_ratio": hough_coverage["bilateral_coverage_ratio"],
        "hough_left_right_balance": hough_coverage["left_right_balance"],
        "left_coverage_ratio": context.clip01(effective_left),
        "right_coverage_ratio": context.clip01(effective_right),
        "total_coverage_ratio": context.clip01(effective_total),
        "bilateral_coverage_ratio": context.clip01(effective_bilateral),
        "left_right_balance": context.clip01(effective_balance),
        "coverage_source": "max_of_hough_intervals_and_full_resolution_support_bins",
    })

    boot_span = max(1.0, float(y_max - y_min))
    normalized_y = float((y_axis - y_min) / boot_span)
    location_center = float(search_cfg.get("location_center_ratio", 0.82))
    location_sigma = max(1e-6, float(search_cfg.get("location_sigma_ratio", 0.095)))
    location_prior = float(math.exp(-0.5 * ((normalized_y - location_center) / location_sigma) ** 2))
    angle = float(refined["angle_deg"])
    horizontal_prior = float(math.exp(-0.5 * (abs(angle) / max(0.2, float(search_cfg.get("max_abs_angle_deg", 5.0)) / 2.0)) ** 2))
    rmse = refined.get("fit_rmse_px")
    p90 = refined.get("fit_p90_abs_residual_px")
    rmse_score = 0.0 if rmse is None else float(math.exp(-float(rmse) / max(1e-6, float(refine_cfg.get("fit_rmse_scale_px", 4.5)))))
    p90_score = 0.0 if p90 is None else float(math.exp(-float(p90) / max(1e-6, float(refine_cfg.get("fit_p90_scale_px", 7.0)))))
    fit_quality = float(math.sqrt(max(0.0, rmse_score * p90_score)))
    edge_support = float(math.sqrt(max(0.0, bins["left_occupied_bin_ratio"] * bins["right_occupied_bin_ratio"])))
    bilateral_score = float(np.clip(coverage["bilateral_coverage_ratio"] / 0.35, 0.0, 1.0))
    total_coverage_score = float(np.clip(coverage["total_coverage_ratio"] / 0.60, 0.0, 1.0))

    components = {
        "total_coverage": total_coverage_score,
        "bilateral_coverage": bilateral_score,
        "left_right_balance": float(coverage["left_right_balance"]),
        "edge_support": edge_support,
        "fit_quality": fit_quality,
        "location_prior": location_prior,
        "horizontal_prior": horizontal_prior,
    }
    weights = {
        "total_coverage": float(score_cfg.get("total_coverage_weight", 0.22)),
        "bilateral_coverage": float(score_cfg.get("bilateral_coverage_weight", 0.20)),
        "left_right_balance": float(score_cfg.get("left_right_balance_weight", 0.10)),
        "edge_support": float(score_cfg.get("edge_support_weight", 0.18)),
        "fit_quality": float(score_cfg.get("fit_quality_weight", 0.14)),
        "location_prior": float(score_cfg.get("location_prior_weight", 0.11)),
        "horizontal_prior": float(score_cfg.get("horizontal_prior_weight", 0.05)),
    }
    denominator = sum(weights.values())
    score = sum(components[name] * weights[name] for name in components) / max(1e-9, denominator)

    valid_reasons = []
    if coverage["total_coverage_ratio"] < float(search_cfg.get("minimum_total_coverage_ratio", 0.34)):
        valid_reasons.append("insufficient_total_coverage")
    if coverage["left_coverage_ratio"] < float(search_cfg.get("minimum_left_coverage_ratio", 0.18)):
        valid_reasons.append("insufficient_left_coverage")
    if coverage["right_coverage_ratio"] < float(search_cfg.get("minimum_right_coverage_ratio", 0.18)):
        valid_reasons.append("insufficient_right_coverage")
    if coverage["left_right_balance"] < float(search_cfg.get("minimum_bilateral_balance", 0.42)):
        valid_reasons.append("unbalanced_left_right_support")
    if refined["support_pixel_count"] < int(refine_cfg.get("minimum_support_pixels", 180)):
        valid_reasons.append("insufficient_support_pixels")
    if bins["occupied_bin_ratio"] < float(refine_cfg.get("minimum_occupied_bin_ratio", 0.30)):
        valid_reasons.append("insufficient_horizontal_continuity")

    return {
        "score": context.clip01(score),
        "score_percent": 100.0 * context.clip01(score),
        "valid": not valid_reasons,
        "rejection_reasons": valid_reasons,
        "slope": slope,
        "intercept": intercept,
        "angle_deg": angle,
        "y_at_axis": y_axis,
        "normalized_y_in_boot": normalized_y,
        "left_boundary_x": float(left_boundary),
        "right_boundary_x": float(right_boundary),
        "cluster_segment_count": int(len(cluster)),
        "cluster_total_segment_length_px": float(sum(segment["length_px"] for segment in cluster)),
        "coverage": coverage,
        "support_bins": bins,
        "components": components,
        "component_weights": weights,
        "refinement": {key: value for key, value in refined.items() if key != "support_points"},
        "support_points": refined["support_points"],
        "segments": cluster,
    }


def detect_table_line_once(
    edge: np.ndarray,
    exclusion_mask: np.ndarray,
    y_min: int,
    y_max: int,
    axis_candidate: dict,
    override: dict | None = None,
) -> dict:
    search_cfg = deepcopy(context.STEP_CONFIG.get("table_search", {}))
    if override:
        search_cfg.update(deepcopy(override))
    search_cfg.setdefault("vertical_range_ratio", search_cfg.get("primary_vertical_range_ratio", [0.68, 0.96]))
    refine_cfg = context.STEP_CONFIG.get("line_refinement", {})
    score_cfg = context.STEP_CONFIG.get("table_score", {})
    segments, search_info, search_edge = extract_horizontal_segments(
        edge,
        exclusion_mask,
        y_min,
        y_max,
        axis_candidate,
        search_cfg,
    )
    clusters = cluster_segments(segments, search_cfg)
    prescored = [
        (
            _pre_score_cluster(cluster, exclusion_mask, y_min, y_max, axis_candidate, search_cfg),
            cluster_index,
            cluster,
        )
        for cluster_index, cluster in enumerate(clusters)
    ]
    prescored.sort(key=lambda item: (item[0], len(item[2]), -item[1]), reverse=True)
    prescored = prescored[: max(1, int(search_cfg.get("max_clusters_to_refine", 10)))]
    ys, xs = np.nonzero(search_edge)
    search_points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    candidates = []
    for pre_score, cluster_index, cluster in prescored:
        candidate = score_cluster(
            cluster,
            edge,
            search_points,
            exclusion_mask,
            y_min,
            y_max,
            axis_candidate,
            search_info,
            search_cfg,
            refine_cfg,
            score_cfg,
        )
        if candidate is not None:
            candidate["pre_score"] = float(pre_score)
            candidate["cluster_index"] = int(cluster_index)
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            bool(item["valid"]),
            float(item["score"]),
            float(item["coverage"]["bilateral_coverage_ratio"]),
            float(item["coverage"]["total_coverage_ratio"]),
            -abs(float(item["angle_deg"])),
            -float(item["y_at_axis"]),
        ),
        reverse=True,
    )
    return {
        "available": bool(candidates),
        "winner": None if not candidates else candidates[0],
        "candidates": candidates,
        "search_info": search_info,
        "search_config": search_cfg,
        "search_edge": search_edge,
    }


def _equivalent(line_a: dict, line_b: dict, angle_tolerance: float, y_tolerance: float) -> bool:
    return (
        abs(float(line_a["angle_deg"]) - float(line_b["angle_deg"])) <= angle_tolerance
        and abs(float(line_a["y_at_axis"]) - float(line_b["y_at_axis"])) <= y_tolerance
    )


def detect_table_line_with_stability(
    edge: np.ndarray,
    exclusion_mask: np.ndarray,
    y_min: int,
    y_max: int,
    axis_candidate: dict,
) -> dict:
    stability_cfg = context.STEP_CONFIG.get("stability", {})
    variants = stability_cfg.get("variants", [{"name": "default", "override": {}}])
    runs = []
    for variant in variants:
        result = detect_table_line_once(
            edge,
            exclusion_mask,
            y_min,
            y_max,
            axis_candidate,
            variant.get("override", {}),
        )
        winner = result.get("winner")
        runs.append({
            "name": str(variant.get("name", "variant")),
            "available": bool(winner is not None),
            "winner": winner,
            "candidates": result.get("candidates", []),
            "search_info": result.get("search_info"),
        })

    available = [run for run in runs if run["available"] and run["winner"] is not None]
    if not available:
        # One wider emergency fallback, still physically tied to the lower boot region.
        fallback_cfg = deepcopy(context.STEP_CONFIG.get("table_search", {}))
        fallback_cfg["vertical_range_ratio"] = fallback_cfg.get("fallback_vertical_range_ratio", [0.58, 1.02])
        fallback = detect_table_line_once(edge, exclusion_mask, y_min, y_max, axis_candidate, fallback_cfg)
        return {
            "available": bool(fallback.get("winner")),
            "winner": fallback.get("winner"),
            "variant_runs": runs,
            "stability": {
                "available_variant_count": 0,
                "variant_count": int(len(runs)),
                "equivalent_variant_ratio": 0.0,
                "angle_std_deg": None,
                "y_std_px": None,
                "score": 0.0,
                "fallback_used": True,
            },
            "runner_up": None if len(fallback.get("candidates", [])) < 2 else fallback["candidates"][1],
        }

    # Find the strongest consensus group among variant winners.
    angle_tol = float(stability_cfg.get("equivalent_angle_tolerance_deg", 0.22))
    y_tol = float(stability_cfg.get("equivalent_y_tolerance_px", 14.0))
    groups: list[list[dict]] = []
    for run in sorted(available, key=lambda item: (-float(item["winner"]["score"]), item["name"])):
        placed = False
        for group in groups:
            representative = group[0]["winner"]
            if _equivalent(run["winner"], representative, angle_tol, y_tol):
                group.append(run)
                placed = True
                break
        if not placed:
            groups.append([run])
    groups.sort(
        key=lambda group: (
            len(group),
            sum(float(run["winner"]["score"]) for run in group),
            max(float(run["winner"]["score"]) for run in group),
        ),
        reverse=True,
    )
    consensus_group = groups[0]
    representative_run = max(consensus_group, key=lambda item: float(item["winner"]["score"]))
    winner = representative_run["winner"]
    angles = np.asarray([float(run["winner"]["angle_deg"]) for run in consensus_group], dtype=np.float64)
    ys = np.asarray([float(run["winner"]["y_at_axis"]) for run in consensus_group], dtype=np.float64)
    equivalent_ratio = float(len(consensus_group) / max(1, len(available)))
    angle_std = float(np.std(angles)) if angles.size > 1 else 0.0
    y_std = float(np.std(ys)) if ys.size > 1 else 0.0
    angle_stability = math.exp(-angle_std / max(1e-6, float(stability_cfg.get("angle_std_scale_deg", 0.30))))
    y_stability = math.exp(-y_std / max(1e-6, float(stability_cfg.get("y_std_scale_px", 18.0))))
    stability_score = context.clip01(equivalent_ratio * math.sqrt(angle_stability * y_stability))

    # Runner-up is selected from the already-computed representative run and
    # must be geometrically distinct. No Hough work is repeated here.
    runner_up = None
    for candidate in representative_run.get("candidates", []):
        if not _equivalent(candidate, winner, angle_tol, y_tol):
            runner_up = candidate
            break
    if runner_up is None and len(groups) > 1:
        runner_up = max(groups[1], key=lambda item: float(item["winner"]["score"]))["winner"]

    margin = 1.0 if runner_up is None else max(0.0, float(winner["score"]) - float(runner_up["score"]))
    margin_component = context.clip01(margin / 0.12)
    table_quality = context.clip01(
        0.72 * float(winner["score"])
        + 0.18 * stability_score
        + 0.10 * margin_component
    )
    winner["table_line_quality_score"] = table_quality
    winner["table_line_quality_percent"] = 100.0 * table_quality
    winner["candidate_margin"] = float(margin)
    winner["candidate_margin_percent"] = 100.0 * float(margin)
    winner["candidate_margin_component"] = margin_component

    return {
        "available": True,
        "winner": winner,
        "runner_up": runner_up,
        "variant_runs": runs,
        "stability": {
            "available_variant_count": int(len(available)),
            "variant_count": int(len(runs)),
            "consensus_variant_count": int(len(consensus_group)),
            "equivalent_variant_ratio": equivalent_ratio,
            "angle_std_deg": angle_std,
            "y_std_px": y_std,
            "angle_stability_score": float(angle_stability),
            "y_stability_score": float(y_stability),
            "score": stability_score,
            "score_percent": 100.0 * stability_score,
            "fallback_used": False,
            "consensus_variant_names": [run["name"] for run in consensus_group],
        },
    }
