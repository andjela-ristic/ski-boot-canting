from __future__ import annotations

import math

import cv2
import numpy as np

from .context import cfg, clip01
from .geometry import (
    mask_boundary,
    rectify_about_axis,
    segment_ranges,
    split_mirrored_sides,
)


def _dice_score(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first > 0
    second_bool = second > 0
    denominator = int(np.count_nonzero(first_bool)) + int(np.count_nonzero(second_bool))
    if denominator == 0:
        return 0.0
    intersection = int(np.count_nonzero(first_bool & second_bool))
    return float(2.0 * intersection / denominator)


def _bidirectional_chamfer(
    first: np.ndarray,
    second: np.ndarray,
    max_distance_px: float,
    min_pixels_per_side: int,
) -> dict:
    first_bool = first > 0
    second_bool = second > 0
    first_count = int(np.count_nonzero(first_bool))
    second_count = int(np.count_nonzero(second_bool))
    if first_count < min_pixels_per_side or second_count < min_pixels_per_side:
        return {
            "score": None,
            "left_to_right_score": None,
            "right_to_left_score": None,
            "median_distance_px": None,
            "first_pixel_count": first_count,
            "second_pixel_count": second_count,
            "reliable": False,
        }

    # distanceTransform measures distance to zero-valued pixels. The target
    # foreground is therefore encoded as zero and all other pixels as one.
    first_distance = cv2.distanceTransform(
        np.where(first_bool, 0, 1).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    second_distance = cv2.distanceTransform(
        np.where(second_bool, 0, 1).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    max_distance = max(1e-6, float(max_distance_px))
    first_to_second = np.clip(second_distance[first_bool], 0.0, max_distance)
    second_to_first = np.clip(first_distance[second_bool], 0.0, max_distance)
    ltr = clip01(1.0 - float(np.mean(first_to_second)) / max_distance)
    rtl = clip01(1.0 - float(np.mean(second_to_first)) / max_distance)
    distances = np.concatenate([first_to_second, second_to_first])
    return {
        "score": float(0.5 * (ltr + rtl)),
        "left_to_right_score": float(ltr),
        "right_to_left_score": float(rtl),
        "median_distance_px": float(np.median(distances)),
        "first_pixel_count": first_count,
        "second_pixel_count": second_count,
        "reliable": True,
    }


def _weighted_average(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if not np.any(weight_array > 0):
        return float(np.mean(value_array))
    return float(np.average(value_array, weights=np.maximum(weight_array, 1e-9)))


def _normalize_channel_weights(first: float, second: float) -> tuple[float, float]:
    first = max(0.0, float(first))
    second = max(0.0, float(second))
    total = first + second
    if total <= 1e-9:
        return 1.0, 0.0
    return first / total, second / total


def _segment_score(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    center_inside_rows: np.ndarray,
    segment_index: int,
    y_start: int,
    y_end: int,
) -> dict:
    left_mask_pixels = int(np.count_nonzero(left_mask))
    right_mask_pixels = int(np.count_nonzero(right_mask))
    min_mask_pixels = int(cfg("mirror", "min_mask_pixels_per_side", default=120))
    inside_ratio = float(np.mean(center_inside_rows)) if center_inside_rows.size else 0.0
    min_inside_ratio = float(
        cfg("mirror", "min_axis_inside_row_ratio", default=0.55)
    )

    occupancy_score = _dice_score(left_mask, right_mask)
    left_boundary = mask_boundary(left_mask)
    right_boundary = mask_boundary(right_mask)
    boundary_result = _bidirectional_chamfer(
        left_boundary,
        right_boundary,
        float(cfg("mirror", "max_chamfer_distance_px", default=12.0)),
        int(cfg("mirror", "min_boundary_pixels_per_side", default=8)),
    )
    if boundary_result["reliable"]:
        boundary_weight, occupancy_weight = _normalize_channel_weights(
            float(cfg("mirror", "silhouette_boundary_weight", default=0.72)),
            float(cfg("mirror", "silhouette_occupancy_weight", default=0.28)),
        )
        silhouette_score = (
            boundary_weight * float(boundary_result["score"])
            + occupancy_weight * occupancy_score
        )
    else:
        silhouette_score = occupancy_score

    # Only edge pixels lying in the corresponding ROI half are compared.
    left_edge_in_mask = np.where(left_mask > 0, left_edge, 0).astype(np.uint8)
    right_edge_in_mask = np.where(right_mask > 0, right_edge, 0).astype(np.uint8)
    edge_result = _bidirectional_chamfer(
        left_edge_in_mask,
        right_edge_in_mask,
        float(cfg("mirror", "max_chamfer_distance_px", default=12.0)),
        int(cfg("mirror", "min_edge_pixels_per_side", default=12)),
    )

    silhouette_weight, edge_weight = _normalize_channel_weights(
        float(cfg("mirror", "silhouette_weight", default=0.72)),
        float(cfg("mirror", "internal_edge_weight", default=0.28)),
    )
    if edge_result["reliable"]:
        combined_score = (
            silhouette_weight * silhouette_score
            + edge_weight * float(edge_result["score"])
        )
    else:
        combined_score = silhouette_score

    valid = bool(
        left_mask_pixels >= min_mask_pixels
        and right_mask_pixels >= min_mask_pixels
        and inside_ratio >= min_inside_ratio
    )
    # sqrt keeps high-information segments influential without letting the
    # widest part of a boot dominate all other segments.
    evidence_weight = math.sqrt(max(1.0, min(left_mask_pixels, right_mask_pixels)))
    evidence_weight *= max(0.10, inside_ratio)

    return {
        "segment_index": int(segment_index),
        "segment_label": f"S{segment_index + 1:02d}",
        "rectified_y_start": int(y_start),
        "rectified_y_end": int(y_end - 1),
        "valid": valid,
        "score": float(clip01(combined_score)),
        "symmetry_percent": float(100.0 * clip01(combined_score)),
        "silhouette_score": float(clip01(silhouette_score)),
        "silhouette_percent": float(100.0 * clip01(silhouette_score)),
        "occupancy_dice_score": float(clip01(occupancy_score)),
        "boundary_chamfer_score": (
            None
            if boundary_result["score"] is None
            else float(boundary_result["score"])
        ),
        "internal_edge_score": (
            None if edge_result["score"] is None else float(edge_result["score"])
        ),
        "edge_left_to_right_score": edge_result["left_to_right_score"],
        "edge_right_to_left_score": edge_result["right_to_left_score"],
        "boundary_median_distance_px": boundary_result["median_distance_px"],
        "edge_median_distance_px": edge_result["median_distance_px"],
        "left_mask_pixel_count": left_mask_pixels,
        "right_mask_pixel_count": right_mask_pixels,
        "left_edge_pixel_count": int(edge_result["first_pixel_count"]),
        "right_edge_pixel_count": int(edge_result["second_pixel_count"]),
        "axis_inside_row_ratio": float(inside_ratio),
        "evidence_weight": float(evidence_weight),
    }


def _aggregate_segments(segments: list[dict]) -> dict:
    valid = [segment for segment in segments if bool(segment["valid"])]
    valid_count = len(valid)
    total_count = len(segments)
    min_valid = min(
        total_count,
        max(1, int(cfg("segment_aggregation", "min_valid_segments", default=8))),
    )
    discard_count = max(
        0,
        int(cfg("segment_aggregation", "discard_worst_segments", default=2)),
    )

    sorted_valid = sorted(valid, key=lambda item: float(item["score"]))
    discarded = sorted_valid[: min(discard_count, max(0, valid_count - 1))]
    retained = sorted_valid[len(discarded) :]

    if retained:
        trimmed_mean = _weighted_average(
            [float(item["score"]) for item in retained],
            [float(item["evidence_weight"]) for item in retained],
        )
    else:
        trimmed_mean = 0.0
    median_score = (
        float(np.median([float(item["score"]) for item in valid]))
        if valid
        else 0.0
    )
    trimmed_weight, median_weight = _normalize_channel_weights(
        float(cfg("segment_aggregation", "trimmed_mean_weight", default=0.75)),
        float(cfg("segment_aggregation", "median_weight", default=0.25)),
    )
    robust_score = trimmed_weight * trimmed_mean + median_weight * median_score

    valid_ratio = valid_count / max(1, total_count)
    valid_power = max(
        0.0,
        float(cfg("segment_aggregation", "valid_segment_ratio_power", default=0.35)),
    )
    coverage_factor = valid_ratio ** valid_power if valid_ratio > 0 else 0.0
    final_score = robust_score * coverage_factor
    is_valid = valid_count >= min_valid
    if not is_valid:
        final_score *= valid_count / max(1, min_valid)
    final_score = max(
        float(cfg("segment_aggregation", "minimum_candidate_score", default=0.0)),
        clip01(final_score),
    )

    return {
        "verification_score": float(final_score),
        "symmetry_percent": float(100.0 * final_score),
        "robust_trimmed_mean_score": float(trimmed_mean),
        "median_segment_score": float(median_score),
        "valid_segment_count": int(valid_count),
        "segment_count": int(total_count),
        "valid_segment_ratio": float(valid_ratio),
        "discarded_segment_indices": [
            int(item["segment_index"]) for item in discarded
        ],
        "retained_segment_indices": [
            int(item["segment_index"]) for item in retained
        ],
        "verification_valid": bool(is_valid),
    }


def verify_candidate(
    candidate: dict,
    candidate_index: int,
    core_roi_mask: np.ndarray,
    edge_mask: np.ndarray,
    y_min: int,
    y_max: int,
    half_width: int,
    segment_count: int,
) -> dict:
    center_exclusion = int(cfg("mirror", "center_exclusion_px", default=4))
    rectified_mask = rectify_about_axis(
        core_roi_mask,
        candidate,
        y_min,
        y_max,
        half_width,
        interpolation=cv2.INTER_NEAREST,
    )
    rectified_edge = rectify_about_axis(
        edge_mask,
        candidate,
        y_min,
        y_max,
        half_width,
        interpolation=cv2.INTER_NEAREST,
    )
    left_mask, right_mask = split_mirrored_sides(rectified_mask, center_exclusion)
    left_edge, right_edge = split_mirrored_sides(rectified_edge, center_exclusion)
    center_column = rectified_mask.shape[1] // 2
    center_inside_rows = rectified_mask[:, center_column] > 0

    segments: list[dict] = []
    for segment_index, (start, end) in enumerate(
        segment_ranges(rectified_mask.shape[0], segment_count)
    ):
        result = _segment_score(
            left_mask[start:end],
            right_mask[start:end],
            left_edge[start:end],
            right_edge[start:end],
            center_inside_rows[start:end],
            segment_index,
            start,
            end,
        )
        result["image_y_start"] = int(y_min + start)
        result["image_y_end"] = int(y_min + end - 1)
        segments.append(result)

    aggregate = _aggregate_segments(segments)
    result = {
        "candidate_index": int(candidate_index),
        "candidate_label": f"C{candidate_index + 1:02d}",
        "source_rank": int(candidate_index + 1),
        "x_ref": float(candidate.get("x_ref", 0.0)),
        "y_ref": float(candidate.get("y_ref", 0.0)),
        "a": float(candidate.get("a", 0.0)),
        "b": float(candidate.get("b", candidate.get("x_ref", 0.0))),
        "tilt_deg": float(candidate.get("tilt_deg", 0.0)),
        "step_06_final_score": float(candidate.get("final_score", candidate.get("score", 0.0))),
        "step_06_geometry_score": float(candidate.get("geometry_score", 0.0)),
        "step_06_mirror_score": float(candidate.get("mirror_symmetry_score", 0.0)),
        "hypothesis_source": candidate.get(
            "hypothesis_source",
            candidate.get("source_hypothesis_label"),
        ),
        **aggregate,
        "segments": segments,
        # Retained for rendering only; removed from JSON by sanitize_result.
        "_rectified_mask": rectified_mask,
        "_rectified_edge": rectified_edge,
    }
    return result


def verify_candidates(
    candidates: list[dict],
    core_roi_mask: np.ndarray,
    edge_mask: np.ndarray,
    y_min: int,
    y_max: int,
    half_width: int,
    segment_count: int,
) -> list[dict]:
    verified = [
        verify_candidate(
            candidate,
            index,
            core_roi_mask,
            edge_mask,
            y_min,
            y_max,
            half_width,
            segment_count,
        )
        for index, candidate in enumerate(candidates)
    ]
    # Step 06 score is only a deterministic tie-breaker. Step 07 symmetry is
    # always the primary ranking criterion.
    return sorted(
        verified,
        key=lambda item: (
            float(item["verification_score"]),
            bool(item["verification_valid"]),
            int(item["valid_segment_count"]),
            float(item["median_segment_score"]),
            float(item["step_06_final_score"]),
        ),
        reverse=True,
    )


def sanitize_result(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }
