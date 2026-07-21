from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .context import cfg, clip01
from .geometry import (
    axis_tilt_deg,
    axis_x_at_y,
    build_rectification_grid,
    rectify_about_axis,
    segment_ranges,
    split_mirrored_sides,
)


def _normalize_weights(values: list[float]) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(array))
    if total <= 1e-12:
        return np.full(array.shape, 1.0 / max(1, array.size), dtype=np.float64)
    return array / total


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if not np.any(w > 0):
        return float(np.mean(v))
    return float(np.average(v, weights=w))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    if float(np.sum(w)) <= 1e-12:
        return float(np.median(v))
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * np.sum(w), side="left")])


def _partial_bidirectional_chamfer(first: np.ndarray, second: np.ndarray, match_fraction: float) -> dict:
    first_bool, second_bool = first > 0, second > 0
    n_first, n_second = cv2.countNonZero(first), cv2.countNonZero(second)
    minimum = int(cfg("mirror", "min_edge_pixels_per_side", default=14))
    if n_first < minimum or n_second < minimum:
        return {"reliable": False, "score": None, "mean_distance_px": None, "first_count": n_first, "second_count": n_second}
    first_dt = cv2.distanceTransform(cv2.compare(first, 0, cv2.CMP_EQ), cv2.DIST_L2, 3)
    second_dt = cv2.distanceTransform(cv2.compare(second, 0, cv2.CMP_EQ), cv2.DIST_L2, 3)
    maximum = max(1.0, float(cfg("mirror", "max_chamfer_distance_px", default=16.0)))
    scale = max(1e-6, float(cfg("mirror", "distance_score_scale_px", default=7.0)))
    d12 = np.clip(second_dt[first_bool], 0.0, maximum)
    d21 = np.clip(first_dt[second_bool], 0.0, maximum)
    fraction = float(np.clip(match_fraction, 0.20, 1.0))

    def trimmed_mean(values: np.ndarray) -> float:
        keep = max(1, int(math.ceil(values.size * fraction)))
        if keep >= values.size:
            return float(np.mean(values))
        return float(np.mean(np.partition(values, keep - 1)[:keep]))

    m12, m21 = trimmed_mean(d12), trimmed_mean(d21)
    mean_distance = 0.5 * (m12 + m21)
    score = math.exp(-mean_distance / scale)
    return {
        "reliable": True,
        "score": float(clip01(score)),
        "first_to_second_score": float(clip01(math.exp(-m12 / scale))),
        "second_to_first_score": float(clip01(math.exp(-m21 / scale))),
        "mean_distance_px": float(mean_distance),
        "first_count": n_first,
        "second_count": n_second,
    }


def _radial_histogram_similarity(first: np.ndarray, second: np.ndarray) -> float:
    h1 = np.count_nonzero(first, axis=0).astype(np.float64)
    h2 = np.count_nonzero(second, axis=0).astype(np.float64)
    sigma = max(0.0, float(cfg("mirror", "histogram_smoothing_sigma", default=2.0)))
    if sigma > 0 and h1.size > 2:
        h1 = cv2.GaussianBlur(h1.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        h2 = cv2.GaussianBlur(h2.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
    s1, s2 = float(np.sum(h1)), float(np.sum(h2))
    if s1 <= 0 or s2 <= 0:
        return 0.0
    h1, h2 = h1 / s1, h2 / s2
    return float(clip01(np.sum(np.minimum(h1, h2))))


def _row_coverage(edge: np.ndarray) -> float:
    return float(np.mean(np.any(edge, axis=1))) if edge.shape[0] else 0.0


def _band_score(left: np.ndarray, right: np.ndarray, match_fraction: float) -> dict:
    chamfer = _partial_bidirectional_chamfer(left, right, match_fraction)
    left_count = int(chamfer["first_count"])
    right_count = int(chamfer["second_count"])
    count_balance = min(left_count, right_count) / max(1, max(left_count, right_count))
    left_rows, right_rows = _row_coverage(left), _row_coverage(right)
    row_balance = min(left_rows, right_rows) / max(1e-9, max(left_rows, right_rows))
    radial = _radial_histogram_similarity(left, right)
    if not chamfer["reliable"]:
        return {
            "valid": False,
            "score": 0.0,
            "chamfer_score": None,
            "radial_histogram_score": float(radial),
            "edge_count_balance": float(count_balance),
            "row_coverage_balance": float(row_balance),
            "left_edge_count": left_count,
            "right_edge_count": right_count,
            "left_row_coverage": float(left_rows),
            "right_row_coverage": float(right_rows),
            "mean_chamfer_distance_px": None,
        }
    weights = _normalize_weights([
        float(cfg("mirror", "chamfer_weight", default=0.64)),
        float(cfg("mirror", "radial_histogram_weight", default=0.24)),
        float(cfg("mirror", "edge_count_balance_weight", default=0.12)),
    ])
    raw = float(weights[0] * chamfer["score"] + weights[1] * radial + weights[2] * count_balance)
    reliability = math.sqrt(max(0.0, count_balance * row_balance))
    power = max(0.0, float(cfg("mirror", "coverage_reliability_power", default=0.20)))
    score = raw * (reliability ** power if reliability > 0 else 0.0)
    minimum_rows = float(cfg("mirror", "min_edge_rows_per_side_ratio", default=0.16))
    valid = left_rows >= minimum_rows and right_rows >= minimum_rows
    return {
        "valid": bool(valid),
        "score": float(clip01(score)),
        "chamfer_score": float(chamfer["score"]),
        "radial_histogram_score": float(radial),
        "edge_count_balance": float(count_balance),
        "row_coverage_balance": float(row_balance),
        "left_edge_count": left_count,
        "right_edge_count": right_count,
        "left_row_coverage": float(left_rows),
        "right_row_coverage": float(right_rows),
        "mean_chamfer_distance_px": chamfer["mean_distance_px"],
    }


def _segment_score(
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    center_inside_rows: np.ndarray,
    segment_index: int,
    start: int,
    end: int,
) -> dict:
    boundaries = list(cfg("mirror", "band_boundaries", default=[0.0, 0.22, 0.52, 1.0]))
    band_weights = list(cfg("mirror", "band_weights", default=[0.46, 0.39, 0.15]))
    fractions = list(cfg("mirror", "band_match_fractions", default=[0.90, 0.76, 0.58]))
    side_width = min(left_edge.shape[1], right_edge.shape[1])
    bands: list[dict] = []
    for index in range(min(len(boundaries) - 1, len(band_weights), len(fractions))):
        x0 = int(round(side_width * float(boundaries[index])))
        x1 = int(round(side_width * float(boundaries[index + 1])))
        x0 = max(0, min(side_width, x0))
        x1 = max(x0 + 1, min(side_width, x1))
        result = _band_score(left_edge[:, x0:x1], right_edge[:, x0:x1], float(fractions[index]))
        result.update({"band_index": index, "x_start": x0, "x_end": x1 - 1, "configured_weight": float(band_weights[index])})
        bands.append(result)
    valid_bands = [band for band in bands if band["valid"]]
    if valid_bands:
        score = _weighted_mean(
            [float(band["score"]) for band in valid_bands],
            [float(band["configured_weight"]) for band in valid_bands],
        )
        evidence_weight = math.sqrt(max(1.0, sum(min(band["left_edge_count"], band["right_edge_count"]) for band in valid_bands)))
        bilateral_coverage = _weighted_mean(
            [math.sqrt(max(0.0, band["edge_count_balance"] * band["row_coverage_balance"])) for band in valid_bands],
            [float(band["configured_weight"]) for band in valid_bands],
        )
    else:
        score, evidence_weight, bilateral_coverage = 0.0, 0.0, 0.0
    inside_ratio = float(np.mean(center_inside_rows)) if center_inside_rows.size else 0.0
    valid = bool(valid_bands and inside_ratio >= 0.80)
    return {
        "segment_index": int(segment_index),
        "segment_label": f"S{segment_index + 1:02d}",
        "rectified_y_start": int(start),
        "rectified_y_end": int(end - 1),
        "valid": valid,
        "score": float(clip01(score)),
        "mirror_symmetry_percent": float(100.0 * clip01(score)),
        "bilateral_coverage_score": float(clip01(bilateral_coverage)),
        "axis_inside_corridor_row_ratio": float(inside_ratio),
        "evidence_weight": float(evidence_weight * max(0.1, bilateral_coverage)),
        "valid_band_count": int(len(valid_bands)),
        "bands": bands,
    }


def _zone_result(zone_segments: list[dict], min_valid: int) -> dict:
    valid = [segment for segment in zone_segments if segment["valid"]]
    scores = [float(segment["score"]) for segment in valid]
    weights = [float(segment["evidence_weight"]) for segment in valid]
    if valid:
        mean = _weighted_mean(scores, weights)
        median = _weighted_median(scores, weights)
        score = 0.55 * mean + 0.45 * median
        coverage = _weighted_mean([float(s["bilateral_coverage_score"]) for s in valid], weights)
    else:
        mean = median = score = coverage = 0.0
    return {
        "valid": len(valid) >= min_valid,
        "score": float(clip01(score)),
        "weighted_mean_score": float(mean),
        "weighted_median_score": float(median),
        "bilateral_coverage_score": float(clip01(coverage)),
        "valid_segment_count": int(len(valid)),
        "segment_indices": [int(s["segment_index"]) for s in zone_segments],
    }


def _aggregate_segments(segments: list[dict]) -> dict:
    total = len(segments)
    valid = [s for s in segments if s["valid"]]
    values = [float(s["score"]) for s in valid]
    weights = [float(s["evidence_weight"]) for s in valid]
    global_mean = _weighted_mean(values, weights)
    global_median = _weighted_median(values, weights)
    gm_w = _normalize_weights([
        float(cfg("segment_aggregation", "global_weighted_mean_weight", default=0.55)),
        float(cfg("segment_aggregation", "global_weighted_median_weight", default=0.45)),
    ])
    global_score = float(gm_w[0] * global_mean + gm_w[1] * global_median)

    # K=12 -> top/middle/bottom each receives four contiguous segments.
    zone_indexes = np.array_split(np.arange(total, dtype=np.int32), 3)
    min_zone = max(1, int(cfg("segment_aggregation", "min_valid_segments_per_zone", default=2)))
    zone_names = ["top", "middle", "bottom"]
    zones = {
        name: _zone_result([segments[int(i)] for i in indexes], min_zone)
        for name, indexes in zip(zone_names, zone_indexes)
    }
    zone_weights = _normalize_weights(list(cfg("segment_aggregation", "zone_weights", default=[0.30, 0.35, 0.35])))
    zone_scores = np.asarray([max(0.02, float(zones[name]["score"])) for name in zone_names], dtype=np.float64)
    zone_harmonic = float(1.0 / np.sum(zone_weights / zone_scores))
    combination = _normalize_weights([
        float(cfg("segment_aggregation", "global_score_weight", default=0.55)),
        float(cfg("segment_aggregation", "zone_harmonic_weight", default=0.45)),
    ])
    mirror_score = float(combination[0] * global_score + combination[1] * zone_harmonic)
    coverage = _weighted_mean([float(s["bilateral_coverage_score"]) for s in valid], weights) if valid else 0.0
    valid_ratio = len(valid) / max(1, total)
    coverage *= valid_ratio
    min_valid = max(1, int(cfg("segment_aggregation", "min_valid_segments", default=8)))
    zones_valid = all(zones[name]["valid"] for name in zone_names)
    valid_result = len(valid) >= min_valid and zones_valid
    return {
        "mirror_symmetry_score": float(clip01(mirror_score)),
        "mirror_symmetry_percent": float(100.0 * clip01(mirror_score)),
        "bilateral_coverage_score": float(clip01(coverage)),
        "global_segment_score": float(clip01(global_score)),
        "zone_harmonic_score": float(clip01(zone_harmonic)),
        "valid_segment_count": int(len(valid)),
        "segment_count": int(total),
        "valid_segment_ratio": float(valid_ratio),
        "zones": zones,
        "segments_valid": bool(valid_result),
    }


def _candidate_geometry_reliability(
    candidate: dict,
    consensus_info: dict,
    row_half_widths: np.ndarray,
    y_min: int,
    y_max: int,
) -> dict:
    rows = np.arange(y_min, y_max + 1, dtype=np.float64)
    candidate_x = np.asarray(axis_x_at_y(candidate, rows), dtype=np.float64)
    consensus_axis = consensus_info["consensus_axis"]
    consensus_x = np.asarray(axis_x_at_y(consensus_axis, rows), dtype=np.float64)
    widths = np.maximum(1.0, row_half_widths[y_min : y_max + 1].astype(np.float64))
    normalized_offsets = np.abs(candidate_x - consensus_x) / widths
    median_offset = float(np.median(normalized_offsets))
    p90_offset = float(np.quantile(normalized_offsets, 0.90))
    centrality_scale = max(1e-6, float(cfg("consensus_corridor", "max_normalized_candidate_offset", default=0.60)) * 0.65)
    centrality = math.exp(-((median_offset / centrality_scale) ** 2))

    candidate_tilt = axis_tilt_deg(candidate)
    tilt_median = float(consensus_axis.get("candidate_tilt_median_deg", consensus_axis.get("tilt_deg", 0.0)))
    tilt_mad = max(
        float(cfg("consensus_corridor", "tilt_mad_floor_deg", default=0.75)),
        float(consensus_axis.get("candidate_tilt_mad_deg", 0.0)),
    )
    allowed_delta = max(
        float(cfg("consensus_corridor", "absolute_max_tilt_delta_deg", default=5.0)),
        float(cfg("consensus_corridor", "tilt_mad_multiplier", default=6.0)) * tilt_mad,
    )
    tilt_delta = abs(candidate_tilt - tilt_median)
    return {
        "consensus_centrality_score": float(clip01(centrality)),
        "median_normalized_consensus_offset": median_offset,
        "p90_normalized_consensus_offset": p90_offset,
        "candidate_tilt_deg": float(candidate_tilt),
        "consensus_tilt_median_deg": float(tilt_median),
        "tilt_delta_from_consensus_deg": float(tilt_delta),
        "allowed_tilt_delta_deg": float(allowed_delta),
        "offset_valid": median_offset <= float(cfg("consensus_corridor", "max_normalized_candidate_offset", default=0.60)),
        "tilt_valid": tilt_delta <= allowed_delta,
    }


def _step06_prior(candidate: dict, all_candidates: list[dict]) -> float:
    scores = np.asarray([float(c.get("final_score", c.get("score", 0.0))) for c in all_candidates], dtype=np.float64)
    value = float(candidate.get("final_score", candidate.get("score", 0.0)))
    if scores.size <= 1 or float(np.max(scores) - np.min(scores)) <= 1e-9:
        return 0.5
    q10, q90 = float(np.quantile(scores, 0.10)), float(np.quantile(scores, 0.90))
    return float(clip01((value - q10) / max(1e-9, q90 - q10)))


def verify_candidate(
    candidate: dict,
    candidate_index: int,
    all_candidates: list[dict],
    corridor_mask: np.ndarray,
    edge_mask: np.ndarray,
    consensus_info: dict,
    row_half_widths: np.ndarray,
    y_min: int,
    y_max: int,
    half_width: int,
    segment_count: int,
    _rectification_source: np.ndarray | None = None,
    _rectification_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    _segment_ranges: list[tuple[int, int]] | None = None,
    _step06_prior_score: float | None = None,
    _final_score_weights: np.ndarray | None = None,
) -> dict:
    center_exclusion = int(cfg("mirror", "center_exclusion_px", default=4))
    if _rectification_source is None:
        _rectification_source = cv2.merge((corridor_mask, edge_mask))
    rectified_pair = rectify_about_axis(
        _rectification_source,
        candidate,
        y_min,
        y_max,
        half_width,
        cv2.INTER_NEAREST,
        rectification_grid=_rectification_grid,
    )
    rectified_corridor = rectified_pair[:, :, 0]
    rectified_edge = rectified_pair[:, :, 1]
    left_edge, right_edge = split_mirrored_sides(rectified_edge, center_exclusion)
    center_column = rectified_corridor.shape[1] // 2
    center_inside = rectified_corridor[:, center_column] > 0

    segments: list[dict] = []
    ranges = _segment_ranges if _segment_ranges is not None else segment_ranges(rectified_edge.shape[0], segment_count)
    for index, (start, end) in enumerate(ranges):
        result = _segment_score(left_edge[start:end], right_edge[start:end], center_inside[start:end], index, start, end)
        result["image_y_start"] = int(y_min + start)
        result["image_y_end"] = int(y_min + end - 1)
        segments.append(result)
    aggregate = _aggregate_segments(segments)
    geometry = _candidate_geometry_reliability(candidate, consensus_info, row_half_widths, y_min, y_max)
    axis_inside_ratio = float(np.mean(center_inside)) if center_inside.size else 0.0
    step06_prior = (
        float(_step06_prior_score)
        if _step06_prior_score is not None
        else _step06_prior(candidate, all_candidates)
    )
    score_weights = (
        _final_score_weights
        if _final_score_weights is not None
        else _normalize_weights([
            float(cfg("final_scoring", "mirror_symmetry_weight", default=0.72)),
            float(cfg("final_scoring", "bilateral_coverage_weight", default=0.10)),
            float(cfg("final_scoring", "consensus_centrality_weight", default=0.08)),
            float(cfg("final_scoring", "step_06_prior_weight", default=0.10)),
        ])
    )
    verification_score = float(
        score_weights[0] * aggregate["mirror_symmetry_score"]
        + score_weights[1] * aggregate["bilateral_coverage_score"]
        + score_weights[2] * geometry["consensus_centrality_score"]
        + score_weights[3] * step06_prior
    )
    rejection_reasons: list[str] = []
    if not aggregate["segments_valid"]:
        rejection_reasons.append("insufficient_bilateral_segments_or_zone_coverage")
    if axis_inside_ratio < float(cfg("consensus_corridor", "min_axis_inside_corridor_ratio", default=0.82)):
        rejection_reasons.append("axis_leaves_consensus_corridor")
    if not geometry["offset_valid"]:
        rejection_reasons.append("axis_too_far_from_candidate_consensus")
    if not geometry["tilt_valid"]:
        rejection_reasons.append("tilt_outlier_against_candidate_consensus")
    valid = not rejection_reasons
    if not valid:
        # Keep diagnostics meaningful while ensuring an invalid accidental match
        # cannot outrank a valid candidate.
        verification_score *= 0.50
    result = {
        "candidate_index": int(candidate_index),
        "candidate_label": f"C{candidate_index + 1:02d}",
        "source_rank": int(candidate_index + 1),
        "x_ref": float(candidate.get("x_ref", 0.0)),
        "y_ref": float(candidate.get("y_ref", 0.0)),
        "a": float(candidate.get("a", 0.0)),
        "b": float(candidate.get("b", candidate.get("x_ref", 0.0))),
        "tilt_deg": float(candidate.get("tilt_deg", axis_tilt_deg(candidate))),
        "step_06_final_score": float(candidate.get("final_score", candidate.get("score", 0.0))),
        "step_06_geometry_score": float(candidate.get("geometry_score", 0.0)),
        "step_06_mirror_score": float(candidate.get("mirror_symmetry_score", 0.0)),
        "step_06_prior_score": float(step06_prior),
        "hypothesis_source": candidate.get("hypothesis_source", candidate.get("source_hypothesis_label")),
        **aggregate,
        **geometry,
        "axis_inside_corridor_ratio": float(axis_inside_ratio),
        "verification_score": float(clip01(verification_score)),
        "verification_percent": float(100.0 * clip01(verification_score)),
        # Compatibility: this is the single final Step 07 measure.
        "symmetry_percent": float(100.0 * clip01(verification_score)),
        "verification_valid": bool(valid),
        "rejection_reasons": rejection_reasons,
        "segments": segments,
        "_rectified_edge": rectified_edge,
        "_left_edge": left_edge,
        "_right_edge": right_edge,
    }
    return result


def verify_candidates(
    candidates: list[dict],
    corridor_mask: np.ndarray,
    edge_mask: np.ndarray,
    consensus_info: dict,
    row_half_widths: np.ndarray,
    y_min: int,
    y_max: int,
    half_width: int,
    segment_count: int,
) -> list[dict]:
    rectification_source = cv2.merge((corridor_mask, edge_mask))
    rectification_grid = build_rectification_grid(y_min, y_max, half_width)
    prepared_segment_ranges = segment_ranges(y_max - y_min + 1, segment_count)
    step06_priors = [_step06_prior(candidate, candidates) for candidate in candidates]
    final_score_weights = _normalize_weights([
        float(cfg("final_scoring", "mirror_symmetry_weight", default=0.72)),
        float(cfg("final_scoring", "bilateral_coverage_weight", default=0.10)),
        float(cfg("final_scoring", "consensus_centrality_weight", default=0.08)),
        float(cfg("final_scoring", "step_06_prior_weight", default=0.10)),
    ])

    def run_candidate(item: tuple[int, dict]) -> dict:
        index, candidate = item
        return verify_candidate(
            candidate,
            index,
            candidates,
            corridor_mask,
            edge_mask,
            consensus_info,
            row_half_widths,
            y_min,
            y_max,
            half_width,
            segment_count,
            _rectification_source=rectification_source,
            _rectification_grid=rectification_grid,
            _segment_ranges=prepared_segment_ranges,
            _step06_prior_score=step06_priors[index],
            _final_score_weights=final_score_weights,
        )

    configured_workers = max(1, int(cfg("performance", "candidate_workers", default=2)))
    available_cpus = os.cpu_count() or 1
    worker_count = min(configured_workers, available_cpus, len(candidates))
    if worker_count <= 1:
        verified = [run_candidate(item) for item in enumerate(candidates)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="step07-candidate") as executor:
            # executor.map preserves source order, so labels and deterministic tie
            # handling remain exactly the same as in the serial implementation.
            verified = list(executor.map(run_candidate, enumerate(candidates)))
    # Validity must be compared before score. The previous implementation put
    # score first and could allow an invalid accidental mirror match to win.
    return sorted(
        verified,
        key=lambda item: (
            bool(item["verification_valid"]),
            float(item["verification_score"]),
            float(item["mirror_symmetry_score"]),
            float(item["bilateral_coverage_score"]),
            float(item["step_06_final_score"]),
            -int(item["source_rank"]),
        ),
        reverse=True,
    )


def sanitize_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}
