from __future__ import annotations

import math

import numpy as np

from .context import cfg, clip01


def axis_x_at_y(axis: dict, y: float | np.ndarray) -> float | np.ndarray:
    if "a" in axis and "b" in axis:
        return float(axis["a"]) * y + float(axis["b"])
    return float(axis.get("x_ref", 0.0)) + float(axis.get("a", 0.0)) * (
        y - float(axis.get("y_ref", 0.0))
    )


def axis_tilt_deg(axis: dict) -> float:
    if "tilt_deg" in axis:
        return float(axis["tilt_deg"])
    return float(math.degrees(math.atan(float(axis.get("a", 0.0)))))


def mean_axis_distance(first: dict, second: dict, y_min: int, y_max: int, sample_count: int | None = None) -> float:
    count = int(sample_count or cfg("equivalence", "sample_count", default=24))
    ys = np.linspace(float(y_min), float(y_max), max(2, count))
    first_x = np.asarray(axis_x_at_y(first, ys), dtype=np.float64)
    second_x = np.asarray(axis_x_at_y(second, ys), dtype=np.float64)
    return float(np.mean(np.abs(first_x - second_x)))


def candidates_equivalent(first: dict, second: dict, y_min: int, y_max: int) -> bool:
    distance = mean_axis_distance(first, second, y_min, y_max)
    tilt = abs(axis_tilt_deg(first) - axis_tilt_deg(second))
    return (
        distance <= float(cfg("equivalence", "max_mean_axis_distance_px", default=5.0))
        and tilt <= float(cfg("equivalence", "max_tilt_difference_deg", default=0.25))
    )


def evaluation_mask_axis_support(candidate: dict, evaluation_mask: np.ndarray, y_min: int, y_max: int) -> dict:
    """Check only whether the axis remains inside Step 07's support mask.

    The mask is intentionally *not* scored by distance to its boundary because
    Step 07 created it symmetrically around a candidate consensus. Treating its
    boundary as object geometry would be circular.
    """
    height, width = evaluation_mask.shape[:2]
    y_min = max(0, int(y_min))
    y_max = min(height - 1, int(y_max))
    step = max(1, int(cfg("evaluation_mask", "sample_row_step_px", default=4)))
    checked = 0
    inside = 0
    missing_rows = 0
    for y in range(y_min, y_max + 1, step):
        row = evaluation_mask[y] > 0
        if not np.any(row):
            missing_rows += 1
            continue
        checked += 1
        x = int(round(float(axis_x_at_y(candidate, float(y)))))
        if 0 <= x < width and bool(row[x]):
            inside += 1
    if checked == 0:
        return {
            "available": False,
            "score": None,
            "axis_inside_ratio": 0.0,
            "reason": "evaluation_mask_has_no_valid_rows",
        }
    ratio = inside / checked
    minimum = float(cfg("evaluation_mask", "minimum_axis_inside_ratio", default=0.82))
    return {
        "available": True,
        "score": clip01(ratio),
        "axis_inside_ratio": float(ratio),
        "checked_row_count": int(checked),
        "missing_mask_row_count": int(missing_rows),
        "valid": bool(ratio >= minimum),
        "minimum_required_ratio": float(minimum),
    }
