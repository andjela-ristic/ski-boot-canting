from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def axis_x(candidate: dict, y: float) -> float:
    return float(candidate.get("a", 0.0)) * float(y) + float(candidate.get("b", candidate.get("x_ref", 0.0)))


def resolve_axis_candidate(step08: dict) -> dict:
    label = str(step08.get("final_candidate", ""))
    winner = step08.get("winner")
    if isinstance(winner, dict) and (not label or str(winner.get("candidate_label", "")) == label):
        return winner
    for candidate in step08.get("ranked_candidates", []):
        if str(candidate.get("candidate_label", "")) == label:
            return candidate
    if isinstance(winner, dict):
        return winner
    raise RuntimeError("Step 08 metadata does not contain the final axis candidate geometry")


def axis_up_vector(candidate: dict, y_top: float, y_bottom: float) -> np.ndarray:
    x_top = axis_x(candidate, y_top)
    x_bottom = axis_x(candidate, y_bottom)
    vector = np.asarray([x_top - x_bottom, float(y_top) - float(y_bottom)], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("Degenerate final axis vector")
    return vector / norm


def table_direction_vector(slope: float) -> np.ndarray:
    vector = np.asarray([1.0, float(slope)], dtype=np.float64)
    return vector / float(np.linalg.norm(vector))


def table_up_normal(slope: float) -> np.ndarray:
    direction = table_direction_vector(slope)
    normal = np.asarray([direction[1], -direction[0]], dtype=np.float64)
    return normal / float(np.linalg.norm(normal))


def signed_angle_deg(vector_from: np.ndarray, vector_to: np.ndarray) -> float:
    a = np.asarray(vector_from, dtype=np.float64)
    b = np.asarray(vector_to, dtype=np.float64)
    a /= max(1e-12, float(np.linalg.norm(a)))
    b /= max(1e-12, float(np.linalg.norm(b)))
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.atan2(cross, dot)))


def acute_line_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    a /= max(1e-12, float(np.linalg.norm(a)))
    b /= max(1e-12, float(np.linalg.norm(b)))
    dot = abs(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    return float(math.degrees(math.acos(dot)))


def union_length(intervals: Iterable[tuple[float, float]], gap_tolerance: float = 0.0) -> float:
    normalized: list[tuple[float, float]] = []
    for start, end in intervals:
        a, b = sorted((float(start), float(end)))
        if b > a:
            normalized.append((a, b))
    if not normalized:
        return 0.0
    normalized.sort()
    current_start, current_end = normalized[0]
    total = 0.0
    for start, end in normalized[1:]:
        if start <= current_end + float(gap_tolerance):
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return float(total + current_end - current_start)


def weighted_mean(values: list[tuple[float, float]]) -> float | None:
    valid = [(float(value), float(weight)) for value, weight in values if np.isfinite(value) and weight > 0]
    if not valid:
        return None
    denominator = sum(weight for _, weight in valid)
    return float(sum(value * weight for value, weight in valid) / denominator)
