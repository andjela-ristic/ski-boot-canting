from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .context import (
    COLOR_CORRIDOR,
    COLOR_LEFT,
    COLOR_OTHER,
    COLOR_OVERLAP,
    COLOR_RIGHT,
    COLOR_SEGMENT_BAD,
    COLOR_SEGMENT_GOOD,
    COLOR_SEGMENT_MEDIUM,
    COLOR_TEXT,
    COLOR_WINNER,
    WORKING_PNG_DIR,
    cfg,
    load_color,
    normalize_path_value,
    relative_project_path,
)
from .geometry import axis_x_at_y, mask_boundary


def to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def extract_corridor_contours(corridor_mask: np.ndarray) -> list[np.ndarray]:
    boundary = mask_boundary(corridor_mask)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def load_visual_background(metadata: dict, fallback: np.ndarray) -> tuple[np.ndarray, str | None]:
    image_name = str(metadata.get("image_name", ""))
    candidates = [
        normalize_path_value(metadata.get("source_file")),
        WORKING_PNG_DIR / image_name if image_name else None,
        normalize_path_value(metadata.get("base_edge_file")),
    ]
    for path in candidates:
        image = load_color(path)
        if image is not None and image.shape[:2] == fallback.shape[:2]:
            return image, relative_project_path(path)
    return to_bgr(fallback), None


def _segment_color(score: float, valid: bool) -> tuple[int, int, int]:
    if not valid or score < 0.42:
        return COLOR_SEGMENT_BAD
    if score < 0.62:
        return COLOR_SEGMENT_MEDIUM
    return COLOR_SEGMENT_GOOD


def _put_text(image: np.ndarray, text: str, x: int, y: int, color=COLOR_TEXT, scale: float | None = None, thickness: int = 1) -> None:
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(cfg("drawing", "font_scale", default=0.62) if scale is None else scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_axis(image: np.ndarray, candidate: dict, y_min: int, y_max: int, color: tuple[int, int, int], thickness: int) -> None:
    cv2.line(
        image,
        (int(round(float(axis_x_at_y(candidate, y_min)))), y_min),
        (int(round(float(axis_x_at_y(candidate, y_max)))), y_max),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_winner_overlay(
    background: np.ndarray,
    corridor_mask: np.ndarray,
    ranked: list[dict],
    consensus_axis: dict,
    y_min: int,
    y_max: int,
    image_name: str,
    confidence: str,
    corridor_contours: list[np.ndarray] | None = None,
) -> np.ndarray:
    canvas = (to_bgr(background).astype(np.float32) * float(cfg("drawing", "background_alpha", default=0.82))).astype(np.uint8)
    contours = corridor_contours if corridor_contours is not None else extract_corridor_contours(corridor_mask)
    cv2.drawContours(canvas, contours, -1, COLOR_CORRIDOR, int(cfg("drawing", "corridor_boundary_thickness", default=2)), cv2.LINE_AA)
    _draw_axis(canvas, consensus_axis, y_min, y_max, (160, 160, 0), 1)
    for candidate in ranked[1:]:
        _draw_axis(canvas, candidate, y_min, y_max, COLOR_OTHER, int(cfg("drawing", "other_axis_thickness", default=1)))
    winner = ranked[0]
    for segment in winner["segments"]:
        y = int(segment["image_y_start"])
        color = _segment_color(float(segment["score"]), bool(segment["valid"]))
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), color, int(cfg("drawing", "segment_line_thickness", default=1)), cv2.LINE_AA)
        y_mid = int(0.5 * (segment["image_y_start"] + segment["image_y_end"]))
        x = int(round(float(axis_x_at_y(winner, y_mid))))
        _put_text(canvas, f"{segment['segment_label']} {segment['mirror_symmetry_percent']:.1f}%", x + 16, y_mid, color, scale=0.46)
    _draw_axis(canvas, winner, y_min, y_max, COLOR_WINNER, int(cfg("drawing", "axis_thickness", default=4)))
    _put_text(canvas, image_name, 22, 34, COLOR_TEXT, scale=0.72, thickness=2)
    _put_text(canvas, f"WINNER {winner['candidate_label']}  verified={winner['verification_percent']:.2f}%", 22, 64, COLOR_WINNER, scale=0.66, thickness=2)
    _put_text(canvas, f"mirror={winner['mirror_symmetry_percent']:.2f}%  margin={winner.get('winner_margin_percent', 0.0):.2f}%  confidence={confidence}", 22, 92, COLOR_TEXT, scale=0.52)
    return canvas


def build_rectified_diagnostic(candidate: dict) -> np.ndarray:
    left, right = candidate["_left_edge"], candidate["_right_edge"]
    height, width = left.shape
    diagnostic = np.zeros((height, width, 3), dtype=np.uint8)
    left_bool, right_bool = left > 0, right > 0
    diagnostic[left_bool] = COLOR_LEFT
    diagnostic[right_bool] = COLOR_RIGHT
    diagnostic[left_bool & right_bool] = COLOR_OVERLAP
    for segment in candidate["segments"]:
        start, end = int(segment["rectified_y_start"]), int(segment["rectified_y_end"])
        color = _segment_color(float(segment["score"]), bool(segment["valid"]))
        cv2.rectangle(diagnostic, (0, start), (width - 1, end), color, 1)
        _put_text(diagnostic, f"{segment['segment_label']} {segment['mirror_symmetry_percent']:.1f}%", 6, min(end - 2, start + 18), color, scale=0.40)
    _put_text(diagnostic, "green=mirrored left  magenta=right  white=overlap", 6, height - 8, COLOR_TEXT, scale=0.34)
    return diagnostic


def draw_candidate_snapshot(
    background: np.ndarray,
    corridor_mask: np.ndarray,
    candidate: dict,
    consensus_axis: dict,
    y_min: int,
    y_max: int,
    image_name: str,
    corridor_contours: list[np.ndarray] | None = None,
) -> np.ndarray:
    left_panel = to_bgr(background)
    contours = corridor_contours if corridor_contours is not None else extract_corridor_contours(corridor_mask)
    cv2.drawContours(left_panel, contours, -1, COLOR_CORRIDOR, 1, cv2.LINE_AA)
    _draw_axis(left_panel, consensus_axis, y_min, y_max, (160, 160, 0), 1)
    _draw_axis(left_panel, candidate, y_min, y_max, COLOR_WINNER, int(cfg("drawing", "axis_thickness", default=4)))
    for segment in candidate["segments"]:
        cv2.line(left_panel, (0, int(segment["image_y_start"])), (left_panel.shape[1] - 1, int(segment["image_y_start"])), _segment_color(float(segment["score"]), bool(segment["valid"])), 1)
    _put_text(left_panel, image_name, 20, 34, COLOR_TEXT, scale=0.68, thickness=2)
    _put_text(left_panel, f"{candidate['candidate_label']} verified={candidate['verification_percent']:.2f}% mirror={candidate['mirror_symmetry_percent']:.2f}%", 20, 64, COLOR_WINNER, scale=0.56, thickness=2)
    _put_text(left_panel, f"valid={candidate['verification_valid']} seg={candidate['valid_segment_count']}/{candidate['segment_count']} centrality={candidate['consensus_centrality_score']:.3f} step06={candidate['step_06_final_score']:.3f}", 20, 91, COLOR_TEXT, scale=0.44)
    diagnostic = build_rectified_diagnostic(candidate)
    if diagnostic.shape[0] != left_panel.shape[0]:
        scale = left_panel.shape[0] / max(1, diagnostic.shape[0])
        diagnostic = cv2.resize(diagnostic, (max(1, int(round(diagnostic.shape[1] * scale))), left_panel.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.hstack([left_panel, diagnostic])


def build_ranking_board(ranked: list[dict], width: int, height: int) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    _put_text(board, "STEP 07 ROBUST MIRROR VERIFICATION", 18, 34, COLOR_TEXT, scale=0.64, thickness=2)
    y = 72
    row_height = max(48, min(78, (height - 90) // max(1, len(ranked))))
    for rank, c in enumerate(ranked, start=1):
        color = COLOR_WINNER if rank == 1 else (COLOR_TEXT if c["verification_valid"] else COLOR_SEGMENT_BAD)
        _put_text(board, f"#{rank:02d} {c['candidate_label']} verified {c['verification_percent']:.2f}%  mirror {c['mirror_symmetry_percent']:.2f}%", 20, y, color, scale=0.54, thickness=2 if rank == 1 else 1)
        _put_text(board, f"valid {c['verification_valid']} seg {c['valid_segment_count']}/{c['segment_count']} cov {c['bilateral_coverage_score']:.3f} cent {c['consensus_centrality_score']:.3f} step06 {c['step_06_final_score']:.3f}", 38, y + 23, COLOR_OTHER if rank > 1 else COLOR_TEXT, scale=0.39)
        bar_width = max(1, int((width - 45) * float(c["verification_score"])))
        cv2.rectangle(board, (20, y + 31), (20 + bar_width, y + 37), color, -1)
        y += row_height
        if y >= height - 20:
            break
    return board


def create_comparison(overlay: np.ndarray, ranked: list[dict]) -> np.ndarray:
    return np.hstack([overlay, build_ranking_board(ranked, max(560, overlay.shape[1] // 2), overlay.shape[0])])
