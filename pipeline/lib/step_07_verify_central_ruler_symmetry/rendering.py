from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .context import (
    COLOR_OTHER,
    COLOR_ROI,
    COLOR_SEGMENT_BAD,
    COLOR_SEGMENT_GOOD,
    COLOR_SEGMENT_MEDIUM,
    COLOR_TEXT,
    COLOR_WINNER,
    WORKING_PNG_DIR,
    cfg,
    load_color,
    normalize_path_value,
)
from .geometry import axis_x_at_y, mask_boundary


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def load_visual_background(metadata: dict, edge_image: np.ndarray) -> tuple[np.ndarray, str | None]:
    image_name = str(metadata.get("image_name", ""))
    candidates = [
        WORKING_PNG_DIR / image_name,
        normalize_path_value(metadata.get("visual_file")),
    ]
    for path in candidates:
        image = load_color(path)
        if image is not None and image.shape[:2] == edge_image.shape[:2]:
            return image, str(path)
    return to_bgr(edge_image), None


def _segment_color(score: float, valid: bool) -> tuple[int, int, int]:
    if not valid or score < 0.55:
        return COLOR_SEGMENT_BAD
    if score < 0.75:
        return COLOR_SEGMENT_MEDIUM
    return COLOR_SEGMENT_GOOD


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = COLOR_TEXT,
    scale: float | None = None,
    thickness: int = 1,
) -> None:
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


def _draw_axis(
    image: np.ndarray,
    candidate: dict,
    y_min: int,
    y_max: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x0 = int(round(float(axis_x_at_y(candidate, y_min))))
    x1 = int(round(float(axis_x_at_y(candidate, y_max))))
    cv2.line(image, (x0, y_min), (x1, y_max), color, thickness, cv2.LINE_AA)


def draw_winner_overlay(
    background: np.ndarray,
    core_roi_mask: np.ndarray,
    ranked: list[dict],
    y_min: int,
    y_max: int,
    image_name: str,
    confidence: str,
) -> np.ndarray:
    canvas = to_bgr(background)
    dimmed = (canvas.astype(np.float32) * float(cfg("drawing", "background_alpha", default=0.82))).astype(np.uint8)
    canvas = dimmed

    boundary = mask_boundary(core_roi_mask)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
        canvas,
        contours,
        -1,
        COLOR_ROI,
        int(cfg("drawing", "roi_boundary_thickness", default=2)),
        cv2.LINE_AA,
    )

    for candidate in ranked[1:]:
        _draw_axis(
            canvas,
            candidate,
            y_min,
            y_max,
            COLOR_OTHER,
            int(cfg("drawing", "other_axis_thickness", default=1)),
        )

    winner = ranked[0]
    for segment in winner["segments"]:
        y = int(segment["image_y_start"])
        color = _segment_color(float(segment["score"]), bool(segment["valid"]))
        cv2.line(
            canvas,
            (0, y),
            (canvas.shape[1] - 1, y),
            color,
            int(cfg("drawing", "segment_line_thickness", default=1)),
            cv2.LINE_AA,
        )
        axis_x = int(round(float(axis_x_at_y(winner, 0.5 * (segment["image_y_start"] + segment["image_y_end"])))))
        _put_text(
            canvas,
            f"{segment['segment_label']} {segment['symmetry_percent']:.1f}%",
            axis_x + 16,
            int(0.5 * (segment["image_y_start"] + segment["image_y_end"])),
            color,
            scale=0.48,
        )

    _draw_axis(
        canvas,
        winner,
        y_min,
        y_max,
        COLOR_WINNER,
        int(cfg("drawing", "axis_thickness", default=4)),
    )
    margin = float(winner.get("winner_margin_percent", 0.0))
    _put_text(canvas, image_name, 22, 34, COLOR_TEXT, scale=0.72, thickness=2)
    _put_text(
        canvas,
        f"WINNER {winner['candidate_label']}  symmetry={winner['symmetry_percent']:.2f}%",
        22,
        64,
        COLOR_WINNER,
        scale=0.68,
        thickness=2,
    )
    _put_text(
        canvas,
        f"margin={margin:.2f}%  valid={winner['valid_segment_count']}/{winner['segment_count']}  confidence={confidence}",
        22,
        92,
        COLOR_TEXT,
        scale=0.55,
    )
    return canvas


def build_rectified_diagnostic(candidate: dict) -> np.ndarray:
    mask = candidate["_rectified_mask"]
    edge = candidate["_rectified_edge"]
    height, width = mask.shape
    center = width // 2
    diagnostic = np.zeros((height, width, 3), dtype=np.uint8)

    # Original rectified mask is a subtle gray background.
    diagnostic[mask > 0] = (34, 34, 34)
    diagnostic[edge > 0] = (235, 235, 235)
    cv2.line(diagnostic, (center, 0), (center, height - 1), COLOR_WINNER, 2, cv2.LINE_AA)

    for segment in candidate["segments"]:
        start = int(segment["rectified_y_start"])
        end = int(segment["rectified_y_end"])
        color = _segment_color(float(segment["score"]), bool(segment["valid"]))
        cv2.rectangle(diagnostic, (0, start), (width - 1, end), color, 1)
        _put_text(
            diagnostic,
            f"{segment['segment_label']} {segment['symmetry_percent']:.1f}%",
            6,
            min(end - 2, start + 18),
            color,
            scale=0.42,
        )
    return diagnostic


def draw_candidate_snapshot(
    background: np.ndarray,
    core_roi_mask: np.ndarray,
    candidate: dict,
    y_min: int,
    y_max: int,
    image_name: str,
) -> np.ndarray:
    left_panel = to_bgr(background)
    boundary = mask_boundary(core_roi_mask)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left_panel, contours, -1, COLOR_ROI, 1, cv2.LINE_AA)
    _draw_axis(
        left_panel,
        candidate,
        y_min,
        y_max,
        COLOR_WINNER,
        int(cfg("drawing", "axis_thickness", default=4)),
    )
    for segment in candidate["segments"]:
        y = int(segment["image_y_start"])
        color = _segment_color(float(segment["score"]), bool(segment["valid"]))
        cv2.line(left_panel, (0, y), (left_panel.shape[1] - 1, y), color, 1)

    _put_text(left_panel, image_name, 20, 34, COLOR_TEXT, scale=0.68, thickness=2)
    _put_text(
        left_panel,
        f"{candidate['candidate_label']}  symmetry={candidate['symmetry_percent']:.2f}%",
        20,
        64,
        COLOR_WINNER,
        scale=0.64,
        thickness=2,
    )
    _put_text(
        left_panel,
        f"valid={candidate['valid_segment_count']}/{candidate['segment_count']}  step06={candidate['step_06_final_score']:.3f}",
        20,
        91,
        COLOR_TEXT,
        scale=0.50,
    )

    diagnostic = build_rectified_diagnostic(candidate)
    target_height = left_panel.shape[0]
    if diagnostic.shape[0] != target_height:
        scale = target_height / max(1, diagnostic.shape[0])
        diagnostic = cv2.resize(
            diagnostic,
            (max(1, int(round(diagnostic.shape[1] * scale))), target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.hstack([left_panel, diagnostic])


def build_ranking_board(ranked: list[dict], width: int, height: int) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    _put_text(board, "STEP 07 SYMMETRY RANKING", 18, 34, COLOR_TEXT, scale=0.68, thickness=2)
    y = 72
    row_height = max(42, min(70, (height - 90) // max(1, len(ranked))))
    for rank, candidate in enumerate(ranked, start=1):
        color = COLOR_WINNER if rank == 1 else COLOR_TEXT
        _put_text(
            board,
            f"#{rank:02d} {candidate['candidate_label']}  {candidate['symmetry_percent']:.2f}%",
            20,
            y,
            color,
            scale=0.60,
            thickness=2 if rank == 1 else 1,
        )
        _put_text(
            board,
            f"valid {candidate['valid_segment_count']}/{candidate['segment_count']}   tilt {candidate['tilt_deg']:.2f} deg   step06 {candidate['step_06_final_score']:.3f}",
            42,
            y + 23,
            COLOR_OTHER if rank > 1 else COLOR_TEXT,
            scale=0.43,
        )
        bar_x0 = 20
        bar_y = y + 31
        bar_width = max(1, int((width - 45) * float(candidate["verification_score"])))
        cv2.rectangle(board, (bar_x0, bar_y), (bar_x0 + bar_width, bar_y + 6), color, -1)
        y += row_height
        if y >= height - 20:
            break
    return board


def create_comparison(overlay: np.ndarray, ranked: list[dict]) -> np.ndarray:
    board_width = max(480, overlay.shape[1] // 2)
    board = build_ranking_board(ranked, board_width, overlay.shape[0])
    return np.hstack([overlay, board])
