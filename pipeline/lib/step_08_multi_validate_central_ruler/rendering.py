from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .context import WORKING_PNG_DIR, cfg, load_color, normalize_path_value, relative_project_path
from .geometry import axis_x_at_y


COLOR_WINNER = (40, 220, 40)
COLOR_OTHER = (105, 105, 105)
COLOR_MEDIAL = (0, 220, 255)
COLOR_ANCHOR = (255, 190, 30)
COLOR_TEXT = (245, 245, 245)
COLOR_BAD = (70, 70, 230)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def load_visual_background(step06_metadata: dict, step07_metadata: dict, fallback: np.ndarray) -> tuple[np.ndarray, str | None]:
    image_name = str(step07_metadata.get("image_name", step06_metadata.get("image_name", "")))
    paths = [
        normalize_path_value(step06_metadata.get("source_file")),
        normalize_path_value(step07_metadata.get("visual_file")),
        WORKING_PNG_DIR / image_name if image_name else None,
        normalize_path_value(step06_metadata.get("base_edge_file")),
    ]
    for path in paths:
        image = load_color(path)
        if image is not None and image.shape[:2] == fallback.shape[:2]:
            return image, relative_project_path(path)
    return _to_bgr(fallback), None


def _put_text(image: np.ndarray, text: str, x: int, y: int, color=COLOR_TEXT, scale: float | None = None, thickness: int = 1) -> None:
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(cfg("drawing", "font_scale", default=0.60) if scale is None else scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_axis(image: np.ndarray, candidate: dict, y_min: int, y_max: int, color: tuple[int, int, int], thickness: int) -> None:
    cv2.line(
        image,
        (int(round(axis_x_at_y(candidate, y_min))), int(y_min)),
        (int(round(axis_x_at_y(candidate, y_max))), int(y_max)),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def draw_overlay(background: np.ndarray, ranked: list[dict], medial: dict, anchors: dict, y_min: int, y_max: int, image_name: str, confidence: dict) -> np.ndarray:
    canvas = (_to_bgr(background).astype(np.float32) * float(cfg("drawing", "background_alpha", default=0.84))).astype(np.uint8)
    for candidate in ranked[1:]:
        _draw_axis(canvas, candidate, y_min, y_max, COLOR_OTHER, int(cfg("drawing", "other_axis_thickness", default=1)))
    if medial.get("available"):
        _draw_axis(canvas, medial, y_min, y_max, COLOR_MEDIAL, int(cfg("drawing", "medial_axis_thickness", default=2)))
    radius = int(cfg("drawing", "anchor_radius", default=9))
    for anchor in anchors.get("anchors", []):
        if anchor.get("available"):
            cv2.circle(canvas, (int(round(anchor["x"])), int(round(anchor["y"]))), radius, COLOR_ANCHOR, 2, cv2.LINE_AA)
    winner = ranked[0]
    _draw_axis(canvas, winner, y_min, y_max, COLOR_WINNER, int(cfg("drawing", "axis_thickness", default=4)))
    _put_text(canvas, image_name, 22, 34, COLOR_TEXT, scale=0.72, thickness=2)
    _put_text(canvas, f"FINAL {winner['candidate_label']}  confidence={confidence['confidence_percent']:.2f}%", 22, 66, COLOR_WINNER, scale=0.66, thickness=2)
    _put_text(canvas, f"multi={winner['multi_validation_percent']:.2f}%  symmetry={winner['validators']['step_07_symmetry']['score'] * 100.0:.2f}%  decision={confidence['decision']}", 22, 94, COLOR_TEXT, scale=0.50)
    _put_text(canvas, "green=final  yellow=medial reference  orange=structural anchors", 22, 120, COLOR_TEXT, scale=0.42)
    return canvas


def _validator_abbreviation(name: str) -> str:
    return {
        "step_07_symmetry": "SYM",
        "medial_axis": "MED",
        "structural_anchors": "ANC",
        "fragment_evidence": "FRG",
        "roi_balance": "ROI",
    }.get(name, name[:3].upper())


def build_ranking_board(ranked: list[dict], validator_names: list[str], width: int, height: int, confidence: dict) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    _put_text(board, "STEP 08 MULTI-VALIDATION", 18, 34, COLOR_TEXT, scale=0.66, thickness=2)
    _put_text(board, f"confidence {confidence['confidence_percent']:.2f}% | agreement {confidence['validator_agreement_percent']:.1f}% | stability {confidence['ensemble_stability_percent']:.1f}%", 18, 61, COLOR_TEXT, scale=0.42)
    y = 98
    row_height = max(60, min(88, (height - 115) // max(1, len(ranked))))
    for rank, candidate in enumerate(ranked, start=1):
        color = COLOR_WINNER if rank == 1 else (COLOR_TEXT if candidate["validation_valid"] else COLOR_BAD)
        _put_text(board, f"#{rank:02d} {candidate['candidate_label']} multi {candidate['multi_validation_percent']:.2f}%", 18, y, color, scale=0.54, thickness=2 if rank == 1 else 1)
        parts = []
        for validator in validator_names:
            result = candidate["validators"].get(validator, {})
            if result.get("available") and result.get("score") is not None:
                parts.append(f"{_validator_abbreviation(validator)} {100.0 * float(result['score']):.0f}")
            else:
                parts.append(f"{_validator_abbreviation(validator)} --")
        _put_text(board, "  ".join(parts), 34, y + 23, COLOR_TEXT if rank == 1 else COLOR_OTHER, scale=0.39)
        bar = max(1, int((width - 40) * float(candidate["multi_validation_score"])))
        cv2.rectangle(board, (18, y + 32), (18 + bar, y + 39), color, -1)
        y += row_height
        if y >= height - 15:
            break
    return board


def create_comparison(overlay: np.ndarray, ranked: list[dict], validator_names: list[str], confidence: dict) -> np.ndarray:
    board = build_ranking_board(ranked, validator_names, max(680, overlay.shape[1] // 2), overlay.shape[0], confidence)
    return np.hstack([overlay, board])


def draw_diagnostic(core_mask: np.ndarray, medial: dict, anchors: dict, ranked: list[dict], y_min: int, y_max: int) -> np.ndarray:
    image = _to_bgr(core_mask)
    if medial.get("available"):
        for point in medial.get("points", []):
            cv2.circle(image, (int(round(point["x"])), int(round(point["y"]))), 2, COLOR_MEDIAL, -1)
        _draw_axis(image, medial, y_min, y_max, COLOR_MEDIAL, 2)
    for anchor in anchors.get("anchors", []):
        if anchor.get("available"):
            cv2.circle(image, (int(round(anchor["x"])), int(round(anchor["y"]))), 9, COLOR_ANCHOR, 2)
    if ranked:
        _draw_axis(image, ranked[0], y_min, y_max, COLOR_WINNER, 3)
    return image
