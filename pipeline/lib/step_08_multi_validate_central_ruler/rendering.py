from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .context import PROCESSED_DIR, WORKING_PNG_DIR, cfg, load_color, normalize_path_value, relative_project_path
from .geometry import axis_x_at_y


COLOR_FINAL = (40, 220, 40)
COLOR_OTHER = (105, 105, 105)
COLOR_MASK = (220, 220, 220)
COLOR_ENSEMBLE = (0, 210, 255)
COLOR_TEXT = (245, 245, 245)
COLOR_BAD = (70, 70, 230)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def load_visual_background(step06_metadata: dict, step07_metadata: dict, fallback: np.ndarray) -> tuple[np.ndarray, str | None]:
    image_name = str(step07_metadata.get("image_name", step06_metadata.get("image_name", "")))
    normalized_background_subdir = str(cfg("visual_background_subdir", default="01_illumination_normalized") or "").strip()
    paths = [
        (PROCESSED_DIR / normalized_background_subdir / image_name) if image_name and normalized_background_subdir else None,
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
        (int(round(float(axis_x_at_y(candidate, y_min)))), int(y_min)),
        (int(round(float(axis_x_at_y(candidate, y_max)))), int(y_max)),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return np.where((binary > 0) & (eroded == 0), 255, 0).astype(np.uint8)


def draw_overlay(
    background: np.ndarray,
    evaluation_mask: np.ndarray,
    ranked: list[dict],
    final_candidate: dict,
    y_min: int,
    y_max: int,
    image_name: str,
    confidence: dict,
    selection_info: dict,
) -> np.ndarray:
    canvas = (_to_bgr(background).astype(np.float32) * float(cfg("drawing", "background_alpha", default=0.84))).astype(np.uint8)
    boundary = _mask_boundary(evaluation_mask)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
        canvas,
        contours,
        -1,
        COLOR_MASK,
        int(cfg("drawing", "evaluation_mask_boundary_thickness", default=2)),
        cv2.LINE_AA,
    )
    for candidate in ranked:
        if str(candidate["candidate_label"]) == str(final_candidate["candidate_label"]):
            continue
        _draw_axis(canvas, candidate, y_min, y_max, COLOR_OTHER, int(cfg("drawing", "other_axis_thickness", default=1)))
    ensemble_label = str(selection_info.get("ensemble_recommendation_label", ""))
    ensemble = next((candidate for candidate in ranked if str(candidate["candidate_label"]) == ensemble_label), None)
    if ensemble is not None and str(ensemble["candidate_label"]) != str(final_candidate["candidate_label"]):
        _draw_axis(canvas, ensemble, y_min, y_max, COLOR_ENSEMBLE, 2)
    _draw_axis(canvas, final_candidate, y_min, y_max, COLOR_FINAL, int(cfg("drawing", "axis_thickness", default=4)))
    _put_text(canvas, image_name, 22, 34, COLOR_TEXT, scale=0.72, thickness=2)
    _put_text(
        canvas,
        f"FINAL {final_candidate['candidate_label']}  confidence={confidence['confidence_percent']:.2f}%",
        22,
        66,
        COLOR_FINAL,
        scale=0.66,
        thickness=2,
    )
    _put_text(
        canvas,
        f"symmetry={100.0 * float(final_candidate['validators']['step_07_symmetry']['score']):.2f}%  multi={final_candidate['multi_validation_percent']:.2f}%  decision={confidence['decision']}",
        22,
        94,
        COLOR_TEXT,
        scale=0.50,
    )
    _put_text(
        canvas,
        f"selection={selection_info['selection_reason']}  ensemble={ensemble_label}",
        22,
        120,
        COLOR_TEXT,
        scale=0.42,
    )
    _put_text(canvas, "white=Step 07 evaluation mask; it clips evidence but its boundary is not scored", 22, 145, COLOR_TEXT, scale=0.40)
    return canvas


def _abbr(name: str) -> str:
    return {
        "step_07_symmetry": "SYM",
        "segment_consistency": "SEG",
        "perturbation_stability": "STB",
        "fragment_evidence": "FRG",
        "evaluation_mask_support": "MSK",
    }.get(name, name[:3].upper())


def build_ranking_board(
    ranked: list[dict],
    final_candidate: dict,
    width: int,
    height: int,
    confidence: dict,
    selection_info: dict,
) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    _put_text(board, "STEP 08 CONFIDENCE VALIDATION V2", 18, 34, COLOR_TEXT, scale=0.66, thickness=2)
    _put_text(
        board,
        f"confidence {confidence['confidence_percent']:.2f}% | agreement {confidence['validator_agreement_percent']:.1f}% | margin {confidence['distinct_step_07_margin_percent']:.2f}%",
        18,
        61,
        COLOR_TEXT,
        scale=0.42,
    )
    y = 98
    row_height = max(60, min(88, (height - 115) // max(1, len(ranked))))
    validator_names = (
        "step_07_symmetry",
        "segment_consistency",
        "perturbation_stability",
        "fragment_evidence",
        "evaluation_mask_support",
    )
    for rank, candidate in enumerate(ranked, start=1):
        is_final = str(candidate["candidate_label"]) == str(final_candidate["candidate_label"])
        color = COLOR_FINAL if is_final else (COLOR_TEXT if candidate["validation_valid"] else COLOR_BAD)
        prefix = "FINAL" if is_final else f"#{rank:02d}"
        _put_text(board, f"{prefix} {candidate['candidate_label']} multi {candidate['multi_validation_percent']:.2f}%", 18, y, color, scale=0.54, thickness=2 if is_final else 1)
        parts = []
        for validator in validator_names:
            result = candidate.get("validators", {}).get(validator, {})
            if result.get("available") and result.get("score") is not None:
                parts.append(f"{_abbr(validator)} {100.0 * float(result['score']):.0f}")
            else:
                parts.append(f"{_abbr(validator)} --")
        _put_text(board, "  ".join(parts), 34, y + 23, COLOR_TEXT if is_final else COLOR_OTHER, scale=0.39)
        bar = max(1, int((width - 40) * float(candidate["multi_validation_score"])))
        cv2.rectangle(board, (18, y + 32), (18 + bar, y + 39), color, -1)
        y += row_height
        if y >= height - 15:
            break
    return board


def create_comparison(
    overlay: np.ndarray,
    ranked: list[dict],
    final_candidate: dict,
    confidence: dict,
    selection_info: dict,
) -> np.ndarray:
    board = build_ranking_board(ranked, final_candidate, max(720, overlay.shape[1] // 2), overlay.shape[0], confidence, selection_info)
    return np.hstack([overlay, board])


def draw_diagnostic(evaluation_mask: np.ndarray, ranked: list[dict], final_candidate: dict, y_min: int, y_max: int) -> np.ndarray:
    image = _to_bgr(evaluation_mask)
    for candidate in ranked:
        if str(candidate["candidate_label"]) == str(final_candidate["candidate_label"]):
            continue
        _draw_axis(image, candidate, y_min, y_max, COLOR_OTHER, 1)
    _draw_axis(image, final_candidate, y_min, y_max, COLOR_FINAL, 3)
    return image
