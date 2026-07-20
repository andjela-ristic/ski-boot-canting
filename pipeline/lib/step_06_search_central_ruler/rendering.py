from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .context import COLOR_ALL_FRAGMENTS, COLOR_CANDIDATE, COLOR_FINAL_AXIS, COLOR_SELECTED_FRAGMENTS, COLOR_TEXT, PROCESSED_DIR, PROJECT_ROOT, WORKING_PNG_DIR, cfg, get_step_dirs, put_text, resolve_project_path, to_bgr
from .geometry import line_x_at_y


def load_base_edge_image(data: dict) -> tuple[np.ndarray, str | None]:
    source_path = resolve_project_path(data.get("source_file"))
    if source_path is not None and source_path.exists():
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image, str(source_path.relative_to(PROJECT_ROOT))

    image_name = data.get("image_name", "")
    for path in [
        WORKING_PNG_DIR / image_name,
        PROCESSED_DIR / "03_edges" / "cleaned" / image_name,
    ]:
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image, str(path.relative_to(PROJECT_ROOT))

    height = int(data.get("height", 4032))
    width = int(data.get("width", 3024))
    return np.zeros((height, width), dtype=np.uint8), None

def load_step05_overlay(image_name: str) -> np.ndarray | None:
    overlay_path = get_step_dirs()["input_overlay_dir"] / image_name
    if not overlay_path.exists():
        return None
    return cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)

def draw_fragment(image: np.ndarray, line: dict, color: tuple[int, int, int], thickness: int) -> None:
    p1 = (int(round(float(line["x1"]))), int(round(float(line["y1"]))))
    p2 = (int(round(float(line["x2"]))), int(round(float(line["y2"]))))
    cv2.line(image, p1, p2, color, int(thickness), cv2.LINE_AA)

def draw_axis(image: np.ndarray, axis: dict[str, float], roi_profile: dict, color: tuple[int, int, int], thickness: int) -> None:
    y_start = int(roi_profile["trimmed_y_min"])
    y_end = int(roi_profile["trimmed_y_max"])
    p1 = (int(round(line_x_at_y(axis, y_start))), y_start)
    p2 = (int(round(line_x_at_y(axis, y_end))), y_end)
    cv2.line(image, p1, p2, color, int(thickness), cv2.LINE_AA)

def build_fragment_background(base_edge_image: np.ndarray, filtered_lines: list[dict]) -> np.ndarray:
    overlay = to_bgr(base_edge_image)
    alpha = float(cfg("drawing", "background_alpha", default=0.78))
    overlay = cv2.addWeighted(overlay, alpha, np.zeros_like(overlay), 1.0 - alpha, 0)
    for line in filtered_lines:
        draw_fragment(
            overlay,
            line,
            COLOR_ALL_FRAGMENTS,
            int(cfg("drawing", "all_fragment_thickness", default=2)),
        )
    return overlay

def draw_overlay(
    fragment_background: np.ndarray,
    filtered_line_count: int,
    best_candidate: dict | None,
    fine_candidates: list[dict],
    roi_profile: dict,
    image_name: str,
) -> np.ndarray:
    overlay = fragment_background.copy()

    if bool(cfg("drawing", "show_candidate_lines", default=True)):
        for candidate in fine_candidates[: int(cfg("drawing", "candidate_count_to_draw", default=3))]:
            draw_axis(
                overlay,
                candidate,
                roi_profile,
                COLOR_CANDIDATE,
                int(cfg("drawing", "candidate_thickness", default=2)),
            )

    if best_candidate is not None:
        for item in best_candidate.get("selected_support", []):
            draw_fragment(
                overlay,
                item.get("effective_line", item["line"]),
                COLOR_SELECTED_FRAGMENTS,
                int(cfg("drawing", "selected_fragment_thickness", default=3)),
            )
        draw_axis(
            overlay,
            best_candidate,
            roi_profile,
            COLOR_FINAL_AXIS,
            int(cfg("drawing", "final_line_thickness", default=4)),
        )

    put_text(overlay, image_name, 26, 34, COLOR_TEXT, scale=0.82)
    put_text(overlay, f"filtered fragments={int(filtered_line_count)}", 26, 62)
    if best_candidate is not None:
        put_text(
            overlay,
            (
                f"selected={best_candidate['selected_fragment_count']} "
                f"bins={best_candidate['supported_bin_count']}/{best_candidate['bin_count']} "
                f"score={best_candidate['score']:.3f}"
            ),
            26,
            90,
        )
        put_text(
            overlay,
            (
                f"tilt={best_candidate['tilt_deg']:.2f}deg "
                f"geom={best_candidate.get('geometry_score', 0.0):.3f} "
                f"mirror={best_candidate.get('mirror_symmetry_score', 0.5):.3f}"
            ),
            26,
            118,
        )
        put_text(
            overlay,
            (
                f"coverage={best_candidate.get('unique_vertical_coverage', 0.0):.3f} "
                f"span={best_candidate.get('chain_span_ratio', 0.0):.3f} "
                f"rmse={best_candidate.get('fit_rmse_px', 0.0):.1f}px"
            ),
            26,
            146,
        )
        put_text(
            overlay,
            (
                f"balance={best_candidate.get('above_below_balance_score', 0.0):.3f} "
                f"continuity={best_candidate.get('chain_continuity_ratio', 0.0):.3f} "
                f"valid={'yes' if best_candidate.get('validation_passed', False) else 'no'}"
            ),
            26,
            174,
        )

    if bool(cfg("drawing", "label_candidates", default=True)):
        for index, candidate in enumerate(fine_candidates[: int(cfg("drawing", "candidate_count_to_draw", default=3))], start=1):
            label_y = int(roi_profile["trimmed_y_min"]) + 22 + (index - 1) * 18
            label_x = int(round(line_x_at_y(candidate, label_y))) + 8
            put_text(overlay, f"C{index}:{candidate['score']:.2f}", label_x, label_y, COLOR_CANDIDATE, scale=0.54)

    return overlay

def draw_candidate_snapshot(
    fragment_background: np.ndarray,
    candidate: dict,
    roi_profile: dict,
    image_name: str,
    candidate_label: str,
) -> np.ndarray:
    overlay = fragment_background.copy()

    for item in candidate.get("selected_support", []):
        draw_fragment(
            overlay,
            item.get("effective_line", item["line"]),
            COLOR_SELECTED_FRAGMENTS,
            int(cfg("drawing", "selected_fragment_thickness", default=3)),
        )

    draw_axis(
        overlay,
        candidate,
        roi_profile,
        COLOR_CANDIDATE,
        int(cfg("drawing", "candidate_thickness", default=2)),
    )

    put_text(overlay, image_name, 26, 34, COLOR_TEXT, scale=0.82)
    put_text(
        overlay,
        (
            f"{candidate_label} score={candidate['score']:.3f} "
            f"tilt={candidate['tilt_deg']:.2f}deg "
            f"sel={candidate['selected_fragment_count']}"
        ),
        26,
        62,
    )
    put_text(
        overlay,
        (
            f"geom={candidate.get('geometry_score', 0.0):.3f} "
            f"mirror={candidate.get('mirror_symmetry_score', 0.5):.3f} "
            f"rmse={candidate.get('fit_rmse_px', 0.0):.1f}px"
        ),
        26,
        90,
    )
    put_text(
        overlay,
        (
            f"src={candidate.get('source_hypothesis_label', '?')}/"
            f"{candidate.get('hypothesis_source', 'grid')} "
            f"mode={candidate.get('support_distribution_mode', '?')} "
            f"cov={candidate.get('unique_vertical_coverage', 0.0):.2f} "
            f"span={candidate.get('chain_span_ratio', 0.0):.2f} "
            f"cont={candidate.get('chain_continuity_ratio', 0.0):.2f}"
        ),
        26,
        118,
    )

    return overlay

def create_comparison(step05_overlay: np.ndarray | None, step07_overlay: np.ndarray) -> np.ndarray:
    left_image = step05_overlay if step05_overlay is not None else step07_overlay
    left_image = to_bgr(left_image)
    right_image = to_bgr(step07_overlay)

    height, width = left_image.shape[:2]
    max_width = 1300
    if width > max_width:
        scale = max_width / max(1, width)
        size = (int(width * scale), int(height * scale))
        left_image = cv2.resize(left_image, size, interpolation=cv2.INTER_AREA)
        right_image = cv2.resize(right_image, size, interpolation=cv2.INTER_AREA)

    separator = np.full((left_image.shape[0], 10, 3), 255, dtype=np.uint8)
    return np.hstack([left_image, separator, right_image])

def sanitize_candidate(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None

    return {
        "x_ref": float(candidate["x_ref"]),
        "y_ref": float(candidate["y_ref"]),
        "a": float(candidate["a"]),
        "b": float(candidate["b"]),
        "tilt_deg": float(candidate["tilt_deg"]),
        "score": float(candidate["score"]),
        "final_score": float(candidate.get("final_score", candidate["score"])),
        "geometry_score": float(candidate.get("geometry_score", 0.0)),
        "mirror_symmetry_score": float(candidate.get("mirror_symmetry_score", 0.5)),
        "mirror_left_to_right_score": float(candidate.get("mirror_left_to_right_score", 0.5)),
        "mirror_right_to_left_score": float(candidate.get("mirror_right_to_left_score", 0.5)),
        "mirror_valid_row_ratio": float(candidate.get("mirror_valid_row_ratio", 0.0)),
        "mirror_median_distance_px": float(candidate.get("mirror_median_distance_px", 0.0)),
        "mirror_is_reliable": bool(candidate.get("mirror_is_reliable", False)),
        "validation_passed": bool(candidate.get("validation_passed", True)),
        "rejection_reasons": list(candidate.get("rejection_reasons", [])),
        "dense_support_validation_passed": bool(
            candidate.get("dense_support_validation_passed", False)
        ),
        "sparse_support_validation_passed": bool(
            candidate.get("sparse_support_validation_passed", False)
        ),
        "support_distribution_passed": bool(
            candidate.get("support_distribution_passed", False)
        ),
        "support_distribution_mode": candidate.get(
            "support_distribution_mode", "unknown"
        ),
        "chain_span_ratio": float(candidate.get("chain_span_ratio", 0.0)),
        "unique_vertical_coverage": float(candidate.get("unique_vertical_coverage", 0.0)),
        "fit_rmse_px": float(candidate.get("fit_rmse_px", 0.0)),
        "fit_median_abs_residual_px": float(candidate.get("fit_median_abs_residual_px", 0.0)),
        "fit_p90_abs_residual_px": float(candidate.get("fit_p90_abs_residual_px", 0.0)),
        "fit_consistency_score": float(candidate.get("fit_consistency_score", 0.0)),
        "fragment_alignment_score": float(candidate.get("fragment_alignment_score", 0.0)),
        "support_above_y_ref_ratio": float(candidate.get("support_above_y_ref_ratio", 0.0)),
        "support_below_y_ref_ratio": float(candidate.get("support_below_y_ref_ratio", 0.0)),
        "above_below_balance_score": float(candidate.get("above_below_balance_score", 0.0)),
        "roi_balance_score": float(candidate.get("roi_balance_score", candidate.get("symmetry_score", 0.0))),
        "gap_outlier_penalty": float(candidate.get("gap_outlier_penalty", 0.0)),
        "selected_fragment_count": int(candidate["selected_fragment_count"]),
        "selected_fragment_line_indices": [int(value) for value in candidate["selected_fragment_line_indices"]],
        "selected_total_length_px": float(candidate["selected_total_length_px"]),
        "selected_total_support_strength": float(candidate["selected_total_support_strength"]),
        "fragment_support_score": float(candidate["fragment_support_score"]),
        "vertical_coverage_score": float(candidate["vertical_coverage_score"]),
        "supported_bin_count": int(candidate["supported_bin_count"]),
        "bin_count": int(candidate["bin_count"]),
        "gap_penalty": float(candidate["gap_penalty"]),
        "largest_gap_px": float(candidate["largest_gap_px"]),
        "support_y_min": float(candidate["support_y_min"]),
        "support_y_max": float(candidate["support_y_max"]),
        "support_span_px": float(candidate["support_span_px"]),
        "endpoint_band_px": float(candidate["endpoint_band_px"]),
        "top_endpoint_coverage": float(candidate["top_endpoint_coverage"]),
        "bottom_endpoint_coverage": float(candidate["bottom_endpoint_coverage"]),
        "top_endpoint_alignment_score": float(candidate["top_endpoint_alignment_score"]),
        "bottom_endpoint_alignment_score": float(candidate["bottom_endpoint_alignment_score"]),
        "top_endpoint_best_fragment_overlap_px": float(candidate.get("top_endpoint_best_fragment_overlap_px", 0.0)),
        "bottom_endpoint_best_fragment_overlap_px": float(
            candidate.get("bottom_endpoint_best_fragment_overlap_px", 0.0)
        ),
        "top_endpoint_best_fragment_ratio": float(candidate.get("top_endpoint_best_fragment_ratio", 0.0)),
        "bottom_endpoint_best_fragment_ratio": float(candidate.get("bottom_endpoint_best_fragment_ratio", 0.0)),
        "top_original_endpoint_coverage": float(candidate.get("top_original_endpoint_coverage", 0.0)),
        "bottom_original_endpoint_coverage": float(candidate.get("bottom_original_endpoint_coverage", 0.0)),
        "top_original_endpoint_best_fragment_overlap_px": float(
            candidate.get("top_original_endpoint_best_fragment_overlap_px", 0.0)
        ),
        "bottom_original_endpoint_best_fragment_overlap_px": float(
            candidate.get("bottom_original_endpoint_best_fragment_overlap_px", 0.0)
        ),
        "top_original_endpoint_fragment_ratio": float(candidate.get("top_original_endpoint_fragment_ratio", 0.0)),
        "bottom_original_endpoint_fragment_ratio": float(
            candidate.get("bottom_original_endpoint_fragment_ratio", 0.0)
        ),
        "endpoint_anchor_score": float(candidate["endpoint_anchor_score"]),
        "top_reach_gap_px": float(candidate.get("top_reach_gap_px", 0.0)),
        "bottom_reach_gap_px": float(candidate.get("bottom_reach_gap_px", 0.0)),
        "has_top_anchor": bool(candidate.get("has_top_anchor", False)),
        "has_bottom_anchor": bool(candidate.get("has_bottom_anchor", False)),
        "has_top_bottom_anchor": bool(candidate.get("has_top_bottom_anchor", False)),
        "has_top_original_anchor": bool(candidate.get("has_top_original_anchor", False)),
        "has_bottom_original_anchor": bool(candidate.get("has_bottom_original_anchor", False)),
        "has_top_bottom_original_anchor": bool(candidate.get("has_top_bottom_original_anchor", False)),
        "merged_interval_count": int(candidate.get("merged_interval_count", 0)),
        "total_merged_length_px": float(candidate.get("total_merged_length_px", 0.0)),
        "longest_merged_interval_px": float(candidate.get("longest_merged_interval_px", 0.0)),
        "chain_total_gap_px": float(candidate.get("chain_total_gap_px", 0.0)),
        "chain_continuity_ratio": float(candidate.get("chain_continuity_ratio", 0.0)),
        "chain_fragment_count": int(candidate.get("chain_fragment_count", 0)),
        "chain_total_length_px": float(candidate.get("chain_total_length_px", 0.0)),
        "outside_chain_length_ratio": float(candidate.get("outside_chain_length_ratio", 0.0)),
        "outside_chain_fragment_ratio": float(candidate.get("outside_chain_fragment_ratio", 0.0)),
        "outside_mask_penalty": float(candidate["outside_mask_penalty"]),
        "symmetry_score": float(candidate["symmetry_score"]),
        "roi_center_score": float(candidate["roi_center_score"]),
        "rows_inside_mask_count": int(candidate["rows_inside_mask_count"]),
        "sampled_row_count": int(candidate["sampled_row_count"]),
        "adjusted_fragment_count": int(candidate["adjusted_fragment_count"]),
        "adjusted_fragment_ratio": float(candidate["adjusted_fragment_ratio"]),
        "mean_abs_support_shift_px": float(candidate["mean_abs_support_shift_px"]),
        "length_weighted_mean_abs_support_shift_px": float(candidate["length_weighted_mean_abs_support_shift_px"]),
        "max_abs_support_shift_px": float(candidate["max_abs_support_shift_px"]),
        "mean_abs_support_tilt_delta_deg": float(candidate["mean_abs_support_tilt_delta_deg"]),
        "max_abs_support_tilt_delta_deg": float(candidate["max_abs_support_tilt_delta_deg"]),
        "support_adjustment_penalty": float(candidate["support_adjustment_penalty"]),
        "selection_score": float(candidate.get("selection_score", candidate["score"])),
        "source_hypothesis_rank": candidate.get("source_hypothesis_rank"),
        "source_hypothesis_label": candidate.get("source_hypothesis_label"),
        "hypothesis_source": candidate.get("hypothesis_source", "grid"),
        "structural_seed_score": float(
            candidate.get("structural_seed_score", 0.0)
        ),
        "seed_line_indices": [
            int(value) for value in candidate.get("seed_line_indices", [])
        ],
        "seed_vertical_separation_ratio": float(
            candidate.get("seed_vertical_separation_ratio", 0.0)
        ),
        "structural_seed_inside_roi_ratio": float(
            candidate.get("structural_seed_inside_roi_ratio", 0.0)
        ),
        "roi_prior_x_offset_ratio": candidate.get("roi_prior_x_offset_ratio"),
        "roi_prior_angle_offset_deg": candidate.get(
            "roi_prior_angle_offset_deg"
        ),
        "hypothesis_band_half_width_px": candidate.get(
            "hypothesis_band_half_width_px"
        ),
        "hypothesis_max_angle_error_deg": candidate.get(
            "hypothesis_max_angle_error_deg"
        ),
        "search_stage": candidate.get("search_stage"),
        "hypothesis_x_ref": candidate.get("hypothesis_x_ref"),
        "hypothesis_tilt_deg": candidate.get("hypothesis_tilt_deg"),
        "hypothesis_score": candidate.get("hypothesis_score"),
        "hypothesis_x_ref_delta_px": float(candidate.get("hypothesis_x_ref_delta_px", 0.0)),
        "hypothesis_tilt_delta_deg": float(candidate.get("hypothesis_tilt_delta_deg", 0.0)),
        "selected_support": [
            {
                "line_index": int(item["line"]["line_index"]),
                "source_line_indices": [
                    int(value) for value in item["line"].get("source_line_indices", [item["line"]["line_index"]])
                ],
                "nms_duplicate_count": int(item["line"].get("nms_duplicate_count", 0)),
                "length": float(item["line"]["length"]),
                "axis_distance_px": float(item["axis_distance_px"]),
                "angle_error_deg": float(item["angle_error_deg"]),
                "support_strength": float(item["support_strength"]),
                "effective_tilt_deg": float(item.get("effective_line", item["line"])["signed_tilt_deg"]),
                "is_adjusted": bool(item.get("adjustment", {}).get("is_adjusted", False)),
                "midpoint_shift_px": float(item.get("adjustment", {}).get("midpoint_shift_px", 0.0)),
                "mean_abs_shift_px": float(item.get("adjustment", {}).get("mean_abs_shift_px", 0.0)),
                "max_abs_shift_px": float(item.get("adjustment", {}).get("max_abs_shift_px", 0.0)),
                "tilt_delta_deg": float(item.get("adjustment", {}).get("tilt_delta_deg", 0.0)),
            }
            for item in candidate.get("selected_support", [])
        ],
    }
