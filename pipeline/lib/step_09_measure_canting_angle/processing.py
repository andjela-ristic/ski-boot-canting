from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from . import context
from .axis_metrics import build_axis_quality
from .geometry import resolve_axis_candidate
from .measurement import combine_measurement_confidence, compute_canting, compute_line_angle_uncertainty
from .rendering import create_comparison, draw_diagnostic, draw_overlay, load_background
from .table_line import build_exclusion_mask, detect_table_line_with_stability


def _persistence_enabled(key: str, default: bool = True) -> bool:
    return bool(context.STEP_CONFIG.get("persistence", {}).get(key, default))


def collect_metadata_files(image_filter: str | None = None, limit: int | None = None) -> list[Path]:
    input_dir = context.get_step_dirs()["input_metadata_dir"]
    if not input_dir.exists():
        raise FileNotFoundError(f"Step 08 metadata directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*_multi_validation.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [path for path in files if wanted in path.stem.lower() or path.stem.lower().startswith(wanted)]
    return files[: int(limit)] if limit is not None else files


def _resolve_cleaned_edge(step08: dict, image_name: str) -> tuple[np.ndarray, Path]:
    candidates = [
        context.get_step_dirs()["cleaned_edge_dir"] / image_name,
        context.normalize_path_value(step08.get("visual_file")),
    ]
    for path in candidates:
        image = context.load_grayscale(path)
        if image is not None:
            return (image > 0).astype(np.uint8) * 255, path
    raise FileNotFoundError(f"Could not resolve cleaned edge image for {image_name}")


def _resolve_step07_mask(step08: dict, image_name: str) -> tuple[np.ndarray | None, Path | None]:
    paths = [
        context.normalize_path_value(step08.get("step_07_evaluation_mask_file")),
    ]
    step07_path = context.normalize_path_value(step08.get("step_07_metadata_file"))
    if step07_path is not None and step07_path.exists():
        try:
            step07 = context.load_json(step07_path)
            paths.append(context.normalize_path_value(step07.get("output_evaluation_mask_file")))
        except Exception:
            pass
    paths.extend([
        context.PROCESSED_DIR / "07_verify_central_ruler_symmetry" / "evaluation_mask" / image_name,
    ])
    for path in paths:
        image = context.load_grayscale(path)
        if image is not None:
            return (image > 0).astype(np.uint8) * 255, path
    if bool(context.STEP_CONFIG.get("evaluation_mask", {}).get("require_step_07_evaluation_mask", False)):
        raise FileNotFoundError(f"Step 07 evaluation mask is required but missing for {image_name}")
    return None, None


def _vertical_range(step08: dict, edge: np.ndarray) -> tuple[int, int]:
    vertical = step08.get("vertical_range", {})
    if "y_min" in vertical and "y_max" in vertical:
        return (
            max(0, int(vertical["y_min"])),
            min(edge.shape[0] - 1, int(vertical["y_max"])),
        )
    winner = step08.get("winner", {})
    segments = winner.get("segments", []) if isinstance(winner, dict) else []
    y_values = []
    for segment in segments:
        for key in ("y_start", "y_end", "y_min", "y_max"):
            if key in segment:
                y_values.append(int(segment[key]))
    if y_values:
        return max(0, min(y_values)), min(edge.shape[0] - 1, max(y_values))
    raise RuntimeError("Step 08 metadata does not contain a usable vertical range")


def build_analysis(metadata_path: Path) -> dict:
    started = time.perf_counter()
    step08 = context.load_json(metadata_path)
    image_name = str(step08.get("image_name", metadata_path.stem.replace("_multi_validation", "") + ".png"))
    edge, edge_path = _resolve_cleaned_edge(step08, image_name)
    axis_candidate = resolve_axis_candidate(step08)
    y_min, y_max = _vertical_range(step08, edge)
    evaluation_mask, evaluation_mask_path = _resolve_step07_mask(step08, image_name)
    exclusion_mask, exclusion_info = build_exclusion_mask(edge.shape, evaluation_mask, axis_candidate, y_min, y_max)
    axis_quality = build_axis_quality(step08, axis_candidate)
    table_result = detect_table_line_with_stability(edge, exclusion_mask, y_min, y_max, axis_candidate)

    canting = None
    uncertainty = {
        "table_angle_standard_error_deg": None,
        "table_angle_uncertainty_95_deg": None,
        "effective_support_span_px": None,
    }
    measurement_confidence = None
    if table_result.get("available") and table_result.get("winner") is not None:
        uncertainty = compute_line_angle_uncertainty(table_result["winner"])
        canting = compute_canting(axis_candidate, table_result["winner"], y_min, y_max)
        measurement_confidence = combine_measurement_confidence(axis_quality, table_result, uncertainty)

    background, background_path = load_background(image_name, edge)
    overlay = draw_overlay(
        background,
        edge,
        exclusion_mask,
        axis_candidate,
        y_min,
        y_max,
        table_result,
        canting,
        axis_quality,
        measurement_confidence,
    )
    save_diagnostic = _persistence_enabled("save_diagnostic", True)
    save_comparison = _persistence_enabled("save_comparison", True)
    diagnostic = draw_diagnostic(edge, exclusion_mask, table_result) if (save_diagnostic or save_comparison) else None

    table_line = table_result.get("winner")
    output_metadata = {
        "image_name": image_name,
        "processing_step": "09_measure_canting_angle",
        "algorithm_version": "bilateral_table_line_consensus_and_full_step08_metric_fusion_v1",
        "source_step": step08.get("processing_step", "08_multi_validate_central_ruler"),
        "step_08_metadata_file": context.relative_project_path(metadata_path),
        "cleaned_edge_file": context.relative_project_path(edge_path),
        "step_07_evaluation_mask_file": None if evaluation_mask_path is None else context.relative_project_path(evaluation_mask_path),
        "background_file": background_path,
        "final_axis_candidate": str(step08.get("final_candidate", axis_candidate.get("candidate_label", ""))),
        "vertical_range": {"y_min": int(y_min), "y_max": int(y_max)},
        "axis": {
            "a": float(axis_candidate.get("a", 0.0)),
            "b": float(axis_candidate.get("b", axis_candidate.get("x_ref", 0.0))),
            "x_ref": axis_candidate.get("x_ref"),
            "y_ref": axis_candidate.get("y_ref"),
            "tilt_deg": axis_candidate.get("tilt_deg", axis_candidate.get("candidate_tilt_deg")),
        },
        "axis_validation": axis_quality,
        "exclusion_mask": exclusion_info,
        "table_line_available": bool(table_line is not None),
        "table_line": None if table_line is None else {
            key: value
            for key, value in table_line.items()
            if key not in {"support_points", "segments"}
        },
        "table_line_support_segments": [] if table_line is None else table_line.get("segments", []),
        "table_line_support_points_count": 0 if table_line is None else int(table_line.get("refinement", {}).get("support_pixel_count", 0)),
        "table_line_stability": table_result.get("stability"),
        "table_line_variant_runs": [
            {
                "name": run.get("name"),
                "available": run.get("available"),
                "search_info": run.get("search_info"),
                "winner": None if run.get("winner") is None else {
                    key: value
                    for key, value in run["winner"].items()
                    if key not in {"support_points", "segments"}
                },
            }
            for run in table_result.get("variant_runs", [])
        ],
        "table_line_runner_up": None if table_result.get("runner_up") is None else {
            key: value
            for key, value in table_result["runner_up"].items()
            if key not in {"support_points", "segments"}
        },
        "angle_measurement": canting,
        "angle_uncertainty": uncertainty,
        "measurement_validation": measurement_confidence or {
            "measurement_confidence_score": 0.0,
            "measurement_confidence_percent": 0.0,
            "decision": "reference_line_not_reliable",
            "calibration_status": str(context.STEP_CONFIG.get("measurement_confidence", {}).get("calibration_status")),
        },
        "source_step_08_summary": {
            key: step08.get(key)
            for key in (
                "final_candidate",
                "step_07_candidate",
                "step_07_agrees",
                "symmetry_percent",
                "multi_validation_percent",
                "confidence_percent",
                "decision",
            )
            if key in step08
        },
        "parameters": deepcopy(context.STEP_CONFIG),
        "timings_sec": {"analysis_total": float(time.perf_counter() - started)},
    }
    comparison = create_comparison(overlay, diagnostic, output_metadata) if save_comparison and diagnostic is not None else None
    return {
        "image_name": image_name,
        "metadata": output_metadata,
        "overlay": overlay,
        "diagnostic": diagnostic,
        "comparison": comparison,
    }


def process_metadata_file(metadata_path: Path) -> dict:
    started = time.perf_counter()
    analysis = build_analysis(metadata_path)
    dirs = context.get_step_dirs()
    image_name = analysis["image_name"]
    stem = Path(image_name).stem
    overlay_path = dirs["output_overlay_dir"] / image_name
    comparison_path = dirs["output_comparison_dir"] / image_name
    diagnostic_path = dirs["output_diagnostics_dir"] / image_name
    metadata_output_path = dirs["output_metadata_dir"] / f"{stem}_canting_angle.json"
    save_comparison = _persistence_enabled("save_comparison", True)
    save_diagnostic = _persistence_enabled("save_diagnostic", True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_comparison:
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
    if save_diagnostic:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    for path, image in (
        (overlay_path, analysis["overlay"]),
        (comparison_path, analysis["comparison"]) if save_comparison else (None, None),
        (diagnostic_path, analysis["diagnostic"]) if save_diagnostic else (None, None),
    ):
        if path is None:
            continue
        if image is None or not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save image: {path}")
    metadata = deepcopy(analysis["metadata"])
    metadata.update({
        "output_overlay_file": context.relative_project_path(overlay_path),
        "output_comparison_file": context.relative_project_path(comparison_path) if save_comparison else None,
        "output_diagnostic_file": context.relative_project_path(diagnostic_path) if save_diagnostic else None,
    })
    metadata["timings_sec"]["process_total"] = float(time.perf_counter() - started)
    context.save_json(metadata_output_path, metadata)
    angle = metadata.get("angle_measurement") or {}
    measurement = metadata.get("measurement_validation", {})
    table_line = metadata.get("table_line") or {}
    return {
        "image_name": image_name,
        "final_axis_candidate": metadata.get("final_axis_candidate"),
        "canting_angle_deg": angle.get("canting_angle_deg"),
        "absolute_canting_angle_deg": angle.get("absolute_canting_angle_deg"),
        "canting_direction": angle.get("canting_direction"),
        "axis_table_angle_deg": angle.get("axis_table_angle_deg"),
        "axis_quality_percent": metadata.get("axis_validation", {}).get("axis_quality_percent"),
        "table_line_angle_deg": table_line.get("angle_deg"),
        "table_line_quality_percent": table_line.get("table_line_quality_percent", table_line.get("score_percent")),
        "table_line_stability_percent": metadata.get("table_line_stability", {}).get("score_percent"),
        "measurement_confidence_percent": measurement.get("measurement_confidence_percent", 0.0),
        "uncertainty_95_deg": metadata.get("angle_uncertainty", {}).get("table_angle_uncertainty_95_deg"),
        "decision": measurement.get("decision", "reference_line_not_reliable"),
        "overlay_path": context.relative_project_path(overlay_path),
        "comparison_path": context.relative_project_path(comparison_path) if save_comparison else None,
        "metadata_path": context.relative_project_path(metadata_output_path),
    }


def show_image(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return
    max_height = int(context.DISPLAY_CONFIG.get("max_height", 900))
    if image.shape[0] > max_height:
        scale = max_height / image.shape[0]
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max_height))
    cv2.imshow("Step 09 canting angle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
