from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import time

import cv2

from . import context
from .context import (
    DISPLAY_CONFIG,
    cfg,
    get_step_dirs,
    load_grayscale,
    load_json,
    normalize_path_value,
    relative_project_path,
    save_json,
)
from .geometry import (
    build_symmetric_evaluation_corridor,
    prepare_edge_mask,
    prepare_roi_mask,
    resolve_vertical_range,
)
from .rendering import (
    create_comparison,
    draw_candidate_snapshot,
    draw_winner_overlay,
    extract_corridor_contours,
    load_visual_background,
)
from .symmetry import sanitize_result, verify_candidates


def collect_metadata_files(image_filter: str | None = None, limit: int | None = None) -> list[Path]:
    input_dir = get_step_dirs()["input_metadata_dir"]
    if not input_dir.exists():
        raise FileNotFoundError(f"Step 06 metadata directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*_central_ruler.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [p for p in files if wanted in p.stem.lower() or p.stem.lower().startswith(wanted)]
    return files[: int(limit)] if limit is not None else files


def _load_inputs(metadata: dict) -> tuple:
    roi_path = normalize_path_value(metadata.get("roi_mask_file"))
    edge_path = normalize_path_value(metadata.get("base_edge_file") or metadata.get("source_file"))
    roi, edge = load_grayscale(roi_path), load_grayscale(edge_path)
    if roi is None:
        raise FileNotFoundError(f"Could not load ROI mask: {roi_path}")
    if edge is None:
        raise FileNotFoundError(f"Could not load cleaned edge image: {edge_path}")
    if roi.shape != edge.shape:
        raise ValueError(f"ROI mask and edge image shape mismatch: {roi.shape} vs {edge.shape}")
    return roi, edge, roi_path, edge_path


def _confidence_label(percent: float, margin: float, valid: bool) -> str:
    if not valid:
        return "low"
    if percent >= float(cfg("confidence", "high_min_verification_percent", default=72.0)) and margin >= float(cfg("confidence", "high_min_margin_percent", default=2.0)):
        return "high"
    if percent >= float(cfg("confidence", "medium_min_verification_percent", default=60.0)) and margin >= float(cfg("confidence", "medium_min_margin_percent", default=0.8)):
        return "medium"
    return "low"


def build_analysis(metadata_path: Path) -> dict:
    started = time.perf_counter()
    metadata = load_json(metadata_path)
    image_name = str(metadata.get("image_name", metadata_path.stem + ".png"))
    candidates = list(metadata.get("top_candidates", []))
    if not candidates and metadata.get("best_candidate"):
        candidates = [metadata["best_candidate"]]
    candidates = candidates[: max(1, int(context.STEP_CONFIG.get("candidate_limit", 10)))]
    if not candidates:
        raise RuntimeError(f"No Step 06 candidates found for {image_name}")

    roi_raw, edge_raw, roi_path, edge_path = _load_inputs(metadata)
    roi_mask = prepare_roi_mask(roi_raw)
    y_min, y_max = resolve_vertical_range(roi_mask, metadata)
    corridor_mask, corridor_info, row_half_widths = build_symmetric_evaluation_corridor(roi_mask, metadata, candidates, y_min, y_max)
    edge_mask = prepare_edge_mask(edge_raw, corridor_mask, corridor_info["consensus_axis"], y_min, y_max)
    segment_count = max(3, int(context.STEP_CONFIG.get("segment_count", 12)))
    half_width = max(16, int(round(corridor_info["corridor_half_width_cap_px"])))

    verification_started = time.perf_counter()
    ranked = verify_candidates(candidates, corridor_mask, edge_mask, corridor_info, row_half_widths, y_min, y_max, half_width, segment_count)
    verification_duration = time.perf_counter() - verification_started
    winner = ranked[0]
    runner_up = float(ranked[1]["verification_percent"]) if len(ranked) > 1 else 0.0
    margin = max(0.0, float(winner["verification_percent"]) - runner_up)
    confidence = _confidence_label(float(winner["verification_percent"]), margin, bool(winner["verification_valid"]))
    for rank, candidate in enumerate(ranked, start=1):
        candidate["verification_rank"] = int(rank)
        candidate["winner_margin_percent"] = float(margin if rank == 1 else 0.0)

    rendering_started = time.perf_counter()
    background, visual_path = load_visual_background(metadata, edge_raw)
    corridor_contours = extract_corridor_contours(corridor_mask)
    overlay = draw_winner_overlay(
        background,
        corridor_mask,
        ranked,
        corridor_info["consensus_axis"],
        y_min,
        y_max,
        image_name,
        confidence,
        corridor_contours=corridor_contours,
    )
    comparison = create_comparison(overlay, ranked)

    def render_snapshot(candidate: dict) -> dict:
        return {
            "candidate_label": candidate["candidate_label"],
            "image": draw_candidate_snapshot(
                background,
                corridor_mask,
                candidate,
                corridor_info["consensus_axis"],
                y_min,
                y_max,
                image_name,
                corridor_contours=corridor_contours,
            ),
        }

    configured_workers = max(1, int(cfg("performance", "snapshot_workers", default=2)))
    worker_count = min(configured_workers, os.cpu_count() or 1, len(ranked))
    if worker_count <= 1:
        snapshots = [render_snapshot(candidate) for candidate in ranked]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="step07-snapshot") as executor:
            snapshots = list(executor.map(render_snapshot, ranked))
    rendering_duration = time.perf_counter() - rendering_started

    output_metadata = {
        "image_name": image_name,
        "processing_step": "07_verify_central_ruler_symmetry_v2",
        "algorithm_version": "robust_edge_mirror_consensus_corridor_v2",
        "source_step": metadata.get("processing_step", "06_search_central_ruler"),
        "step_06_metadata_file": relative_project_path(metadata_path),
        "edge_file": relative_project_path(edge_path),
        "roi_mask_file": relative_project_path(roi_path),
        "visual_file": visual_path,
        "segment_count": int(segment_count),
        "candidate_count": int(len(ranked)),
        "vertical_range": {"y_min": int(y_min), "y_max": int(y_max)},
        "consensus_corridor": corridor_info,
        "winner": sanitize_result(winner),
        "ranked_candidates": [sanitize_result(c) for c in ranked],
        "winner_label": winner["candidate_label"],
        "verification_percent": float(winner["verification_percent"]),
        "symmetry_percent": float(winner["verification_percent"]),
        "mirror_symmetry_percent": float(winner["mirror_symmetry_percent"]),
        "winner_margin_percent": float(margin),
        "confidence": confidence,
        "parameters": deepcopy(context.STEP_CONFIG),
        "timings_sec": {
            "verification": float(verification_duration),
            "rendering": float(rendering_duration),
            "analysis_total": float(time.perf_counter() - started),
        },
    }
    return {
        "image_name": image_name,
        "metadata": output_metadata,
        "overlay": overlay,
        "comparison": comparison,
        "snapshots": snapshots,
        "winner_rectified_edge": winner["_rectified_edge"],
        "evaluation_mask": corridor_mask,
        "evaluation_edge_mask": edge_mask,
    }


def process_metadata_file(metadata_path: Path) -> dict:
    started = time.perf_counter()
    analysis = build_analysis(metadata_path)
    dirs = get_step_dirs()
    image_name, stem = analysis["image_name"], Path(analysis["image_name"]).stem
    overlay_path = dirs["output_overlay_dir"] / image_name
    comparison_path = dirs["output_comparison_dir"] / image_name
    output_metadata_path = dirs["output_metadata_dir"] / f"{stem}_symmetry.json"
    snapshot_dir = dirs["output_candidate_snapshot_dir"] / stem
    rectified_path = dirs["output_rectified_dir"] / image_name
    evaluation_mask_path = dirs["output_evaluation_mask_dir"] / image_name
    evaluation_edge_path = dirs["output_evaluation_edge_dir"] / image_name
    for path in [overlay_path.parent, comparison_path.parent, output_metadata_path.parent, snapshot_dir, rectified_path.parent, evaluation_mask_path.parent, evaluation_edge_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    write_jobs: list[tuple[Path, object, str]] = [
        (overlay_path, analysis["overlay"], "overlay"),
        (comparison_path, analysis["comparison"], "comparison"),
        (rectified_path, analysis["winner_rectified_edge"], "rectified winner"),
        (evaluation_mask_path, analysis["evaluation_mask"], "Step 07 evaluation mask"),
        (evaluation_edge_path, analysis["evaluation_edge_mask"], "Step 07 evaluation edge mask"),
    ]
    snapshot_paths: list[Path] = []
    for snapshot in analysis["snapshots"]:
        path = snapshot_dir / f"{snapshot['candidate_label']}_{image_name}"
        snapshot_paths.append(path)
        write_jobs.append((path, snapshot["image"], "candidate snapshot"))

    def write_image(job: tuple[Path, object, str]) -> Path:
        path, image, description = job
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save {description}: {path}")
        return path

    configured_workers = max(1, int(cfg("performance", "image_write_workers", default=4)))
    worker_count = min(configured_workers, os.cpu_count() or 1, len(write_jobs))
    if worker_count <= 1:
        for job in write_jobs:
            write_image(job)
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="step07-write") as executor:
            list(executor.map(write_image, write_jobs))
    snapshot_files = [relative_project_path(path) for path in snapshot_paths]
    metadata = deepcopy(analysis["metadata"])
    metadata.update({
        "output_overlay_file": relative_project_path(overlay_path),
        "output_comparison_file": relative_project_path(comparison_path),
        "output_rectified_file": relative_project_path(rectified_path),
        "output_evaluation_mask_file": relative_project_path(evaluation_mask_path),
        "output_evaluation_edge_file": relative_project_path(evaluation_edge_path),
        "evaluation_mask_role": "candidate_independent_support_mask_not_shape_ground_truth",
        "candidate_snapshot_files": snapshot_files,
    })
    metadata["timings_sec"]["process_total"] = float(time.perf_counter() - started)
    save_json(output_metadata_path, metadata)
    winner = metadata["winner"]
    return {
        "image_name": image_name,
        "candidate_count": int(metadata["candidate_count"]),
        "winner_label": str(metadata["winner_label"]),
        "symmetry_percent": float(metadata["symmetry_percent"]),
        "mirror_symmetry_percent": float(metadata["mirror_symmetry_percent"]),
        "winner_margin_percent": float(metadata["winner_margin_percent"]),
        "confidence": str(metadata["confidence"]),
        "winner_tilt_deg": float(winner["tilt_deg"]),
        "overlay_path": relative_project_path(overlay_path),
        "comparison_path": relative_project_path(comparison_path),
        "metadata_path": relative_project_path(output_metadata_path),
        "candidate_snapshot_dir": relative_project_path(snapshot_dir),
        "evaluation_mask_path": relative_project_path(evaluation_mask_path),
        "evaluation_edge_path": relative_project_path(evaluation_edge_path),
    }


def show_image(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return
    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    if image.shape[0] > max_height:
        scale = max_height / image.shape[0]
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max_height))
    cv2.imshow("Step 07 robust mirror verification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
