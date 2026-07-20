from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import cv2

from . import context
from .context import (
    DISPLAY_CONFIG,
    PROJECT_ROOT,
    WORKING_PNG_DIR,
    cfg,
    ensure_dirs,
    get_step_dirs,
    load_color,
    load_grayscale,
    load_json,
    normalize_path_value,
    relative_project_path,
    save_json,
)
from .geometry import (
    build_core_roi_mask,
    prepare_edge_mask,
    prepare_roi_mask,
    resolve_vertical_range,
)
from .rendering import (
    create_comparison,
    draw_candidate_snapshot,
    draw_winner_overlay,
    load_visual_background,
)
from .symmetry import sanitize_result, verify_candidates


def collect_metadata_files(
    image_filter: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    input_dir = get_step_dirs()["input_metadata_dir"]
    if not input_dir.exists():
        raise FileNotFoundError(f"Step 06 metadata directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*_central_ruler.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [
            path
            for path in files
            if wanted in path.stem.lower()
            or path.stem.lower().startswith(wanted)
        ]
    if limit is not None:
        files = files[: int(limit)]
    return files


def _load_inputs(metadata: dict) -> tuple:
    roi_path = normalize_path_value(metadata.get("roi_mask_file"))
    edge_path = normalize_path_value(
        metadata.get("base_edge_file") or metadata.get("source_file")
    )
    roi = load_grayscale(roi_path)
    edge = load_grayscale(edge_path)
    if roi is None:
        raise FileNotFoundError(f"Could not load ROI mask: {roi_path}")
    if edge is None:
        raise FileNotFoundError(f"Could not load cleaned edge image: {edge_path}")
    if roi.shape != edge.shape:
        raise ValueError(
            f"ROI mask and edge image shape mismatch: {roi.shape} vs {edge.shape}"
        )
    return roi, edge, roi_path, edge_path


def _confidence_label(symmetry_percent: float, margin_percent: float) -> str:
    if (
        symmetry_percent
        >= float(cfg("confidence", "high_min_symmetry_percent", default=82.0))
        and margin_percent
        >= float(cfg("confidence", "high_min_margin_percent", default=2.0))
    ):
        return "high"
    if (
        symmetry_percent
        >= float(cfg("confidence", "medium_min_symmetry_percent", default=68.0))
        and margin_percent
        >= float(cfg("confidence", "medium_min_margin_percent", default=0.8))
    ):
        return "medium"
    return "low"


def build_analysis(metadata_path: Path) -> dict:
    started = time.perf_counter()
    metadata = load_json(metadata_path)
    image_name = str(metadata.get("image_name", metadata_path.stem + ".png"))
    candidates = list(metadata.get("top_candidates", []))
    if not candidates and metadata.get("best_candidate"):
        candidates = [metadata["best_candidate"]]
    candidate_limit = max(1, int(context.STEP_CONFIG.get("candidate_limit", 10)))
    candidates = candidates[:candidate_limit]
    if not candidates:
        raise RuntimeError(f"No Step 06 candidates found for {image_name}")

    roi_raw, edge_raw, roi_path, edge_path = _load_inputs(metadata)
    roi_mask = prepare_roi_mask(roi_raw)
    y_min, y_max = resolve_vertical_range(roi_mask, metadata)
    core_roi_mask, core_info = build_core_roi_mask(
        roi_mask,
        metadata,
        y_min,
        y_max,
    )
    edge_mask = prepare_edge_mask(edge_raw, core_roi_mask)
    segment_count = max(3, int(context.STEP_CONFIG.get("segment_count", 12)))

    verification_started = time.perf_counter()
    ranked = verify_candidates(
        candidates,
        core_roi_mask,
        edge_mask,
        y_min,
        y_max,
        int(core_info["evaluation_half_width_px"]),
        segment_count,
    )
    verification_duration = time.perf_counter() - verification_started

    winner = ranked[0]
    runner_up_score = float(ranked[1]["symmetry_percent"]) if len(ranked) > 1 else 0.0
    margin = max(0.0, float(winner["symmetry_percent"]) - runner_up_score)
    confidence = _confidence_label(float(winner["symmetry_percent"]), margin)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["verification_rank"] = int(rank)
        candidate["winner_margin_percent"] = float(margin if rank == 1 else 0.0)

    rendering_started = time.perf_counter()
    background, visual_path = load_visual_background(metadata, edge_raw)
    overlay = draw_winner_overlay(
        background,
        core_roi_mask,
        ranked,
        y_min,
        y_max,
        image_name,
        confidence,
    )
    comparison = create_comparison(overlay, ranked)
    snapshots = [
        {
            "candidate_label": candidate["candidate_label"],
            "image": draw_candidate_snapshot(
                background,
                core_roi_mask,
                candidate,
                y_min,
                y_max,
                image_name,
            ),
        }
        for candidate in ranked
    ]
    rectified_winner = ranked[0]["_rectified_edge"]
    rendering_duration = time.perf_counter() - rendering_started

    output_metadata = {
        "image_name": image_name,
        "processing_step": "07_verify_central_ruler_symmetry",
        "source_step": metadata.get("processing_step", "06_search_central_ruler"),
        "step_06_metadata_file": relative_project_path(metadata_path),
        "edge_file": relative_project_path(edge_path),
        "roi_mask_file": relative_project_path(roi_path),
        "visual_file": visual_path,
        "segment_count": int(segment_count),
        "candidate_count": int(len(ranked)),
        "vertical_range": {"y_min": int(y_min), "y_max": int(y_max)},
        "core_roi": core_info,
        "winner": sanitize_result(ranked[0]),
        "ranked_candidates": [sanitize_result(item) for item in ranked],
        "winner_label": winner["candidate_label"],
        "symmetry_percent": float(winner["symmetry_percent"]),
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
        "winner_rectified_edge": rectified_winner,
        "ranked": ranked,
    }


def process_metadata_file(metadata_path: Path) -> dict:
    started = time.perf_counter()
    analysis = build_analysis(metadata_path)
    dirs = get_step_dirs()
    image_name = analysis["image_name"]
    stem = Path(image_name).stem

    overlay_path = dirs["output_overlay_dir"] / image_name
    comparison_path = dirs["output_comparison_dir"] / image_name
    output_metadata_path = dirs["output_metadata_dir"] / f"{stem}_symmetry.json"
    snapshot_dir = dirs["output_candidate_snapshot_dir"] / stem
    rectified_path = dirs["output_rectified_dir"] / image_name

    for path in [overlay_path.parent, comparison_path.parent, output_metadata_path.parent, snapshot_dir, rectified_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(overlay_path), analysis["overlay"]):
        raise RuntimeError(f"Could not save overlay: {overlay_path}")
    if not cv2.imwrite(str(comparison_path), analysis["comparison"]):
        raise RuntimeError(f"Could not save comparison: {comparison_path}")
    if not cv2.imwrite(str(rectified_path), analysis["winner_rectified_edge"]):
        raise RuntimeError(f"Could not save rectified winner: {rectified_path}")

    snapshot_files: list[str] = []
    for snapshot in analysis["snapshots"]:
        path = snapshot_dir / f"{snapshot['candidate_label']}_{image_name}"
        if not cv2.imwrite(str(path), snapshot["image"]):
            raise RuntimeError(f"Could not save candidate snapshot: {path}")
        snapshot_files.append(relative_project_path(path))

    metadata = deepcopy(analysis["metadata"])
    metadata["output_overlay_file"] = relative_project_path(overlay_path)
    metadata["output_comparison_file"] = relative_project_path(comparison_path)
    metadata["output_rectified_file"] = relative_project_path(rectified_path)
    metadata["candidate_snapshot_files"] = snapshot_files
    metadata["timings_sec"]["save"] = float(
        time.perf_counter()
        - started
        - metadata["timings_sec"].get("analysis_total", 0.0)
    )
    metadata["timings_sec"]["process_total"] = float(time.perf_counter() - started)
    save_json(output_metadata_path, metadata)

    winner = metadata["winner"]
    return {
        "image_name": image_name,
        "candidate_count": int(metadata["candidate_count"]),
        "winner_label": str(metadata["winner_label"]),
        "symmetry_percent": float(metadata["symmetry_percent"]),
        "winner_margin_percent": float(metadata["winner_margin_percent"]),
        "confidence": str(metadata["confidence"]),
        "winner_tilt_deg": float(winner["tilt_deg"]),
        "overlay_path": relative_project_path(overlay_path),
        "comparison_path": relative_project_path(comparison_path),
        "metadata_path": relative_project_path(output_metadata_path),
        "candidate_snapshot_dir": relative_project_path(snapshot_dir),
    }


def show_image(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return
    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / max(1, height)
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max_height),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imshow(path.name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
