from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from . import context
from .context import DISPLAY_CONFIG, PROJECT_ROOT, cfg, get_pipeline_config_path, get_step_dirs, load_json, load_roi_mask, relative_project_path, resolve_project_path, save_json, sha256_file, sha256_json
from .geometry import build_row_profile, canonicalize_lines, filter_fragments, normalize_line
from .rendering import build_fragment_background, create_comparison, draw_candidate_snapshot, draw_overlay, load_base_edge_image, load_step05_overlay, sanitize_candidate
from .search import search_best_candidate

def _persistence_enabled(key: str, default: bool = True) -> bool:
    return bool(context.STEP_CONFIG.get("persistence", {}).get(key, default))

def process_image(json_path: Path | str) -> dict:
    return process_json_file(Path(json_path))

def build_analysis( json_path: Path, *,materialize_candidate_snapshots: bool = True,) -> dict:
    analysis_started_at = time.perf_counter()
    data = load_json(json_path)
    image_name = data.get("image_name", json_path.stem + ".png")
    width = int(data.get("width", 0))
    height = int(data.get("height", 0)) or 4032

    raw_lines = data.get("valid_lines", [])
    normalized_lines = [normalize_line(line, index) for index, line in enumerate(raw_lines, start=1)]
    lines = canonicalize_lines(normalized_lines)
    filtered_lines, rejected_lines = filter_fragments(lines)

    roi_mask_path = resolve_project_path(data.get("roi_mask_file"))
    roi_mask = load_roi_mask(roi_mask_path)
    roi_profile = build_row_profile(roi_mask)
    if roi_profile is None:
        raise RuntimeError(f"Could not build ROI profile for {image_name}")

    # Mirror symmetry is part of final candidate ranking, so load the edge image
    # before the search rather than only during rendering.
    base_edge_image, base_edge_path = load_base_edge_image(data)
    search_started_at = time.perf_counter()
    search_result = search_best_candidate(filtered_lines, roi_profile, edge_image=base_edge_image)
    search_duration_sec = time.perf_counter() - search_started_at
    best_candidate = search_result["best_candidate"]
    fine_candidates = search_result["fine_candidates"]
    ranked_candidates = search_result.get("ranked_candidates", fine_candidates)
    ranked_candidate_total_count = int(search_result.get("ranked_candidate_total_count", len(ranked_candidates)))

    save_overlay = _persistence_enabled("save_overlay", True)
    save_comparison = _persistence_enabled("save_comparison", True)
    save_candidate_snapshots = _persistence_enabled("save_candidate_snapshots", True)
    needs_fragment_background = save_overlay or save_comparison or save_candidate_snapshots
    rendering_started_at = time.perf_counter()
    fragment_background = (
        build_fragment_background(base_edge_image, filtered_lines)
        if needs_fragment_background
        else None
    )
    overlay = None
    if (save_overlay or save_comparison) and fragment_background is not None:
        overlay = draw_overlay(
            fragment_background=fragment_background,
            filtered_line_count=len(filtered_lines),
            best_candidate=best_candidate,
            fine_candidates=ranked_candidates,
            roi_profile=roi_profile,
            image_name=image_name,
        )
    comparison = None
    if save_comparison and overlay is not None:
        step05_overlay = load_step05_overlay(image_name)
        comparison = create_comparison(step05_overlay, overlay)
    saved_candidate_count = int(cfg("candidate_deduplication", "max_saved_candidates", default=8))
    if bool(context.STEP_CONFIG.get("save_all_final_candidates", False)):
        saved_candidate_count = len(ranked_candidates)
    candidate_snapshot_images = []
    if materialize_candidate_snapshots and save_candidate_snapshots and fragment_background is not None:
        for index, candidate in enumerate(ranked_candidates[:saved_candidate_count],start=1,):
            candidate_snapshot_images.append(
                {
                    "index": index,
                    "image": draw_candidate_snapshot(
                        fragment_background=fragment_background,
                        candidate=candidate,
                        roi_profile=roi_profile,
                        image_name=image_name,
                        candidate_label=f"C{index}",
                    ),
                }
            )
    rendering_duration_sec = time.perf_counter() - rendering_started_at
    total_analysis_duration_sec = time.perf_counter() - analysis_started_at

    pipeline_config_path = get_pipeline_config_path()
    roi_mask_hash = sha256_file(roi_mask_path)
    step06_source_paths = [PROJECT_ROOT / "pipeline" / "06_search_central_ruler.py",*sorted((PROJECT_ROOT / "pipeline" / "lib" / "step_06_search_central_ruler").glob("*.py")),]
    step06_source_files = [
        {"path": relative_project_path(path),"sha256": sha256_file(path),}
        for path in step06_source_paths
        if path.exists()
    ]
    reproducibility = {
        "effective_step_config_sha256": sha256_json(context.STEP_CONFIG),
        "pipeline_config_file": relative_project_path(pipeline_config_path),
        "pipeline_config_sha256": sha256_file(pipeline_config_path),
        "input_json_sha256": sha256_file(json_path),
        "base_edge_sha256": sha256_file(resolve_project_path(base_edge_path) if base_edge_path else None),
        "roi_mask_sha256": roi_mask_hash,
        "step06_source_sha256": sha256_json(step06_source_files),
        "step06_source_files": step06_source_files,
        "library_versions": {"numpy": str(np.__version__),"opencv": str(cv2.__version__),},
    }

    metadata = {
        "image_name": image_name,
        "processing_step": "06_search_central_ruler",
        "source_step": data.get("processing_step", "05_valid_hough_lines_in_roi"),
        "width": width,
        "height": height,
        "input_json_file": relative_project_path(json_path),
        "base_edge_file": base_edge_path,
        "source_file": data.get("source_file"),
        "roi_mask_file": data.get("roi_mask_file"),
        "resolved_input_dir": relative_project_path(get_step_dirs()["input_dir"]),
        "input_line_count": len(lines),
        "filtered_line_count": len(filtered_lines),
        "nms_line_count": int(search_result.get("nms_line_count", len(filtered_lines))),
        "nms_removed_line_count": int(search_result.get("nms_removed_line_count", 0)),
        "rejected_fragment_count": len(rejected_lines),
        "coarse_candidate_count": len(search_result["coarse_candidates"]),
        "fine_candidate_count": len(fine_candidates),
        "structural_seed_count": int(search_result.get("structural_seed_count", 0)),
        "evaluated_hypothesis_count": int(search_result.get("evaluated_hypothesis_count", 0)),
        "ranked_candidate_count": ranked_candidate_total_count,
        "saved_candidate_count": min(saved_candidate_count, len(ranked_candidates)),
        "best_candidate": sanitize_candidate(best_candidate),
        "top_candidates": [
            sanitize_candidate(candidate)
            for candidate in ranked_candidates[:saved_candidate_count]
        ],
        "roi_profile": {
            "y_min": int(roi_profile["y_min"]),
            "y_max": int(roi_profile["y_max"]),
            "trimmed_y_min": int(roi_profile["trimmed_y_min"]),
            "trimmed_y_max": int(roi_profile["trimmed_y_max"]),
            "y_ref": float(roi_profile["y_ref"]),
            "reference_width_px": float(roi_profile["reference_width_px"]),
            "median_center_x": float(roi_profile["median_center_x"]),
            "center_fit": {
                "a": float(roi_profile["center_fit"]["a"]),
                "b": float(roi_profile["center_fit"]["b"]),
                "tilt_deg": float(roi_profile["center_fit"]["tilt_deg"]),
            },
        },
        "timings_sec": {
            "search": float(search_duration_sec),
            "rendering": float(rendering_duration_sec),
            "analysis_total": float(total_analysis_duration_sec),
        },
        "parameters": context.STEP_CONFIG,
        "reproducibility": reproducibility,
    }

    return {
        "image_name": image_name,
        "metadata": metadata,
        "overlay": overlay,
        "comparison": comparison,
        "best_candidate": best_candidate,
        "fine_candidates": fine_candidates,
        "ranked_candidates": ranked_candidates,
        "filtered_lines": filtered_lines,
        "candidate_snapshot_images": candidate_snapshot_images,
        "_fragment_background": fragment_background,
        "_roi_profile": roi_profile,
        "_saved_candidate_count": min(saved_candidate_count, len(ranked_candidates)),
    }

def process_json_file(json_path: Path) -> dict:
    process_started_at = time.perf_counter()
    analysis = build_analysis(json_path,materialize_candidate_snapshots=False,)
    dirs = get_step_dirs()
    image_name = analysis["image_name"]

    overlay_path = dirs["output_overlay_dir"] / image_name
    comparison_path = dirs["output_comparison_dir"] / image_name
    metadata_path = dirs["output_metadata_dir"] / f"{Path(image_name).stem}_central_ruler.json"
    candidate_snapshot_dir = dirs["output_candidate_snapshot_dir"] / Path(image_name).stem
    save_overlay = _persistence_enabled("save_overlay", True)
    save_comparison = _persistence_enabled("save_comparison", True)
    save_candidate_snapshots = _persistence_enabled("save_candidate_snapshots", True)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if save_comparison:
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
    if save_candidate_snapshots:
        candidate_snapshot_dir.mkdir(parents=True, exist_ok=True)

    if save_overlay:
        if analysis["overlay"] is None or not cv2.imwrite(str(overlay_path), analysis["overlay"]):
            raise RuntimeError(f"Could not save overlay: {overlay_path}")
    if save_comparison:
        if analysis["comparison"] is None or not cv2.imwrite(str(comparison_path), analysis["comparison"]):
            raise RuntimeError(f"Could not save comparison: {comparison_path}")

    candidate_snapshot_files = []
    snapshot_rendering_duration_sec = 0.0
    fragment_background = analysis["_fragment_background"]
    roi_profile = analysis["_roi_profile"]
    saved_candidate_count = int(analysis["_saved_candidate_count"])
    if save_candidate_snapshots and fragment_background is not None:
        for snapshot_index, candidate in enumerate(analysis["ranked_candidates"][:saved_candidate_count],start=1,):
            snapshot_render_started_at = time.perf_counter()
            snapshot_image = draw_candidate_snapshot(fragment_background=fragment_background,candidate=candidate,
                roi_profile=roi_profile,image_name=image_name,candidate_label=f"C{snapshot_index}",)
            snapshot_rendering_duration_sec += (time.perf_counter() - snapshot_render_started_at)
            snapshot_path = candidate_snapshot_dir / f"C{snapshot_index:02d}_{image_name}"
            if not cv2.imwrite(str(snapshot_path), snapshot_image): raise RuntimeError(f"Could not save candidate snapshot: {snapshot_path}")
            candidate_snapshot_files.append(relative_project_path(snapshot_path))
            del snapshot_image

    metadata = deepcopy(analysis["metadata"])
    metadata.setdefault("timings_sec", {})
    metadata["timings_sec"]["rendering"] = float(metadata["timings_sec"].get("rendering", 0.0)+ snapshot_rendering_duration_sec)
    metadata["timings_sec"]["analysis_total"] = float(metadata["timings_sec"].get("analysis_total", 0.0)+ snapshot_rendering_duration_sec)
    metadata["output_overlay_file"] = relative_project_path(overlay_path) if save_overlay else None
    metadata["output_comparison_file"] = relative_project_path(comparison_path) if save_comparison else None
    metadata["candidate_snapshot_files"] = candidate_snapshot_files
    metadata["timings_sec"]["save"] = float(time.perf_counter()- process_started_at- metadata["timings_sec"].get("analysis_total", 0.0))
    metadata["timings_sec"]["process_total"] = float(time.perf_counter() - process_started_at)
    save_json(metadata_path, metadata)

    best_candidate = analysis["best_candidate"]
    return {
        "image_name": image_name,
        "filtered_line_count": len(analysis["filtered_lines"]),
        "candidate_count": len(analysis["fine_candidates"]),
        "selected_fragment_count": best_candidate["selected_fragment_count"] if best_candidate else 0,
        "best_score": best_candidate["score"] if best_candidate else None,
        "best_tilt_deg": best_candidate["tilt_deg"] if best_candidate else None,
        "timings_sec": metadata.get("timings_sec", {}),
        "overlay_path": relative_project_path(overlay_path) if save_overlay else None,
        "metadata_path": relative_project_path(metadata_path),
        "comparison_path": relative_project_path(comparison_path) if save_comparison else None,
        "candidate_snapshot_dir": relative_project_path(candidate_snapshot_dir),
    }

def collect_json_files(image_filter: str | None = None, limit: int | None = None) -> list[Path]:
    input_json_dir = get_step_dirs()["input_json_dir"]
    if not input_json_dir.exists(): raise FileNotFoundError(f"Input JSON dir does not exist: {input_json_dir}")

    files = sorted(input_json_dir.glob("*.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [path for path in files if path.stem.lower() == wanted or wanted in path.stem.lower()]
    if limit is not None: files = files[:limit]
    return files

def show_image(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:return

    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / max(1, height)
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow(path.name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
