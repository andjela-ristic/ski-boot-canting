from __future__ import annotations

from copy import deepcopy
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
from .fusion import compute_candidate_fusion, compute_confidence
from .geometry import (
    candidate_anchor_score,
    candidate_medial_score,
    candidate_roi_balance_score,
    compute_medial_reference,
    compute_structural_anchors,
    prepare_roi_core,
)
from .rendering import create_comparison, draw_diagnostic, draw_overlay, load_visual_background
from .validators import fragment_evidence_validator, step07_symmetry_validator


def collect_metadata_files(image_filter: str | None = None, limit: int | None = None) -> list[Path]:
    input_dir = get_step_dirs()["input_metadata_dir"]
    if not input_dir.exists():
        raise FileNotFoundError(f"Step 07 metadata directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*_symmetry.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [p for p in files if wanted in p.stem.lower() or p.stem.lower().startswith(wanted)]
    return files[: int(limit)] if limit is not None else files


def _resolve_step06_metadata(step07: dict, step07_path: Path) -> tuple[dict, Path]:
    path = normalize_path_value(step07.get("step_06_metadata_file"))
    if path is None or not path.exists():
        image_stem = Path(str(step07.get("image_name", step07_path.stem))).stem
        candidate = context.PROCESSED_DIR / "06_search_central_ruler" / "metadata" / f"{image_stem}_central_ruler.json"
        path = candidate if candidate.exists() else path
    if path is None or not path.exists():
        raise FileNotFoundError(f"Could not resolve Step 06 metadata referenced by: {step07_path}")
    return load_json(path), path


def _resolve_roi(step07: dict, step06: dict) -> tuple:
    path = normalize_path_value(step07.get("roi_mask_file") or step06.get("roi_mask_file"))
    roi = load_grayscale(path)
    if roi is None:
        raise FileNotFoundError(f"Could not load ROI mask: {path}")
    return roi, path


def _vertical_range(step07: dict, step06: dict, roi) -> tuple[int, int]:
    vertical = step07.get("vertical_range", {})
    if "y_min" in vertical and "y_max" in vertical:
        return max(0, int(vertical["y_min"])), min(roi.shape[0] - 1, int(vertical["y_max"]))
    profile = step06.get("roi_profile", {})
    return (
        max(0, int(profile.get("trimmed_y_min", profile.get("y_min", 0)))),
        min(roi.shape[0] - 1, int(profile.get("trimmed_y_max", profile.get("y_max", roi.shape[0] - 1)))),
    )


def _match_step06_candidate(step07_candidate: dict, step06_candidates: list[dict]) -> dict | None:
    source_rank = int(step07_candidate.get("source_rank", 0) or 0)
    if 1 <= source_rank <= len(step06_candidates):
        return step06_candidates[source_rank - 1]
    if not step06_candidates:
        return None
    x_ref = float(step07_candidate.get("x_ref", 0.0))
    a = float(step07_candidate.get("a", 0.0))
    return min(
        step06_candidates,
        key=lambda candidate: abs(float(candidate.get("x_ref", 0.0)) - x_ref) + 500.0 * abs(float(candidate.get("a", 0.0)) - a),
    )


def build_analysis(metadata_path: Path) -> dict:
    started = time.perf_counter()
    step07 = load_json(metadata_path)
    step06, step06_path = _resolve_step06_metadata(step07, metadata_path)
    image_name = str(step07.get("image_name", step06.get("image_name", metadata_path.stem + ".png")))
    step07_candidates = list(step07.get("ranked_candidates", []))
    if not step07_candidates and step07.get("winner"):
        step07_candidates = [step07["winner"]]
    step07_candidates = step07_candidates[: max(1, int(context.STEP_CONFIG.get("candidate_limit", 10)))]
    if not step07_candidates:
        raise RuntimeError(f"No Step 07 ranked candidates found for {image_name}")

    roi_raw, roi_path = _resolve_roi(step07, step06)
    y_min, y_max = _vertical_range(step07, step06, roi_raw)
    core_mask, core_info = prepare_roi_core(roi_raw, y_min, y_max)
    medial = compute_medial_reference(core_mask, core_info) if bool(cfg("medial_axis", "enabled", default=True)) else {"available": False, "reason": "disabled"}
    anchors = compute_structural_anchors(core_info) if bool(cfg("structural_anchors", "enabled", default=True)) else {"available": False, "anchors": []}
    step06_candidates = list(step06.get("top_candidates", []))

    candidates = []
    for candidate in step07_candidates:
        item = deepcopy(candidate)
        item.setdefault("candidate_label", f"C{int(item.get('source_rank', len(candidates) + 1)):02d}")
        matched = _match_step06_candidate(item, step06_candidates)
        validators = {"step_07_symmetry": step07_symmetry_validator(item)}
        validators["medial_axis"] = candidate_medial_score(item, medial) if bool(cfg("medial_axis", "enabled", default=True)) else {"available": False, "score": None, "reason": "disabled"}
        validators["structural_anchors"] = candidate_anchor_score(item, anchors) if bool(cfg("structural_anchors", "enabled", default=True)) else {"available": False, "score": None, "reason": "disabled"}
        validators["fragment_evidence"] = fragment_evidence_validator(matched) if bool(cfg("fragment_evidence", "enabled", default=True)) else {"available": False, "score": None, "reason": "disabled"}
        validators["roi_balance"] = candidate_roi_balance_score(item, core_info) if bool(cfg("roi_balance", "enabled", default=True)) else {"available": False, "score": None, "reason": "disabled"}
        item["validators"] = validators
        item["step_06_candidate"] = {
            "source_rank": int(item.get("source_rank", 0) or 0),
            "final_score": None if matched is None else matched.get("final_score", matched.get("score")),
            "hypothesis_source": None if matched is None else matched.get("hypothesis_source"),
            "selected_fragment_count": 0 if matched is None else int(matched.get("selected_fragment_count", 0) or 0),
        }
        candidates.append(item)

    validator_names = list(cfg("fusion", "validator_names", default=[]))
    ranked = compute_candidate_fusion(candidates, validator_names)
    confidence = compute_confidence(ranked, validator_names, y_min, y_max)
    winner = ranked[0]
    step07_winner_label = str(step07.get("winner_label", step07.get("winner", {}).get("candidate_label", "")))

    background, visual_path = load_visual_background(step06, step07, roi_raw)
    overlay = draw_overlay(background, ranked, medial, anchors, y_min, y_max, image_name, confidence)
    comparison = create_comparison(overlay, ranked, validator_names, confidence)
    diagnostic = draw_diagnostic(core_mask, medial, anchors, ranked, y_min, y_max)

    serial_medial = {key: value for key, value in medial.items() if key not in {"y", "x", "radius", "half_width"}}
    output_metadata = {
        "image_name": image_name,
        "processing_step": "08_multi_validate_central_ruler",
        "algorithm_version": "independent_validator_rank_fusion_v1",
        "source_step": step07.get("processing_step", "07_verify_central_ruler_symmetry"),
        "step_07_metadata_file": relative_project_path(metadata_path),
        "step_06_metadata_file": relative_project_path(step06_path),
        "roi_mask_file": relative_project_path(roi_path),
        "visual_file": visual_path,
        "candidate_count": int(len(ranked)),
        "validator_names": validator_names,
        "vertical_range": {"y_min": int(y_min), "y_max": int(y_max)},
        "final_candidate": winner["candidate_label"],
        "step_07_candidate": step07_winner_label,
        "step_07_agrees": bool(winner["candidate_label"] == step07_winner_label),
        "symmetry_percent": 100.0 * float(winner["validators"]["step_07_symmetry"]["score"]),
        "multi_validation_score": float(winner["multi_validation_score"]),
        "multi_validation_percent": float(winner["multi_validation_percent"]),
        **confidence,
        "winner": winner,
        "ranked_candidates": ranked,
        "medial_reference": serial_medial,
        "structural_anchor_reference": anchors,
        "parameters": deepcopy(context.STEP_CONFIG),
        "timings_sec": {"analysis_total": float(time.perf_counter() - started)},
    }
    return {
        "image_name": image_name,
        "metadata": output_metadata,
        "overlay": overlay,
        "comparison": comparison,
        "diagnostic": diagnostic,
    }


def process_metadata_file(metadata_path: Path) -> dict:
    started = time.perf_counter()
    analysis = build_analysis(metadata_path)
    dirs = get_step_dirs()
    image_name = analysis["image_name"]
    stem = Path(image_name).stem
    overlay_path = dirs["output_overlay_dir"] / image_name
    comparison_path = dirs["output_comparison_dir"] / image_name
    diagnostic_path = dirs["output_diagnostics_dir"] / image_name
    metadata_output_path = dirs["output_metadata_dir"] / f"{stem}_multi_validation.json"
    for path in [overlay_path, comparison_path, diagnostic_path, metadata_output_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, image in [(overlay_path, analysis["overlay"]), (comparison_path, analysis["comparison"]), (diagnostic_path, analysis["diagnostic"])]:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save image: {path}")
    metadata = deepcopy(analysis["metadata"])
    metadata.update({
        "output_overlay_file": relative_project_path(overlay_path),
        "output_comparison_file": relative_project_path(comparison_path),
        "output_diagnostic_file": relative_project_path(diagnostic_path),
    })
    metadata["timings_sec"]["process_total"] = float(time.perf_counter() - started)
    save_json(metadata_output_path, metadata)
    return {
        "image_name": image_name,
        "final_candidate": str(metadata["final_candidate"]),
        "step_07_candidate": str(metadata["step_07_candidate"]),
        "step_07_agrees": bool(metadata["step_07_agrees"]),
        "symmetry_percent": float(metadata["symmetry_percent"]),
        "multi_validation_percent": float(metadata["multi_validation_percent"]),
        "confidence_percent": float(metadata["confidence_percent"]),
        "decision": str(metadata["decision"]),
        "overlay_path": relative_project_path(overlay_path),
        "comparison_path": relative_project_path(comparison_path),
        "metadata_path": relative_project_path(metadata_output_path),
    }


def show_image(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return
    max_height = int(DISPLAY_CONFIG.get("max_height", 900))
    if image.shape[0] > max_height:
        scale = max_height / image.shape[0]
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max_height))
    cv2.imshow("Step 08 multi-validation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
