from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import cv2

from . import context
from .context import (
    DISPLAY_CONFIG,
    get_step_dirs,
    load_grayscale,
    load_json,
    normalize_path_value,
    relative_project_path,
    save_json,
)
from .fusion import compute_candidate_validation_scores, compute_confidence, select_final_candidate
from .geometry import candidates_equivalent, evaluation_mask_axis_support
from .rendering import create_comparison, draw_diagnostic, draw_overlay, load_visual_background
from .validators import (
    compute_segment_perturbation_stability,
    fragment_evidence_validator,
    segment_consistency_validator,
    step07_symmetry_validator,
)


def collect_metadata_files(image_filter: str | None = None, limit: int | None = None) -> list[Path]:
    input_dir = get_step_dirs()["input_metadata_dir"]
    if not input_dir.exists():
        raise FileNotFoundError(f"Step 07 metadata directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*_symmetry.json"))
    if image_filter:
        wanted = Path(image_filter).stem.lower()
        files = [path for path in files if wanted in path.stem.lower() or path.stem.lower().startswith(wanted)]
    return files[: int(limit)] if limit is not None else files


def _resolve_step06_metadata(step07: dict, step07_path: Path) -> tuple[dict, Path]:
    path = normalize_path_value(step07.get("step_06_metadata_file"))
    if path is None or not path.exists():
        image_stem = Path(str(step07.get("image_name", step07_path.stem))).stem
        fallback = context.PROCESSED_DIR / "06_search_central_ruler" / "metadata" / f"{image_stem}_central_ruler.json"
        path = fallback if fallback.exists() else path
    if path is None or not path.exists():
        raise FileNotFoundError(f"Could not resolve Step 06 metadata referenced by: {step07_path}")
    return load_json(path), path


def _resolve_step07_evaluation_mask(step07: dict, image_name: str) -> tuple:
    path = normalize_path_value(step07.get("output_evaluation_mask_file"))
    if path is None or not path.exists():
        fallback = get_step_dirs()["input_dir"] / "evaluation_mask" / image_name
        path = fallback if fallback.exists() else path
    mask = load_grayscale(path)
    if mask is None:
        raise FileNotFoundError(
            "Step 08 requires the exact candidate-independent evaluation mask produced by Step 07. "
            "Install the included Step 07 persistence patch and rerun Step 07. "
            f"Missing mask for {image_name}: {path}"
        )
    return (mask > 0).astype("uint8") * 255, path


def _vertical_range(step07: dict, mask) -> tuple[int, int]:
    vertical = step07.get("vertical_range", {})
    if "y_min" in vertical and "y_max" in vertical:
        return max(0, int(vertical["y_min"])), min(mask.shape[0] - 1, int(vertical["y_max"]))
    rows = (mask > 0).any(axis=1).nonzero()[0]
    if rows.size == 0:
        raise RuntimeError("Step 07 evaluation mask is empty")
    return int(rows[0]), int(rows[-1])


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
        key=lambda candidate: (
            abs(float(candidate.get("x_ref", 0.0)) - x_ref)
            + 500.0 * abs(float(candidate.get("a", 0.0)) - a)
        ),
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

    evaluation_mask, evaluation_mask_path = _resolve_step07_evaluation_mask(step07, image_name)
    y_min, y_max = _vertical_range(step07, evaluation_mask)
    step06_candidates = list(step06.get("top_candidates", []))

    candidates = []
    for index, candidate in enumerate(step07_candidates):
        item = deepcopy(candidate)
        item.setdefault("candidate_label", f"C{int(item.get('source_rank', index + 1)):02d}")
        item.setdefault("source_rank", int(item.get("verification_rank", index + 1) or index + 1))
        matched = _match_step06_candidate(item, step06_candidates)
        item["validators"] = {
            "step_07_symmetry": step07_symmetry_validator(item),
            "segment_consistency": segment_consistency_validator(item),
            "fragment_evidence": fragment_evidence_validator(matched),
            # This is a hard support check only. The mask boundary is not used
            # as an independent geometry score because it was created by Step 07.
            "evaluation_mask_support": evaluation_mask_axis_support(item, evaluation_mask, y_min, y_max),
        }
        item["step_06_candidate"] = {
            "source_rank": int(item.get("source_rank", 0) or 0),
            "final_score": None if matched is None else matched.get("final_score", matched.get("score")),
            "hypothesis_source": None if matched is None else matched.get("hypothesis_source"),
            "selected_fragment_count": 0 if matched is None else int(matched.get("selected_fragment_count", 0) or 0),
        }
        candidates.append(item)

    stability_by_label, perturbation_scenarios = compute_segment_perturbation_stability(candidates, y_min, y_max)
    for candidate in candidates:
        candidate["validators"]["perturbation_stability"] = stability_by_label[str(candidate["candidate_label"])]

    ranked = compute_candidate_validation_scores(candidates)
    step07_winner_label = str(step07.get("winner_label", step07.get("winner", {}).get("candidate_label", "")))
    final_candidate, selection_info = select_final_candidate(ranked, step07_winner_label)
    ensemble_candidate = ranked[0]
    selection_info["ensemble_recommends_same_or_equivalent"] = bool(
        candidates_equivalent(final_candidate, ensemble_candidate, y_min, y_max)
    )
    confidence = compute_confidence(final_candidate, ranked, y_min, y_max)

    background, visual_path = load_visual_background(step06, step07, evaluation_mask)
    overlay = draw_overlay(
        background,
        evaluation_mask,
        ranked,
        final_candidate,
        y_min,
        y_max,
        image_name,
        confidence,
        selection_info,
    )
    comparison = create_comparison(overlay, ranked, final_candidate, confidence, selection_info)
    diagnostic = draw_diagnostic(evaluation_mask, ranked, final_candidate, y_min, y_max)

    output_metadata = {
        "image_name": image_name,
        "processing_step": "08_multi_validate_central_ruler_v2",
        "algorithm_version": "step07_mask_reuse_segment_stability_confidence_v2",
        "source_step": step07.get("processing_step", "07_verify_central_ruler_symmetry"),
        "step_07_metadata_file": relative_project_path(metadata_path),
        "step_06_metadata_file": relative_project_path(step06_path),
        "step_07_evaluation_mask_file": relative_project_path(evaluation_mask_path),
        "evaluation_mask_role": "support_and_exclusion_only_not_scored_shape_geometry",
        "visual_file": visual_path,
        "candidate_count": int(len(ranked)),
        "vertical_range": {"y_min": int(y_min), "y_max": int(y_max)},
        "final_candidate": str(final_candidate["candidate_label"]),
        "step_07_candidate": step07_winner_label,
        "step_07_agrees": bool(str(final_candidate["candidate_label"]) == step07_winner_label),
        "symmetry_percent": 100.0 * float(final_candidate["validators"]["step_07_symmetry"]["score"]),
        "multi_validation_score": float(final_candidate["multi_validation_score"]),
        "multi_validation_percent": float(final_candidate["multi_validation_percent"]),
        **selection_info,
        **confidence,
        "winner": final_candidate,
        "ranked_candidates": ranked,
        "perturbation_scenarios": perturbation_scenarios,
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
    for path in (overlay_path, comparison_path, diagnostic_path, metadata_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, image in (
        (overlay_path, analysis["overlay"]),
        (comparison_path, analysis["comparison"]),
        (diagnostic_path, analysis["diagnostic"]),
    ):
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
        "candidate_count": int(metadata["candidate_count"]),
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
    cv2.imshow("Step 08 multi-validation confidence", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
