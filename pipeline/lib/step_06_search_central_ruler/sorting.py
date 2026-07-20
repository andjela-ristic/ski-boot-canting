from __future__ import annotations

import numpy as np

from .context import cfg
from .calculations import candidate_endpoint_strengths


def mean_axis_distance_px(
    candidate_a: dict,
    candidate_b: dict,
    roi_profile: dict | None = None,
    probe_rows: np.ndarray | None = None,
) -> float:
    if probe_rows is None:
        if roi_profile is None:
            raise ValueError("roi_profile or probe_rows must be provided")
        probe_rows = np.linspace(
            float(roi_profile["trimmed_y_min"]),
            float(roi_profile["trimmed_y_max"]),
            8,
            dtype=np.float64,
        )
    delta_a = float(candidate_a["a"]) - float(candidate_b["a"])
    delta_b = float(candidate_a["b"]) - float(candidate_b["b"])
    return float(np.mean(np.abs(delta_a * probe_rows + delta_b)))

def deduplicate_candidates(
    candidates: list[dict],
    roi_profile: dict,
    max_candidates: int | None = None,
    sort_key=None,
) -> list[dict]:
    kept: list[dict] = []
    max_mean_axis_distance_px = float(cfg("candidate_deduplication", "max_mean_axis_distance_px", default=8.0))
    max_angle_difference_deg = float(cfg("candidate_deduplication", "max_angle_difference_deg", default=0.45))
    resolved_max_candidates = (
        int(cfg("candidate_deduplication", "max_saved_candidates", default=8))
        if max_candidates is None
        else max(0, int(max_candidates))
    )
    if resolved_max_candidates <= 0 or not candidates:
        return []
    probe_rows = np.linspace(
        float(roi_profile["trimmed_y_min"]),
        float(roi_profile["trimmed_y_max"]),
        8,
        dtype=np.float64,
    )
    angle_bucket_size = max(1e-6, max_angle_difference_deg)
    x_bucket_size = max(1e-6, max_mean_axis_distance_px)
    kept_buckets: dict[tuple[int, int], list[dict]] = {}

    if sort_key is None:
        ordered_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    else:
        ordered_candidates = sorted(candidates, key=sort_key, reverse=True)

    for candidate in ordered_candidates:
        is_duplicate = False
        angle_bucket = int(round(float(candidate["tilt_deg"]) / angle_bucket_size))
        x_bucket = int(round(float(candidate["x_ref"]) / x_bucket_size))
        for angle_offset in (-1, 0, 1):
            if is_duplicate:
                break
            for x_offset in (-1, 0, 1):
                bucket_key = (angle_bucket + angle_offset, x_bucket + x_offset)
                for existing in kept_buckets.get(bucket_key, []):
                    if abs(float(candidate["tilt_deg"]) - float(existing["tilt_deg"])) > max_angle_difference_deg:
                        continue
                    if (
                        mean_axis_distance_px(candidate, existing, probe_rows=probe_rows)
                        <= max_mean_axis_distance_px
                    ):
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
        if not is_duplicate:
            kept.append(candidate)
            kept_buckets.setdefault((angle_bucket, x_bucket), []).append(candidate)
            if len(kept) >= resolved_max_candidates:
                break

    return kept[:resolved_max_candidates]



def select_ranked_candidate_portfolio(
    candidates: list[dict],
    roi_profile: dict,
    max_candidates: int,
    sort_key=None,
) -> list[dict]:
    """Select a score-led but axis-diverse final snapshot portfolio.

    C01 remains the highest scoring candidate. The remaining slots are not
    allowed to collapse around one local optimum: a few source-family
    representatives and farthest useful axes are retained so a correct sparse
    fallback can still appear in C01-C10.
    """
    if max_candidates <= 0 or not candidates:
        return []
    if sort_key is None:
        ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    else:
        ordered = sorted(candidates, key=sort_key, reverse=True)
    if not bool(cfg("candidate_portfolio", "enabled", default=True)):
        return select_saved_candidates_progressive(
            ordered,
            roi_profile,
            max_candidates=max_candidates,
            sort_key=sort_key,
        )

    best_score = float(
        ordered[0].get("final_score", ordered[0].get("score", 0.0))
    )
    min_score_ratio = float(
        cfg("candidate_portfolio", "min_score_ratio", default=0.55)
    )
    max_score_drop = float(
        cfg("candidate_portfolio", "max_score_drop", default=0.24)
    )
    score_floor = max(best_score * min_score_ratio, best_score - max_score_drop)
    eligible = [
        candidate
        for candidate in ordered
        if float(candidate.get("final_score", candidate.get("score", 0.0)))
        >= score_floor
    ]
    if not eligible:
        eligible = ordered

    probe_rows = np.linspace(
        float(roi_profile["trimmed_y_min"]),
        float(roi_profile["trimmed_y_max"]),
        8,
        dtype=np.float64,
    )
    duplicate_distance = float(
        cfg(
            "candidate_deduplication",
            "max_mean_axis_distance_px",
            default=5.0,
        )
    )
    duplicate_angle = float(
        cfg(
            "candidate_deduplication",
            "max_angle_difference_deg",
            default=0.25,
        )
    )
    selected: list[dict] = []
    exact_axes: set[tuple[float, float]] = set()

    def exact_key(candidate: dict) -> tuple[float, float]:
        return (
            round(float(candidate["x_ref"]), 6),
            round(float(candidate["tilt_deg"]), 6),
        )

    def is_normal_duplicate(candidate: dict) -> bool:
        for existing in selected:
            if (
                abs(float(candidate["tilt_deg"]) - float(existing["tilt_deg"]))
                <= duplicate_angle
                and mean_axis_distance_px(
                    candidate, existing, probe_rows=probe_rows
                )
                <= duplicate_distance
            ):
                return True
        return False

    def add_candidate(candidate: dict, allow_near: bool = False) -> bool:
        key = exact_key(candidate)
        if key in exact_axes:
            return False
        if not allow_near and is_normal_duplicate(candidate):
            return False
        selected.append(candidate)
        exact_axes.add(key)
        return True

    top_score_slots = max(
        1,
        min(
            max_candidates,
            int(cfg("candidate_portfolio", "top_score_slots", default=4)),
        ),
    )
    for candidate in ordered:
        add_candidate(candidate)
        if len(selected) >= top_score_slots:
            break

    if bool(
        cfg("candidate_portfolio", "reserve_best_per_source", default=True)
    ):
        source_order = cfg(
            "candidate_portfolio",
            "source_order",
            default=["fragment_pair", "fragment_axis", "roi_prior"],
        )
        for source in source_order:
            if len(selected) >= max_candidates:
                break
            for candidate in eligible:
                if candidate.get("hypothesis_source", "grid") != source:
                    continue
                if add_candidate(candidate):
                    break

    novelty_target = max(
        1.0,
        float(roi_profile.get("reference_width_px", 1.0))
        * float(
            cfg(
                "candidate_portfolio",
                "novelty_target_width_ratio",
                default=0.08,
            )
        ),
    )
    score_weight = max(
        0.0, float(cfg("candidate_portfolio", "score_weight", default=0.55))
    )
    novelty_weight = max(
        0.0,
        float(cfg("candidate_portfolio", "novelty_weight", default=0.45)),
    )
    weight_sum = max(1e-9, score_weight + novelty_weight)
    score_weight /= weight_sum
    novelty_weight /= weight_sum

    while len(selected) < max_candidates:
        remaining = [
            candidate for candidate in eligible if exact_key(candidate) not in exact_axes
        ]
        if not remaining:
            break

        best_candidate = None
        best_utility = None
        for candidate in remaining:
            min_distance = min(
                (
                    mean_axis_distance_px(
                        candidate, existing, probe_rows=probe_rows
                    )
                    for existing in selected
                ),
                default=novelty_target,
            )
            novelty = min(1.0, min_distance / novelty_target)
            score = float(
                candidate.get("final_score", candidate.get("score", 0.0))
            )
            score_normalized = (
                (score - score_floor) / max(1e-9, best_score - score_floor)
                if best_score > score_floor
                else 1.0
            )
            score_normalized = float(np.clip(score_normalized, 0.0, 1.0))
            utility = score_weight * score_normalized + novelty_weight * novelty
            tie_key = (
                utility,
                score,
                min_distance,
                float(candidate.get("geometry_score", 0.0)),
            )
            if best_utility is None or tie_key > best_utility:
                best_utility = tie_key
                best_candidate = candidate

        if best_candidate is None:
            break
        # At this stage novelty is intentional. Only exact duplicates are
        # rejected, because normal deduplication was already applied to the
        # top-score and source-reserved slots.
        add_candidate(best_candidate, allow_near=True)

    if len(selected) < max_candidates:
        for candidate in ordered:
            if len(selected) >= max_candidates:
                break
            add_candidate(candidate, allow_near=True)

    if sort_key is None:
        return sorted(selected, key=lambda item: item["score"], reverse=True)
    return sorted(selected, key=sort_key, reverse=True)

def select_saved_candidates_progressive(
    candidates: list[dict],
    roi_profile: dict,
    max_candidates: int,
    sort_key=None,
) -> list[dict]:
    """Fill the requested snapshot count without abandoning deduplication.

    The first pass uses the normal geometric duplicate thresholds. If an image
    has only a few clusters (as happened for 508), later passes progressively
    reduce those thresholds and expose useful nearby alternatives. Exact axis
    duplicates are never returned.
    """
    if max_candidates <= 0 or not candidates:
        return []
    if sort_key is None:
        ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    else:
        ordered = sorted(candidates, key=sort_key, reverse=True)

    base_distance = float(
        cfg(
            "candidate_deduplication",
            "max_mean_axis_distance_px",
            default=5.0,
        )
    )
    base_angle = float(
        cfg(
            "candidate_deduplication",
            "max_angle_difference_deg",
            default=0.25,
        )
    )
    scales = cfg(
        "candidate_deduplication",
        "progressive_distance_scales",
        default=[1.0, 0.65, 0.35, 0.0],
    )
    if not bool(
        cfg(
            "candidate_deduplication",
            "progressive_fill_enabled",
            default=True,
        )
    ):
        scales = [1.0]

    probe_rows = np.linspace(
        float(roi_profile["trimmed_y_min"]),
        float(roi_profile["trimmed_y_max"]),
        8,
        dtype=np.float64,
    )
    kept: list[dict] = []
    exact_axes: set[tuple[float, float]] = set()

    for raw_scale in scales:
        scale = max(0.0, float(raw_scale))
        distance_threshold = base_distance * scale
        angle_threshold = base_angle * scale
        for candidate in ordered:
            exact_key = (
                round(float(candidate["x_ref"]), 6),
                round(float(candidate["tilt_deg"]), 6),
            )
            if exact_key in exact_axes:
                continue

            is_duplicate = False
            if scale > 0.0:
                for existing in kept:
                    if (
                        abs(
                            float(candidate["tilt_deg"])
                            - float(existing["tilt_deg"])
                        )
                        <= angle_threshold
                        and mean_axis_distance_px(
                            candidate, existing, probe_rows=probe_rows
                        )
                        <= distance_threshold
                    ):
                        is_duplicate = True
                        break
            if is_duplicate:
                continue

            kept.append(candidate)
            exact_axes.add(exact_key)
            if len(kept) >= max_candidates:
                return kept

    return kept[:max_candidates]

def sort_candidates(
    candidates: list[dict],
    sort_key=None,
) -> list[dict]:
    if sort_key is None:
        return sorted(candidates, key=lambda item: item["score"], reverse=True)
    return sorted(candidates, key=sort_key, reverse=True)

def unique_candidates_by_axis(
    candidates: list[dict],
) -> list[dict]:
    unique: list[dict] = []
    seen_axes: set[tuple[float, float]] = set()
    for candidate in candidates:
        axis_key = (
            round(float(candidate["x_ref"]), 6),
            round(float(candidate["tilt_deg"]), 6),
        )
        if axis_key in seen_axes:
            continue
        seen_axes.add(axis_key)
        unique.append(candidate)
    return unique

def filter_candidates_by_score_ratio(
    candidates: list[dict],
    score_ratio_floor: float,
) -> list[dict]:
    if not candidates:
        return []
    if score_ratio_floor <= 0.0:
        return candidates

    best_score = float(candidates[0]["score"])
    min_score = best_score * score_ratio_floor
    kept = [candidate for candidate in candidates if float(candidate["score"]) >= min_score]
    return kept if kept else candidates[:1]

def select_diverse_candidates_by_angle(
    candidates: list[dict],
    max_candidates: int,
    angle_bucket_deg: float,
    max_per_bucket: int,
) -> list[dict]:
    if max_candidates <= 0 or not candidates:
        return []
    if angle_bucket_deg <= 0.0 or max_per_bucket <= 0:
        return candidates[:max_candidates]

    selected: list[dict] = []
    selected_ids: set[int] = set()
    bucket_counts: dict[int, int] = {}

    for candidate in candidates:
        bucket = int(round(float(candidate["tilt_deg"]) / angle_bucket_deg))
        current_count = bucket_counts.get(bucket, 0)
        if current_count >= max_per_bucket:
            continue

        selected.append(candidate)
        selected_ids.add(id(candidate))
        bucket_counts[bucket] = current_count + 1
        if len(selected) >= max_candidates:
            return selected

    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    return selected

def select_diverse_candidates(
    candidates: list[dict],
    max_candidates: int,
    angle_bucket_deg: float,
    max_per_angle_bucket: int,
    x_bucket_px: float,
    max_per_x_bucket: int,
) -> list[dict]:
    if max_candidates <= 0 or not candidates:
        return []
    if (
        angle_bucket_deg <= 0.0
        or max_per_angle_bucket <= 0
        or x_bucket_px <= 0.0
        or max_per_x_bucket <= 0
    ):
        return candidates[:max_candidates]

    selected: list[dict] = []
    selected_ids: set[int] = set()
    angle_bucket_counts: dict[int, int] = {}
    x_bucket_counts: dict[int, int] = {}

    for candidate in candidates:
        angle_bucket = int(round(float(candidate["tilt_deg"]) / angle_bucket_deg))
        x_bucket = int(round(float(candidate["x_ref"]) / x_bucket_px))
        current_angle_count = angle_bucket_counts.get(angle_bucket, 0)
        current_x_count = x_bucket_counts.get(x_bucket, 0)
        if current_angle_count >= max_per_angle_bucket or current_x_count >= max_per_x_bucket:
            continue

        selected.append(candidate)
        selected_ids.add(id(candidate))
        angle_bucket_counts[angle_bucket] = current_angle_count + 1
        x_bucket_counts[x_bucket] = current_x_count + 1
        if len(selected) >= max_candidates:
            return selected

    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    return selected

def compute_best_fit_selection_score(candidate: dict) -> float:
    """Return the single normalized score used for final ranking."""
    return float(candidate.get("final_score", candidate.get("score", 0.0)))

def annotate_candidate_selection(
    candidate: dict,
    hypothesis: dict,
    hypothesis_rank: int,
    stage_name: str,
) -> dict:
    result = dict(candidate)
    result["search_stage"] = stage_name
    result["hypothesis_x_ref"] = float(hypothesis["x_ref"])
    result["hypothesis_tilt_deg"] = float(hypothesis["tilt_deg"])
    result["hypothesis_score"] = float(hypothesis.get("final_score", hypothesis.get("score", 0.0)))
    result["hypothesis_x_ref_delta_px"] = abs(float(result["x_ref"]) - float(hypothesis["x_ref"]))
    result["hypothesis_tilt_delta_deg"] = abs(float(result["tilt_deg"]) - float(hypothesis["tilt_deg"]))
    result["source_hypothesis_rank"] = int(hypothesis_rank)
    result["source_hypothesis_label"] = f"C{hypothesis_rank:02d}"
    result["hypothesis_source"] = str(
        hypothesis.get("hypothesis_source", result.get("hypothesis_source", "grid"))
    )
    for metadata_key in (
        "structural_seed_score",
        "seed_line_indices",
        "seed_vertical_separation_ratio",
        "structural_seed_inside_roi_ratio",
        "roi_prior_x_offset_ratio",
        "roi_prior_angle_offset_deg",
        "hypothesis_band_half_width_px",
        "hypothesis_max_angle_error_deg",
    ):
        if metadata_key in hypothesis:
            result[metadata_key] = hypothesis[metadata_key]
    result["selection_score"] = float(compute_best_fit_selection_score(result))
    return result

def candidate_selection_key(candidate: dict) -> tuple[float, ...]:
    return candidate_ranking_key(candidate)

def candidate_ranking_key(candidate: dict) -> tuple[float, ...]:
    """Final score is authoritative; remaining fields are deterministic tie-breakers."""
    return (
        float(candidate.get("final_score", candidate.get("selection_score", candidate.get("score", 0.0)))),
        1.0 if bool(candidate.get("validation_passed", True)) else 0.0,
        float(candidate.get("geometry_score", 0.0)),
        float(candidate.get("mirror_symmetry_score", 0.5)),
        float(candidate.get("unique_vertical_coverage", candidate.get("vertical_coverage_score", 0.0))),
        float(candidate.get("chain_span_ratio", 0.0)),
        float(candidate.get("chain_continuity_ratio", 0.0)),
        -float(candidate.get("fit_rmse_px", 1e9)),
        -float(candidate.get("largest_gap_px", 1e9)),
    )

