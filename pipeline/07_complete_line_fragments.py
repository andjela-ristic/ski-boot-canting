from pathlib import Path
import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()

PATHS = CONFIG["paths"]
DISPLAY = CONFIG.get("display", {})
STEP = CONFIG["step_07_complete_line_fragments"]

PROCESSED_DIR = PROJECT_ROOT / PATHS["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS["metadata_dir"]

VISUAL_INPUT_DIR = PROCESSED_DIR / STEP["input_visual_subdir"]
EDGE_INPUT_DIR = PROCESSED_DIR / STEP["input_edges_subdir"]
ROI_INPUT_DIR = PROCESSED_DIR / STEP["input_roi_subdir"]

OUTPUT_DIR = PROCESSED_DIR / STEP["output_subdir"]
ORIGINAL_DIR = OUTPUT_DIR / "01_original_hough_fragments"
MASKED_DIR = OUTPUT_DIR / "02_mask_filtered_fragments"
VERTICAL_DIR = OUTPUT_DIR / "03_vertical_fragments"
COMPLETED_DIR = OUTPUT_DIR / "04_completed_lines"
DEBUG_DIR = OUTPUT_DIR / "debug"
JSON_DIR = OUTPUT_DIR / "json"

CSV_PATH = METADATA_DIR / "processing_07_complete_line_fragments.csv"


@dataclass
class Fragment:
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deviation_from_vertical_deg: float
    a: float
    b: float
    mask_coverage: float


@dataclass
class Transform:
    fragment_id: str
    dx_normal_px: float
    dtheta_deg: float
    merge_type: str


@dataclass
class CompletedLine:
    id: str
    a: float
    b: float
    x1: float
    y1: float
    x2: float
    y2: float
    theta_deviation_from_vertical_deg: float
    fragment_ids: list[str]
    fragment_count: int
    total_original_length: float
    completed_length: float
    y_min_original: float
    y_max_original: float
    max_gap_filled_px: float
    total_shift_px: float
    max_shift_px: float
    total_angle_adjust_deg: float
    max_angle_adjust_deg: float
    edge_support: float
    support_score: float
    transforms: list[dict[str, Any]]


def cfg(*keys: str, default: Any) -> Any:
    cur: Any = STEP
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_dirs() -> None:
    for d in [
        OUTPUT_DIR,
        ORIGINAL_DIR,
        MASKED_DIR,
        VERTICAL_DIR,
        COMPLETED_DIR,
        DEBUG_DIR,
        JSON_DIR,
        METADATA_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def collect_images() -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg"}
    if not VISUAL_INPUT_DIR.exists():
        return []
    return sorted(
        p for p in VISUAL_INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )


def read_gray(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if shape_hw is not None and img.shape[:2] != shape_hw:
        h, w = shape_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    return img


def make_edge_fallback(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 75, 100, apertureSize=3, L2gradient=True)


def load_edge(image_path: Path, image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    edge_path = EDGE_INPUT_DIR / image_path.name

    edge = read_gray(edge_path, (h, w))
    if edge is None:
        edge = make_edge_fallback(image_bgr)

    _, edge = cv2.threshold(edge, 1, 255, cv2.THRESH_BINARY)
    return edge


def find_mask_file(image_name: str, mode: str) -> Path | None:
    preferred_dirs = []

    if mode == "boot_mask":
        preferred_dirs = [
            ROI_INPUT_DIR / "boot_mask",
            ROI_INPUT_DIR / "selected_boot_mask",
            ROI_INPUT_DIR / "mask",
            ROI_INPUT_DIR,
        ]
    elif mode == "roi_mask":
        preferred_dirs = [
            ROI_INPUT_DIR / "roi_mask",
            ROI_INPUT_DIR / "selected_roi_mask",
            ROI_INPUT_DIR / "mask",
            ROI_INPUT_DIR,
        ]
    else:
        preferred_dirs = [
            ROI_INPUT_DIR / "boot_mask",
            ROI_INPUT_DIR / "roi_mask",
            ROI_INPUT_DIR / "mask",
            ROI_INPUT_DIR,
        ]

    for d in preferred_dirs:
        p = d / image_name
        if p.exists():
            return p

    if ROI_INPUT_DIR.exists():
        for p in ROI_INPUT_DIR.rglob(image_name):
            lowered = str(p).lower()
            if "mask" in lowered or "roi" in lowered:
                return p

    return None


def load_roi_mask(image_name: str, shape_hw: tuple[int, int]) -> tuple[np.ndarray, str]:
    h, w = shape_hw
    mode = str(cfg("mask_filter", "mode", default="boot_mask")).lower()

    mask_file = find_mask_file(image_name, mode)
    if mask_file is None and mode == "boot_mask":
        mask_file = find_mask_file(image_name, "roi_mask")

    if mask_file is None:
        mask = np.full((h, w), 255, dtype=np.uint8)
        return mask, "full_image_fallback"

    mask = read_gray(mask_file, (h, w))
    if mask is None:
        mask = np.full((h, w), 255, dtype=np.uint8)
        return mask, "full_image_fallback"

    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask, str(mask_file.relative_to(PROJECT_ROOT))


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    if not bool(cfg("mask_filter", "enabled", default=True)):
        return mask

    k = int(cfg("mask_filter", "dilate_kernel_size", default=21))
    iterations = int(cfg("mask_filter", "dilate_iterations", default=1))

    if k <= 1 or iterations <= 0:
        return mask

    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=iterations)


def line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x2 - x1, y2 - y1))


def fit_x_as_function_of_y(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """
    Fits x = a*y + b.

    This is intentionally used because Step 7 only works with near-vertical lines.
    For vertical lines, a is close to 0, and b is approximately the x position.
    """
    if len(points) < 2:
        return 0.0, 0.0

    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ws = np.array([max(1e-6, p[2]) for p in points], dtype=np.float64)

    y_mean = float(np.average(ys, weights=ws))
    x_mean = float(np.average(xs, weights=ws))

    denom = float(np.sum(ws * (ys - y_mean) ** 2))
    if denom < 1e-6:
        return 0.0, x_mean

    a = float(np.sum(ws * (ys - y_mean) * (xs - x_mean)) / denom)
    b = float(x_mean - a * y_mean)
    return a, b


def model_from_segment(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    length = line_length(x1, y1, x2, y2)
    return fit_x_as_function_of_y([
        (x1, y1, length / 2.0),
        (x2, y2, length / 2.0),
    ])


def angle_deviation_from_vertical(a: float) -> float:
    return float(abs(math.degrees(math.atan(a))))


def x_at_y(a: float, b: float, y: float) -> float:
    return float(a * y + b)


def segment_mask_coverage(mask: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    length = max(1.0, line_length(x1, y1, x2, y2))
    samples = max(8, int(round(length / 4.0)))

    h, w = mask.shape[:2]
    inside = 0
    total = 0

    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))

        if 0 <= x < w and 0 <= y < h:
            total += 1
            if mask[y, x] > 0:
                inside += 1

    if total == 0:
        return 0.0

    return float(inside / total)


def detect_hough_lines(masked_edge: np.ndarray) -> list[tuple[int, int, int, int]]:
    hough = STEP.get("hough_lines_p", {})

    rho = float(hough.get("rho", 1))
    theta_divisor = float(hough.get("theta_divisor", 180))
    theta = np.pi / theta_divisor

    threshold = int(hough.get("threshold", 45))
    min_line_length = int(hough.get("min_line_length", 70))
    max_line_gap = int(hough.get("max_line_gap", 20))

    raw = cv2.HoughLinesP(
        masked_edge,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if raw is None:
        return []

    lines: list[tuple[int, int, int, int]] = []
    for item in raw[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in item]
        lines.append((x1, y1, x2, y2))

    return lines


def make_fragment(
    idx: int,
    line: tuple[int, int, int, int],
    mask: np.ndarray,
) -> Fragment | None:
    x1, y1, x2, y2 = line
    length = line_length(x1, y1, x2, y2)

    min_len = float(cfg("line_selection", "min_fragment_length_px", default=35))
    if length < min_len:
        return None

    a, b = model_from_segment(x1, y1, x2, y2)
    angle_dev = angle_deviation_from_vertical(a)

    coverage = segment_mask_coverage(mask, x1, y1, x2, y2)

    return Fragment(
        id=f"fragment_{idx:03d}",
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        length=float(length),
        angle_deviation_from_vertical_deg=float(angle_dev),
        a=float(a),
        b=float(b),
        mask_coverage=float(coverage),
    )


def filter_mask_supported(fragments: list[Fragment]) -> list[Fragment]:
    min_cov = float(cfg("mask_filter", "min_line_mask_coverage", default=0.45))
    return [f for f in fragments if f.mask_coverage >= min_cov]


def filter_vertical(fragments: list[Fragment]) -> list[Fragment]:
    if not bool(cfg("line_selection", "use_only_vertical", default=True)):
        return fragments

    tol = float(cfg("line_selection", "vertical_angle_tolerance_degrees", default=15))
    return [f for f in fragments if f.angle_deviation_from_vertical_deg <= tol]


def fragment_y_range(f: Fragment) -> tuple[float, float]:
    return min(f.y1, f.y2), max(f.y1, f.y2)


def group_y_range(group: list[Fragment]) -> tuple[float, float]:
    ys = []
    for f in group:
        ys.extend([f.y1, f.y2])
    return float(min(ys)), float(max(ys))


def y_gap_between_groups(g1: list[Fragment], g2: list[Fragment]) -> float:
    a1, a2 = group_y_range(g1)
    b1, b2 = group_y_range(g2)

    if a2 < b1:
        return float(b1 - a2)
    if b2 < a1:
        return float(a1 - b2)
    return 0.0


def fit_group_model(group: list[Fragment]) -> tuple[float, float]:
    points: list[tuple[float, float, float]] = []

    for f in group:
        w = max(1.0, f.length / 2.0)
        points.append((f.x1, f.y1, w))
        points.append((f.x2, f.y2, w))

    return fit_x_as_function_of_y(points)


def classify_fragment_to_model(f: Fragment, model_a: float, model_b: float) -> tuple[bool, str, float, float]:
    mid_y = (f.y1 + f.y2) / 2.0
    mid_x = (f.x1 + f.x2) / 2.0

    dx = float(mid_x - x_at_y(model_a, model_b, mid_y))
    dtheta = float(f.angle_deviation_from_vertical_deg - angle_deviation_from_vertical(model_a))

    exact_angle = float(cfg("merge", "exact", "max_angle_diff_degrees", default=2.0))
    exact_shift = float(cfg("merge", "exact", "max_rho_diff_px", default=6.0))

    shift_enabled = bool(cfg("merge", "shift", "enabled", default=True))
    shift_angle = float(cfg("merge", "shift", "max_angle_diff_degrees", default=2.0))
    max_shift = float(cfg("merge", "shift", "max_parallel_shift_px", default=18.0))

    rotate_enabled = bool(cfg("merge", "rotate", "enabled", default=True))
    max_angle_adjust = float(cfg("merge", "rotate", "max_angle_adjust_degrees", default=3.0))
    max_shift_after_rotation = float(cfg("merge", "rotate", "max_parallel_shift_after_rotation_px", default=12.0))

    abs_dx = abs(dx)
    abs_dt = abs(dtheta)

    if abs_dt <= exact_angle and abs_dx <= exact_shift:
        return True, "exact", dx, dtheta

    if shift_enabled and abs_dt <= shift_angle and abs_dx <= max_shift:
        return True, "shift", dx, dtheta

    if rotate_enabled and abs_dt <= max_angle_adjust and abs_dx <= max_shift_after_rotation:
        return True, "rotate", dx, dtheta

    return False, "incompatible", dx, dtheta


def groups_are_compatible(g1: list[Fragment], g2: list[Fragment]) -> tuple[bool, float]:
    max_gap = float(cfg("merge", "max_gap_px", default=90))
    gap = y_gap_between_groups(g1, g2)

    if gap > max_gap:
        return False, gap

    merged = g1 + g2
    model_a, model_b = fit_group_model(merged)

    for f in merged:
        ok, _, _, _ = classify_fragment_to_model(f, model_a, model_b)
        if not ok:
            return False, gap

    return True, gap


def merge_groups(fragments: list[Fragment]) -> list[list[Fragment]]:
    groups: list[list[Fragment]] = [[f] for f in fragments]

    if not bool(cfg("merge", "iterative_merge", default=True)):
        return groups

    max_iterations = int(cfg("merge", "max_iterations", default=8))

    for _ in range(max_iterations):
        best_pair: tuple[int, int] | None = None
        best_score = -1e9

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ok, gap = groups_are_compatible(groups[i], groups[j])
                if not ok:
                    continue

                total_len = sum(f.length for f in groups[i] + groups[j])
                fragment_bonus = len(groups[i]) + len(groups[j])
                score = total_len + 25.0 * fragment_bonus - 2.0 * gap

                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        groups[i] = groups[i] + groups[j]
        del groups[j]

    return groups


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w - 1, h - 1

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def completed_line_endpoints(
    a: float,
    b: float,
    group: list[Fragment],
    mask: np.ndarray,
) -> tuple[float, float, float, float]:
    h, w = mask.shape[:2]
    _, mask_y1, _, mask_y2 = mask_bbox(mask)

    y1_original, y2_original = group_y_range(group)

    if bool(cfg("extension", "extend_to_mask_bounds", default=True)):
        pad = float(cfg("extension", "extend_padding_px", default=20))
        y1 = max(0.0, float(mask_y1) - pad)
        y2 = min(float(h - 1), float(mask_y2) + pad)
    else:
        y1 = y1_original
        y2 = y2_original

    x1 = float(np.clip(x_at_y(a, b, y1), 0, w - 1))
    x2 = float(np.clip(x_at_y(a, b, y2), 0, w - 1))

    return x1, y1, x2, y2


def edge_support_along_line(edge: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    length = max(1.0, line_length(x1, y1, x2, y2))
    samples = max(20, int(round(length / 3.0)))

    edge_radius = int(cfg("scoring", "edge_support_radius_px", default=2))
    if edge_radius > 0:
        k = edge_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge_eval = cv2.dilate(edge, kernel, iterations=1)
    else:
        edge_eval = edge

    h, w = edge_eval.shape[:2]
    hits = 0
    total = 0

    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))

        if 0 <= x < w and 0 <= y < h:
            total += 1
            if edge_eval[y, x] > 0:
                hits += 1

    if total == 0:
        return 0.0

    return float(hits / total)


def max_gap_inside_group(group: list[Fragment]) -> float:
    intervals = sorted(fragment_y_range(f) for f in group)
    if len(intervals) <= 1:
        return 0.0

    max_gap = 0.0
    current_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start > current_end:
            max_gap = max(max_gap, start - current_end)
        current_end = max(current_end, end)

    return float(max_gap)


def build_completed_line(idx: int, group: list[Fragment], mask: np.ndarray, edge: np.ndarray) -> CompletedLine:
    a, b = fit_group_model(group)
    x1, y1, x2, y2 = completed_line_endpoints(a, b, group, mask)

    transforms: list[Transform] = []
    total_shift = 0.0
    max_shift = 0.0
    total_angle_adjust = 0.0
    max_angle_adjust = 0.0

    for f in group:
        ok, merge_type, dx, dtheta = classify_fragment_to_model(f, a, b)
        abs_dx = abs(dx)
        abs_dt = abs(dtheta)

        total_shift += abs_dx
        max_shift = max(max_shift, abs_dx)
        total_angle_adjust += abs_dt
        max_angle_adjust = max(max_angle_adjust, abs_dt)

        transforms.append(
            Transform(
                fragment_id=f.id,
                dx_normal_px=float(dx),
                dtheta_deg=float(dtheta),
                merge_type=merge_type if ok else "kept_but_incompatible_debug",
            )
        )

    total_original_length = float(sum(f.length for f in group))
    completed_length = float(line_length(x1, y1, x2, y2))
    edge_support = edge_support_along_line(edge, x1, y1, x2, y2)

    mask_h = max(1.0, float(mask.shape[0]))
    length_score = min(1.0, total_original_length / mask_h)
    fragment_count_score = min(1.0, len(group) / 3.0)

    max_shift_allowed = max(1.0, float(cfg("merge", "shift", "max_parallel_shift_px", default=18.0)))
    max_angle_allowed = max(1.0, float(cfg("merge", "rotate", "max_angle_adjust_degrees", default=3.0)))

    shift_penalty = min(1.0, total_shift / (max(1, len(group)) * max_shift_allowed))
    angle_penalty = min(1.0, total_angle_adjust / (max(1, len(group)) * max_angle_allowed))

    w_len = float(cfg("dominance", "length_weight", default=0.40))
    w_edge = float(cfg("dominance", "edge_support_weight", default=0.25))
    w_count = float(cfg("dominance", "fragment_count_weight", default=0.20))
    w_shift = float(cfg("dominance", "shift_penalty_weight", default=0.10))
    w_angle = float(cfg("dominance", "angle_penalty_weight", default=0.05))

    support_score = (
        w_len * length_score
        + w_edge * edge_support
        + w_count * fragment_count_score
        - w_shift * shift_penalty
        - w_angle * angle_penalty
    )

    y_min_original, y_max_original = group_y_range(group)

    return CompletedLine(
        id=f"completed_line_{idx:03d}",
        a=float(a),
        b=float(b),
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        theta_deviation_from_vertical_deg=float(angle_deviation_from_vertical(a)),
        fragment_ids=[f.id for f in group],
        fragment_count=len(group),
        total_original_length=float(total_original_length),
        completed_length=float(completed_length),
        y_min_original=float(y_min_original),
        y_max_original=float(y_max_original),
        max_gap_filled_px=float(max_gap_inside_group(group)),
        total_shift_px=float(total_shift),
        max_shift_px=float(max_shift),
        total_angle_adjust_deg=float(total_angle_adjust),
        max_angle_adjust_deg=float(max_angle_adjust),
        edge_support=float(edge_support),
        support_score=float(support_score),
        transforms=[asdict(t) for t in transforms],
    )


def draw_fragments(
    visual: np.ndarray,
    fragments: list[Fragment],
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = visual.copy()

    for f in fragments:
        p1 = (int(round(f.x1)), int(round(f.y1)))
        p2 = (int(round(f.x2)), int(round(f.y2)))
        cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)

        mx = int(round((f.x1 + f.x2) / 2.0))
        my = int(round((f.y1 + f.y2) / 2.0))
        cv2.putText(
            out,
            f.id.replace("fragment_", "f"),
            (mx + 3, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    add_title(out, label)
    return out


def draw_completed(visual: np.ndarray, completed: list[CompletedLine]) -> np.ndarray:
    out = visual.copy()

    for line in completed:
        p1 = (int(round(line.x1)), int(round(line.y1)))
        p2 = (int(round(line.x2)), int(round(line.y2)))

        thickness = 3 if line.fragment_count > 1 else 2
        color = (0, 255, 0) if line.fragment_count > 1 else (0, 220, 255)

        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

        label = f"{line.id.replace('completed_line_', 'L')} s={line.support_score:.2f} n={line.fragment_count}"
        cv2.putText(
            out,
            label,
            (p1[0] + 5, max(20, p1[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    add_title(out, "completed vertical line hypotheses")
    return out


def draw_mask_debug(visual: np.ndarray, mask: np.ndarray, masked_edge: np.ndarray) -> np.ndarray:
    out = visual.copy()

    tint = np.zeros_like(out)
    tint[:, :, 1] = 180
    inside = mask > 0
    out[inside] = cv2.addWeighted(out[inside], 0.75, tint[inside], 0.25, 0)

    edge_col = cv2.cvtColor(masked_edge, cv2.COLOR_GRAY2BGR)
    edge_pixels = masked_edge > 0
    out[edge_pixels] = (255, 255, 255)

    add_title(out, "dilated ROI/boot mask + masked edge")
    return out


def add_title(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_h = int(DISPLAY.get("max_height", 800))
    h, w = image.shape[:2]

    if h <= max_h:
        return image

    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h), interpolation=cv2.INTER_AREA)


def make_grid(images: list[np.ndarray]) -> np.ndarray:
    resized = [resize_for_display(img) for img in images]
    target_h = min(img.shape[0] for img in resized)

    normalized = []
    for img in resized:
        h, w = img.shape[:2]
        normalized.append(cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA))

    sep = np.full((target_h, 8, 3), 255, dtype=np.uint8)

    out = normalized[0]
    for img in normalized[1:]:
        out = np.hstack([out, sep, img])

    return out


def save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_image(image_path: Path, show: bool) -> tuple[list[dict[str, Any]], bool]:
    visual = cv2.imread(str(image_path))
    if visual is None:
        print(f"Cannot read image: {image_path}")
        return [], False

    h, w = visual.shape[:2]

    edge = load_edge(image_path, visual)
    raw_mask, mask_source = load_roi_mask(image_path.name, (h, w))
    work_mask = dilate_mask(raw_mask)

    masked_edge = cv2.bitwise_and(edge, edge, mask=work_mask)

    raw_hough_lines = detect_hough_lines(masked_edge)

    all_fragments: list[Fragment] = []
    for idx, line in enumerate(raw_hough_lines, start=1):
        fragment = make_fragment(idx, line, work_mask)
        if fragment is not None:
            all_fragments.append(fragment)

    mask_supported = filter_mask_supported(all_fragments)
    vertical = filter_vertical(mask_supported)

    groups = merge_groups(vertical)

    completed = [
        build_completed_line(idx, group, work_mask, edge)
        for idx, group in enumerate(groups, start=1)
    ]

    completed.sort(key=lambda item: item.support_score, reverse=True)

    max_total = int(cfg("hypotheses", "max_total_hypotheses", default=100))
    completed = completed[:max_total]

    original_overlay = draw_fragments(visual, all_fragments, "all Hough fragments inside masked edge", (255, 160, 0))
    masked_overlay = draw_fragments(visual, mask_supported, "mask-supported fragments", (0, 220, 255))
    vertical_overlay = draw_fragments(visual, vertical, "vertical fragments only", (255, 0, 255))
    completed_overlay = draw_completed(visual, completed)
    mask_debug = draw_mask_debug(visual, work_mask, masked_edge)

    cv2.imwrite(str(ORIGINAL_DIR / image_path.name), original_overlay)
    cv2.imwrite(str(MASKED_DIR / image_path.name), masked_overlay)
    cv2.imwrite(str(VERTICAL_DIR / image_path.name), vertical_overlay)
    cv2.imwrite(str(COMPLETED_DIR / image_path.name), completed_overlay)
    cv2.imwrite(str(DEBUG_DIR / image_path.name), mask_debug)

    json_path = JSON_DIR / f"{image_path.stem}.json"
    save_json(
        json_path,
        {
            "source_file": image_path.name,
            "processing_step": "07_complete_line_fragments",
            "mask_source": mask_source,
            "counts": {
                "raw_hough_lines": len(raw_hough_lines),
                "all_fragments_after_min_length": len(all_fragments),
                "mask_supported_fragments": len(mask_supported),
                "vertical_fragments": len(vertical),
                "completed_lines": len(completed),
            },
            "all_fragments": [asdict(f) for f in all_fragments],
            "mask_supported_fragments": [f.id for f in mask_supported],
            "vertical_fragments": [f.id for f in vertical],
            "completed_lines": [asdict(c) for c in completed],
        },
    )

    rows: list[dict[str, Any]] = []
    if completed:
        for line in completed:
            rows.append(
                {
                    "source_file": image_path.name,
                    "mask_source": mask_source,
                    "raw_hough_lines": len(raw_hough_lines),
                    "all_fragments_after_min_length": len(all_fragments),
                    "mask_supported_fragments": len(mask_supported),
                    "vertical_fragments": len(vertical),
                    "completed_lines": len(completed),
                    "line_id": line.id,
                    "fragment_count": line.fragment_count,
                    "fragment_ids": "|".join(line.fragment_ids),
                    "x1": round(line.x1, 3),
                    "y1": round(line.y1, 3),
                    "x2": round(line.x2, 3),
                    "y2": round(line.y2, 3),
                    "theta_deviation_from_vertical_deg": round(line.theta_deviation_from_vertical_deg, 3),
                    "total_original_length": round(line.total_original_length, 3),
                    "completed_length": round(line.completed_length, 3),
                    "max_gap_filled_px": round(line.max_gap_filled_px, 3),
                    "total_shift_px": round(line.total_shift_px, 3),
                    "max_shift_px": round(line.max_shift_px, 3),
                    "total_angle_adjust_deg": round(line.total_angle_adjust_deg, 3),
                    "max_angle_adjust_deg": round(line.max_angle_adjust_deg, 3),
                    "edge_support": round(line.edge_support, 4),
                    "support_score": round(line.support_score, 4),
                }
            )
    else:
        rows.append(
            {
                "source_file": image_path.name,
                "mask_source": mask_source,
                "raw_hough_lines": len(raw_hough_lines),
                "all_fragments_after_min_length": len(all_fragments),
                "mask_supported_fragments": len(mask_supported),
                "vertical_fragments": len(vertical),
                "completed_lines": 0,
                "line_id": "",
                "fragment_count": "",
                "fragment_ids": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "theta_deviation_from_vertical_deg": "",
                "total_original_length": "",
                "completed_length": "",
                "max_gap_filled_px": "",
                "total_shift_px": "",
                "max_shift_px": "",
                "total_angle_adjust_deg": "",
                "max_angle_adjust_deg": "",
                "edge_support": "",
                "support_score": "",
            }
        )

    print(
        f"{image_path.name}: "
        f"hough={len(raw_hough_lines)} "
        f"fragments={len(all_fragments)} "
        f"mask={len(mask_supported)} "
        f"vertical={len(vertical)} "
        f"completed={len(completed)}"
    )

    if show:
        grid = make_grid([
            mask_debug,
            original_overlay,
            vertical_overlay,
            completed_overlay,
        ])

        title = f"07 complete line fragments | {image_path.name}"
        cv2.imshow(title, grid)

        if bool(DISPLAY.get("wait_between_images", True)):
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(500) & 0xFF

        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass

        if key in [ord("q"), 27]:
            return rows, True

    return rows, False


def save_metadata(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_file",
        "mask_source",
        "raw_hough_lines",
        "all_fragments_after_min_length",
        "mask_supported_fragments",
        "vertical_fragments",
        "completed_lines",
        "line_id",
        "fragment_count",
        "fragment_ids",
        "x1",
        "y1",
        "x2",
        "y2",
        "theta_deviation_from_vertical_deg",
        "total_original_length",
        "completed_length",
        "max_gap_filled_px",
        "total_shift_px",
        "max_shift_px",
        "total_angle_adjust_deg",
        "max_angle_adjust_deg",
        "edge_support",
        "support_score",
    ]

    rows = sorted(rows, key=lambda r: (str(r["source_file"]), str(r["line_id"])))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete vertical boot line fragments inside Step 5 ROI/mask.")
    parser.add_argument("--no-show", action="store_true", help="Run without OpenCV windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not bool(STEP.get("enabled", True)):
        print("Step 07 is disabled in config.")
        return

    ensure_dirs()

    images = collect_images()
    if not images:
        print(f"No input images found in: {VISUAL_INPUT_DIR}")
        return

    show = bool(DISPLAY.get("show_windows", True)) and not args.no_show

    print()
    print("Processing step 07: complete vertical line fragments")
    print(f"Visual input: {VISUAL_INPUT_DIR}")
    print(f"Edge input:   {EDGE_INPUT_DIR}")
    print(f"ROI input:    {ROI_INPUT_DIR}")
    print(f"Output:       {OUTPUT_DIR}")
    print(f"Show windows: {show}")
    print()

    all_rows: list[dict[str, Any]] = []

    for image_path in images:
        rows, stop = process_image(image_path, show)
        all_rows.extend(rows)

        if stop:
            print("Stopped by user.")
            break

    save_metadata(all_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Step 07 outputs saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()