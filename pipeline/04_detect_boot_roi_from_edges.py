from __future__ import annotations

import argparse
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]
STEP_CONFIG = CONFIG["step_04_boot_roi_from_edges"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]

if bool(STEP_CONFIG.get("inherit_step_03_output", True)):
    INPUT_ROOT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]
else:
    INPUT_ROOT_DIR = PROCESSED_DIR / STEP_CONFIG["input_subdir"]

SELECTED_INPUT = STEP_CONFIG.get("selected_input")
INPUT_DIR = (
    INPUT_ROOT_DIR / str(SELECTED_INPUT).strip()
    if SELECTED_INPUT is not None
    else INPUT_ROOT_DIR
)
OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

MASK_DIR = OUTPUT_DIR / "mask"
OVERLAY_DIR = OUTPUT_DIR / "overlay"
SELECTED_COMPONENT_DIR = OUTPUT_DIR / "selected_component"
COMPARISON_DIR = OUTPUT_DIR / "comparison"

_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Too many OpenCV worker threads make the large-kernel morphology slower due
# to scheduling and synchronization overhead. Four workers was consistently
# faster for 3024x4032 inputs while preserving byte-identical results.
_OPENCV_THREADS = max(1, min(4, os.cpu_count() or 1))
cv2.setUseOptimized(True)
cv2.setNumThreads(_OPENCV_THREADS)


# ---------------------------------------------------------------------------
# CLI and file discovery
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rough ski boot ROI mask from edge images."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Process only one image, for example: IMG_0502 or IMG_0502.png",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save overlay/component/comparison debug images and open debug "
            "windows. Without this flag only the ROI mask is produced."
        ),
    )

    return parser.parse_args()


def collect_images(selected_image: str | None = None) -> list[Path]:
    if not INPUT_DIR.exists():
        return []

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in _ALLOWED_EXTENSIONS
    )

    if selected_image is None:
        return image_paths

    selected_image = selected_image.strip()
    candidates = {selected_image}

    if "." not in selected_image:
        candidates.update(
            f"{selected_image}{extension}" for extension in _ALLOWED_EXTENSIONS
        )

    return [path for path in image_paths if path.name in candidates]


# ---------------------------------------------------------------------------
# Static processing resources
# ---------------------------------------------------------------------------

def ensure_odd_kernel_size(value: int, name: str) -> int:
    value = int(value)

    if value < 1:
        raise ValueError(f"{name} must be positive. Got: {value}")

    if value % 2 == 0:
        raise ValueError(f"{name} must be odd. Got: {value}")

    return value


@lru_cache(maxsize=32)
def make_ellipse_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = ensure_odd_kernel_size(kernel_size, "kernel_size")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    kernel.setflags(write=False)
    return kernel


def scale_to_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


_DENSITY_CONFIG = STEP_CONFIG["density"]
_NOISE_CONFIG = STEP_CONFIG["noise"]
_MORPHOLOGY_CONFIG = STEP_CONFIG["morphology"]
_SELECTION_CONFIG = STEP_CONFIG["component_selection"]
_HULL_CONFIG = STEP_CONFIG["hull"]
_OVERLAY_CONFIG = STEP_CONFIG["overlay"]
_CENTER_PRIOR_CONFIG = STEP_CONFIG["center_prior"]

_EDGE_THRESHOLD = int(STEP_CONFIG["edge_threshold"])
_MIN_COMPONENT_AREA = int(_NOISE_CONFIG["min_component_area"])

_DENSITY_KERNEL_SIZE = ensure_odd_kernel_size(
    int(_DENSITY_CONFIG["kernel_size"]),
    "density.kernel_size",
)
_DENSITY_SEED_KERNEL = make_ellipse_kernel(
    scale_to_odd(_DENSITY_KERNEL_SIZE // 3)
)

_CLOSE_KERNEL = make_ellipse_kernel(int(_MORPHOLOGY_CONFIG["close_kernel_size"]))
_DILATE_KERNEL = make_ellipse_kernel(int(_MORPHOLOGY_CONFIG["dilate_kernel_size"]))
_SECOND_CLOSE_KERNEL = make_ellipse_kernel(
    int(_MORPHOLOGY_CONFIG["second_close_kernel_size"])
)
_SMOOTH_KERNEL = make_ellipse_kernel(int(_MORPHOLOGY_CONFIG["smooth_kernel_size"]))

_ITERATIONS_CLOSE = int(_MORPHOLOGY_CONFIG["iterations_close"])
_ITERATIONS_DILATE = int(_MORPHOLOGY_CONFIG["iterations_dilate"])
_ITERATIONS_SECOND_CLOSE = int(_MORPHOLOGY_CONFIG["iterations_second_close"])

_HULL_ENABLED = bool(_HULL_CONFIG.get("enabled", False))
_HULL_MODE = str(_HULL_CONFIG.get("mode", "convex")).strip().lower()
_HULL_EPSILON_RATIO = float(_HULL_CONFIG.get("approx_epsilon_ratio", 0.008))

if _HULL_MODE not in {"convex", "approx", "soft"}:
    raise ValueError(
        f"Unsupported hull mode: {_HULL_MODE}. Supported: convex, approx, soft"
    )


def _kernel_radius(kernel: np.ndarray) -> int:
    return max(kernel.shape[0] // 2, kernel.shape[1] // 2)


def _compute_morphology_margin() -> int:
    # Conservative maximum outward propagation. The extra two zero pixels keep
    # floodFill's crop seed outside every reachable foreground pixel.
    margin = (
        _kernel_radius(_CLOSE_KERNEL) * _ITERATIONS_CLOSE
        + _kernel_radius(_DILATE_KERNEL) * _ITERATIONS_DILATE
        + _kernel_radius(_SECOND_CLOSE_KERNEL) * _ITERATIONS_SECOND_CLOSE
        + _kernel_radius(_SMOOTH_KERNEL)
    )

    if _HULL_ENABLED and _HULL_MODE == "soft":
        margin += _kernel_radius(_SMOOTH_KERNEL)

    return margin + 2


_MORPHOLOGY_MARGIN = _compute_morphology_margin()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # GaussianBlur keeps uint8 input as uint8. Avoid an otherwise redundant
    # full-image copy in that common path. Float debug data preserves the old
    # truncating astype behaviour exactly.
    if normalized.dtype == np.uint8:
        return normalized

    return normalized.astype(np.uint8)


def _connected_components_fast(binary_image: np.ndarray, *, preserve_default_order: bool = False):
    if preserve_default_order or not hasattr(cv2, "connectedComponentsWithStatsWithAlgorithm"):
        return cv2.connectedComponentsWithStats(binary_image, 8)
    return cv2.connectedComponentsWithStatsWithAlgorithm(
        binary_image, 8, cv2.CV_32S, cv2.CCL_GRANA
    )


def remove_small_components(binary_image: np.ndarray, min_area: int) -> np.ndarray:
    x, y, width, height = cv2.boundingRect(binary_image)
    if width == 0 or height == 0:
        return np.zeros_like(binary_image)

    binary_crop = binary_image[y:y + height, x:x + width]
    component_count, labels, stats, _ = _connected_components_fast(binary_crop)
    if component_count <= 1:
        return np.zeros_like(binary_image)

    keep_values = np.zeros(component_count, dtype=np.uint8)
    keep_values[
        stats[:, cv2.CC_STAT_AREA].astype(np.int64, copy=False) >= int(min_area)
    ] = 255
    keep_values[0] = 0

    cleaned = np.zeros_like(binary_image)
    np.take(keep_values, labels, out=cleaned[y:y + height, x:x + width])
    return cleaned


def compute_density_threshold(density_image: np.ndarray) -> int:
    """Compute NumPy's linear uint8 percentile from a 256-bin histogram.

    This is mathematically identical to np.percentile(..., method="linear")
    for uint8 data, but it avoids partitioning/copying a 12 MP array.
    """
    threshold_percentile = float(_DENSITY_CONFIG["threshold_percentile"])
    min_threshold = int(_DENSITY_CONFIG["min_threshold"])

    histogram = cv2.calcHist(
        [density_image],
        [0],
        None,
        [256],
        [0, 256],
    ).reshape(-1)
    cumulative = np.cumsum(histogram, dtype=np.int64)

    rank = (density_image.size - 1) * (threshold_percentile / 100.0)
    lower_rank = int(math.floor(rank))
    upper_rank = int(math.ceil(rank))

    lower_value = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper_value = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))

    percentile_value = lower_value + (rank - lower_rank) * (
        upper_value - lower_value
    )
    threshold_value = max(min_threshold, int(round(float(percentile_value))))

    return min(threshold_value, 255)


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill holes with one full-size temporary matrix instead of three."""
    floodfilled = binary_mask.copy()

    # Passing None lets OpenCV manage the (+2, +2) flood-fill mask internally,
    # avoiding another Python-visible full-frame allocation.
    cv2.floodFill(floodfilled, None, (0, 0), 255)
    cv2.bitwise_not(floodfilled, dst=floodfilled)
    cv2.bitwise_or(binary_mask, floodfilled, dst=floodfilled)

    return floodfilled


def component_center_score(
    centroid_x: float,
    centroid_y: float,
    width: int,
    height: int,
) -> float:
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    dx = (centroid_x - center_x) / max(width / 2.0, 1.0)
    dy = (centroid_y - center_y) / max(height / 2.0, 1.0)
    distance = float(np.sqrt(dx * dx + dy * dy))

    return max(0.05, 1.0 - min(distance, 1.0))


def select_components_union(
    binary_mask: np.ndarray,
    *,
    build_debug: bool,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    min_area_ratio = float(_SELECTION_CONFIG["min_area_ratio"])
    center_weight = float(_SELECTION_CONFIG["center_weight"])
    vertical_extent_weight = float(_SELECTION_CONFIG["vertical_extent_weight"])

    height, width = binary_mask.shape[:2]
    image_area = height * width
    scaled_min_area = int(round(image_area * min_area_ratio * 0.05))
    min_area = max(_MIN_COMPONENT_AREA, scaled_min_area)

    x, y, crop_width, crop_height = cv2.boundingRect(binary_mask)
    component_debug = (
        np.zeros((height, width, 3), dtype=np.uint8) if build_debug else None
    )
    if crop_width == 0 or crop_height == 0:
        return np.zeros_like(binary_mask), component_debug, False

    binary_crop = binary_mask[y:y + crop_height, x:x + crop_width]
    component_count, labels, stats, centroids = _connected_components_fast(
        binary_crop,
        preserve_default_order=build_debug,
    )
    if component_count <= 1:
        return np.zeros_like(binary_mask), component_debug, False

    component_records: list[tuple[int, float, float, float]] = []
    color_lookup = (
        np.zeros((component_count, 3), dtype=np.uint8) if build_debug else None
    )

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        centroid_x = float(centroids[component_index][0]) + x
        centroid_y = float(centroids[component_index][1]) + y
        center_score = component_center_score(
            centroid_x, centroid_y, width, height
        )
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        vertical_extent_score = max(0.05, float(component_height) / float(height))
        score = (
            float(area)
            * (center_score ** center_weight)
            * (vertical_extent_score ** vertical_extent_weight)
        )
        component_records.append(
            (component_index, score, center_score, vertical_extent_score)
        )

        if color_lookup is not None:
            color_lookup[component_index] = (
                int((37 * component_index) % 255),
                int((97 * component_index) % 255),
                int((173 * component_index) % 255),
            )

    if not component_records:
        return np.zeros_like(binary_mask), component_debug, False

    best_score = max(record[1] for record in component_records)
    selected_lookup = np.zeros(component_count, dtype=np.uint8)
    has_selected = False
    for component_index, score, center_score, vertical_extent_score in component_records:
        if (
            score >= best_score * 0.12
            or center_score >= 0.55
            or (center_score >= 0.3 and vertical_extent_score >= 0.08)
        ):
            selected_lookup[component_index] = 255
            has_selected = True

    selected_union = np.zeros_like(binary_mask)
    selected_crop = selected_union[y:y + crop_height, x:x + crop_width]
    np.take(selected_lookup, labels, out=selected_crop)

    if build_debug:
        assert component_debug is not None and color_lookup is not None
        component_debug[y:y + crop_height, x:x + crop_width] = color_lookup[labels]
        for component_index, _, _, _ in component_records:
            centroid_x = float(centroids[component_index][0]) + x
            centroid_y = float(centroids[component_index][1]) + y
            cv2.circle(
                component_debug,
                (int(round(centroid_x)), int(round(centroid_y))),
                6,
                (255, 255, 255),
                thickness=-1,
            )
        component_debug[selected_union > 0] = (0, 255, 0)

    return selected_union, component_debug, has_selected


def smooth_with_hull(binary_mask: np.ndarray) -> np.ndarray:
    if not _HULL_ENABLED:
        return binary_mask

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return binary_mask

    largest_contour = max(contours, key=cv2.contourArea)

    if _HULL_MODE == "approx":
        epsilon = _HULL_EPSILON_RATIO * cv2.arcLength(largest_contour, True)
        hull_source = cv2.approxPolyDP(largest_contour, epsilon, True)
    else:
        # The original computed approxPolyDP here even though convex/soft modes
        # never used it. Skipping that dead work cannot affect the result.
        hull_source = largest_contour

    hull = cv2.convexHull(hull_source)
    hull_mask = np.zeros_like(binary_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)

    if _HULL_MODE in {"convex", "approx"}:
        return hull_mask

    blended = cv2.bitwise_or(binary_mask, hull_mask)
    cv2.morphologyEx(
        blended,
        cv2.MORPH_CLOSE,
        _SMOOTH_KERNEL,
        dst=blended,
    )
    return blended


def _apply_morphology_to_working_mask(working_mask: np.ndarray) -> np.ndarray:
    # All operations are intentionally in the same order and use the same
    # kernels/iterations as the original. OpenCV supports safe in-place dst.
    cv2.morphologyEx(
        working_mask,
        cv2.MORPH_CLOSE,
        _CLOSE_KERNEL,
        dst=working_mask,
        iterations=_ITERATIONS_CLOSE,
    )
    cv2.dilate(
        working_mask,
        _DILATE_KERNEL,
        dst=working_mask,
        iterations=_ITERATIONS_DILATE,
    )
    cv2.morphologyEx(
        working_mask,
        cv2.MORPH_CLOSE,
        _SECOND_CLOSE_KERNEL,
        dst=working_mask,
        iterations=_ITERATIONS_SECOND_CLOSE,
    )

    working_mask = fill_holes(working_mask)

    cv2.morphologyEx(
        working_mask,
        cv2.MORPH_OPEN,
        _SMOOTH_KERNEL,
        dst=working_mask,
    )
    cv2.morphologyEx(
        working_mask,
        cv2.MORPH_CLOSE,
        _SMOOTH_KERNEL,
        dst=working_mask,
    )

    working_mask = smooth_with_hull(working_mask)
    return fill_holes(working_mask)


def build_final_mask(selected_component: np.ndarray) -> np.ndarray:
    """Run expensive morphology on an exact, safely padded ROI crop.

    The crop contains every pixel that any configured operation can reach.
    If it would begin at the global (0, 0), full-frame processing is retained
    because the legacy fill_holes implementation deliberately seeds floodFill
    there and therefore has special behaviour when foreground reaches it.
    """
    x, y, width, height = cv2.boundingRect(selected_component)

    if width == 0 or height == 0:
        return np.zeros_like(selected_component)

    image_height, image_width = selected_component.shape[:2]

    x0 = max(0, x - _MORPHOLOGY_MARGIN)
    y0 = max(0, y - _MORPHOLOGY_MARGIN)
    x1 = min(image_width, x + width + _MORPHOLOGY_MARGIN)
    y1 = min(image_height, y + height + _MORPHOLOGY_MARGIN)

    # Preserve the global flood-fill seed edge case exactly.
    use_full_frame = x0 == 0 and y0 == 0

    if use_full_frame:
        return _apply_morphology_to_working_mask(selected_component.copy())

    working_mask = selected_component[y0:y1, x0:x1].copy()
    working_mask = _apply_morphology_to_working_mask(working_mask)

    final_mask = np.zeros_like(selected_component)
    final_mask[y0:y1, x0:x1] = working_mask
    return final_mask


def make_density_blob_mask(
    density_image: np.ndarray,
    threshold_value: int,
) -> np.ndarray:
    return cv2.compare(
        density_image,
        int(threshold_value),
        cv2.CMP_GE,
    )


# ---------------------------------------------------------------------------
# Debug-only helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def build_center_prior(height: int, width: int) -> np.ndarray:
    if not bool(_CENTER_PRIOR_CONFIG.get("enabled", True)):
        prior = np.ones((height, width), dtype=np.float32)
        prior.setflags(write=False)
        return prior

    sigma_x_ratio = float(_CENTER_PRIOR_CONFIG["sigma_x_ratio"])
    sigma_y_ratio = float(_CENTER_PRIOR_CONFIG["sigma_y_ratio"])
    power = float(_CENTER_PRIOR_CONFIG.get("power", 1.0))

    sigma_x = max(width * sigma_x_ratio, 1.0)
    sigma_y = max(height * sigma_y_ratio, 1.0)

    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    x_exponent = ((x_coords - center_x) ** 2) / (2.0 * sigma_x * sigma_x)
    y_exponent = ((y_coords - center_y) ** 2) / (2.0 * sigma_y * sigma_y)
    exponent = y_exponent[:, None] + x_exponent[None, :]
    prior = np.exp(-exponent)

    if power != 1.0:
        prior = np.power(prior, power)

    prior = prior.astype(np.float32, copy=False)
    prior.setflags(write=False)
    return prior


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(
        labeled,
        (0, 0),
        (labeled.shape[1], 45),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        labeled,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _prepare_debug_display(image: np.ndarray, label: str) -> np.ndarray:
    # Resize grayscale first, then expand to BGR. Since all BGR channels would
    # be identical, this produces the same pixels with about one third of the
    # resize traffic and temporary memory.
    resized = resize_for_display(image)
    return add_label(to_bgr(resized), label)


def make_overlay(
    edge_image: np.ndarray,
    final_mask: np.ndarray,
) -> np.ndarray:
    alpha = float(_OVERLAY_CONFIG["alpha"])
    contour_thickness = int(_OVERLAY_CONFIG["contour_thickness"])

    edge_bgr = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    green_fill = np.zeros_like(edge_bgr)
    green_fill[:, :, 1] = final_mask

    overlay = cv2.addWeighted(edge_bgr, 1.0, green_fill, alpha, 0.0)

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 0),
            thickness=contour_thickness,
            lineType=cv2.LINE_AA,
        )

    return overlay


def make_comparison_view(
    edge_image: np.ndarray,
    density_image: np.ndarray,
    weighted_density_image: np.ndarray,
    final_overlay: np.ndarray,
) -> np.ndarray:
    displays = [
        _prepare_debug_display(edge_image, "input edges"),
        _prepare_debug_display(density_image, "density"),
        _prepare_debug_display(weighted_density_image, "weighted density"),
        _prepare_debug_display(final_overlay, "final roi"),
    ]

    target_height = min(image.shape[0] for image in displays)
    normalized_displays: list[np.ndarray] = []

    for image in displays:
        height, width = image.shape[:2]

        if height == target_height:
            normalized_displays.append(image)
        else:
            normalized_displays.append(
                cv2.resize(
                    image,
                    (int(width * target_height / height), target_height),
                    interpolation=cv2.INTER_AREA,
                )
            )

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []

    for index, image in enumerate(normalized_displays):
        if index:
            parts.append(separator)
        parts.append(image)

    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def process_edge_image(
    edge_image: np.ndarray,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    binary_edges = cv2.compare(edge_image, _EDGE_THRESHOLD, cv2.CMP_GE)
    cleaned_edges = remove_small_components(binary_edges, _MIN_COMPONENT_AREA)
    del binary_edges

    density_seed = cv2.dilate(
        cleaned_edges,
        _DENSITY_SEED_KERNEL,
        iterations=1,
    )
    del cleaned_edges

    density_raw = cv2.GaussianBlur(
        density_seed,
        (_DENSITY_KERNEL_SIZE, _DENSITY_KERNEL_SIZE),
        0,
    )
    del density_seed

    density_normalized = normalize_to_uint8(density_raw)
    if density_normalized is not density_raw:
        del density_raw

    density_threshold = compute_density_threshold(density_normalized)
    activity_mask = make_density_blob_mask(
        density_normalized,
        density_threshold,
    )

    selected_component, component_debug, has_selected = select_components_union(
        activity_mask,
        build_debug=debug,
    )

    if not has_selected:
        selected_component = activity_mask.copy()

    del activity_mask

    # weighted_density and center_prior never affect selection or final_mask;
    # they exist only for the comparison visualization. Build them only when
    # the user explicitly requests --debug.
    if not debug:
        del density_normalized

    final_mask = build_final_mask(selected_component)
    del selected_component

    results: dict[str, Any] = {
        "final_mask": final_mask,
        "density_threshold": density_threshold,
    }

    if debug:
        center_prior = build_center_prior(*edge_image.shape[:2])
        weighted_density_float = density_normalized.astype(np.float32)
        np.multiply(
            weighted_density_float,
            center_prior,
            out=weighted_density_float,
        )
        weighted_density = normalize_to_uint8(weighted_density_float)
        del weighted_density_float

        results.update(
            {
                "density_normalized": density_normalized,
                "weighted_density": weighted_density,
                "component_debug": component_debug,
            }
        )

    return results


def _write_image(path: Path, image: np.ndarray) -> None:
    # Compression level changes encoded size/time only, never decoded pixels.
    # Level 1 is substantially faster than OpenCV's default level 3.
    params = (
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            1,
            cv2.IMWRITE_PNG_STRATEGY,
            cv2.IMWRITE_PNG_STRATEGY_RLE,
        ]
        if path.suffix.lower() == ".png"
        else []
    )

    if not cv2.imwrite(str(path), image, params):
        raise IOError(f"Could not write image: {path}")


def save_outputs(
    image_path: Path,
    edge_image: np.ndarray,
    results: dict[str, Any],
    *,
    debug: bool,
) -> np.ndarray | None:
    final_mask = results["final_mask"]
    _write_image(MASK_DIR / image_path.name, final_mask)

    if not debug:
        return None

    overlay = make_overlay(edge_image, final_mask)
    comparison = make_comparison_view(
        edge_image,
        results["density_normalized"],
        results["weighted_density"],
        overlay,
    )

    _write_image(OVERLAY_DIR / image_path.name, overlay)
    _write_image(
        SELECTED_COMPONENT_DIR / image_path.name,
        results["component_debug"],
    )
    _write_image(COMPARISON_DIR / image_path.name, comparison)

    return comparison


def ensure_output_dirs(*, debug: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    if debug:
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        SELECTED_COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
        COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not STEP_CONFIG["enabled"]:
        print("Step 04 is disabled in config.")
        return

    args = parse_args()
    debug = bool(args.debug)

    ensure_output_dirs(debug=debug)
    image_paths = collect_images(args.image)

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    print()
    print("Processing step 04: boot ROI from edges")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Selected image filter: {args.image if args.image else 'all'}")
    print(f"Debug: {'enabled' if debug else 'disabled'}")
    print(f"OpenCV threads: {_OPENCV_THREADS}")
    print()

    for index, image_path in enumerate(image_paths, start=1):
        edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if edge_image is None:
            print(f"Could not read edge image: {image_path}")
            continue

        results = process_edge_image(edge_image, debug=debug)
        comparison = save_outputs(
            image_path,
            edge_image,
            results,
            debug=debug,
        )

        final_mask = results["final_mask"]
        mask_area = int(cv2.countNonZero(final_mask))
        density_threshold = int(results["density_threshold"])

        print(
            f"[{index}/{len(image_paths)}] Saved: {image_path.name} | "
            f"threshold={density_threshold} | "
            f"roi_pixels={mask_area}"
        )

        if debug and comparison is not None:
            title = (
                f"04 Boot ROI from edges | "
                f"{index}/{len(image_paths)} | "
                f"{image_path.name}"
            )
            cv2.imshow(title, comparison)

            if DISPLAY_CONFIG["wait_between_images"]:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(500) & 0xFF

            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass

            if key in [ord("q"), 27]:
                print("Stopped by user.")
                break

        # Release all per-image full-resolution arrays before loading the next
        # image. This matters on 3024x4032 inputs.
        del comparison, results, final_mask, edge_image

    if debug:
        cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Mask saved to: {MASK_DIR}")

    if debug:
        print(f"Overlay saved to: {OVERLAY_DIR}")
        print(f"Selected component debug saved to: {SELECTED_COMPONENT_DIR}")
        print(f"Comparison debug saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()