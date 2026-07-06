from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_CONFIG = CONFIG["step_04_boot_roi_from_edges"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]

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
        "--show",
        action="store_true",
        help="Show debug windows while processing.",
    )

    return parser.parse_args()


def collect_images(selected_image: str | None = None) -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    if not INPUT_DIR.exists():
        return []

    image_paths = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    image_paths = sorted(image_paths)

    if selected_image is None:
        return image_paths

    selected_image = selected_image.strip()
    candidates = {selected_image}

    if "." not in selected_image:
        candidates.update(f"{selected_image}{extension}" for extension in allowed_extensions)

    selected_paths = [path for path in image_paths if path.name in candidates]

    return selected_paths


def ensure_odd_kernel_size(value: int, name: str) -> int:
    value = int(value)

    if value < 1:
        raise ValueError(f"{name} must be positive. Got: {value}")

    if value % 2 == 0:
        raise ValueError(f"{name} must be odd. Got: {value}")

    return value


def make_ellipse_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = ensure_odd_kernel_size(kernel_size, "kernel_size")

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )


def scale_to_odd(value: int) -> int:
    value = max(3, int(value))

    if value % 2 == 0:
        value += 1

    return value


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
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


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return normalized.astype(np.uint8)


def remove_small_components(binary_image: np.ndarray, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, 8)
    cleaned = np.zeros_like(binary_image)

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        cleaned[labels == component_index] = 255

    return cleaned


def build_center_prior(height: int, width: int) -> np.ndarray:
    center_prior_config = STEP_CONFIG["center_prior"]

    if not bool(center_prior_config.get("enabled", True)):
        return np.ones((height, width), dtype=np.float32)

    sigma_x_ratio = float(center_prior_config["sigma_x_ratio"])
    sigma_y_ratio = float(center_prior_config["sigma_y_ratio"])
    power = float(center_prior_config.get("power", 1.0))

    sigma_x = max(width * sigma_x_ratio, 1.0)
    sigma_y = max(height * sigma_y_ratio, 1.0)

    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    exponent = (
        ((xx - center_x) ** 2) / (2.0 * sigma_x * sigma_x)
        + ((yy - center_y) ** 2) / (2.0 * sigma_y * sigma_y)
    )
    prior = np.exp(-exponent)

    if power != 1.0:
        prior = np.power(prior, power)

    return prior.astype(np.float32)


def compute_density_threshold(weighted_density: np.ndarray) -> int:
    density_config = STEP_CONFIG["density"]

    threshold_percentile = float(density_config["threshold_percentile"])
    min_threshold = int(density_config["min_threshold"])
    percentile_value = np.percentile(weighted_density, threshold_percentile)
    threshold_value = max(min_threshold, int(round(float(percentile_value))))

    return min(threshold_value, 255)


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    floodfill_mask = np.zeros((binary_mask.shape[0] + 2, binary_mask.shape[1] + 2), dtype=np.uint8)
    floodfilled = binary_mask.copy()

    cv2.floodFill(floodfilled, floodfill_mask, (0, 0), 255)

    inverted = cv2.bitwise_not(floodfilled)

    return cv2.bitwise_or(binary_mask, inverted)


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


def component_vertical_extent_score(component_mask: np.ndarray) -> float:
    ys = np.where(component_mask > 0)[0]

    if ys.size == 0:
        return 0.0

    extent = (float(ys.max()) - float(ys.min()) + 1.0) / float(component_mask.shape[0])

    return max(0.05, extent)


def select_components_union(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selection_config = STEP_CONFIG["component_selection"]
    min_area_ratio = float(selection_config["min_area_ratio"])
    center_weight = float(selection_config["center_weight"])
    vertical_extent_weight = float(selection_config["vertical_extent_weight"])

    height, width = binary_mask.shape[:2]
    image_area = height * width
    scaled_min_area = int(round(image_area * min_area_ratio * 0.05))
    min_area = max(int(STEP_CONFIG["noise"]["min_component_area"]), scaled_min_area)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)

    component_records: list[dict[str, float | int | np.ndarray]] = []
    component_debug = np.zeros((height, width, 3), dtype=np.uint8)

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        component_mask = np.where(labels == component_index, 255, 0).astype(np.uint8)
        centroid_x = float(centroids[component_index][0])
        centroid_y = float(centroids[component_index][1])
        center_score = component_center_score(centroid_x, centroid_y, width, height)
        vertical_extent_score = component_vertical_extent_score(component_mask)

        score = (
            float(area)
            * (center_score ** center_weight)
            * (vertical_extent_score ** vertical_extent_weight)
        )

        color = (
            int((37 * component_index) % 255),
            int((97 * component_index) % 255),
            int((173 * component_index) % 255),
        )
        component_debug[component_mask > 0] = color

        cv2.circle(
            component_debug,
            (int(round(centroid_x)), int(round(centroid_y))),
            6,
            (255, 255, 255),
            thickness=-1,
        )

        component_records.append({
            "mask": component_mask,
            "score": score,
            "area": area,
            "center_score": center_score,
            "vertical_extent_score": vertical_extent_score,
        })

    if not component_records:
        return np.zeros_like(binary_mask), component_debug

    best_score = max(float(record["score"]) for record in component_records)
    selected_union = np.zeros_like(binary_mask)

    for record in component_records:
        score = float(record["score"])
        center_score = float(record["center_score"])
        vertical_extent_score = float(record["vertical_extent_score"])

        should_keep = (
            score >= best_score * 0.12
            or center_score >= 0.55
            or (center_score >= 0.3 and vertical_extent_score >= 0.08)
        )

        if not should_keep:
            continue

        component_mask = record["mask"]
        selected_union = cv2.bitwise_or(selected_union, component_mask)
        component_debug[component_mask > 0] = (0, 255, 0)

    return selected_union, component_debug


def smooth_with_hull(binary_mask: np.ndarray) -> np.ndarray:
    hull_config = STEP_CONFIG["hull"]

    if not bool(hull_config.get("enabled", False)):
        return binary_mask

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return binary_mask

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon_ratio = float(hull_config.get("approx_epsilon_ratio", 0.008))
    epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
    approximated = cv2.approxPolyDP(largest_contour, epsilon, True)

    mode = str(hull_config.get("mode", "convex")).strip().lower()
    hull_source = approximated if mode == "approx" else largest_contour
    hull = cv2.convexHull(hull_source)

    hull_mask = np.zeros_like(binary_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)

    if mode == "convex" or mode == "approx":
        return hull_mask

    if mode == "soft":
        smooth_kernel_size = int(STEP_CONFIG["morphology"]["smooth_kernel_size"])
        smooth_kernel = make_ellipse_kernel(smooth_kernel_size)
        blended = cv2.bitwise_or(binary_mask, hull_mask)
        blended = cv2.morphologyEx(blended, cv2.MORPH_CLOSE, smooth_kernel)

        return blended

    raise ValueError(
        f"Unsupported hull mode: {mode}. Supported: convex, approx, soft"
    )


def make_overlay(
    edge_image: np.ndarray,
    final_mask: np.ndarray,
) -> np.ndarray:
    overlay_config = STEP_CONFIG["overlay"]
    alpha = float(overlay_config["alpha"])
    contour_thickness = int(overlay_config["contour_thickness"])

    edge_bgr = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    green_fill = np.zeros_like(edge_bgr)
    green_fill[final_mask > 0] = (0, 255, 0)

    overlay = cv2.addWeighted(edge_bgr, 1.0, green_fill, alpha, 0.0)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        add_label(resize_for_display(to_bgr(edge_image)), "input edges"),
        add_label(resize_for_display(density_image), "density"),
        add_label(resize_for_display(weighted_density_image), "weighted density"),
        add_label(resize_for_display(final_overlay), "final roi"),
    ]

    target_height = min(image.shape[0] for image in displays)
    resized = []

    for image in displays:
        height, width = image.shape[:2]
        resized.append(
            cv2.resize(
                image,
                (int(width * target_height / height), target_height),
                interpolation=cv2.INTER_AREA,
            )
        )

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    combined = resized[0]

    for image in resized[1:]:
        combined = np.hstack([combined, separator, image])

    return combined


def make_density_blob_mask(density_image: np.ndarray, threshold_value: int) -> np.ndarray:
    activity_mask = np.where(density_image >= threshold_value, 255, 0).astype(np.uint8)

    return activity_mask


def process_edge_image(edge_image: np.ndarray) -> dict[str, np.ndarray | int]:
    edge_threshold = int(STEP_CONFIG["edge_threshold"])
    min_component_area = int(STEP_CONFIG["noise"]["min_component_area"])

    density_kernel_size = ensure_odd_kernel_size(
        int(STEP_CONFIG["density"]["kernel_size"]),
        "density.kernel_size",
    )
    density_seed_kernel_size = scale_to_odd(density_kernel_size // 3)
    density_seed_kernel = make_ellipse_kernel(density_seed_kernel_size)

    morphology_config = STEP_CONFIG["morphology"]
    close_kernel = make_ellipse_kernel(int(morphology_config["close_kernel_size"]))
    dilate_kernel = make_ellipse_kernel(int(morphology_config["dilate_kernel_size"]))
    second_close_kernel = make_ellipse_kernel(int(morphology_config["second_close_kernel_size"]))
    smooth_kernel = make_ellipse_kernel(int(morphology_config["smooth_kernel_size"]))

    binary_edges = np.where(edge_image >= edge_threshold, 255, 0).astype(np.uint8)
    cleaned_edges = remove_small_components(binary_edges, min_component_area)

    density_seed = cv2.dilate(cleaned_edges, density_seed_kernel, iterations=1)
    density_raw = cv2.GaussianBlur(
        density_seed,
        (density_kernel_size, density_kernel_size),
        0,
    )
    density_normalized = normalize_to_uint8(density_raw)

    center_prior = build_center_prior(*edge_image.shape[:2])
    weighted_density_float = density_normalized.astype(np.float32) * center_prior
    weighted_density = normalize_to_uint8(weighted_density_float)

    density_threshold = compute_density_threshold(density_normalized)
    activity_mask = make_density_blob_mask(density_normalized, density_threshold)

    selected_component, component_debug = select_components_union(activity_mask)

    if not np.any(selected_component):
        selected_component = activity_mask.copy()

    morphed_mask = cv2.morphologyEx(
        selected_component,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=int(morphology_config["iterations_close"]),
    )
    morphed_mask = cv2.dilate(
        morphed_mask,
        dilate_kernel,
        iterations=int(morphology_config["iterations_dilate"]),
    )
    morphed_mask = cv2.morphologyEx(
        morphed_mask,
        cv2.MORPH_CLOSE,
        second_close_kernel,
        iterations=int(morphology_config["iterations_second_close"]),
    )

    filled_mask = fill_holes(morphed_mask)
    smoothed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, smooth_kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, smooth_kernel)
    final_mask = smooth_with_hull(smoothed_mask)
    final_mask = fill_holes(final_mask)

    return {
        "binary_edges": binary_edges,
        "cleaned_edges": cleaned_edges,
        "density_raw": density_normalized,
        "density_normalized": density_normalized,
        "weighted_density": weighted_density,
        "activity_mask": activity_mask,
        "selected_component": selected_component,
        "component_debug": component_debug,
        "final_mask": final_mask,
        "density_threshold": density_threshold,
    }


def save_outputs(
    image_path: Path,
    edge_image: np.ndarray,
    results: dict[str, np.ndarray | int],
) -> None:
    final_mask = results["final_mask"]
    overlay = make_overlay(edge_image, final_mask)
    comparison = make_comparison_view(
        edge_image,
        to_bgr(results["density_raw"]),
        to_bgr(results["weighted_density"]),
        overlay,
    )

    cv2.imwrite(str(MASK_DIR / image_path.name), final_mask)
    cv2.imwrite(str(OVERLAY_DIR / image_path.name), overlay)
    cv2.imwrite(str(SELECTED_COMPONENT_DIR / image_path.name), results["component_debug"])
    cv2.imwrite(str(COMPARISON_DIR / image_path.name), comparison)


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_DIR,
        MASK_DIR,
        OVERLAY_DIR,
        SELECTED_COMPONENT_DIR,
        COMPARISON_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not STEP_CONFIG["enabled"]:
        print("Step 04 is disabled in config.")
        return

    args = parse_args()
    show_windows = bool(args.show)

    ensure_output_dirs()

    image_paths = collect_images(args.image)

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    print()
    print("Processing step 04: boot ROI from edges")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Selected image filter: {args.image if args.image else 'all'}")
    print()

    for index, image_path in enumerate(image_paths, start=1):
        edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if edge_image is None:
            print(f"Could not read edge image: {image_path}")
            continue

        results = process_edge_image(edge_image)
        save_outputs(image_path, edge_image, results)

        final_mask = results["final_mask"]
        mask_area = int(np.count_nonzero(final_mask))
        density_threshold = int(results["density_threshold"])

        print(
            f"[{index}/{len(image_paths)}] Saved: {image_path.name} | "
            f"threshold={density_threshold} | "
            f"roi_pixels={mask_area}"
        )

        if show_windows:
            comparison = cv2.imread(str(COMPARISON_DIR / image_path.name))

            if comparison is not None:
                title = f"04 Boot ROI from edges | {index}/{len(image_paths)} | {image_path.name}"
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

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Mask saved to: {MASK_DIR}")
    print(f"Overlay saved to: {OVERLAY_DIR}")
    print(f"Selected component debug saved to: {SELECTED_COMPONENT_DIR}")
    print(f"Comparison debug saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
