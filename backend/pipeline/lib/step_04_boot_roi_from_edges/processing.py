from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from . import context, runtime
from .display import build_center_prior, make_comparison_view, make_overlay
from .io import load_edge_image, relative_project_path, save_metadata, write_image

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    if normalized.dtype == np.uint8: return normalized
    return normalized.astype(np.uint8)

def _connected_components_fast(binary_image: np.ndarray, *, preserve_default_order: bool = False):
    if preserve_default_order or not hasattr(cv2, "connectedComponentsWithStatsWithAlgorithm"):
        return cv2.connectedComponentsWithStats(binary_image, 8)
    return cv2.connectedComponentsWithStatsWithAlgorithm(binary_image, 8, cv2.CV_32S, cv2.CCL_GRANA)

def remove_small_components(binary_image: np.ndarray, min_area: int) -> np.ndarray:
    x, y, width, height = cv2.boundingRect(binary_image)
    if width == 0 or height == 0: return np.zeros_like(binary_image)
    binary_crop = binary_image[y:y + height, x:x + width]
    component_count, labels, stats, _ = _connected_components_fast(binary_crop)
    if component_count <= 1: return np.zeros_like(binary_image)

    keep_values = np.zeros(component_count, dtype=np.uint8)
    keep_values[stats[:, cv2.CC_STAT_AREA].astype(np.int64, copy=False) >= int(min_area)] = 255
    keep_values[0] = 0
    cleaned = np.zeros_like(binary_image)
    np.take(keep_values, labels, out=cleaned[y:y + height, x:x + width])
    return cleaned

def compute_density_threshold(density_image: np.ndarray) -> int:
    threshold_percentile = float(runtime.DENSITY_CONFIG["threshold_percentile"])
    min_threshold = int(runtime.DENSITY_CONFIG["min_threshold"])
    histogram = cv2.calcHist([density_image], [0], None, [256], [0, 256]).reshape(-1)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    rank = (density_image.size - 1) * (threshold_percentile / 100.0)
    lower_rank = int(math.floor(rank))
    upper_rank = int(math.ceil(rank))
    lower_value = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper_value = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    percentile_value = lower_value + (rank - lower_rank) * (upper_value - lower_value)
    threshold_value = max(min_threshold, int(round(float(percentile_value))))
    return min(threshold_value, 255)

def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    floodfilled = binary_mask.copy()
    cv2.floodFill(floodfilled, None, (0, 0), 255)
    cv2.bitwise_not(floodfilled, dst=floodfilled)
    cv2.bitwise_or(binary_mask, floodfilled, dst=floodfilled)
    return floodfilled

def component_center_score(centroid_x: float, centroid_y: float, width: int, height: int) -> float:
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    dx = (centroid_x - center_x) / max(width / 2.0, 1.0)
    dy = (centroid_y - center_y) / max(height / 2.0, 1.0)
    distance = float(np.sqrt(dx * dx + dy * dy))
    return max(0.05, 1.0 - min(distance, 1.0))

def select_components_union(binary_mask: np.ndarray, *, build_debug: bool) -> tuple[np.ndarray, np.ndarray | None, bool]:
    min_area_ratio = float(runtime.SELECTION_CONFIG["min_area_ratio"])
    center_weight = float(runtime.SELECTION_CONFIG["center_weight"])
    vertical_extent_weight = float(runtime.SELECTION_CONFIG["vertical_extent_weight"])
    height, width = binary_mask.shape[:2]
    image_area = height * width
    scaled_min_area = int(round(image_area * min_area_ratio * 0.05))
    min_area = max(runtime.MIN_COMPONENT_AREA, scaled_min_area)
    x, y, crop_width, crop_height = cv2.boundingRect(binary_mask)
    component_debug = np.zeros((height, width, 3), dtype=np.uint8) if build_debug else None
    if crop_width == 0 or crop_height == 0: return np.zeros_like(binary_mask), component_debug, False

    binary_crop = binary_mask[y:y + crop_height, x:x + crop_width]
    component_count, labels, stats, centroids = _connected_components_fast(binary_crop, preserve_default_order=build_debug)
    if component_count <= 1: return np.zeros_like(binary_mask), component_debug, False

    component_records: list[tuple[int, float, float, float]] = []
    color_lookup = np.zeros((component_count, 3), dtype=np.uint8) if build_debug else None

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_area: continue
        centroid_x = float(centroids[component_index][0]) + x
        centroid_y = float(centroids[component_index][1]) + y
        center_score = component_center_score(centroid_x, centroid_y, width, height)
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        vertical_extent_score = max(0.05, float(component_height) / float(height))
        score = float(area) * (center_score ** center_weight) * (vertical_extent_score ** vertical_extent_weight)
        component_records.append((component_index, score, center_score, vertical_extent_score))
        if color_lookup is not None: color_lookup[component_index] = (int((37 * component_index) % 255), int((97 * component_index) % 255), int((173 * component_index) % 255))

    if not component_records: return np.zeros_like(binary_mask), component_debug, False

    best_score = max(record[1] for record in component_records)
    selected_lookup = np.zeros(component_count, dtype=np.uint8)
    has_selected = False
    for component_index, score, center_score, vertical_extent_score in component_records:
        if score >= best_score * 0.12 or center_score >= 0.55 or (center_score >= 0.3 and vertical_extent_score >= 0.08):
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
            cv2.circle(component_debug, (int(round(centroid_x)), int(round(centroid_y))), 6, (255, 255, 255), thickness=-1)
        component_debug[selected_union > 0] = (0, 255, 0)

    return selected_union, component_debug, has_selected

def smooth_with_hull(binary_mask: np.ndarray) -> np.ndarray:
    if not runtime.HULL_ENABLED: return binary_mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return binary_mask

    largest_contour = max(contours, key=cv2.contourArea)
    if runtime.HULL_MODE == "approx":
        epsilon = runtime.HULL_EPSILON_RATIO * cv2.arcLength(largest_contour, True)
        hull_source = cv2.approxPolyDP(largest_contour, epsilon, True)
    else:
        hull_source = largest_contour

    hull = cv2.convexHull(hull_source)
    hull_mask = np.zeros_like(binary_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
    if runtime.HULL_MODE in {"convex", "approx"}: return hull_mask
    blended = cv2.bitwise_or(binary_mask, hull_mask)
    cv2.morphologyEx(blended, cv2.MORPH_CLOSE, runtime.SMOOTH_KERNEL, dst=blended)
    return blended

def _apply_morphology_to_working_mask(working_mask: np.ndarray) -> np.ndarray:
    cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, runtime.CLOSE_KERNEL, dst=working_mask, iterations=runtime.ITERATIONS_CLOSE)
    cv2.dilate(working_mask, runtime.DILATE_KERNEL, dst=working_mask, iterations=runtime.ITERATIONS_DILATE)
    cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, runtime.SECOND_CLOSE_KERNEL, dst=working_mask, iterations=runtime.ITERATIONS_SECOND_CLOSE)
    working_mask = fill_holes(working_mask)
    cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, runtime.SMOOTH_KERNEL, dst=working_mask)
    cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, runtime.SMOOTH_KERNEL, dst=working_mask)
    working_mask = smooth_with_hull(working_mask)
    return fill_holes(working_mask)

def build_final_mask(selected_component: np.ndarray) -> np.ndarray:
    x, y, width, height = cv2.boundingRect(selected_component)
    if width == 0 or height == 0: return np.zeros_like(selected_component)

    image_height, image_width = selected_component.shape[:2]
    x0 = max(0, x - runtime.MORPHOLOGY_MARGIN)
    y0 = max(0, y - runtime.MORPHOLOGY_MARGIN)
    x1 = min(image_width, x + width + runtime.MORPHOLOGY_MARGIN)
    y1 = min(image_height, y + height + runtime.MORPHOLOGY_MARGIN)
    use_full_frame = x0 == 0 and y0 == 0
    if use_full_frame: return _apply_morphology_to_working_mask(selected_component.copy())

    working_mask = selected_component[y0:y1, x0:x1].copy()
    working_mask = _apply_morphology_to_working_mask(working_mask)
    final_mask = np.zeros_like(selected_component)
    final_mask[y0:y1, x0:x1] = working_mask
    return final_mask

def make_density_blob_mask(density_image: np.ndarray, threshold_value: int) -> np.ndarray:
    return cv2.compare(density_image, int(threshold_value), cv2.CMP_GE)

def process_edge_image(edge_image: np.ndarray, *, debug: bool = False) -> dict[str, Any]:
    binary_edges = cv2.compare(edge_image, runtime.EDGE_THRESHOLD, cv2.CMP_GE)
    cleaned_edges = remove_small_components(binary_edges, runtime.MIN_COMPONENT_AREA)
    del binary_edges
    density_seed = cv2.dilate(cleaned_edges, runtime.DENSITY_SEED_KERNEL, iterations=1)
    del cleaned_edges
    density_raw = cv2.GaussianBlur(density_seed, (runtime.DENSITY_KERNEL_SIZE, runtime.DENSITY_KERNEL_SIZE), 0)
    del density_seed
    density_normalized = normalize_to_uint8(density_raw)
    if density_normalized is not density_raw: del density_raw
    density_threshold = compute_density_threshold(density_normalized)
    activity_mask = make_density_blob_mask(density_normalized, density_threshold)
    selected_component, component_debug, has_selected = select_components_union(activity_mask, build_debug=debug)
    if not has_selected: selected_component = activity_mask.copy()
    del activity_mask
    if not debug: del density_normalized
    final_mask = build_final_mask(selected_component)
    del selected_component

    results: dict[str, Any] = {"final_mask": final_mask, "density_threshold": density_threshold}
    if debug:
        center_prior = build_center_prior(*edge_image.shape[:2])
        weighted_density_float = density_normalized.astype(np.float32)
        np.multiply(weighted_density_float, center_prior, out=weighted_density_float)
        weighted_density = normalize_to_uint8(weighted_density_float)
        del weighted_density_float
        results.update({"density_normalized": density_normalized, "weighted_density": weighted_density, "component_debug": component_debug})

    return results

def ensure_output_dirs(*, debug: bool) -> None:
    context.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context.MASK_DIR.mkdir(parents=True, exist_ok=True)
    context.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if debug:
        context.OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        context.SELECTED_COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
        context.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

def save_outputs(image_path, edge_image: np.ndarray, results: dict[str, Any], *, debug: bool) -> dict[str, str | None]:
    final_mask = results["final_mask"]
    mask_path = context.MASK_DIR / image_path.name
    write_image(mask_path, final_mask)
    output_paths = {
        "mask_output_file": relative_project_path(mask_path),
        "overlay_output_file": None,
        "selected_component_output_file": None,
        "comparison_output_file": None,
    }
    if not debug: return output_paths

    overlay = make_overlay(edge_image, final_mask)
    comparison = make_comparison_view(edge_image, results["density_normalized"], results["weighted_density"], overlay)
    overlay_path = context.OVERLAY_DIR / image_path.name
    component_path = context.SELECTED_COMPONENT_DIR / image_path.name
    comparison_path = context.COMPARISON_DIR / image_path.name
    write_image(overlay_path, overlay)
    write_image(component_path, results["component_debug"])
    write_image(comparison_path, comparison)
    output_paths["overlay_output_file"] = relative_project_path(overlay_path)
    output_paths["selected_component_output_file"] = relative_project_path(component_path)
    output_paths["comparison_output_file"] = relative_project_path(comparison_path)
    output_paths["comparison_image"] = comparison
    return output_paths

def process_images(image_paths: list, *, debug: bool = False) -> None:
    ensure_output_dirs(debug=debug)
    print()
    print("Processing step 04: boot ROI from edges")
    print(f"Input:  {context.INPUT_DIR}")
    print(f"Output: {context.OUTPUT_DIR}")
    print(f"Debug: {'enabled' if debug else 'disabled'}")
    print(f"OpenCV threads: {context.OPENCV_THREADS}")
    print()

    metadata_rows = []
    total_images = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        total_started = perf_counter()
        try:
            read_started = perf_counter()
            edge_image = load_edge_image(image_path)
            read_time_ms = (perf_counter() - read_started) * 1000.0
        except ValueError:
            print(f"Could not read edge image: {image_path}")
            continue

        processing_started = perf_counter()
        results = process_edge_image(edge_image, debug=debug)
        processing_time_ms = (perf_counter() - processing_started) * 1000.0

        write_started = perf_counter()
        output_paths = save_outputs(image_path, edge_image, results, debug=debug)
        write_time_ms = (perf_counter() - write_started) * 1000.0
        total_time_ms = (perf_counter() - total_started) * 1000.0

        final_mask = results["final_mask"]
        mask_area = int(cv2.countNonZero(final_mask))
        density_threshold = int(results["density_threshold"])
        height, width = edge_image.shape[:2]

        metadata_rows.append(
            {
                "source_file": relative_project_path(image_path),
                "mask_output_file": output_paths["mask_output_file"],
                "overlay_output_file": output_paths["overlay_output_file"] or "",
                "selected_component_output_file": output_paths["selected_component_output_file"] or "",
                "comparison_output_file": output_paths["comparison_output_file"] or "",
                "width": width,
                "height": height,
                "processing_step": "04_boot_roi_from_edges",
                "input_from_step_03": str(context.SELECTED_INPUT).strip() if context.SELECTED_INPUT is not None else "",
                "edge_threshold": runtime.EDGE_THRESHOLD,
                "min_component_area": runtime.MIN_COMPONENT_AREA,
                "density_kernel_size": runtime.DENSITY_KERNEL_SIZE,
                "density_threshold_percentile": float(runtime.DENSITY_CONFIG["threshold_percentile"]),
                "density_min_threshold": int(runtime.DENSITY_CONFIG["min_threshold"]),
                "close_kernel_size": int(runtime.MORPHOLOGY_CONFIG["close_kernel_size"]),
                "dilate_kernel_size": int(runtime.MORPHOLOGY_CONFIG["dilate_kernel_size"]),
                "second_close_kernel_size": int(runtime.MORPHOLOGY_CONFIG["second_close_kernel_size"]),
                "smooth_kernel_size": int(runtime.MORPHOLOGY_CONFIG["smooth_kernel_size"]),
                "iterations_close": runtime.ITERATIONS_CLOSE,
                "iterations_dilate": runtime.ITERATIONS_DILATE,
                "iterations_second_close": runtime.ITERATIONS_SECOND_CLOSE,
                "hull_enabled": runtime.HULL_ENABLED,
                "hull_mode": runtime.HULL_MODE,
                "debug_enabled": debug,
                "density_threshold": density_threshold,
                "roi_pixels": mask_area,
                "read_time_ms": round(read_time_ms, 3),
                "processing_time_ms": round(processing_time_ms, 3),
                "write_time_ms": round(write_time_ms, 3),
                "total_time_ms": round(total_time_ms, 3),
            }
        )

        print(
            f"[{index}/{total_images}] Saved: {image_path.name} | "
            f"threshold={density_threshold} | "
            f"roi_pixels={mask_area} | "
            f"read={read_time_ms:.1f} ms, "
            f"process={processing_time_ms:.1f} ms, "
            f"write={write_time_ms:.1f} ms, "
            f"total={total_time_ms:.1f} ms"
        )

        comparison = output_paths.get("comparison_image")
        if debug and comparison is not None:
            title = f"04 Boot ROI from edges | {index}/{total_images} | {image_path.name}"
            cv2.imshow(title, comparison)
            key = cv2.waitKey(0 if context.DISPLAY_CONFIG["wait_between_images"] else 500) & 0xFF
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass
            if key in [ord("q"), 27]:
                print("Stopped by user.")
                break

        del results, final_mask, edge_image

    save_metadata(metadata_rows)
    if debug: cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Mask saved to: {context.MASK_DIR}")
    if debug:
        print(f"Overlay saved to: {context.OVERLAY_DIR}")
        print(f"Selected component debug saved to: {context.SELECTED_COMPONENT_DIR}")
        print(f"Comparison debug saved to: {context.COMPARISON_DIR}")
    print(f"Metadata saved to: {context.CSV_PATH}")
