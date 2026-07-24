from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from . import context
from .display import draw_lines, make_comparison_view
from .io import collect_images, load_grayscale_image, relative_project_path, save_metadata, save_valid_lines_json


def ensure_odd_kernel_size(value: int, name: str) -> int:
    value = int(value)
    if value < 1: raise ValueError(f"{name} must be positive. Got: {value}")
    if value % 2 == 0: raise ValueError(f"{name} must be odd. Got: {value}")
    return value


def make_ellipse_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = ensure_odd_kernel_size(kernel_size, "kernel_size")
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def make_hough_mask(roi_mask: np.ndarray) -> np.ndarray:
    roi_config = context.STEP_CONFIG["roi"]
    binary_mask = ensure_binary_mask(roi_mask)
    if not bool(roi_config.get("use_inner_mask", True)): return binary_mask

    kernel_size = ensure_odd_kernel_size(int(roi_config["inner_erode_kernel_size"]), "roi.inner_erode_kernel_size")
    iterations = int(roi_config.get("inner_erode_iterations", 1))
    kernel = make_ellipse_kernel(kernel_size)
    eroded = cv2.erode(binary_mask, kernel, iterations=iterations)
    if np.count_nonzero(eroded) == 0: return binary_mask
    return eroded


def build_row_mask_bounds(roi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    height, width = roi_mask.shape[:2]
    left_bounds = np.full(height, width, dtype=np.int32)
    right_bounds = np.full(height, -1, dtype=np.int32)
    row_widths = np.zeros(height, dtype=np.int32)

    for y in range(height):
        row_x = np.flatnonzero(roi_mask[y] > 0)
        if row_x.size == 0: continue
        left_bounds[y] = int(row_x[0])
        right_bounds[y] = int(row_x[-1])
        row_widths[y] = int(row_x[-1] - row_x[0] + 1)

    valid_row_widths = row_widths[row_widths > 0]
    if valid_row_widths.size == 0:
        reference_mask_width = 1
    else:
        quantile = float(context.STEP_CONFIG["validation"].get("reference_mask_width_quantile", 0.25))
        quantile = min(max(quantile, 0.0), 1.0)
        reference_mask_width = int(round(float(np.quantile(valid_row_widths, quantile))))

    return left_bounds, right_bounds, row_widths, max(1, reference_mask_width)


def detect_hough_lines(masked_edge: np.ndarray) -> list[tuple[int, int, int, int]]:
    hough_config = context.STEP_CONFIG["hough_lines_p"]
    theta_radians = math.radians(float(hough_config["theta_degrees"]))
    raw_lines = cv2.HoughLinesP(
        image=masked_edge,
        rho=float(hough_config["rho"]),
        theta=theta_radians,
        threshold=int(hough_config["threshold"]),
        minLineLength=int(hough_config["min_line_length"]),
        maxLineGap=int(hough_config["max_line_gap"]),
    )
    if raw_lines is None: return []
    return [tuple(int(value) for value in line[0]) for line in raw_lines]


def sample_line_points(x1: int, y1: int, x2: int, y2: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = max(abs(x2 - x1), abs(y2 - y1)) + 1
    x_coords = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int32)
    y_coords = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int32)
    return x_coords, y_coords


def build_line_record(index: int, line: tuple[int, int, int, int], roi_mask: np.ndarray, row_mask_bounds: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> dict[str, float | int | bool]:
    x1, y1, x2, y2 = line
    x_coords, y_coords = sample_line_points(x1, y1, x2, y2)
    inside_mask = roi_mask[y_coords, x_coords] > 0
    points_inside_mask = int(np.count_nonzero(inside_mask))
    total_points = int(len(x_coords))
    support_ratio = points_inside_mask / total_points if total_points else 0.0
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    angle_degrees = math.degrees(math.atan2(dy, dx))
    absolute_angle = abs(angle_degrees)
    normalized_angle = absolute_angle if absolute_angle <= 90.0 else 180.0 - absolute_angle
    vertical_deviation = abs(90.0 - normalized_angle)
    left_bounds, right_bounds, row_widths, reference_mask_width = row_mask_bounds
    row_left = left_bounds[y_coords]
    row_right = right_bounds[y_coords]
    sample_row_widths = row_widths[y_coords]
    left_clearance = x_coords - row_left
    right_clearance = row_right - x_coords
    horizontal_clearance = np.minimum(left_clearance, right_clearance)
    valid_clearance_samples = inside_mask & (row_right >= row_left)
    min_horizontal_clearance = int(np.min(horizontal_clearance[valid_clearance_samples])) if np.any(valid_clearance_samples) else -1
    horizontal_clearance_ratio = np.full(len(x_coords), -1.0, dtype=np.float64)
    effective_widths = np.minimum(sample_row_widths, reference_mask_width)
    horizontal_clearance_ratio[valid_clearance_samples] = horizontal_clearance[valid_clearance_samples] / np.maximum(effective_widths[valid_clearance_samples], 1)
    min_horizontal_clearance_ratio = float(np.min(horizontal_clearance_ratio[valid_clearance_samples])) if np.any(valid_clearance_samples) else -1.0

    validation_config = context.STEP_CONFIG["validation"]
    is_valid = (
        support_ratio >= float(validation_config["min_mask_support_ratio"])
        and points_inside_mask >= int(validation_config["min_points_inside_mask"])
        and min_horizontal_clearance_ratio >= float(validation_config.get("min_horizontal_clearance_ratio_of_mask_width", 0.0))
        and vertical_deviation <= float(validation_config["max_deviation_from_vertical_degrees"])
    )

    return {
        "line_index": index,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length": round(length, 2),
        "angle_degrees": round(angle_degrees, 2),
        "vertical_deviation_degrees": round(vertical_deviation, 2),
        "sampled_points": total_points,
        "points_inside_mask": points_inside_mask,
        "mask_support_ratio": round(support_ratio, 4),
        "mask_reference_width_px": int(reference_mask_width),
        "min_horizontal_clearance_from_mask_edge_px": min_horizontal_clearance,
        "min_horizontal_clearance_ratio_of_mask_width": round(min_horizontal_clearance_ratio, 4),
        "is_valid": is_valid,
    }


def ensure_output_dirs() -> None:
    for directory in [context.OUTPUT_DIR, context.RAW_OVERLAY_DIR, context.VALID_OVERLAY_DIR, context.COMPARISON_DIR, context.VALID_LINES_JSON_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def process_images(image_names: list[str], *, debug: bool = False) -> None:
    show_windows = debug
    save_config = context.STEP_CONFIG.get("save", {})
    save_debug_images = bool(save_config.get("debug_images", False))
    save_metadata_csv = bool(save_config.get("metadata_csv", False))
    save_valid_lines_json_enabled = bool(save_config.get("valid_lines_json", False))

    if save_debug_images or save_valid_lines_json_enabled: ensure_output_dirs()
    metadata_rows: list[dict[str, str | int | float]] = []

    print()
    print("Processing step 05: valid Hough lines in ROI")
    print(f"Edges input: {context.EDGE_INPUT_DIR}")
    print(f"ROI input:   {context.ROI_MASK_DIR}")
    print(f"Save debug images: {save_debug_images}")
    print(f"Save metadata CSV: {save_metadata_csv}")
    print(f"Save valid lines JSON: {save_valid_lines_json_enabled}")
    if save_debug_images or save_valid_lines_json_enabled: print(f"Output:      {context.OUTPUT_DIR}")
    print(f"Debug windows: {show_windows}")
    print()

    total_images = len(image_names)
    for index, image_name in enumerate(image_names, start=1):
        total_started = perf_counter()
        edge_path = context.EDGE_INPUT_DIR / image_name
        roi_path = context.ROI_MASK_DIR / image_name

        try:
            read_started = perf_counter()
            edge_image = load_grayscale_image(edge_path, label="cleaned edge image")
            roi_mask = load_grayscale_image(roi_path, label="ROI mask image")
            read_time_ms = (perf_counter() - read_started) * 1000.0
        except ValueError as error:
            print(str(error))
            continue

        processing_started = perf_counter()
        hough_mask = make_hough_mask(roi_mask)
        row_mask_bounds = build_row_mask_bounds(roi_mask)
        masked_edge = cv2.bitwise_and(edge_image, edge_image, mask=hough_mask)
        raw_lines = detect_hough_lines(masked_edge)
        line_records = [build_line_record(record_index, line, roi_mask, row_mask_bounds) for record_index, line in enumerate(raw_lines, start=1)]
        raw_overlay = draw_lines(edge_image, line_records, valid_only=False)
        valid_overlay = draw_lines(edge_image, line_records, valid_only=True)
        comparison = make_comparison_view(edge_image, raw_overlay, valid_overlay)
        valid_line_records = [record for record in line_records if bool(record["is_valid"])]
        processing_time_ms = (perf_counter() - processing_started) * 1000.0

        raw_overlay_output_file = ""
        valid_overlay_output_file = ""
        comparison_output_file = ""

        write_started = perf_counter()
        if save_debug_images:
            raw_overlay_output_path = context.RAW_OVERLAY_DIR / image_name
            valid_overlay_output_path = context.VALID_OVERLAY_DIR / image_name
            comparison_output_path = context.COMPARISON_DIR / image_name
            cv2.imwrite(str(raw_overlay_output_path), raw_overlay)
            cv2.imwrite(str(valid_overlay_output_path), valid_overlay)
            cv2.imwrite(str(comparison_output_path), comparison)
            raw_overlay_output_file = relative_project_path(raw_overlay_output_path)
            valid_overlay_output_file = relative_project_path(valid_overlay_output_path)
            comparison_output_file = relative_project_path(comparison_output_path)

        if save_valid_lines_json_enabled:
            json_output_path = context.VALID_LINES_JSON_DIR / f"{Path(image_name).stem}.json"
            save_valid_lines_json(
                output_path=json_output_path,
                image_name=image_name,
                edge_path=edge_path,
                roi_path=roi_path,
                image_shape=edge_image.shape[:2],
                raw_line_count=len(line_records),
                valid_lines=valid_line_records,
            )
        write_time_ms = (perf_counter() - write_started) * 1000.0
        total_time_ms = (perf_counter() - total_started) * 1000.0

        valid_count = len(valid_line_records)
        height, width = edge_image.shape[:2]
        roi_config = context.STEP_CONFIG["roi"]
        hough_config = context.STEP_CONFIG["hough_lines_p"]
        validation_config = context.STEP_CONFIG["validation"]
        metadata_rows.append(
            {
                "source_file": relative_project_path(edge_path),
                "edge_input_file": relative_project_path(edge_path),
                "roi_mask_file": relative_project_path(roi_path),
                "raw_overlay_output_file": raw_overlay_output_file,
                "valid_overlay_output_file": valid_overlay_output_file,
                "comparison_output_file": comparison_output_file,
                "width": width,
                "height": height,
                "processing_step": "05_valid_hough_lines_in_roi",
                "raw_line_count": len(line_records),
                "valid_line_count": valid_count,
                "roi_use_inner_mask": bool(roi_config.get("use_inner_mask", True)),
                "roi_inner_erode_kernel_size": int(roi_config["inner_erode_kernel_size"]),
                "roi_inner_erode_iterations": int(roi_config.get("inner_erode_iterations", 1)),
                "hough_rho": float(hough_config["rho"]),
                "hough_theta_degrees": float(hough_config["theta_degrees"]),
                "hough_threshold": int(hough_config["threshold"]),
                "hough_min_line_length": int(hough_config["min_line_length"]),
                "hough_max_line_gap": int(hough_config["max_line_gap"]),
                "validation_min_mask_support_ratio": float(validation_config["min_mask_support_ratio"]),
                "validation_min_points_inside_mask": int(validation_config["min_points_inside_mask"]),
                "validation_reference_mask_width_quantile": float(validation_config.get("reference_mask_width_quantile", 0.25)),
                "validation_min_horizontal_clearance_ratio_of_mask_width": float(validation_config.get("min_horizontal_clearance_ratio_of_mask_width", 0.0)),
                "validation_max_deviation_from_vertical_degrees": float(validation_config["max_deviation_from_vertical_degrees"]),
                "read_time_ms": round(read_time_ms, 3),
                "processing_time_ms": round(processing_time_ms, 3),
                "write_time_ms": round(write_time_ms, 3),
                "total_time_ms": round(total_time_ms, 3),
            }
        )

        print(
            f"[{index}/{total_images}] Saved: {image_name} | "
            f"raw_lines={len(line_records)} | valid_lines={valid_count} | "
            f"read={read_time_ms:.1f} ms, "
            f"process={processing_time_ms:.1f} ms, "
            f"write={write_time_ms:.1f} ms, "
            f"total={total_time_ms:.1f} ms"
        )

        if show_windows:
            title = f"05 Valid Hough lines | {index}/{total_images} | {image_name}"
            cv2.imshow(title, comparison)
            key = cv2.waitKey(0 if context.DISPLAY_CONFIG["wait_between_images"] else 500) & 0xFF
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass
            if key in [ord("q"), 27]:
                print("Stopped by user.")
                break

    if save_metadata_csv: save_metadata(metadata_rows)
    cv2.destroyAllWindows()

    print()
    print("Done.")
    if save_debug_images:
        print(f"Raw lines overlay saved to: {context.RAW_OVERLAY_DIR}")
        print(f"Valid lines overlay saved to: {context.VALID_OVERLAY_DIR}")
        print(f"Comparison debug saved to: {context.COMPARISON_DIR}")
    if save_valid_lines_json_enabled: print(f"Valid lines JSON saved to: {context.VALID_LINES_JSON_DIR}")
    if save_metadata_csv: print(f"Metadata saved to: {context.CSV_PATH}")
