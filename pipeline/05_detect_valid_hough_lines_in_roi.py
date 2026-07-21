from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]
STEP_04_CONFIG = CONFIG["step_04_boot_roi_from_edges"]
STEP_CONFIG = CONFIG["step_05_valid_hough_lines_in_roi"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]

if bool(STEP_CONFIG.get("inherit_step_03_output", True)):
    edge_input_name = str(
        STEP_CONFIG.get("edge_input_name", STEP_03_CONFIG.get("selected_output", "cleaned"))
    ).strip()
    EDGE_INPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"] / edge_input_name
else:
    EDGE_INPUT_DIR = PROCESSED_DIR / STEP_CONFIG["edge_input_subdir"]

if bool(STEP_CONFIG.get("inherit_step_04_output", True)):
    roi_mask_subdir_name = str(STEP_CONFIG.get("roi_mask_subdir_name", "mask")).strip()
    ROI_MASK_DIR = PROCESSED_DIR / STEP_04_CONFIG["output_subdir"] / roi_mask_subdir_name
else:
    ROI_MASK_DIR = PROCESSED_DIR / STEP_CONFIG["roi_mask_subdir"]
OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

MASKED_EDGE_DIR = OUTPUT_DIR / "masked_edges"
RAW_OVERLAY_DIR = OUTPUT_DIR / "raw_lines_overlay"
VALID_OVERLAY_DIR = OUTPUT_DIR / "valid_lines_overlay"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
VALID_LINES_JSON_DIR = OUTPUT_DIR / "valid_lines_json"
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]
CSV_PATH = METADATA_DIR / "processing_05_valid_hough_lines_in_roi.csv"


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect Hough line segments from cleaned edges and keep only lines valid inside ROI."
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


def collect_images(selected_image: str | None = None) -> list[str]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    if not EDGE_INPUT_DIR.exists() or not ROI_MASK_DIR.exists():
        return []

    edge_names = {
        path.name
        for path in EDGE_INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    }
    roi_names = {
        path.name
        for path in ROI_MASK_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    }

    image_names = sorted(edge_names & roi_names)

    if selected_image is None:
        return image_names

    selected_image = selected_image.strip()
    candidates = {selected_image}

    if "." not in selected_image:
        candidates.update(f"{selected_image}{extension}" for extension in allowed_extensions)

    return [image_name for image_name in image_names if image_name in candidates]


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


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def make_hough_mask(roi_mask: np.ndarray) -> np.ndarray:
    roi_config = STEP_CONFIG["roi"]
    binary_mask = ensure_binary_mask(roi_mask)

    if not bool(roi_config.get("use_inner_mask", True)):
        return binary_mask

    kernel_size = ensure_odd_kernel_size(
        int(roi_config["inner_erode_kernel_size"]),
        "roi.inner_erode_kernel_size",
    )
    iterations = int(roi_config.get("inner_erode_iterations", 1))
    kernel = make_ellipse_kernel(kernel_size)

    eroded = cv2.erode(binary_mask, kernel, iterations=iterations)

    if np.count_nonzero(eroded) == 0:
        return binary_mask

    return eroded


def build_row_mask_bounds(roi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    height, width = roi_mask.shape[:2]
    left_bounds = np.full(height, width, dtype=np.int32)
    right_bounds = np.full(height, -1, dtype=np.int32)
    row_widths = np.zeros(height, dtype=np.int32)

    for y in range(height):
        row_x = np.flatnonzero(roi_mask[y] > 0)
        if row_x.size == 0:
            continue
        left_bounds[y] = int(row_x[0])
        right_bounds[y] = int(row_x[-1])
        row_widths[y] = int(row_x[-1] - row_x[0] + 1)

    valid_row_widths = row_widths[row_widths > 0]
    if valid_row_widths.size == 0:
        reference_mask_width = 1
    else:
        quantile = float(STEP_CONFIG["validation"].get("reference_mask_width_quantile", 0.25))
        quantile = min(max(quantile, 0.0), 1.0)
        reference_mask_width = int(round(float(np.quantile(valid_row_widths, quantile))))

    return left_bounds, right_bounds, row_widths, max(1, reference_mask_width)


def detect_hough_lines(masked_edge: np.ndarray) -> list[tuple[int, int, int, int]]:
    hough_config = STEP_CONFIG["hough_lines_p"]
    theta_radians = math.radians(float(hough_config["theta_degrees"]))

    raw_lines = cv2.HoughLinesP(
        image=masked_edge,
        rho=float(hough_config["rho"]),
        theta=theta_radians,
        threshold=int(hough_config["threshold"]),
        minLineLength=int(hough_config["min_line_length"]),
        maxLineGap=int(hough_config["max_line_gap"]),
    )

    if raw_lines is None:
        return []

    return [tuple(int(value) for value in line[0]) for line in raw_lines]


def sample_line_points(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = max(abs(x2 - x1), abs(y2 - y1)) + 1

    x_coords = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int32)
    y_coords = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int32)

    return x_coords, y_coords


def build_line_record(
    index: int,
    line: tuple[int, int, int, int],
    roi_mask: np.ndarray,
    row_mask_bounds: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> dict[str, float | int | bool]:
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
    min_horizontal_clearance = (
        int(np.min(horizontal_clearance[valid_clearance_samples]))
        if np.any(valid_clearance_samples)
        else -1
    )
    horizontal_clearance_ratio = np.full(len(x_coords), -1.0, dtype=np.float64)
    effective_widths = np.minimum(sample_row_widths, reference_mask_width)
    horizontal_clearance_ratio[valid_clearance_samples] = (
        horizontal_clearance[valid_clearance_samples]
        / np.maximum(effective_widths[valid_clearance_samples], 1)
    )
    min_horizontal_clearance_ratio = (
        float(np.min(horizontal_clearance_ratio[valid_clearance_samples]))
        if np.any(valid_clearance_samples)
        else -1.0
    )

    validation_config = STEP_CONFIG["validation"]
    is_valid = (
        support_ratio >= float(validation_config["min_mask_support_ratio"])
        and points_inside_mask >= int(validation_config["min_points_inside_mask"])
        and min_horizontal_clearance_ratio
        >= float(validation_config.get("min_horizontal_clearance_ratio_of_mask_width", 0.0))
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


def draw_lines(
    base_image: np.ndarray,
    line_records: list[dict[str, float | int | bool]],
    valid_only: bool,
) -> np.ndarray:
    overlay = to_bgr(base_image).copy()
    drawing_config = STEP_CONFIG["drawing"]
    raw_thickness = int(drawing_config["raw_line_thickness"])
    valid_thickness = int(drawing_config["valid_line_thickness"])

    for record in line_records:
        if valid_only and not bool(record["is_valid"]):
            continue

        x1 = int(record["x1"])
        y1 = int(record["y1"])
        x2 = int(record["x2"])
        y2 = int(record["y2"])
        is_valid = bool(record["is_valid"])

        color = (0, 220, 0) if is_valid else (0, 0, 255)
        thickness = valid_thickness if is_valid else raw_thickness

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )

    return overlay


def make_comparison_view(
    edge_image: np.ndarray,
    raw_overlay: np.ndarray,
    valid_overlay: np.ndarray,
) -> np.ndarray:
    displays = [
        add_label(resize_for_display(to_bgr(edge_image)), "cleaned edges"),
        add_label(resize_for_display(raw_overlay), "raw hough lines"),
        add_label(resize_for_display(valid_overlay), "valid hough lines"),
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


def save_metadata(rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "edge_input_file",
        "roi_mask_file",
        "raw_overlay_output_file",
        "valid_overlay_output_file",
        "comparison_output_file",
        "width",
        "height",
        "processing_step",
        "raw_line_count",
        "valid_line_count",
        "roi_use_inner_mask",
        "roi_inner_erode_kernel_size",
        "roi_inner_erode_iterations",
        "hough_rho",
        "hough_theta_degrees",
        "hough_threshold",
        "hough_min_line_length",
        "hough_max_line_gap",
        "validation_min_mask_support_ratio",
        "validation_min_points_inside_mask",
        "validation_reference_mask_width_quantile",
        "validation_min_horizontal_clearance_ratio_of_mask_width",
        "validation_max_deviation_from_vertical_degrees",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_valid_lines_json(
    output_path: Path,
    image_name: str,
    edge_path: Path,
    roi_path: Path,
    image_shape: tuple[int, int],
    raw_line_count: int,
    valid_lines: list[dict[str, float | int | bool]],
) -> None:
    height, width = image_shape
    roi_config = STEP_CONFIG["roi"]
    hough_config = STEP_CONFIG["hough_lines_p"]
    validation_config = STEP_CONFIG["validation"]

    payload = {
        "image_name": image_name,
        "source_file": relative_project_path(edge_path),
        "roi_mask_file": relative_project_path(roi_path),
        "processing_step": "05_valid_hough_lines_in_roi",
        "width": width,
        "height": height,
        "raw_line_count": raw_line_count,
        "valid_line_count": len(valid_lines),
        "parameters": {
            "roi": {
                "use_inner_mask": bool(roi_config.get("use_inner_mask", True)),
                "inner_erode_kernel_size": int(roi_config["inner_erode_kernel_size"]),
                "inner_erode_iterations": int(roi_config.get("inner_erode_iterations", 1)),
            },
            "hough_lines_p": {
                "rho": float(hough_config["rho"]),
                "theta_degrees": float(hough_config["theta_degrees"]),
                "threshold": int(hough_config["threshold"]),
                "min_line_length": int(hough_config["min_line_length"]),
                "max_line_gap": int(hough_config["max_line_gap"]),
            },
            "validation": {
                "min_mask_support_ratio": float(validation_config["min_mask_support_ratio"]),
                "min_points_inside_mask": int(validation_config["min_points_inside_mask"]),
                "reference_mask_width_quantile": float(
                    validation_config.get("reference_mask_width_quantile", 0.25)
                ),
                "min_horizontal_clearance_ratio_of_mask_width": float(
                    validation_config.get("min_horizontal_clearance_ratio_of_mask_width", 0.0)
                ),
                "max_deviation_from_vertical_degrees": float(
                    validation_config["max_deviation_from_vertical_degrees"]
                ),
            },
        },
        "valid_lines": valid_lines,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_DIR,
        RAW_OVERLAY_DIR,
        VALID_OVERLAY_DIR,
        COMPARISON_DIR,
        VALID_LINES_JSON_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not STEP_CONFIG["enabled"]:
        print("Step 05 is disabled in config.")
        return

    args = parse_args()
    show_windows = bool(args.show)
    save_config = STEP_CONFIG.get("save", {})
    save_debug_images = bool(save_config.get("debug_images", False))
    save_metadata_csv = bool(save_config.get("metadata_csv", False))
    save_valid_lines_json_enabled = bool(save_config.get("valid_lines_json", False))

    if save_debug_images or save_valid_lines_json_enabled:
        ensure_output_dirs()

    image_names = collect_images(args.image)

    if not image_names:
        print(f"No matching images found in: {EDGE_INPUT_DIR} and {ROI_MASK_DIR}")
        return

    metadata_rows: list[dict[str, str | int | float]] = []

    print()
    print("Processing step 05: valid Hough lines in ROI")
    print(f"Edges input: {EDGE_INPUT_DIR}")
    print(f"ROI input:   {ROI_MASK_DIR}")
    print(f"Save debug images: {save_debug_images}")
    print(f"Save metadata CSV: {save_metadata_csv}")
    print(f"Save valid lines JSON: {save_valid_lines_json_enabled}")
    if save_debug_images or save_valid_lines_json_enabled:
        print(f"Output:      {OUTPUT_DIR}")
    print(f"Selected image filter: {args.image if args.image else 'all'}")
    print()

    for index, image_name in enumerate(image_names, start=1):
        edge_path = EDGE_INPUT_DIR / image_name
        roi_path = ROI_MASK_DIR / image_name

        edge_image = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

        if edge_image is None:
            print(f"Could not read cleaned edge image: {edge_path}")
            continue

        if roi_mask is None:
            print(f"Could not read ROI mask image: {roi_path}")
            continue

        hough_mask = make_hough_mask(roi_mask)
        row_mask_bounds = build_row_mask_bounds(roi_mask)
        masked_edge = cv2.bitwise_and(edge_image, edge_image, mask=hough_mask)
        raw_lines = detect_hough_lines(masked_edge)
        line_records = [
            build_line_record(record_index, line, roi_mask, row_mask_bounds)
            for record_index, line in enumerate(raw_lines, start=1)
        ]

        raw_overlay = draw_lines(edge_image, line_records, valid_only=False)
        valid_overlay = draw_lines(edge_image, line_records, valid_only=True)
        comparison = make_comparison_view(edge_image, raw_overlay, valid_overlay)
        valid_line_records = [record for record in line_records if bool(record["is_valid"])]

        raw_overlay_output_file = ""
        valid_overlay_output_file = ""
        comparison_output_file = ""

        if save_debug_images:
            raw_overlay_output_path = RAW_OVERLAY_DIR / image_name
            valid_overlay_output_path = VALID_OVERLAY_DIR / image_name
            comparison_output_path = COMPARISON_DIR / image_name

            cv2.imwrite(str(raw_overlay_output_path), raw_overlay)
            cv2.imwrite(str(valid_overlay_output_path), valid_overlay)
            cv2.imwrite(str(comparison_output_path), comparison)

            raw_overlay_output_file = relative_project_path(raw_overlay_output_path)
            valid_overlay_output_file = relative_project_path(valid_overlay_output_path)
            comparison_output_file = relative_project_path(comparison_output_path)

        if save_valid_lines_json_enabled:
            json_output_path = VALID_LINES_JSON_DIR / f"{Path(image_name).stem}.json"
            save_valid_lines_json(
                output_path=json_output_path,
                image_name=image_name,
                edge_path=edge_path,
                roi_path=roi_path,
                image_shape=edge_image.shape[:2],
                raw_line_count=len(line_records),
                valid_lines=valid_line_records,
            )

        valid_count = len(valid_line_records)
        height, width = edge_image.shape[:2]
        roi_config = STEP_CONFIG["roi"]
        hough_config = STEP_CONFIG["hough_lines_p"]
        validation_config = STEP_CONFIG["validation"]

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
                "validation_reference_mask_width_quantile": float(
                    validation_config.get("reference_mask_width_quantile", 0.25)
                ),
                "validation_min_horizontal_clearance_ratio_of_mask_width": float(
                    validation_config.get("min_horizontal_clearance_ratio_of_mask_width", 0.0)
                ),
                "validation_max_deviation_from_vertical_degrees": float(
                    validation_config["max_deviation_from_vertical_degrees"]
                ),
            }
        )

        print(
            f"[{index}/{len(image_names)}] Saved: {image_name} | "
            f"raw_lines={len(line_records)} | valid_lines={valid_count}"
        )

        if show_windows:
            title = f"05 Valid Hough lines | {index}/{len(image_names)} | {image_name}"
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

    if save_metadata_csv:
        save_metadata(metadata_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    if save_debug_images:
        print(f"Raw lines overlay saved to: {RAW_OVERLAY_DIR}")
        print(f"Valid lines overlay saved to: {VALID_OVERLAY_DIR}")
        print(f"Comparison debug saved to: {COMPARISON_DIR}")
    if save_valid_lines_json_enabled:
        print(f"Valid lines JSON saved to: {VALID_LINES_JSON_DIR}")
    if save_metadata_csv:
        print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
