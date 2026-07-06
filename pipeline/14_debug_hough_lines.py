from pathlib import Path
import csv
import json
import math
from typing import Any

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]
STEP_14_CONFIG = CONFIG["step_14_debug_hough_lines"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

INPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]
OUTPUT_DIR = PROCESSED_DIR / STEP_14_CONFIG["output_subdir"]
JSON_DIR = OUTPUT_DIR / "json"

SUMMARY_CSV_PATH = METADATA_DIR / "processing_14_debug_hough_lines.csv"
LINES_CSV_PATH = METADATA_DIR / "processing_14_debug_hough_lines_lines.csv"


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    if not INPUT_DIR.exists():
        return []

    image_paths = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    return sorted(image_paths)


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def to_bgr_for_display(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    image = image.copy()

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 45),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        image,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def make_side_by_side(edges: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    edges_display = resize_for_display(to_bgr_for_display(edges))
    overlay_display = resize_for_display(overlay)

    height = min(edges_display.shape[0], overlay_display.shape[0])

    edges_display = cv2.resize(
        edges_display,
        (int(edges_display.shape[1] * height / edges_display.shape[0]), height),
        interpolation=cv2.INTER_AREA,
    )

    overlay_display = cv2.resize(
        overlay_display,
        (int(overlay_display.shape[1] * height / overlay_display.shape[0]), height),
        interpolation=cv2.INTER_AREA,
    )

    edges_display = add_label(edges_display, "input edges")
    overlay_display = add_label(overlay_display, "detected Hough lines")

    separator = np.full((height, 10, 3), 255, dtype=np.uint8)

    return np.hstack([edges_display, separator, overlay_display])


def line_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return float(math.hypot(x2 - x1, y2 - y1))


def line_angle_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    while angle > 90.0:
        angle -= 180.0

    while angle < -90.0:
        angle += 180.0

    return float(angle)


def distance_to_vertical_deg(angle_deg: float) -> float:
    return float(abs(90.0 - abs(angle_deg)))


def distance_to_horizontal_deg(angle_deg: float) -> float:
    return float(abs(angle_deg))


def classify_line(angle_degrees: float) -> str:
    config = STEP_14_CONFIG["classification"]

    vertical_tolerance = float(config["vertical_angle_tolerance_degrees"])
    horizontal_tolerance = float(config["horizontal_angle_tolerance_degrees"])

    if distance_to_vertical_deg(angle_degrees) <= vertical_tolerance:
        return "vertical"

    if distance_to_horizontal_deg(angle_degrees) <= horizontal_tolerance:
        return "horizontal"

    return "other"


def detect_lines(edge_image: np.ndarray) -> np.ndarray | None:
    hough_config = STEP_14_CONFIG["hough_lines_p"]

    rho = float(hough_config["rho"])
    theta = np.pi / int(hough_config["theta_divisor"])
    threshold = int(hough_config["threshold"])
    min_line_length = int(hough_config["min_line_length"])
    max_line_gap = int(hough_config["max_line_gap"])

    return cv2.HoughLinesP(
        image=edge_image,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )


def build_line_records(edge_image: np.ndarray) -> list[dict[str, Any]]:
    lines = detect_lines(edge_image)

    if lines is None:
        return []

    records: list[dict[str, Any]] = []

    for index, line in enumerate(lines, start=1):
        x1, y1, x2, y2 = [int(value) for value in line[0]]
        angle = line_angle_degrees(x1, y1, x2, y2)
        length = line_length(x1, y1, x2, y2)
        line_type = classify_line(angle)

        records.append({
            "id": f"l{index:05d}",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length": round(float(length), 4),
            "angle_deg": round(float(angle), 4),
            "distance_to_vertical_deg": round(distance_to_vertical_deg(angle), 4),
            "distance_to_horizontal_deg": round(distance_to_horizontal_deg(angle), 4),
            "line_type": line_type,
        })

    return records


def draw_detected_lines(edge_image: np.ndarray, records: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    overlay = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    counts = {
        "vertical": 0,
        "horizontal": 0,
        "other": 0,
    }

    for record in records:
        x1 = int(record["x1"])
        y1 = int(record["y1"])
        x2 = int(record["x2"])
        y2 = int(record["y2"])
        line_type = str(record["line_type"])
        length = float(record["length"])

        counts[line_type] = counts.get(line_type, 0) + 1

        if line_type == "vertical":
            color = (0, 255, 0)
            thickness = 2
        elif line_type == "horizontal":
            color = (255, 0, 0)
            thickness = 2
        else:
            color = (80, 80, 80)
            thickness = 1

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )

        midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        cv2.putText(
            overlay,
            f"{int(length)}",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay, counts


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_summary_metadata(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "output_file",
        "json_file",
        "width",
        "height",
        "processing_step",
        "rho",
        "theta_divisor",
        "threshold",
        "min_line_length",
        "max_line_gap",
        "vertical_angle_tolerance_degrees",
        "horizontal_angle_tolerance_degrees",
        "vertical_count",
        "horizontal_count",
        "other_count",
        "total_count",
    ]

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_line_metadata(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "json_file",
        "line_id",
        "line_type",
        "x1",
        "y1",
        "x2",
        "y2",
        "length",
        "angle_deg",
        "distance_to_vertical_deg",
        "distance_to_horizontal_deg",
    ]

    with open(LINES_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not STEP_14_CONFIG["enabled"]:
        print("Step 14 is disabled in config.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    hough_config = STEP_14_CONFIG["hough_lines_p"]
    classification_config = STEP_14_CONFIG["classification"]

    summary_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []

    print()
    print("Processing step 14: Debug Hough lines")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"JSON:   {JSON_DIR}")
    print()
    print("Line colors:")
    print("  green = vertical")
    print("  blue  = horizontal")
    print("  gray  = other")
    print()
    print("Controls:")
    print("  n / SPACE / ENTER  -> next image")
    print("  q / ESC            -> quit")
    print()

    for index, image_path in enumerate(image_paths, start=1):
        edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if edge_image is None:
            print(f"Could not read edge image: {image_path}")
            continue

        height, width = edge_image.shape[:2]
        records = build_line_records(edge_image)
        overlay, counts = draw_detected_lines(edge_image, records)

        output_path = OUTPUT_DIR / image_path.name
        json_path = JSON_DIR / f"{image_path.stem}.json"

        cv2.imwrite(str(output_path), overlay)

        json_data = {
            "source_file": image_path.name,
            "source_path": relative(image_path),
            "overlay_file": relative(output_path),
            "width": width,
            "height": height,
            "processing_step": "14_debug_hough_lines",
            "parameters": {
                "rho": hough_config["rho"],
                "theta_divisor": hough_config["theta_divisor"],
                "threshold": hough_config["threshold"],
                "min_line_length": hough_config["min_line_length"],
                "max_line_gap": hough_config["max_line_gap"],
                "vertical_angle_tolerance_degrees": classification_config["vertical_angle_tolerance_degrees"],
                "horizontal_angle_tolerance_degrees": classification_config["horizontal_angle_tolerance_degrees"],
            },
            "counts": {
                "vertical": counts.get("vertical", 0),
                "horizontal": counts.get("horizontal", 0),
                "other": counts.get("other", 0),
                "total": len(records),
            },
            "lines": records,
        }

        save_json(json_path, json_data)

        summary_rows.append({
            "source_file": relative(image_path),
            "output_file": relative(output_path),
            "json_file": relative(json_path),
            "width": width,
            "height": height,
            "processing_step": "14_debug_hough_lines",
            "rho": hough_config["rho"],
            "theta_divisor": hough_config["theta_divisor"],
            "threshold": hough_config["threshold"],
            "min_line_length": hough_config["min_line_length"],
            "max_line_gap": hough_config["max_line_gap"],
            "vertical_angle_tolerance_degrees": classification_config["vertical_angle_tolerance_degrees"],
            "horizontal_angle_tolerance_degrees": classification_config["horizontal_angle_tolerance_degrees"],
            "vertical_count": counts.get("vertical", 0),
            "horizontal_count": counts.get("horizontal", 0),
            "other_count": counts.get("other", 0),
            "total_count": len(records),
        })

        for record in records:
            line_rows.append({
                "source_file": relative(image_path),
                "json_file": relative(json_path),
                "line_id": record["id"],
                "line_type": record["line_type"],
                "x1": record["x1"],
                "y1": record["y1"],
                "x2": record["x2"],
                "y2": record["y2"],
                "length": record["length"],
                "angle_deg": record["angle_deg"],
                "distance_to_vertical_deg": record["distance_to_vertical_deg"],
                "distance_to_horizontal_deg": record["distance_to_horizontal_deg"],
            })

        print(
            f"[{index}/{len(image_paths)}] Saved: {output_path.name} | "
            f"vertical={counts.get('vertical', 0)}, "
            f"horizontal={counts.get('horizontal', 0)}, "
            f"other={counts.get('other', 0)}, "
            f"total={len(records)}"
        )

        if DISPLAY_CONFIG["show_windows"]:
            comparison = make_side_by_side(edge_image, overlay)

            title = f"14 Debug Hough lines | {index}/{len(image_paths)} | {image_path.name}"

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

    save_summary_metadata(summary_rows)
    save_line_metadata(line_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Line overlays saved to: {OUTPUT_DIR}")
    print(f"Line JSON saved to:     {JSON_DIR}")
    print(f"Summary metadata saved to: {SUMMARY_CSV_PATH}")
    print(f"Per-line metadata saved to: {LINES_CSV_PATH}")


if __name__ == "__main__":
    main()
