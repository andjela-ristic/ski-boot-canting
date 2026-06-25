from pathlib import Path
import csv
import math

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]
STEP_04_CONFIG = CONFIG["step_04_line_detection_hough"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

INPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]
OUTPUT_DIR = PROCESSED_DIR / STEP_04_CONFIG["output_subdir"]

CSV_PATH = METADATA_DIR / "processing_04_line_detection_hough.csv"


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

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
        thickness=-1
    )

    cv2.putText(
        image,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


def make_side_by_side(edges: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    edges_display = resize_for_display(to_bgr_for_display(edges))
    overlay_display = resize_for_display(overlay)

    height = min(edges_display.shape[0], overlay_display.shape[0])

    edges_display = cv2.resize(
        edges_display,
        (int(edges_display.shape[1] * height / edges_display.shape[0]), height),
        interpolation=cv2.INTER_AREA
    )

    overlay_display = cv2.resize(
        overlay_display,
        (int(overlay_display.shape[1] * height / overlay_display.shape[0]), height),
        interpolation=cv2.INTER_AREA
    )

    edges_display = add_label(edges_display, "input edges")
    overlay_display = add_label(overlay_display, "detected lines")

    separator = np.full((height, 10, 3), 255, dtype=np.uint8)

    return np.hstack([edges_display, separator, overlay_display])


def line_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def line_angle_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle


def classify_line(angle_degrees: float) -> str | None:
    config = STEP_04_CONFIG["classification"]

    vertical_tolerance = float(config["vertical_angle_tolerance_degrees"])
    horizontal_tolerance = float(config["horizontal_angle_tolerance_degrees"])

    normalized_angle = abs(angle_degrees)

    if normalized_angle > 90:
        normalized_angle = 180 - normalized_angle

    if abs(normalized_angle - 90) <= vertical_tolerance:
        return "vertical"

    if normalized_angle <= horizontal_tolerance:
        return "horizontal"

    return None


def detect_lines(edge_image: np.ndarray):
    hough_config = STEP_04_CONFIG["hough_lines_p"]

    rho = float(hough_config["rho"])
    theta = np.pi / int(hough_config["theta_divisor"])
    threshold = int(hough_config["threshold"])
    min_line_length = int(hough_config["min_line_length"])
    max_line_gap = int(hough_config["max_line_gap"])

    lines = cv2.HoughLinesP(
        image=edge_image,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    return lines


def draw_detected_lines(edge_image: np.ndarray):
    overlay = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    lines = detect_lines(edge_image)

    vertical_count = 0
    horizontal_count = 0
    other_count = 0

    if lines is None:
        return overlay, vertical_count, horizontal_count, other_count

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = line_angle_degrees(x1, y1, x2, y2)
        length = line_length(x1, y1, x2, y2)
        line_type = classify_line(angle)

        if line_type == "vertical":
            vertical_count += 1
            color = (0, 255, 0)
            thickness = 2

        elif line_type == "horizontal":
            horizontal_count += 1
            color = (255, 0, 0)
            thickness = 2

        else:
            other_count += 1
            color = (80, 80, 80)
            thickness = 1

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA
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
            cv2.LINE_AA
        )

    return overlay, vertical_count, horizontal_count, other_count


def save_metadata(rows: list[dict]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "output_file",
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
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not STEP_04_CONFIG["enabled"]:
        print("Step 04 is disabled in config.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    hough_config = STEP_04_CONFIG["hough_lines_p"]
    classification_config = STEP_04_CONFIG["classification"]

    metadata_rows = []

    print()
    print("Processing step 04: Hough line detection")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
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
            print(f"Could not read image: {image_path}")
            continue

        overlay, vertical_count, horizontal_count, other_count = draw_detected_lines(edge_image)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), overlay)

        height, width = edge_image.shape[:2]

        metadata_rows.append({
            "source_file": str(image_path.relative_to(PROJECT_ROOT)),
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            "width": width,
            "height": height,
            "processing_step": "04_line_detection_hough",
            "rho": hough_config["rho"],
            "theta_divisor": hough_config["theta_divisor"],
            "threshold": hough_config["threshold"],
            "min_line_length": hough_config["min_line_length"],
            "max_line_gap": hough_config["max_line_gap"],
            "vertical_angle_tolerance_degrees": classification_config["vertical_angle_tolerance_degrees"],
            "horizontal_angle_tolerance_degrees": classification_config["horizontal_angle_tolerance_degrees"],
            "vertical_count": vertical_count,
            "horizontal_count": horizontal_count,
            "other_count": other_count,
        })

        print(
            f"[{index}/{len(image_paths)}] Saved: {output_path.name} | "
            f"vertical={vertical_count}, horizontal={horizontal_count}, other={other_count}"
        )

        if DISPLAY_CONFIG["show_windows"]:
            comparison = make_side_by_side(edge_image, overlay)

            title = f"04 Hough lines | {index}/{len(image_paths)} | {image_path.name}"

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

    save_metadata(metadata_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Line overlays saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()