from pathlib import Path
import csv

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_02_CONFIG = CONFIG["step_02_grayscale_and_blur"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

STEP_02_OUTPUT_DIR = PROCESSED_DIR / STEP_02_CONFIG["output_subdir"]
SELECTED_STEP_02_OUTPUT = STEP_02_CONFIG["selected_output"]

INPUT_DIR = STEP_02_OUTPUT_DIR / SELECTED_STEP_02_OUTPUT
OUTPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]

CSV_PATH = METADATA_DIR / "processing_03_edge_detection.csv"


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
    image = to_bgr_for_display(image).copy()

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


def make_side_by_side(input_image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    input_display = resize_for_display(to_bgr_for_display(input_image))
    edges_display = resize_for_display(to_bgr_for_display(edges))

    height = min(input_display.shape[0], edges_display.shape[0])

    input_display = cv2.resize(
        input_display,
        (int(input_display.shape[1] * height / input_display.shape[0]), height),
        interpolation=cv2.INTER_AREA
    )

    edges_display = cv2.resize(
        edges_display,
        (int(edges_display.shape[1] * height / edges_display.shape[0]), height),
        interpolation=cv2.INTER_AREA
    )

    input_display = add_label(input_display, f"input: {SELECTED_STEP_02_OUTPUT}")
    edges_display = add_label(edges_display, "canny edges")

    separator = np.full((height, 10, 3), 255, dtype=np.uint8)

    return np.hstack([input_display, separator, edges_display])


def validate_aperture_size(value: int) -> int:
    allowed_values = {3, 5, 7}

    if value not in allowed_values:
        raise ValueError(f"Canny aperture_size must be one of {allowed_values}. Got: {value}")

    return value


def run_canny(image_gray: np.ndarray) -> np.ndarray:
    canny_config = STEP_03_CONFIG["canny"]

    threshold_1 = int(canny_config["threshold_1"])
    threshold_2 = int(canny_config["threshold_2"])
    aperture_size = validate_aperture_size(int(canny_config["aperture_size"]))
    use_l2_gradient = bool(canny_config["use_l2_gradient"])

    edges = cv2.Canny(
        image=image_gray,
        threshold1=threshold_1,
        threshold2=threshold_2,
        apertureSize=aperture_size,
        L2gradient=use_l2_gradient
    )

    return edges


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
        "input_from_step_02",
        "threshold_1",
        "threshold_2",
        "aperture_size",
        "use_l2_gradient",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not STEP_03_CONFIG["enabled"]:
        print("Step 03 is disabled in config.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    canny_config = STEP_03_CONFIG["canny"]

    threshold_1 = int(canny_config["threshold_1"])
    threshold_2 = int(canny_config["threshold_2"])
    aperture_size = validate_aperture_size(int(canny_config["aperture_size"]))
    use_l2_gradient = bool(canny_config["use_l2_gradient"])

    metadata_rows = []

    print()
    print("Processing step 03: edge detection")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Selected step 2 output: {SELECTED_STEP_02_OUTPUT}")
    print()
    print("Controls:")
    print("  n / SPACE / ENTER  -> next image")
    print("  q / ESC            -> quit")
    print()

    for index, image_path in enumerate(image_paths, start=1):
        input_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if input_image is None:
            print(f"Could not read image: {image_path}")
            continue

        edges = run_canny(input_image)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), edges)

        height, width = input_image.shape[:2]

        metadata_rows.append({
            "source_file": str(image_path.relative_to(PROJECT_ROOT)),
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            "width": width,
            "height": height,
            "processing_step": "03_edge_detection",
            "input_from_step_02": SELECTED_STEP_02_OUTPUT,
            "threshold_1": threshold_1,
            "threshold_2": threshold_2,
            "aperture_size": aperture_size,
            "use_l2_gradient": use_l2_gradient,
        })

        print(f"[{index}/{len(image_paths)}] Saved: {output_path.name}")

        if DISPLAY_CONFIG["show_windows"]:
            comparison = make_side_by_side(input_image, edges)

            title = f"03 Edge detection | {index}/{len(image_paths)} | {image_path.name}"

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
    print(f"Edges saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()