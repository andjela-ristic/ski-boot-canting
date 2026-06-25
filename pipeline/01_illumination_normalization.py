from pathlib import Path
import csv

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_CONFIG = CONFIG["step_01_illumination_normalization"]

INPUT_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
OUTPUT_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"] / STEP_CONFIG["output_subdir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

CSV_PATH = METADATA_DIR / "processing_01_illumination_normalization.csv"


def normalize_illumination_bgr(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe_config = STEP_CONFIG["clahe"]

    clip_limit = float(clahe_config["clip_limit"])
    tile_grid_size = tuple(clahe_config["tile_grid_size"])

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    normalized_l = clahe.apply(l_channel)

    normalized_lab = cv2.merge(
        [normalized_l, a_channel, b_channel]
    )

    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    return normalized_bgr

def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])

    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def make_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    original_display = resize_for_display(original)
    processed_display = resize_for_display(processed)

    height = min(original_display.shape[0], processed_display.shape[0])

    original_display = cv2.resize(
        original_display,
        (int(original_display.shape[1] * height / original_display.shape[0]), height)
    )

    processed_display = cv2.resize(
        processed_display,
        (int(processed_display.shape[1] * height / processed_display.shape[0]), height)
    )

    separator = np.full((height, 10, 3), 255, dtype=np.uint8)

    combined = np.hstack([original_display, separator, processed_display])

    return combined


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    image_paths = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    return sorted(image_paths)


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
        "method",
        "clahe_clip_limit",
        "clahe_tile_grid_size",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    metadata_rows = []

    print()
    print("Processing step 01: illumination normalization")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    print("Controls:")
    print("  n / SPACE / ENTER  -> next image")
    print("  q / ESC            -> quit")
    print()
    clahe_config = STEP_CONFIG["clahe"]
    clip_limit = float(clahe_config["clip_limit"])
    tile_grid_size = tuple(clahe_config["tile_grid_size"])
    tile_grid_size_label = f"{tile_grid_size[0]}x{tile_grid_size[1]}"

    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"Could not read image: {image_path}")
            continue

        normalized_bgr = normalize_illumination_bgr(image_bgr)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), normalized_bgr)

        height, width = image_bgr.shape[:2]

        metadata_rows.append({
            "source_file": str(image_path.relative_to(PROJECT_ROOT)),
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            "width": width,
            "height": height,
            "processing_step": "01_illumination_normalization",
            "method":  STEP_CONFIG["method"],
            "clahe_clip_limit": clip_limit,
            "clahe_tile_grid_size": tile_grid_size_label,
        })

        print(f"[{index}/{len(image_paths)}] Saved: {output_path.name}")

        if DISPLAY_CONFIG["show_windows"]:
            comparison = make_side_by_side(image_bgr, normalized_bgr)

            title = f"01 Illumination normalization | {index}/{len(image_paths)} | {image_path.name}"

            cv2.imshow(title, comparison)

            if DISPLAY_CONFIG["wait_between_images"]:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(500) & 0xFF

            cv2.destroyWindow(title)

            if key in [ord("q"), 27]:
                print("Stopped by user.")
                break

    save_metadata(metadata_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Processed images saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()