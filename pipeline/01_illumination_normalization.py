from pathlib import Path
import csv

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "working_png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "01_illumination_normalized"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

CSV_PATH = METADATA_DIR / "processing_01_illumination_normalization.csv"


def normalize_illumination_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Normalizuje osvetljenje bez menjanja geometrije slike.

    Koristi LAB color space:
    - L kanal = osvetljenje
    - A/B kanali = boja

    CLAHE se primenjuje samo na L kanal, tako da popravljamo kontrast/osvetljenje,
    ali ne menjamo agresivno boje slike.
    """

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        # clipLimit=1.5,
        # tileGridSize=(8, 8)
        clipLimit=2.0,
        tileGridSize=(8, 8)
        # clipLimit=2.5,
        # tileGridSize=(8, 8)
    )

    normalized_l = clahe.apply(l_channel)

    normalized_lab = cv2.merge(
        [normalized_l, a_channel, b_channel]
    )

    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    return normalized_bgr


def resize_for_display(image: np.ndarray, max_height: int = 800) -> np.ndarray:
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
    image_paths = []

    for extension in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        image_paths.extend(INPUT_DIR.glob(extension))

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
            "method": "LAB_L_channel_CLAHE",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": "8x8",
        })

        comparison = make_side_by_side(image_bgr, normalized_bgr)

        title = f"01 Illumination normalization | {index}/{len(image_paths)} | {image_path.name}"

        cv2.imshow(title, comparison)

        print(f"[{index}/{len(image_paths)}] Saved: {output_path.name}")

        key = cv2.waitKey(0) & 0xFF

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