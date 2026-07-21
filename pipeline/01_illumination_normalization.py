from pathlib import Path
from time import perf_counter
import argparse
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

CLAHE_CONFIG = STEP_CONFIG["clahe"]
CLAHE_CLIP_LIMIT = float(CLAHE_CONFIG["clip_limit"])
CLAHE_TILE_GRID_SIZE = tuple(int(value) for value in CLAHE_CONFIG["tile_grid_size"])
CLAHE_TILE_GRID_SIZE_LABEL = (
    f"{CLAHE_TILE_GRID_SIZE[0]}x{CLAHE_TILE_GRID_SIZE[1]}"
)

# 0 is fastest and produces larger PNG files. Decoded pixels remain identical.
PNG_COMPRESSION = max(0, min(9, int(STEP_CONFIG.get("png_compression", 0))))

# OpenCV normally enables optimized kernels by default, but make the intent explicit.
cv2.setUseOptimized(True)

# Reuse one CLAHE instance instead of rebuilding it for every image.
CLAHE = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID_SIZE,
)


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_illumination_bgr(
    image_bgr: np.ndarray,
    *,
    preserve_input: bool = False,
) -> np.ndarray:
    """
    Normalize the LAB lightness channel with minimal temporary allocations.

    When preserve_input is False, image_bgr is reused as the destination buffer.
    This lowers peak memory without changing the returned pixel values.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Extract only L. Unlike cv2.split, this does not allocate copies of A and B.
    l_channel = cv2.extractChannel(lab, 0)

    # CLAHE supports an explicit destination; reuse the L-channel allocation.
    CLAHE.apply(l_channel, l_channel)

    # Replace only the normalized L channel in the existing LAB image.
    cv2.insertChannel(l_channel, lab, 0)

    if preserve_input:
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # The original image is no longer needed in production mode, so reuse it.
    cv2.cvtColor(lab, cv2.COLOR_LAB2BGR, dst=image_bgr)
    return image_bgr


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    return cv2.resize(
        image,
        (round(width * scale), max_height),
        interpolation=cv2.INTER_AREA,
    )


def make_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    # Both images have identical source dimensions, so one resize per image is enough.
    original_display = resize_for_display(original)
    processed_display = resize_for_display(processed)

    separator = np.full(
        (original_display.shape[0], 10, 3),
        255,
        dtype=np.uint8,
    )

    return np.concatenate(
        (original_display, separator, processed_display),
        axis=1,
    )


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )


def save_processed_image(output_path: Path, image: np.ndarray) -> None:
    params: list[int] = []

    if output_path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]

    saved = cv2.imwrite(str(output_path), image, params)
    if not saved:
        raise OSError(f"Could not write image: {output_path}")


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
        "read_time_ms",
        "processing_time_ms",
        "write_time_ms",
        "total_time_ms",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize image illumination using CLAHE."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show original and processed images in OpenCV windows.",
    )
    return parser.parse_args()


def main(*, debug: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    show_windows = debug
    wait_between_images = bool(DISPLAY_CONFIG.get("wait_between_images", True))
    image_count = len(image_paths)
    metadata_rows: list[dict] = []

    print()
    print("Processing step 01: illumination normalization")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"PNG compression: {PNG_COMPRESSION}")

    if show_windows:
        print()
        print("Controls:")
        print("  n / SPACE / ENTER  -> next image")
        print("  q / ESC            -> quit")

    for index, image_path in enumerate(image_paths, start=1):
        total_started = perf_counter()

        read_started = perf_counter()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        read_time_ms = (perf_counter() - read_started) * 1000.0

        if image_bgr is None:
            print(f"Could not read image: {image_path}")
            continue

        processing_started = perf_counter()
        normalized_bgr = normalize_illumination_bgr(
            image_bgr,
            preserve_input=show_windows,
        )
        processing_time_ms = (perf_counter() - processing_started) * 1000.0

        output_path = OUTPUT_DIR / image_path.name

        write_started = perf_counter()
        save_processed_image(output_path, normalized_bgr)
        write_time_ms = (perf_counter() - write_started) * 1000.0

        total_time_ms = (perf_counter() - total_started) * 1000.0
        height, width = normalized_bgr.shape[:2]

        metadata_rows.append({
            "source_file": relative_project_path(image_path),
            "output_file": relative_project_path(output_path),
            "width": width,
            "height": height,
            "processing_step": "01_illumination_normalization",
            "method": STEP_CONFIG["method"],
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid_size": CLAHE_TILE_GRID_SIZE_LABEL,
            "read_time_ms": round(read_time_ms, 3),
            "processing_time_ms": round(processing_time_ms, 3),
            "write_time_ms": round(write_time_ms, 3),
            "total_time_ms": round(total_time_ms, 3),
        })

        print(
            f"[{index}/{image_count}] Saved: {output_path.name} | "
            f"read={read_time_ms:.1f} ms, "
            f"process={processing_time_ms:.1f} ms, "
            f"write={write_time_ms:.1f} ms, "
            f"total={total_time_ms:.1f} ms"
        )

        if show_windows:
            comparison = make_side_by_side(image_bgr, normalized_bgr)
            title = (
                f"01 Illumination normalization | "
                f"{index}/{image_count} | {image_path.name}"
            )

            cv2.imshow(title, comparison)
            key = cv2.waitKey(0 if wait_between_images else 500) & 0xFF
            cv2.destroyWindow(title)

            if key in (ord("q"), 27):
                print("Stopped by user.")
                break

    save_metadata(metadata_rows)

    if show_windows:
        cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Processed images saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    args = parse_arguments()
    main(debug=args.debug)
