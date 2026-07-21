from __future__ import annotations

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_01_CONFIG = CONFIG["step_01_illumination_normalization"]
STEP_CONFIG = CONFIG["step_02_grayscale_and_blur"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

if bool(STEP_CONFIG.get("inherit_step_01_output", True)):
    INPUT_DIR = PROCESSED_DIR / STEP_01_CONFIG["output_subdir"]
else:
    INPUT_DIR = PROCESSED_DIR / STEP_CONFIG["input_subdir"]

OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

# Active outputs.
# - Step 03 reads grayscale_lab_l.
# - The API pipeline runner currently validates that bilateral_filter exists.
GRAYSCALE_LAB_L_DIR = OUTPUT_DIR / "grayscale_lab_l"
BILATERAL_DIR = OUTPUT_DIR / "bilateral_filter"

# Disabled outputs: they are not consumed by the current pipeline.
# GRAYSCALE_DIR = OUTPUT_DIR / "grayscale"
# GRAYSCALE_BGR2GRAY_DIR = OUTPUT_DIR / "grayscale_bgr2gray"
# GRAYSCALE_YCRCB_Y_DIR = OUTPUT_DIR / "grayscale_ycrcb_y"
# GAUSSIAN_DIR = OUTPUT_DIR / "gaussian_blur"

CSV_PATH = METADATA_DIR / "processing_02_grayscale_and_blur.csv"

# PNG remains lossless. Compression level 1 reduces CPU time at the cost of
# somewhat larger intermediate files.
PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    if not INPUT_DIR.exists():
        return []

    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])
    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_bgr_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
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


def make_comparison_view(
    original_bgr: np.ndarray,
    grayscale_lab_l: np.ndarray,
    bilateral: np.ndarray,
) -> np.ndarray:
    images = [
        resize_for_display(original_bgr),
        resize_for_display(to_bgr_for_display(grayscale_lab_l)),
        resize_for_display(to_bgr_for_display(bilateral)),
    ]

    target_height = min(image.shape[0] for image in images)
    resized_images: list[np.ndarray] = []

    for image in images:
        height, width = image.shape[:2]

        if height == target_height:
            resized_images.append(image)
            continue

        new_width = int(width * target_height / height)
        resized_images.append(
            cv2.resize(
                image,
                (new_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        )

    labeled_images = [
        add_label(resized_images[0], "01 normalized"),
        add_label(resized_images[1], "grayscale: lab_l"),
        add_label(resized_images[2], "bilateral filter"),
    ]

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    return np.hstack(
        [
            labeled_images[0],
            separator,
            labeled_images[1],
            separator,
            labeled_images[2],
        ]
    )


def convert_to_lab_l(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to the LAB L channel without allocating all split channels."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    grayscale_lab_l = cv2.extractChannel(lab, 0)
    del lab
    return grayscale_lab_l


# Disabled conversions: the current pipeline does not consume these outputs.
#
# def convert_to_bgr2gray(image_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#
#
# def convert_to_ycrcb_y(image_bgr: np.ndarray) -> np.ndarray:
#     ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
#     y_channel = cv2.extractChannel(ycrcb, 0)
#     del ycrcb
#     return y_channel


def build_bilateral(image_bgr: np.ndarray) -> np.ndarray:
    """Preserve the existing bilateral output semantics: filter BGR2GRAY."""
    bilateral_config = STEP_CONFIG["bilateral_filter"]

    grayscale_bgr2gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(
        grayscale_bgr2gray,
        int(bilateral_config["diameter"]),
        float(bilateral_config["sigma_color"]),
        float(bilateral_config["sigma_space"]),
    )
    del grayscale_bgr2gray

    return bilateral


# Disabled Gaussian processing: Step 03 currently reads grayscale_lab_l directly.
#
# def build_gaussian(grayscale: np.ndarray) -> np.ndarray:
#     gaussian_config = STEP_CONFIG["gaussian_blur"]
#     kernel_size = int(gaussian_config["kernel_size"])
#     sigma_x = float(gaussian_config["sigma_x"])
#     return cv2.GaussianBlur(
#         grayscale,
#         (kernel_size, kernel_size),
#         sigma_x,
#     )


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, PNG_WRITE_PARAMS):
        raise RuntimeError(f"Could not write image: {path}")


def save_metadata(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the previous CSV schema for compatibility. Disabled output columns
    # are intentionally written as empty strings.
    fieldnames = [
        "source_file",
        "grayscale_file",
        "grayscale_bgr2gray_file",
        "grayscale_lab_l_file",
        "grayscale_ycrcb_y_file",
        "gaussian_file",
        "bilateral_file",
        "width",
        "height",
        "processing_step",
        "grayscale_method",
        "gaussian_kernel_size",
        "gaussian_sigma_x",
        "bilateral_diameter",
        "bilateral_sigma_color",
        "bilateral_sigma_space",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 02 grayscale and blur processing."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show comparison windows while processing images.",
    )
    return parser.parse_args()


def main(debug: bool = False) -> None:
    if not STEP_CONFIG["enabled"]:
        print("Step 02 is disabled in config.")
        return

    GRAYSCALE_LAB_L_DIR.mkdir(parents=True, exist_ok=True)
    BILATERAL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Disabled directories are intentionally not created.
    # GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
    # GRAYSCALE_BGR2GRAY_DIR.mkdir(parents=True, exist_ok=True)
    # GRAYSCALE_YCRCB_Y_DIR.mkdir(parents=True, exist_ok=True)
    # GAUSSIAN_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    bilateral_config = STEP_CONFIG["bilateral_filter"]
    metadata_rows: list[dict[str, object]] = []
    show_windows = debug
    wait_between_images = bool(DISPLAY_CONFIG.get("wait_between_images", False))

    print()
    print("Processing step 02: active grayscale and blur outputs")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("Active outputs: grayscale_lab_l, bilateral_filter")

    if show_windows:
        print()
        print("Controls:")
        print("  n / SPACE / ENTER  -> next image")
        print("  q / ESC            -> quit")

    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            print(f"Could not read image: {image_path}")
            continue

        height, width = image_bgr.shape[:2]

        grayscale_lab_l = convert_to_lab_l(image_bgr)
        grayscale_lab_l_path = GRAYSCALE_LAB_L_DIR / image_path.name
        write_image(grayscale_lab_l_path, grayscale_lab_l)

        bilateral = build_bilateral(image_bgr)
        bilateral_path = BILATERAL_DIR / image_path.name
        write_image(bilateral_path, bilateral)

        metadata_rows.append(
            {
                "source_file": relative_project_path(image_path),
                "grayscale_file": "",
                "grayscale_bgr2gray_file": "",
                "grayscale_lab_l_file": relative_project_path(grayscale_lab_l_path),
                "grayscale_ycrcb_y_file": "",
                "gaussian_file": "",
                "bilateral_file": relative_project_path(bilateral_path),
                "width": width,
                "height": height,
                "processing_step": "02_grayscale_and_blur",
                "grayscale_method": "lab_l",
                "gaussian_kernel_size": "",
                "gaussian_sigma_x": "",
                "bilateral_diameter": int(bilateral_config["diameter"]),
                "bilateral_sigma_color": float(bilateral_config["sigma_color"]),
                "bilateral_sigma_space": float(bilateral_config["sigma_space"]),
            }
        )

        print(f"[{index}/{len(image_paths)}] Saved: {image_path.name}")

        if show_windows:
            comparison = make_comparison_view(
                image_bgr,
                grayscale_lab_l,
                bilateral,
            )
            title = (
                "02 Grayscale and blur | "
                f"{index}/{len(image_paths)} | {image_path.name}"
            )

            cv2.imshow(title, comparison)
            key = cv2.waitKey(0 if wait_between_images else 500) & 0xFF
            cv2.destroyWindow(title)

            del comparison

            if key in (ord("q"), 27):
                print("Stopped by user.")
                del grayscale_lab_l, bilateral, image_bgr
                break

        # Release full-resolution arrays before reading the next image.
        del grayscale_lab_l, bilateral, image_bgr

    save_metadata(metadata_rows)
    if show_windows:
        cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Grayscale LAB L saved to: {GRAYSCALE_LAB_L_DIR}")
    print(f"Bilateral filter saved to: {BILATERAL_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    args = parse_args()
    main(debug=args.debug)
