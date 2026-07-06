from pathlib import Path
import csv

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_CONFIG = CONFIG["step_02_grayscale_and_blur"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

INPUT_DIR = PROCESSED_DIR / STEP_CONFIG["input_subdir"]
OUTPUT_DIR = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

GRAYSCALE_DIR = OUTPUT_DIR / "grayscale"
GAUSSIAN_DIR = OUTPUT_DIR / "gaussian_blur"
BILATERAL_DIR = OUTPUT_DIR / "bilateral_filter"

CSV_PATH = METADATA_DIR / "processing_02_grayscale_and_blur.csv"


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


def make_comparison_view(
    original_bgr: np.ndarray,
    grayscale: np.ndarray,
    gaussian: np.ndarray,
    bilateral: np.ndarray,
) -> np.ndarray:
    original_display = resize_for_display(original_bgr)
    grayscale_display = resize_for_display(to_bgr_for_display(grayscale))
    gaussian_display = resize_for_display(to_bgr_for_display(gaussian))
    bilateral_display = resize_for_display(to_bgr_for_display(bilateral))

    target_height = min(
        original_display.shape[0],
        grayscale_display.shape[0],
        gaussian_display.shape[0],
        bilateral_display.shape[0],
    )

    images = [
        original_display,
        grayscale_display,
        gaussian_display,
        bilateral_display,
    ]

    resized_images = []

    for image in images:
        height, width = image.shape[:2]
        new_width = int(width * target_height / height)

        resized = cv2.resize(
            image,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        resized_images.append(resized)

    grayscale_method = get_grayscale_method()

    labeled_images = [
        add_label(resized_images[0], "01 normalized"),
        add_label(resized_images[1], f"grayscale: {grayscale_method}"),
        add_label(resized_images[2], "gaussian blur"),
        add_label(resized_images[3], "bilateral filter"),
    ]

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)

    combined = labeled_images[0]

    for image in labeled_images[1:]:
        combined = np.hstack([combined, separator, image])

    return combined


def ensure_odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError(f"Kernel size must be positive. Got: {value}")

    if value % 2 == 0:
        raise ValueError(f"Kernel size must be odd. Got: {value}")

    return value


def get_grayscale_method() -> str:
    grayscale_config = STEP_CONFIG.get("grayscale", {})
    method = grayscale_config.get("method", "bgr2gray")

    return str(method).strip().lower()


def convert_to_grayscale(image_bgr: np.ndarray, method: str) -> np.ndarray:
    """
    Supported methods:

    bgr2gray:
        Standard OpenCV luminance-weighted grayscale conversion.
        This is NOT a simple RGB average.

    lab_l:
        Uses the L channel from LAB color space.
        Good for geometry/edge detection because it isolates luminance.

    ycrcb_y:
        Uses the Y channel from YCrCb color space.
        Another luminance-based alternative.
    """

    if method == "bgr2gray":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if method == "lab_l":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        return l_channel

    if method == "ycrcb_y":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel, _, _ = cv2.split(ycrcb)
        return y_channel

    raise ValueError(
        "Unsupported grayscale method: "
        f"{method}. Supported: bgr2gray, lab_l, ycrcb_y"
    )


def process_image(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grayscale_method = get_grayscale_method()
    grayscale = convert_to_grayscale(image_bgr, grayscale_method)

    gaussian_config = STEP_CONFIG["gaussian_blur"]
    gaussian_kernel_size = ensure_odd_kernel_size(
        int(gaussian_config["kernel_size"])
    )
    gaussian_sigma_x = float(gaussian_config["sigma_x"])

    gaussian = cv2.GaussianBlur(
        grayscale,
        (gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma_x,
    )

    bilateral_config = STEP_CONFIG["bilateral_filter"]
    bilateral_diameter = int(bilateral_config["diameter"])
    bilateral_sigma_color = float(bilateral_config["sigma_color"])
    bilateral_sigma_space = float(bilateral_config["sigma_space"])

    bilateral = cv2.bilateralFilter(
        grayscale,
        bilateral_diameter,
        bilateral_sigma_color,
        bilateral_sigma_space,
    )

    return grayscale, gaussian, bilateral


def save_metadata(rows: list[dict]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "grayscale_file",
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

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not STEP_CONFIG["enabled"]:
        print("Step 02 is disabled in config.")
        return

    GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
    GAUSSIAN_DIR.mkdir(parents=True, exist_ok=True)
    BILATERAL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    grayscale_method = get_grayscale_method()

    gaussian_config = STEP_CONFIG["gaussian_blur"]
    bilateral_config = STEP_CONFIG["bilateral_filter"]

    gaussian_kernel_size = ensure_odd_kernel_size(
        int(gaussian_config["kernel_size"])
    )
    gaussian_sigma_x = float(gaussian_config["sigma_x"])

    bilateral_diameter = int(bilateral_config["diameter"])
    bilateral_sigma_color = float(bilateral_config["sigma_color"])
    bilateral_sigma_space = float(bilateral_config["sigma_space"])

    metadata_rows = []

    print()
    print("Processing step 02: grayscale and blur")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Grayscale method: {grayscale_method}")
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

        grayscale, gaussian, bilateral = process_image(image_bgr)

        grayscale_output_path = GRAYSCALE_DIR / image_path.name
        gaussian_output_path = GAUSSIAN_DIR / image_path.name
        bilateral_output_path = BILATERAL_DIR / image_path.name

        cv2.imwrite(str(grayscale_output_path), grayscale)
        cv2.imwrite(str(gaussian_output_path), gaussian)
        cv2.imwrite(str(bilateral_output_path), bilateral)

        height, width = image_bgr.shape[:2]

        metadata_rows.append({
            "source_file": str(image_path.relative_to(PROJECT_ROOT)),
            "grayscale_file": str(grayscale_output_path.relative_to(PROJECT_ROOT)),
            "gaussian_file": str(gaussian_output_path.relative_to(PROJECT_ROOT)),
            "bilateral_file": str(bilateral_output_path.relative_to(PROJECT_ROOT)),
            "width": width,
            "height": height,
            "processing_step": "02_grayscale_and_blur",
            "grayscale_method": grayscale_method,
            "gaussian_kernel_size": gaussian_kernel_size,
            "gaussian_sigma_x": gaussian_sigma_x,
            "bilateral_diameter": bilateral_diameter,
            "bilateral_sigma_color": bilateral_sigma_color,
            "bilateral_sigma_space": bilateral_sigma_space,
        })

        print(f"[{index}/{len(image_paths)}] Saved: {image_path.name}")

        if DISPLAY_CONFIG["show_windows"]:
            comparison = make_comparison_view(
                image_bgr,
                grayscale,
                gaussian,
                bilateral,
            )

            title = (
                f"02 Grayscale and blur | "
                f"{index}/{len(image_paths)} | "
                f"{image_path.name} | "
                f"{grayscale_method}"
            )

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
    print(f"Grayscale method: {grayscale_method}")
    print(f"Grayscale saved to: {GRAYSCALE_DIR}")
    print(f"Gaussian blur saved to: {GAUSSIAN_DIR}")
    print(f"Bilateral filter saved to: {BILATERAL_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()