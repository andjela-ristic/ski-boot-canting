from pathlib import Path
import sys
import argparse
import itertools

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "pipeline"))

from config_loader import load_config


CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]

WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]


MAX_IMAGES_PER_WINDOW = 6


def resolve_image_path(step: int, image_name: str) -> Path:
    if step == 1:
        image_path = WORKING_PNG_DIR / image_name

    elif step == 2:
        step_config = CONFIG["step_02_grayscale_and_blur"]
        image_path = PROCESSED_DIR / step_config["input_subdir"] / image_name

    else:
        raise ValueError(f"Unsupported step: {step}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def resize_for_display(image: np.ndarray, max_height: int = 520) -> np.ndarray:
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
    image = to_bgr(image).copy()

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 52),
        (0, 0, 0),
        thickness=-1
    )

    cv2.putText(
        image,
        label,
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


def pad_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    padded = np.full((height, width, 3), 255, dtype=np.uint8)

    image_height, image_width = image.shape[:2]

    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    padded[
        y_offset:y_offset + image_height,
        x_offset:x_offset + image_width
    ] = image

    return padded


def make_grid(images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    prepared_images = []

    for label, image in images:
        display_image = resize_for_display(to_bgr(image))
        display_image = add_label(display_image, label)
        prepared_images.append(display_image)

    max_width = max(image.shape[1] for image in prepared_images)
    max_height = max(image.shape[0] for image in prepared_images)

    padded_images = [
        pad_to_size(image, max_width, max_height)
        for image in prepared_images
    ]

    while len(padded_images) < MAX_IMAGES_PER_WINDOW:
        empty = np.full((max_height, max_width, 3), 245, dtype=np.uint8)
        padded_images.append(empty)

    row_1 = np.hstack(padded_images[:3])
    row_2 = np.hstack(padded_images[3:6])

    grid = np.vstack([row_1, row_2])

    return grid


def show_variation_pages(
    title_prefix: str,
    results: list[tuple[str, np.ndarray]]
) -> None:
    if not results:
        print("No results to display.")
        return

    total_pages = (len(results) + MAX_IMAGES_PER_WINDOW - 1) // MAX_IMAGES_PER_WINDOW

    page_index = 0

    while page_index < total_pages:
        start = page_index * MAX_IMAGES_PER_WINDOW
        end = start + MAX_IMAGES_PER_WINDOW

        page_results = results[start:end]
        grid = make_grid(page_results)

        title = f"{title_prefix} | page {page_index + 1}/{total_pages}"

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, grid)

        print(f"Showing page {page_index + 1}/{total_pages}")
        print("SPACE / ENTER / n -> next page")
        print("b / LEFT          -> previous page")
        print("q / ESC           -> quit")
        print()

        key = cv2.waitKey(0) & 0xFF

        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass

        if key in [ord("q"), 27]:
            break

        if key in [ord("b"), 81]:
            page_index = max(0, page_index - 1)
        else:
            page_index += 1

def normalize_illumination_bgr(
    image_bgr: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int]
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

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


def run_step_01_variations(image_path: Path) -> None:
    step_config = CONFIG["step_01_illumination_normalization"]
    clahe_config = step_config["clahe"]

    clip_values = clahe_config["clip_limit_test_values"]
    tile_values = clahe_config["tile_grid_size_test_values"]

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = []

    for clip_limit, tile_grid_size_list in itertools.product(clip_values, tile_values):
        tile_grid_size = tuple(tile_grid_size_list)

        processed = normalize_illumination_bgr(
            image_bgr=image_bgr,
            clip_limit=float(clip_limit),
            tile_grid_size=tile_grid_size
        )

        tile_label = f"{tile_grid_size[0]}x{tile_grid_size[1]}"
        label = f"clip={clip_limit}, tile={tile_label}"

        results.append((label, processed))

    show_variation_pages(
        title_prefix=f"Step 01 variations | {image_path.name}",
        results=results
    )


def ensure_odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError(f"Kernel size must be positive. Got: {value}")

    if value % 2 == 0:
        raise ValueError(f"Kernel size must be odd. Got: {value}")

    return value


def run_step_02_variations(image_path: Path) -> None:
    step_config = CONFIG["step_02_grayscale_and_blur"]

    gaussian_config = step_config["gaussian_blur"]
    bilateral_config = step_config["bilateral_filter"]

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gaussian_results = [("grayscale", grayscale)]

    for kernel_size, sigma_x in itertools.product(
        gaussian_config["kernel_size_test_values"],
        gaussian_config["sigma_x_test_values"]
    ):
        kernel_size = ensure_odd_kernel_size(int(kernel_size))
        sigma_x = float(sigma_x)

        gaussian = cv2.GaussianBlur(
            grayscale,
            (kernel_size, kernel_size),
            sigma_x
        )

        label = f"gaussian k={kernel_size}, sigma={sigma_x}"
        gaussian_results.append((label, gaussian))

    bilateral_results = [("grayscale", grayscale)]

    for diameter, sigma_color, sigma_space in itertools.product(
        bilateral_config["diameter_test_values"],
        bilateral_config["sigma_color_test_values"],
        bilateral_config["sigma_space_test_values"]
    ):
        diameter = int(diameter)
        sigma_color = float(sigma_color)
        sigma_space = float(sigma_space)

        bilateral = cv2.bilateralFilter(
            grayscale,
            diameter,
            sigma_color,
            sigma_space
        )

        label = f"bilateral d={diameter}, sc={sigma_color}, ss={sigma_space}"
        bilateral_results.append((label, bilateral))

    show_variation_pages(
        title_prefix=f"Step 02 Gaussian variations | {image_path.name}",
        results=gaussian_results
    )

    show_variation_pages(
        title_prefix=f"Step 02 Bilateral variations | {image_path.name}",
        results=bilateral_results
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visually test all configured parameter variations for one step and one image."
    )

    parser.add_argument(
        "--step",
        type=int,
        required=True,
        choices=[1, 2],
        help="Pipeline step to test. Supported: 1, 2."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image filename, for example IMG_0502.png."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = resolve_image_path(
        step=args.step,
        image_name=args.image
    )

    print()
    print("Running visual test variations")
    print(f"Step:  {args.step}")
    print(f"Image: {image_path}")
    print("No files will be saved.")
    print()

    if args.step == 1:
        run_step_01_variations(image_path)

    elif args.step == 2:
        run_step_02_variations(image_path)

    cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()