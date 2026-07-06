from pathlib import Path
import sys
import json
import argparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "pipeline"))

from config_loader import load_config


CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG.get("display", {})

WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]

STEP_CONFIG = CONFIG["step_04_detect_boot_roi"]

MAX_IMAGES_PER_WINDOW = 6


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_kernel_size(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    raise ValueError(f"Invalid kernel size: {value}")


def make_kernel(value) -> np.ndarray:
    kernel_size = normalize_kernel_size(value)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)


def resolve_input_dir() -> Path:
    input_subdir = STEP_CONFIG["input_subdir"]
    selected_input = STEP_CONFIG.get("selected_input")

    base_dir = PROCESSED_DIR / input_subdir

    if selected_input:
        selected_dir = base_dir / selected_input
        if selected_dir.exists():
            return selected_dir

    return base_dir


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def create_initial_mask(gray: np.ndarray, threshold_config: dict) -> np.ndarray:
    method = threshold_config.get("method", "adaptive")

    if method == "fixed":
        threshold_value = int(threshold_config.get("fixed_threshold", 120))
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    elif method == "otsu":
        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    elif method == "adaptive":
        adaptive_method_name = threshold_config.get("adaptive_method", "gaussian")
        block_size = int(threshold_config.get("adaptive_block_size", 51))
        c_value = int(threshold_config.get("adaptive_c", 3))

        if block_size % 2 == 0:
            block_size += 1

        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if adaptive_method_name == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )

        mask = cv2.adaptiveThreshold(
            gray,
            255,
            adaptive_method,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )

    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    if bool(threshold_config.get("invert", False)):
        mask = cv2.bitwise_not(mask)

    return mask


def build_boundary_masks(mask: np.ndarray, boundary_config: dict) -> dict[str, np.ndarray]:
    erode_kernel = make_kernel(boundary_config.get("erode_kernel_size", [3, 3]))
    erode_iterations = int(boundary_config.get("erode_iterations", 1))
    eroded = cv2.erode(mask, erode_kernel, iterations=erode_iterations)

    dilate_kernel = make_kernel(boundary_config.get("dilate_kernel_size", [3, 3]))
    dilate_iterations = int(boundary_config.get("dilate_iterations", 1))
    dilated = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)

    erosion_boundary = cv2.subtract(mask, eroded)
    dilation_boundary = cv2.subtract(dilated, mask)

    mode = str(boundary_config.get("mode", "erosion")).lower()
    if mode == "erosion":
        boundary = erosion_boundary
    elif mode == "dilation":
        boundary = dilation_boundary
    elif mode == "both":
        boundary = cv2.bitwise_or(erosion_boundary, dilation_boundary)
    else:
        raise ValueError(f"Unsupported step_04 boundary mode: {mode}")

    if bool(boundary_config.get("postprocess_enabled", True)):
        close_kernel = make_kernel(boundary_config.get("post_close_kernel_size", [9, 9]))
        close_iterations = int(boundary_config.get("post_close_iterations", 1))
        boundary = cv2.morphologyEx(
            boundary,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iterations,
        )

        dilate_kernel = make_kernel(boundary_config.get("post_dilate_kernel_size", [5, 5]))
        dilate_iterations = int(boundary_config.get("post_dilate_iterations", 1))
        boundary = cv2.dilate(boundary, dilate_kernel, iterations=dilate_iterations)

    return {
        "eroded": eroded,
        "dilated": dilated,
        "erosion_boundary": erosion_boundary,
        "dilation_boundary": dilation_boundary,
        "boundary": boundary,
    }


def contour_score(contour, image_shape: tuple[int, int], contour_filter_config: dict) -> float:
    height, width = image_shape[:2]
    image_area = height * width

    area = cv2.contourArea(contour)
    if area <= 0:
        return -1.0

    min_area = float(contour_filter_config.get("min_area_ratio", 0.015)) * image_area
    max_area = float(contour_filter_config.get("max_area_ratio", 0.65)) * image_area

    if area < min_area or area > max_area:
        return -1.0

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return -1.0

    center_x = x + w / 2
    center_y = y + h / 2

    center_distance = abs(center_x - width / 2) / (width / 2)
    center_score_value = 1.0 - min(center_distance, 1.0)

    vertical_score_value = min(h / max(w, 1), 5.0) / 5.0
    area_score_value = min(area / image_area, 1.0)

    lower_half_score = 0.0
    if bool(contour_filter_config.get("prefer_lower_half", True)):
        lower_half_score = center_y / height

    center_weight = float(contour_filter_config.get("center_weight", 2.0))
    vertical_weight = float(contour_filter_config.get("vertical_weight", 1.5))

    score = (
        area_score_value
        + center_weight * center_score_value
        + vertical_weight * vertical_score_value
        + lower_half_score
    )

    return float(score)


def select_best_contour(mask: np.ndarray, contour_filter_config: dict):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, []

    scored_contours = []
    for contour in contours:
        score = contour_score(contour, mask.shape, contour_filter_config)
        if score >= 0:
            scored_contours.append((score, contour))

    if not scored_contours:
        return None, []

    scored_contours.sort(key=lambda item: item[0], reverse=True)
    return scored_contours[0][1], scored_contours


def build_final_mask(
    boundary_mask: np.ndarray,
    contour_filter_config: dict,
    mask_config: dict,
) -> tuple[np.ndarray, dict]:
    best_contour, scored_contours = select_best_contour(boundary_mask, contour_filter_config)

    final_mask = np.zeros_like(boundary_mask)

    metadata = {
        "contours_found": len(scored_contours),
        "selected_contour_area": None,
        "selected_bounding_rect": None,
    }

    if best_contour is None:
        return final_mask, metadata

    selected_contour = best_contour

    if bool(mask_config.get("convex_hull", True)):
        selected_contour = cv2.convexHull(selected_contour)

    if bool(mask_config.get("fill_contour", True)):
        cv2.drawContours(final_mask, [selected_contour], -1, 255, thickness=cv2.FILLED)
    else:
        cv2.drawContours(final_mask, [selected_contour], -1, 255, thickness=2)

    final_close_kernel = make_kernel(mask_config.get("final_close_kernel_size", [31, 31]))
    final_close_iterations = int(mask_config.get("final_close_iterations", 1))

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        final_close_kernel,
        iterations=final_close_iterations,
    )

    x, y, w, h = cv2.boundingRect(selected_contour)

    metadata["selected_contour_area"] = float(cv2.contourArea(selected_contour))
    metadata["selected_bounding_rect"] = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
    }

    return final_mask, metadata


def make_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    green = np.zeros_like(base)
    green[:, :, 1] = 255

    overlay = base.copy()
    mask_bool = mask > 0

    overlay[mask_bool] = cv2.addWeighted(
        base[mask_bool],
        0.55,
        green[mask_bool],
        0.45,
        0,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def process_image(image_path: Path, output_dirs: dict[str, Path]) -> dict:
    gray = load_grayscale_image(image_path)

    initial_mask = create_initial_mask(gray, STEP_CONFIG["threshold"])
    boundary_results = build_boundary_masks(initial_mask, STEP_CONFIG["boundary"])
    boundary_mask = boundary_results["boundary"]
    final_mask, metadata = build_final_mask(
        boundary_mask,
        STEP_CONFIG["contour_filter"],
        STEP_CONFIG["mask"],
    )

    overlay = make_overlay(gray, final_mask)

    stem = image_path.stem

    cv2.imwrite(str(output_dirs["initial_mask"] / f"{stem}_initial_mask.png"), initial_mask)
    cv2.imwrite(str(output_dirs["clean_mask"] / f"{stem}_clean_mask.png"), boundary_mask)
    cv2.imwrite(str(output_dirs["final_mask"] / f"{stem}_final_mask.png"), final_mask)
    cv2.imwrite(str(output_dirs["overlay"] / f"{stem}_mask_overlay.png"), overlay)

    metadata.update(
        {
            "image": image_path.name,
            "input_path": str(image_path),
            "mask_nonzero_pixels": int(np.count_nonzero(final_mask)),
            "mask_area_ratio": float(np.count_nonzero(final_mask) / final_mask.size),
        }
    )

    if STEP_CONFIG.get("debug", {}).get("save_metadata", True):
        with open(output_dirs["metadata"] / f"{stem}_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    return metadata


def collect_images(input_dir: Path, image_name: str | None = None) -> list[Path]:
    valid_extensions = {".png", ".jpg", ".jpeg"}

    if image_name:
        candidate_stems = [image_name]

        if image_name.isdigit():
            candidate_stems.append(f"IMG_{int(image_name):04d}")

        for stem in candidate_stems:
            for extension in valid_extensions:
                candidate = input_dir / f"{stem}{extension}"
                if candidate.exists():
                    return [candidate]

        raise FileNotFoundError(f"Image {image_name} not found in {input_dir}")

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in valid_extensions
    )


def build_output_dirs() -> dict[str, Path]:
    output_root = PROCESSED_DIR / STEP_CONFIG["output_subdir"]

    output_dirs = {
        "root": output_root,
        "initial_mask": output_root / "initial_mask",
        "clean_mask": output_root / "clean_mask",
        "final_mask": output_root / "final_mask",
        "overlay": output_root / "mask_overlay",
        "metadata": output_root / "metadata",
    }

    for path in output_dirs.values():
        ensure_dir(path)

    return output_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Optional image stem, for example 504")
    args = parser.parse_args()

    if not STEP_CONFIG.get("enabled", True):
        print("Step 04 is disabled in config.")
        return

    input_dir = resolve_input_dir()
    output_dirs = build_output_dirs()

    image_paths = collect_images(input_dir, args.image)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    all_metadata = []

    for image_path in image_paths:
        print(f"Processing {image_path.name}")
        metadata = process_image(image_path, output_dirs)
        all_metadata.append(metadata)

    summary_path = output_dirs["metadata"] / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_metadata, file, indent=2)

    print(f"Done. Output saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()
