from pathlib import Path
import sys
import json
import argparse
import itertools
import copy
import math

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "pipeline"))

from config_loader import load_config


CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
TEST_DIR = PROJECT_ROOT / "data" / "test"

BASE_STEP_CONFIG = CONFIG["step_04_detect_boot_roi"]


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


def resolve_input_dir(step_config: dict) -> Path:
    input_subdir = step_config["input_subdir"]
    selected_input = step_config.get("selected_input")

    base_dir = PROCESSED_DIR / input_subdir

    if selected_input:
        selected_dir = base_dir / selected_input
        if selected_dir.exists():
            return selected_dir

    return base_dir


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


def set_nested_value(config: dict, path: tuple[str, ...], value) -> None:
    current = config

    for key in path[:-1]:
        current = current[key]

    current[path[-1]] = value


def collect_test_values(config: dict) -> list[tuple[tuple[str, ...], str, list]]:
    test_values = []

    def walk(node, path=()):
        if not isinstance(node, dict):
            return

        for key, value in node.items():
            if key.endswith("_test"):
                target_key = key.removesuffix("_test")
                target_path = path + (target_key,)
                test_values.append((target_path, key, value))
            elif isinstance(value, dict):
                walk(value, path + (key,))

    walk(config)
    return test_values


def count_config_combinations(base_config: dict) -> int:
    test_values = collect_test_values(base_config)

    if not test_values:
        return 1

    return math.prod(len(item[2]) for item in test_values)


def iter_config_combinations(
    base_config: dict,
    start_index: int = 0,
    end_index: int | None = None,
):
    test_values = collect_test_values(base_config)

    if not test_values:
        if start_index <= 0 and (end_index is None or end_index >= 0):
            yield 0, copy.deepcopy(base_config), {}
        return

    paths = [item[0] for item in test_values]
    names = [".".join(item[0]) for item in test_values]
    value_lists = [item[2] for item in test_values]

    stop_index = None if end_index is None else end_index + 1
    value_iterator = itertools.islice(
        itertools.product(*value_lists),
        start_index,
        stop_index,
    )

    for index, values in enumerate(value_iterator, start=start_index):
        config = copy.deepcopy(base_config)
        params = {}

        for path, name, value in zip(paths, names, values):
            set_nested_value(config, path, value)
            params[name] = value

        yield index, config, params


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
    eroded = cv2.erode(
        mask,
        make_kernel(boundary_config.get("erode_kernel_size", [3, 3])),
        iterations=int(boundary_config.get("erode_iterations", 1)),
    )
    dilated = cv2.dilate(
        mask,
        make_kernel(boundary_config.get("dilate_kernel_size", [3, 3])),
        iterations=int(boundary_config.get("dilate_iterations", 1)),
    )

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
        boundary = cv2.morphologyEx(
            boundary,
            cv2.MORPH_CLOSE,
            make_kernel(boundary_config.get("post_close_kernel_size", [9, 9])),
            iterations=int(boundary_config.get("post_close_iterations", 1)),
        )
        boundary = cv2.dilate(
            boundary,
            make_kernel(boundary_config.get("post_dilate_kernel_size", [5, 5])),
            iterations=int(boundary_config.get("post_dilate_iterations", 1)),
        )

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

    return float(
        area_score_value
        + center_weight * center_score_value
        + vertical_weight * vertical_score_value
        + lower_half_score
    )


def select_best_contour(mask: np.ndarray, contour_filter_config: dict):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    scored = []
    for contour in contours:
        score = contour_score(contour, mask.shape, contour_filter_config)
        if score >= 0:
            scored.append((score, contour))

    if not scored:
        return None, []

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored


def build_final_mask(boundary_mask: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    contour_filter_config = config["contour_filter"]
    mask_config = config["mask"]

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

    cv2.drawContours(final_mask, [selected_contour], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        make_kernel(mask_config.get("final_close_kernel_size", [31, 31])),
        iterations=int(mask_config.get("final_close_iterations", 1)),
    )

    x, y, w, h = cv2.boundingRect(selected_contour)
    metadata["selected_contour_area"] = float(cv2.contourArea(selected_contour))
    metadata["selected_bounding_rect"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    return final_mask, metadata


def make_green_overlay(gray: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    green = np.zeros_like(base)
    green[:, :, 1] = 255

    overlay = base.copy()
    mask_bool = mask > 0

    if np.any(mask_bool):
        overlay[mask_bool] = cv2.addWeighted(base[mask_bool], 0.55, green[mask_bool], 0.45, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def add_label_to_image(img: np.ndarray, text: str) -> np.ndarray:
    """Pomoćna funkcija za dodavanje teksta u gornji levi ugao pod-slike."""
    res = img.copy()
    cv2.putText(res, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if len(img.shape)==2 or img.shape[2]==1 else (0, 255, 0), 2, cv2.LINE_AA)
    return res


def process_with_config(image_path: Path, config: dict, combination_index: int) -> tuple[np.ndarray, dict]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Koraci obrade
    initial_mask = create_initial_mask(gray, config["threshold"])
    boundary_results = build_boundary_masks(initial_mask, config["boundary"])
    final_mask, metadata = build_final_mask(boundary_results["boundary"], config)

    mask_area_ratio = float(np.count_nonzero(final_mask) / final_mask.size)
    metadata.update({
        "image": image_path.name,
        "combination_index": combination_index,
        "mask_nonzero_pixels": int(np.count_nonzero(final_mask)),
        "mask_area_ratio": mask_area_ratio,
    })

    # Priprema pod-slika za kolaž (sve moraju biti BGR formata iste veličine)
    img_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_eroded = cv2.cvtColor(boundary_results["eroded"], cv2.COLOR_GRAY2BGR)
    img_dilated = cv2.cvtColor(boundary_results["dilated"], cv2.COLOR_GRAY2BGR)
    
    label_combo = f"combo {combination_index}"
    img_final_overlay = make_green_overlay(gray, final_mask, label_combo)

    # Dodavanje oznaka na svaku fazu
    view_1 = add_label_to_image(img_gray_bgr, "1. Original")
    view_2 = add_label_to_image(img_eroded, "2. Erozija")
    view_3 = add_label_to_image(img_dilated, "3. Dilacija")
    view_4 = img_final_overlay  # Već ima combo oznaku na sebi

    # Kreiranje 2x2 kolaža spajanjem matrica
    top_row = np.hstack((view_1, view_2))
    bottom_row = np.hstack((view_3, view_4))
    collage = np.vstack((top_row, bottom_row))

    return collage, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Optional image stem, for example 504")
    parser.add_argument("--start", type=int, default=None, help="First combination index to process")
    parser.add_argument("--end", type=int, default=None, help="Last combination index to process, inclusive")
    args = parser.parse_args()

    input_dir = resolve_input_dir(BASE_STEP_CONFIG)
    image_paths = collect_images(input_dir, args.image)

    print("Loading configuration...", flush=True)

    total_combinations = count_config_combinations(BASE_STEP_CONFIG)
    start_index = args.start if args.start is not None else 0
    end_index = args.end if args.end is not None else total_combinations - 1

    if start_index < 0:
        raise ValueError("--start must be >= 0")
    if end_index < start_index:
        raise ValueError("--end must be >= --start")
    if start_index >= total_combinations:
        raise ValueError(f"--start {start_index} is outside the available range 0..{total_combinations - 1}")

    end_index = min(end_index, total_combinations - 1)
    selected_count = end_index - start_index + 1

    output_root = TEST_DIR / "14_test_boot_roi_mask_combinations"
    overlay_root = output_root / "overlays"
    metadata_root = output_root / "metadata"

    ensure_dir(overlay_root)
    ensure_dir(metadata_root)

    print(f"Images: {len(image_paths)}", flush=True)
    print(f"Ukupno mogucih kombinacija: {total_combinations}", flush=True)
    print(f"Selected combinations: {selected_count} (indexes {start_index}..{end_index})", flush=True)
    print(f"Output: {output_root}", flush=True)

    summary = []

    for image_path in image_paths:
        image_overlay_dir = overlay_root / image_path.stem
        ensure_dir(image_overlay_dir)

        for combination_index, config, params in iter_config_combinations(
            BASE_STEP_CONFIG,
            start_index=start_index,
            end_index=end_index,
        ):
            print(f"Image {image_path.stem} | combo {combination_index}", flush=True)

            collage, metadata = process_with_config(
                image_path=image_path,
                config=config,
                combination_index=combination_index,
            )

            output_name = f"{image_path.stem}_combo_{combination_index:05d}.png"
            cv2.imwrite(str(image_overlay_dir / output_name), collage)

            metadata["params"] = params
            summary.append(metadata)

    summary_path = metadata_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
