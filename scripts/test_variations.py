from pathlib import Path
import sys
import argparse
import itertools
import re

import cv2
import numpy as np
import math
import copy
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "pipeline"))

from config_loader import load_config


CONFIG = load_config()
PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]


MAX_IMAGES_PER_WINDOW = 6
MAX_PAGE_WIDTH = 1600
SAVED_TEST_WINDOWS_DIR = PROJECT_ROOT / "data" / "test"


def set_config(config_path: str | None) -> None:
    global CONFIG, PATHS_CONFIG, DISPLAY_CONFIG, WORKING_PNG_DIR, PROCESSED_DIR

    CONFIG = load_config(config_path)
    PATHS_CONFIG = CONFIG["paths"]
    DISPLAY_CONFIG = CONFIG["display"]
    WORKING_PNG_DIR = PROJECT_ROOT / PATHS_CONFIG["working_png_dir"]
    PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]


def resolve_image_path(step: int, image_name: str) -> Path:
    if step == 1:
        image_path = WORKING_PNG_DIR / image_name

    elif step == 2:
        step_config = CONFIG["step_02_grayscale_and_blur"]
        image_path = PROCESSED_DIR / step_config["input_subdir"] / image_name

    elif step == 3:
        step_02_config = CONFIG["step_02_grayscale_and_blur"]
        step_02_output_dir = PROCESSED_DIR / step_02_config["output_subdir"]
        selected_step_02_output = step_02_config["selected_output"]

        image_path = step_02_output_dir / selected_step_02_output / image_name

    elif step == 4:
        step_03_config = CONFIG["step_03_edge_detection"]
        image_path = PROCESSED_DIR / step_03_config["output_subdir"] / image_name

    elif step == 14:
        step_14_config = CONFIG["step_14_debug_hough_lines"]
        image_path = PROCESSED_DIR / step_14_config["input_visual_subdir"] / image_name

    elif step == 7:
        step_config = CONFIG["step_07_complete_line_fragments"]
        image_path = PROCESSED_DIR / step_config["input_visual_subdir"] / image_name

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


def scale_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / max(1, width), max_height / max(1, height))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def scale_image(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    if abs(zoom_factor - 1.0) < 1e-6:
        return image

    height, width = image.shape[:2]
    new_width = max(1, int(width * zoom_factor))
    new_height = max(1, int(height * zoom_factor))

    interpolation = cv2.INTER_LINEAR if zoom_factor >= 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

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


def safe_filename(value: str) -> str:
    value = value.replace("|", "_")
    value = re.sub(r"[^A-Za-z0-9_. -]+", "_", value)
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def test_image_output_dir(image_path: Path) -> Path:
    digits = "".join(ch for ch in image_path.stem if ch.isdigit())
    suffix = digits[-3:] if len(digits) >= 3 else (digits or image_path.stem)
    return SAVED_TEST_WINDOWS_DIR / f"test_{suffix}"


def save_labeled_results(output_dir: Path, results: list[tuple[str, np.ndarray]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (label, image) in enumerate(results, start=1):
        labeled = add_label(image, label)
        filename = f"{index:02d}__{safe_filename(label)}.png"
        output_path = output_dir / filename

        ok = cv2.imwrite(str(output_path), labeled)
        if not ok:
            raise RuntimeError(f"Could not save result: {output_path}")

        print(f"Saved: {output_path}")

def make_grid(
    images: list[tuple[str, np.ndarray]],
    images_per_window: int = MAX_IMAGES_PER_WINDOW,
    columns: int = 3,
) -> np.ndarray:
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

    while len(padded_images) < images_per_window:
        empty = np.full((max_height, max_width, 3), 245, dtype=np.uint8)
        padded_images.append(empty)

    columns = max(1, min(columns, images_per_window))
    rows = []
    for row_start in range(0, images_per_window, columns):
        row_images = padded_images[row_start:row_start + columns]
        if len(row_images) < columns:
            empty = np.full((max_height, max_width, 3), 245, dtype=np.uint8)
            while len(row_images) < columns:
                row_images.append(empty)
        rows.append(np.hstack(row_images))

    grid = rows[0] if len(rows) == 1 else np.vstack(rows)

    return grid

def show_variation_pages(
    title_prefix: str,
    results: list[tuple[str, np.ndarray]],
    images_per_window: int = MAX_IMAGES_PER_WINDOW,
    columns: int = 3,
    fit_to_screen: bool = False,
    initial_zoom: float = 1.0,
) -> None:
    if not results:
        print("No results to display.")
        return

    total_pages = (len(results) + images_per_window - 1) // images_per_window

    page_index = 0
    zoom_factor = max(0.25, float(initial_zoom))

    while page_index < total_pages:
        start = page_index * images_per_window
        end = start + images_per_window

        page_results = results[start:end]
        grid = make_grid(
            page_results,
            images_per_window=images_per_window,
            columns=columns,
        )

        if fit_to_screen:
            max_height = int(DISPLAY_CONFIG.get("max_height", 800))
            grid = scale_to_fit(grid, max_width=MAX_PAGE_WIDTH, max_height=max_height)

        grid = scale_image(grid, zoom_factor)

        title = f"{title_prefix} | page {page_index + 1}/{total_pages}"

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, grid)

        print(f"Showing page {page_index + 1}/{total_pages}")
        print("SPACE / ENTER / n -> next page")
        print("b / LEFT          -> previous page")
        print("+ / =             -> zoom in")
        print("- / _             -> zoom out")
        print("q / ESC           -> quit")
        print(f"zoom={zoom_factor:.2f}x")
        print()

        key = cv2.waitKey(0) & 0xFF

        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass

        if key in [ord("q"), 27]:
            break

        if key in [ord("+"), ord("=")]:
            zoom_factor = min(4.0, zoom_factor * 1.25)
            page_index = page_index
            continue

        if key in [ord("-"), ord("_")]:
            zoom_factor = max(0.25, zoom_factor / 1.25)
            page_index = page_index
            continue

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

def validate_aperture_size(value: int) -> int:
    allowed_values = {3, 5, 7}

    if value not in allowed_values:
        raise ValueError(f"Canny aperture_size must be one of {allowed_values}. Got: {value}")

    return value

def run_step_03_variations(image_path: Path) -> None:
    step_config = CONFIG["step_03_edge_detection"]
    canny_config = step_config["canny"]

    threshold_1_values = canny_config["threshold_1_test_values"]
    threshold_2_values = canny_config["threshold_2_test_values"]
    aperture_size_values = canny_config["aperture_size_test_values"]
    use_l2_gradient_values = canny_config["use_l2_gradient_test_values"]

    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = []

    for threshold_1, threshold_2, aperture_size, use_l2_gradient in itertools.product(
        threshold_1_values,
        threshold_2_values,
        aperture_size_values,
        use_l2_gradient_values
    ):
        threshold_1 = int(threshold_1)
        threshold_2 = int(threshold_2)
        aperture_size = validate_aperture_size(int(aperture_size))
        use_l2_gradient = bool(use_l2_gradient)

        if threshold_1 >= threshold_2:
            continue

        edges = cv2.Canny(
            image=image_gray,
            threshold1=threshold_1,
            threshold2=threshold_2,
            apertureSize=aperture_size,
            L2gradient=use_l2_gradient
        )

        label = (
            f"t1={threshold_1}, t2={threshold_2}, "
            f"ap={aperture_size}, l2={use_l2_gradient}"
        )

        results.append((label, edges))

    show_variation_pages(
        title_prefix=f"Step 03 Canny variations | {image_path.name}",
        results=results
    )

def classify_hough_line_for_test(
    angle_degrees: float,
    vertical_tolerance: float,
    horizontal_tolerance: float
) -> str | None:
    normalized_angle = abs(angle_degrees)

    if normalized_angle > 90:
        normalized_angle = 180 - normalized_angle

    if abs(normalized_angle - 90) <= vertical_tolerance:
        return "vertical"

    if normalized_angle <= horizontal_tolerance:
        return "horizontal"

    return None

def draw_hough_lines_for_test(
    edge_image: np.ndarray,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
    vertical_tolerance: float,
    horizontal_tolerance: float,
) -> np.ndarray:
    overlay = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(
        image=edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return overlay

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        line_type = classify_hough_line_for_test(
            angle_degrees=angle,
            vertical_tolerance=vertical_tolerance,
            horizontal_tolerance=horizontal_tolerance
        )

        if line_type == "vertical":
            color = (0, 255, 0)
            thickness = 2
        elif line_type == "horizontal":
            color = (255, 0, 0)
            thickness = 2
        else:
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

    return overlay

def run_step_14_variations(image_path: Path) -> None:
    step_config = CONFIG["step_14_debug_hough_lines"]

    hough_config = step_config["hough_lines_p"]
    classification_config = step_config["classification"]

    threshold_values = hough_config["threshold_test_values"]
    min_line_length_values = hough_config["min_line_length_test_values"]
    max_line_gap_values = hough_config["max_line_gap_test_values"]

    vertical_tolerance_values = classification_config["vertical_angle_tolerance_degrees_test_values"]
    horizontal_tolerance_values = classification_config["horizontal_angle_tolerance_degrees_test_values"]

    edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if edge_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = []

    for threshold, min_line_length, max_line_gap, vertical_tolerance, horizontal_tolerance in itertools.product(
        threshold_values,
        min_line_length_values,
        max_line_gap_values,
        vertical_tolerance_values,
        horizontal_tolerance_values
    ):
        threshold = int(threshold)
        min_line_length = int(min_line_length)
        max_line_gap = int(max_line_gap)
        vertical_tolerance = float(vertical_tolerance)
        horizontal_tolerance = float(horizontal_tolerance)

        overlay = draw_hough_lines_for_test(
            edge_image=edge_image,
            threshold=threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            vertical_tolerance=vertical_tolerance,
            horizontal_tolerance=horizontal_tolerance,
        )

        label = (
            f"thr={threshold}, len={min_line_length}, gap={max_line_gap}, "
            f"v={vertical_tolerance}, h={horizontal_tolerance}"
        )

        results.append((label, overlay))

    show_variation_pages(
        title_prefix=f"Step 14 Debug Hough lines | {image_path.name}",
        results=results
    )

def deep_merge_dict(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result

def load_step_06_module():
    module_path = PROJECT_ROOT / "pipeline" / "06_detect_boot_landmarks.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Step 06 file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "step06_landmarks",
        str(module_path)
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def load_step_07_module():
    module_path = PROJECT_ROOT / "pipeline" / "07_complete_line_fragments.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Step 07 file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "step07_complete_fragments",
        str(module_path)
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def run_step_06_single_preset(
    lm,
    image_path: Path,
    preset: dict,
) -> tuple[str, np.ndarray]:
    base_step_config = CONFIG["step_06_detect_boot_landmarks"]
    preset_name = str(preset.get("name", "unnamed variation"))
    preset_without_name = {key: value for key, value in preset.items() if key != "name"}

    lm.STEP = deep_merge_dict(base_step_config, preset_without_name)

    visual = cv2.imread(str(image_path))

    if visual is None:
        raise ValueError(f"Could not read image: {image_path}")

    detection_input, input_mode = lm.prepare_detection_input(image_path, visual)
    circles, _ = lm.find_circles(detection_input, input_mode)
    overlay = lm.draw_circles(visual, circles)
    label = f"{preset_name} | n={len(circles)}"

    return label, overlay

def build_step_06_presets() -> list[dict]:
    step_config = CONFIG["step_06_detect_boot_landmarks"]
    hough_config = step_config["hough"]
    explicit_presets = step_config.get("test_presets", [])

    if explicit_presets:
        presets = []

        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("detection_input", step_config["detection_input"])
            presets.append(preset_copy)

        return presets

    presets = [
        {
            "name": f"default ({step_config['detection_input']})",
            "detection_input": step_config["detection_input"],
        }
    ]

    for key, value in hough_config.items():
        if not key.endswith("_test_values"):
            continue

        base_key = key.removesuffix("_test_values")
        test_values = value

        if base_key not in hough_config:
            continue

        for test_value in test_values:
            if test_value == hough_config[base_key]:
                continue

            presets.append(
                {
                    "name": f"{base_key}={test_value}",
                    "detection_input": step_config["detection_input"],
                    "hough": {base_key: test_value},
                }
            )

    return presets

def run_step_06_variations(image_path: Path) -> None:
    presets = build_step_06_presets()
    lm = load_step_06_module()

    results = []

    for preset in presets:
        label, overlay = run_step_06_single_preset(
            lm=lm,
            image_path=image_path,
            preset=preset
        )

        results.append((label, overlay))

    show_variation_pages(
        title_prefix=f"Step 06 circle variations | {image_path.name}",
        results=results
    )

def collect_step_07_test_presets(
    current_value,
    test_values,
    path: tuple[str, ...],
    presets: list[dict],
) -> None:
    for test_value in test_values:
        if test_value == current_value:
            continue

        override = current = {}

        for key in path[:-1]:
            next_level = {}
            current[key] = next_level
            current = next_level

        current[path[-1]] = copy.deepcopy(test_value)

        presets.append(
            {
                "name": f"{'.'.join(path)}={test_value}",
                "override": override,
            }
        )

def build_step_07_presets() -> list[dict]:
    step_config = CONFIG["step_07_complete_line_fragments"]
    explicit_presets = step_config.get("test_presets", [])

    if explicit_presets:
        presets = []

        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)

        return presets

    presets = [{"name": "default", "override": {}}]

    def walk(node: dict, path: tuple[str, ...] = ()) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, path + (key,))
                continue

            if not key.endswith("_test_values"):
                continue

            base_key = key.removesuffix("_test_values")
            if base_key not in node:
                continue

            collect_step_07_test_presets(
                current_value=node[base_key],
                test_values=value,
                path=path + (base_key,),
                presets=presets
            )

    walk(step_config)

    return presets

def run_step_07_single_preset(
    lm,
    image_path: Path,
    preset: dict,
) -> tuple[str, np.ndarray]:
    base_step_config = CONFIG["step_07_complete_line_fragments"]
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))

    lm.STEP = deep_merge_dict(base_step_config, override)

    visual = cv2.imread(str(image_path))

    if visual is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = visual.shape[:2]
    edge = lm.load_edge(image_path, visual)
    raw_mask, _ = lm.load_roi_mask(image_path.name, (h, w))
    outer_mask = lm.dilate_mask(raw_mask)
    hough_mask = lm.make_inner_mask(raw_mask)
    masked_edge = cv2.bitwise_and(edge, edge, mask=hough_mask)

    raw_hough_lines = lm.detect_hough_lines(masked_edge)

    all_fragments = []
    for idx, line in enumerate(raw_hough_lines, start=1):
        fragment = lm.make_fragment(idx, line, outer_mask)
        if fragment is not None:
            all_fragments.append(fragment)

    mask_supported = lm.filter_mask_supported(all_fragments)
    vertical = lm.filter_vertical(mask_supported)
    groups = lm.merge_groups(vertical)
    completed = [
        lm.build_completed_line(idx, group, outer_mask, edge)
        for idx, group in enumerate(groups, start=1)
    ]
    completed.sort(key=lambda item: item.support_score, reverse=True)

    completed_overlay = lm.draw_completed(visual, completed, vertical)

    label = (
        f"{preset_name} | "
        f"h={len(raw_hough_lines)} f={len(all_fragments)} "
        f"v={len(vertical)} c={len(completed)}"
    )

    return label, completed_overlay

def run_step_07_variations(image_path: Path) -> None:
    presets = build_step_07_presets()
    lm = load_step_07_module()

    results = []

    for preset in presets:
        label, panel = run_step_07_single_preset(
            lm=lm,
            image_path=image_path,
            preset=preset
        )
        results.append((label, panel))

    output_dir = test_image_output_dir(image_path)
    save_labeled_results(output_dir=output_dir, results=results)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visually test all configured parameter variations for one step and one image."
    )

    parser.add_argument(
        "--step",
        type=int,
        required=True,
        choices=[1, 2, 3, 14, 6, 7],
        help="Pipeline step to test. Supported: 1, 2, 3, 14, 6, 7."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image filename, for example IMG_0502.png."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path relative to project root, for example config/pipeline_config_step06_test.yaml."
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_config(args.config)

    image_path = resolve_image_path(
        step=args.step,
        image_name=args.image
    )

    print()
    print("Running visual test variations")
    print(f"Step:  {args.step}")
    print(f"Image: {image_path}")
    print(f"Config: {args.config or 'config/pipeline_config.yaml'}")
    if args.step == 7:
        print(f"Step 7 results will be saved to: {test_image_output_dir(image_path)}")
    else:
        print("No files will be saved.")
    print()

    if args.step == 1:
        run_step_01_variations(image_path)

    elif args.step == 2:
        run_step_02_variations(image_path)

    elif args.step == 3:
        run_step_03_variations(image_path)
    
    elif args.step == 14:
        run_step_14_variations(image_path)

    elif args.step == 6:
        run_step_06_variations(image_path)
    elif args.step == 7:
        run_step_07_variations(image_path)

    cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()

