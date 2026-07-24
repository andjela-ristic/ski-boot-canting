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
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT.parent))

sys.path.append(str(PROJECT_ROOT / "pipeline"))

from backend.pipeline.config_loader import load_config
from backend.pipeline.lib.step_01_illumination_normalization import normalize_illumination_variant
from backend.pipeline.lib.step_02_grayscale_and_blur import build_bilateral_variant, convert_to_bgr2gray
from backend.pipeline.lib import step_03_edge_detection as step03_lib
from backend.pipeline.lib import step_04_boot_roi_from_edges as step04_lib
from backend.pipeline.lib import step_05_valid_hough_lines_in_roi as step05_lib
from backend.pipeline.lib import step_06_search_central_ruler as step06_lib
from backend.pipeline.lib import step_07_verify_central_ruler_symmetry as step07_lib
from backend.pipeline.lib import step_08_multi_validate_central_ruler as step08_lib
from backend.pipeline.lib import step_09_measure_canting_angle as step09_lib

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

def get_step_02_input_dir(step_config: dict | None = None) -> Path:
    step_config = CONFIG["step_02_grayscale_and_blur"] if step_config is None else step_config
    if step_config.get("inherit_step_01_output", True):
        step_01_config = CONFIG["step_01_illumination_normalization"]
        return PROCESSED_DIR / step_01_config["output_subdir"]
    return PROCESSED_DIR / step_config["input_subdir"]

# TODO: implement the following functions and other implementation for step 3
def get_step_03_input_dir(step_config: dict | None = None) -> Path:
    ...

def get_step_04_input_dir(step_config: dict | None = None) -> Path:
    step_config = CONFIG["step_04_boot_roi_from_edges"] if step_config is None else step_config
    if step_config.get("inherit_step_03_output", True):
        step_03_config = CONFIG["step_03_edge_detection"]
        input_root_dir = PROCESSED_DIR / step_03_config["output_subdir"]
    else:
        input_root_dir = PROCESSED_DIR / step_config["input_subdir"]

    selected_input = step_config.get("selected_input")
    return input_root_dir / str(selected_input).strip() if selected_input is not None else input_root_dir

def get_step_05_edge_input_dir(step_config: dict | None = None) -> Path:
    step_config = CONFIG["step_05_valid_hough_lines_in_roi"] if step_config is None else step_config
    if step_config.get("inherit_step_03_output", True):
        step_03_config = CONFIG["step_03_edge_detection"]
        edge_input_name = str(
            step_config.get("edge_input_name", step_03_config.get("selected_output", "cleaned"))
        ).strip()
        return PROCESSED_DIR / step_03_config["output_subdir"] / edge_input_name
    return PROCESSED_DIR / step_config["edge_input_subdir"]

def get_step_05_roi_dir(step_config: dict | None = None) -> Path:
    step_config = CONFIG["step_05_valid_hough_lines_in_roi"] if step_config is None else step_config
    if step_config.get("inherit_step_04_output", True):
        step_04_config = CONFIG["step_04_boot_roi_from_edges"]
        roi_mask_subdir_name = str(step_config.get("roi_mask_subdir_name", "mask")).strip()
        return PROCESSED_DIR / step_04_config["output_subdir"] / roi_mask_subdir_name
    return PROCESSED_DIR / step_config["roi_mask_subdir"]

def get_step_06_input_dir(step_config: dict | None = None) -> Path:
    step_config = CONFIG["step_06_search_central_ruler"] if step_config is None else step_config
    if step_config.get("inherit_step_05_output", True):
        step_05_config = CONFIG["step_05_valid_hough_lines_in_roi"]
        return PROCESSED_DIR / step_05_config["output_subdir"]
    return PROCESSED_DIR / step_config["input_subdir"]

def get_step_09_base_config(lm=None) -> dict:
    if lm is None: lm = step09_lib
    step_config = CONFIG.get("step_09_measure_canting_angle")
    if step_config is None: return copy.deepcopy(lm.context.DEFAULT_STEP_CONFIG)
    return lm.context.deep_merge(lm.context.DEFAULT_STEP_CONFIG, step_config)

def get_step_09_metadata_dir(step_config: dict | None = None) -> Path:
    step_config = get_step_09_base_config() if step_config is None else step_config
    step_08_output = str(CONFIG["step_08_multi_validate_central_ruler"]["output_subdir"])
    input_subdir = step_08_output if step_config.get("inherit_step_08_output", True) else str(step_config["input_subdir"])
    return PROCESSED_DIR / input_subdir / str(step_config.get("input_metadata_subdir", "metadata"))

def resolve_step_09_metadata_path(image_name: str, step_config: dict | None = None) -> Path:
    # metadata comes from step 08
    metadata_dir = get_step_09_metadata_dir(step_config)
    metadata_path = metadata_dir / f"{Path(image_name).stem}_multi_validation.json"
    if not metadata_path.exists(): raise FileNotFoundError(f"Step 09 input metadata not found: {metadata_path}")
    return metadata_path

def get_step_07_input_dir(step_config: dict | None = None) -> Path:
    step_config = get_step_07_base_config() if step_config is None else step_config
    if step_config.get("inherit_step_06_output", True):
        step_06_output = str(get_step_06_base_config().get("output_subdir", "06_search_central_ruler"))
        return PROCESSED_DIR / step_06_output
    return PROCESSED_DIR / str(step_config["input_subdir"])

def get_step_08_input_dir(step_config: dict | None = None) -> Path:
    step_config = get_step_08_base_config() if step_config is None else step_config
    if step_config.get("inherit_step_07_output", True):
        step_07_output = str(get_step_07_base_config().get("output_subdir", "07_verify_central_ruler_symmetry"))
        return PROCESSED_DIR / step_07_output
    return PROCESSED_DIR / str(step_config["input_subdir"])

def get_step_07_base_config(lm=None) -> dict:
    if lm is None: lm = step07_lib
    step_config = CONFIG.get("step_07_verify_central_ruler_symmetry")
    if step_config is None: return copy.deepcopy(lm.context.DEFAULT_STEP_CONFIG)
    return lm.context.deep_merge(lm.context.DEFAULT_STEP_CONFIG, step_config)

def get_step_08_base_config(lm=None) -> dict:
    if lm is None: lm = step08_lib
    step_config = CONFIG.get("step_08_multi_validate_central_ruler")
    if step_config is None: return copy.deepcopy(lm.context.DEFAULT_STEP_CONFIG)
    return lm.context.deep_merge(lm.context.DEFAULT_STEP_CONFIG, step_config)

def resolve_step_07_metadata_path(image_name: str, step_config: dict | None = None) -> Path:
    metadata_dir = get_step_07_input_dir(step_config) / "metadata"
    metadata_path = metadata_dir / f"{Path(image_name).stem}_central_ruler.json"
    if not metadata_path.exists(): raise FileNotFoundError(f"Step 07 input metadata not found: {metadata_path}")
    return metadata_path

def resolve_step_08_metadata_path(image_name: str, step_config: dict | None = None) -> Path:
    metadata_dir = get_step_08_input_dir(step_config) / "metadata"
    metadata_path = metadata_dir / f"{Path(image_name).stem}_symmetry.json"
    if not metadata_path.exists(): raise FileNotFoundError(f"Step 08 input metadata not found: {metadata_path}")
    return metadata_path

def select_combo_results( results: list[tuple[str, np.ndarray]], combo: int | None,) -> list[tuple[str, np.ndarray]]:
    if combo is None: return results

    if combo < 1 or combo > len(results):
        raise ValueError(
            f"--combo must be between 1 and {len(results)}. Got: {combo}"
        )

    return [results[combo - 1]]

def select_combo_items(items: list, combo: int | None) -> list:
    if combo is None: return items

    if combo < 1 or combo > len(items):
        raise ValueError(
            f"--combo must be between 1 and {len(items)}. Got: {combo}"
        )

    return [items[combo - 1]]

def resolve_existing_image_path(base_dir: Path, image_name: str) -> Path:
    candidate = base_dir / image_name

    if candidate.exists():
        return candidate

    allowed_extensions = [".png", ".jpg", ".jpeg"]
    requested_path = Path(image_name)
    requested_stem = requested_path.stem if requested_path.suffix else requested_path.name
    requested_suffix = requested_path.suffix.lower()

    matches = []

    for extension in allowed_extensions:
        if requested_suffix and not extension.startswith(requested_suffix): continue

        extension_candidate = base_dir / f"{requested_stem}{extension}"
        if extension_candidate.exists(): matches.append(extension_candidate)

    if len(matches) == 1: return matches[0]

    if not requested_suffix:
        for extension in allowed_extensions:
            extension_candidate = base_dir / f"{requested_stem}{extension}"
            if extension_candidate.exists(): matches.append(extension_candidate)

        unique_matches = []
        seen = set()
        for match in matches:
            match_key = str(match)
            if match_key in seen: continue
            seen.add(match_key)
            unique_matches.append(match)

        if len(unique_matches) == 1:
            return unique_matches[0]

    raise FileNotFoundError(f"Image not found: {candidate}")

def resolve_image_path(step: int, image_name: str) -> Path:
    if step == 1: image_path = resolve_existing_image_path(WORKING_PNG_DIR, image_name)
    elif step == 2:
        step_config = CONFIG["step_02_grayscale_and_blur"]
        image_path = resolve_existing_image_path(get_step_02_input_dir(step_config),image_name)
    elif step == 3:
        step_02_config = CONFIG["step_02_grayscale_and_blur"]
        step_02_output_dir = PROCESSED_DIR / step_02_config["output_subdir"]
        selected_step_02_output = step_02_config["selected_output"]
        image_path = resolve_existing_image_path(step_02_output_dir / selected_step_02_output,image_name)
    elif step == 4:
        step_config = CONFIG["step_04_boot_roi_from_edges"]
        image_path = resolve_existing_image_path(get_step_04_input_dir(step_config),image_name)
    elif step == 5:
        step_config = CONFIG["step_05_valid_hough_lines_in_roi"]
        image_path = resolve_existing_image_path(get_step_05_edge_input_dir(step_config),image_name)
    elif step == 6:
        step_config = CONFIG["step_06_search_central_ruler"]
        input_root_dir = get_step_06_input_dir(step_config)
        input_visual_dir = input_root_dir / step_config.get("input_overlay_subdir", "valid_lines_overlay")
        image_path = resolve_existing_image_path(input_visual_dir,image_name)
    elif step == 7:
        step_config = get_step_07_base_config()
        resolve_step_07_metadata_path(image_name, step_config)
        image_path = Path(image_name)
    elif step == 8:
        step_config = get_step_08_base_config()
        resolve_step_08_metadata_path(image_name, step_config)
        image_path = Path(image_name)
    elif step == 9:
        step_config = get_step_09_base_config()
        resolve_step_09_metadata_path(image_name, step_config)
        image_path = Path(image_name)
    else:
        raise ValueError(f"Unsupported step: {step}")
    return image_path

def resize_for_display(image: np.ndarray, max_height: int = 520) -> np.ndarray:
    height, width = image.shape[:2]

    if height <= max_height: return image

    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def scale_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]

    if width <= max_width and height <= max_height: return image

    scale = min(max_width / max(1, width), max_height / max(1, height))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def scale_image(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    if abs(zoom_factor - 1.0) < 1e-6: return image

    height, width = image.shape[:2]
    new_width = max(1, int(width * zoom_factor))
    new_height = max(1, int(height * zoom_factor))

    interpolation = cv2.INTER_LINEAR if zoom_factor >= 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def add_label(image: np.ndarray, label: str) -> np.ndarray:
    image = to_bgr(image).copy()
    # draw a black rectangle at the top of the image for the label
    cv2.rectangle(image,(0, 0),(image.shape[1], 52),(0, 0, 0),thickness=-1)
    # add resulting text on top
    cv2.putText(image,label,(12, 33),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2,cv2.LINE_AA)

    return image

def pad_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    padded = np.full((height, width, 3), 255, dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    padded[y_offset:y_offset + image_height,x_offset:x_offset + image_width] = image
    return padded

def safe_filename(value: str) -> str:
    value = value.replace("|", "_")
    value = re.sub(r"[^A-Za-z0-9_. -]+", "_", value)
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_")

def step_image_output_dir(step: int, image_path: Path) -> Path:
    return SAVED_TEST_WINDOWS_DIR / f"step_{step:02d}" / image_path.stem

def save_labeled_results(output_dir: Path, results: list[tuple[str, np.ndarray]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (label, image) in enumerate(results, start=1):
        labeled = add_label(image, label)
        filename = f"{index:02d}__{safe_filename(label)}.png"
        output_path = output_dir / filename

        ok = cv2.imwrite(str(output_path), labeled)
        if not ok: raise RuntimeError(f"Could not save result: {output_path}")
        print(f"Saved: {output_path}")

def make_grid( images: list[tuple[str, np.ndarray]], images_per_window: int = MAX_IMAGES_PER_WINDOW, columns: int = 3,) -> np.ndarray:
    prepared_images = []
    for label, image in images:
        display_image = resize_for_display(to_bgr(image))
        display_image = add_label(display_image, label)
        prepared_images.append(display_image)

    max_width = max(image.shape[1] for image in prepared_images)
    max_height = max(image.shape[0] for image in prepared_images)

    padded_images = [pad_to_size(image, max_width, max_height) for image in prepared_images]

    while len(padded_images) < images_per_window:
        empty = np.full((max_height, max_width, 3), 245, dtype=np.uint8)
        padded_images.append(empty)

    columns = max(1, min(columns, images_per_window))
    rows = []
    for row_start in range(0, images_per_window, columns):
        row_images = padded_images[row_start:row_start + columns]
        if len(row_images) < columns:
            empty = np.full((max_height, max_width, 3), 245, dtype=np.uint8)
            while len(row_images) < columns: row_images.append(empty)
        rows.append(np.hstack(row_images))

    grid = rows[0] if len(rows) == 1 else np.vstack(rows)
    return grid

def show_variation_pages(title_prefix: str, results: list[tuple[str, np.ndarray]], images_per_window: int = MAX_IMAGES_PER_WINDOW, columns: int = 3,
    fit_to_screen: bool = False, initial_zoom: float = 1.0, save_dir: Path | None = None) -> None:
    if not results:
        print("No results to display.")
        return

    total_pages = (len(results) + images_per_window - 1) // images_per_window

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        for page_index in range(total_pages):
            start = page_index * images_per_window
            end = start + images_per_window
            page_results = results[start:end]
            grid = make_grid( page_results, images_per_window=images_per_window, columns=columns,)

            if fit_to_screen:
                max_height = int(DISPLAY_CONFIG.get("max_height", 800))
                grid = scale_to_fit(grid, max_width=MAX_PAGE_WIDTH, max_height=max_height)

            grid = scale_image(grid, max(0.25, float(initial_zoom)))

            output_path = save_dir / f"window_{page_index + 1:02d}.png"
            ok = cv2.imwrite(str(output_path), grid)
            if not ok:
                raise RuntimeError(f"Could not save window page: {output_path}")

            print(f"Saved window page {page_index + 1}/{total_pages}: {output_path}")

        return

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

def deep_merge_dict(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (isinstance(value, dict)and isinstance(result.get(key), dict)):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def ensure_odd_kernel_size(value: int) -> int:
    if value < 1: raise ValueError(f"Kernel size must be positive. Got: {value}")
    if value % 2 == 0: raise ValueError(f"Kernel size must be odd. Got: {value}")
    return value

def run_step_01_variations(image_path: Path, combo: int | None = None) -> None:
    step_config = CONFIG["step_01_illumination_normalization"]
    clahe_config = step_config["clahe"]

    clip_values = clahe_config["clip_limit_test_values"]
    tile_values = clahe_config["tile_grid_size_test_values"]

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None: raise ValueError(f"Could not read image: {image_path}")

    results = []
    for clip_limit, tile_grid_size_list in itertools.product(clip_values, tile_values):
        tile_grid_size = tuple(tile_grid_size_list)
        processed = normalize_illumination_variant(image_bgr=image_bgr, clip_limit=float(clip_limit), tile_grid_size=tile_grid_size)
        tile_label = f"{tile_grid_size[0]}x{tile_grid_size[1]}"
        label = f"clip={clip_limit}, tile={tile_label}"
        results.append((label, processed))
    results = select_combo_results(results, combo)
    show_variation_pages(title_prefix=f"Step 01 variations | {image_path.name}",results=results)

def run_step_02_variations(image_path: Path, combo: int | None = None) -> None:
    step_config = CONFIG["step_02_grayscale_and_blur"]
    gaussian_config = step_config["gaussian_blur"]
    bilateral_config = step_config["bilateral_filter"]

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None: raise ValueError(f"Could not read image: {image_path}")
    grayscale = convert_to_bgr2gray(image_bgr)
    gaussian_results = [("grayscale", grayscale)]
    for kernel_size, sigma_x in itertools.product(gaussian_config["kernel_size_test_values"], gaussian_config["sigma_x_test_values"]):
        kernel_size = ensure_odd_kernel_size(int(kernel_size))
        sigma_x = float(sigma_x)
        gaussian = cv2.GaussianBlur( grayscale,(kernel_size, kernel_size),sigma_x)
        label = f"gaussian k={kernel_size}, sigma={sigma_x}"
        gaussian_results.append((label, gaussian))
    bilateral_results = [("grayscale", grayscale)]

    for diameter, sigma_color, sigma_space in itertools.product(bilateral_config["diameter_test_values"],
        bilateral_config["sigma_color_test_values"],bilateral_config["sigma_space_test_values"]):
        diameter = int(diameter)
        sigma_color = float(sigma_color)
        sigma_space = float(sigma_space)
        bilateral = build_bilateral_variant(grayscale, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)
        label = f"bilateral d={diameter}, sc={sigma_color}, ss={sigma_space}"
        bilateral_results.append((label, bilateral))

    gaussian_results = select_combo_results(gaussian_results, combo)
    bilateral_results = select_combo_results(bilateral_results, combo)

    show_variation_pages(title_prefix=f"Step 02 Gaussian variations | {image_path.name}",results=gaussian_results)
    show_variation_pages(title_prefix=f"Step 02 Bilateral variations | {image_path.name}",results=bilateral_results)

def load_step_03_module():
    return step03_lib

def build_step_03_presets() -> list[dict]:
    step_config = CONFIG["step_03_edge_detection"]
    explicit_presets = step_config.get("test_presets", [])
    if explicit_presets:
        presets = []
        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)
        return presets
    test_dimensions = []

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
            current_value = copy.deepcopy(node[base_key])
            test_values = [copy.deepcopy(item) for item in value]
            if current_value not in test_values:
                test_values.insert(0, current_value)
            test_dimensions.append((path + (base_key,), test_values))
    walk(step_config)

    if not test_dimensions: return [{"name": "default", "override": {}}]

    presets = []
    for combination in itertools.product(*(values for _, values in test_dimensions)):
        override = {}
        label_parts = []

        for (path, _), selected_value in zip(test_dimensions, combination):
            current = override
            for key in path[:-1]: current = current.setdefault(key, {})
            current[path[-1]] = copy.deepcopy(selected_value)
            label_parts.append(f"{'.'.join(path)}={selected_value}")

        presets.append({ "name": " | ".join(label_parts), "override": override,})
    return presets

def run_step_03_single_preset(lm,image_path: Path, preset: dict,) -> tuple[str, np.ndarray]:
    base_step_config = CONFIG["step_03_edge_detection"]
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))

    step_config = deep_merge_dict(base_step_config, override)
    lm.context.set_step_config(step_config)
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None: raise ValueError(f"Could not read image: {image_path}")

    raw_edges, threshold_1, threshold_2 = lm.run_canny(image_gray)
    cleaned_edges = lm.clean_edges(raw_edges)

    selected_output = lm.get_selected_edge_output()
    result_image = raw_edges if selected_output == "raw" else cleaned_edges

    canny_config = step_config["canny"]
    preprocessing_config = step_config.get("preprocessing", {})
    postprocessing_config = step_config.get("postprocessing", {})

    label = (
        f"{preset_name} | mode={lm.get_canny_mode()} "
        f"t1={threshold_1} t2={threshold_2} "
        f"ap={canny_config['aperture_size']} "
        f"l2={bool(canny_config['use_l2_gradient'])} "
        f"pre={bool(preprocessing_config.get('enabled', False))} "
        f"post={bool(postprocessing_config.get('enabled', False))} "
        f"out={selected_output}"
    )

    return label, result_image

def run_step_03_variations(image_path: Path, combo: int | None = None) -> None:
    presets = build_step_03_presets()
    presets = select_combo_items(presets, combo)
    lm = load_step_03_module()
    results = []
    print("Step 03 test variations:")

    for index, preset in enumerate(presets, start=1):
        label, edge_image = run_step_03_single_preset(lm=lm,image_path=image_path,preset=preset)
        print(f"  #{index}: {label}")
        results.append((f"#{index} | {label}", edge_image))

    print()
    show_variation_pages(title_prefix=f"Step 03 edge variations | {image_path.name}", results=results )

def load_step_04_module():
    return step04_lib

def build_step_04_presets() -> list[dict]:
    step_config = CONFIG["step_04_boot_roi_from_edges"]
    explicit_presets = step_config.get("test_presets", [])
    if explicit_presets:
        presets = []
        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)
        return presets
    return [{"name": "default", "override": {}}]

def run_step_04_single_preset( lm, image_path: Path,preset: dict,) -> tuple[str, np.ndarray]:
    base_step_config = CONFIG["step_04_boot_roi_from_edges"]
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))
    step_config = deep_merge_dict(base_step_config, override)
    lm.set_step_config(step_config)
    edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if edge_image is None: raise ValueError(f"Could not read image: {image_path}")
    results = lm.process_edge_image(edge_image)
    overlay = lm.make_overlay(edge_image, results["final_mask"])
    area = int(np.count_nonzero(results["final_mask"]))
    threshold = int(results["density_threshold"])
    label = f"{preset_name} | thr={threshold} area={area}"
    return label, overlay

def run_step_04_variations(image_path: Path, combo: int | None = None) -> None:
    presets = build_step_04_presets()
    presets = select_combo_items(presets, combo)
    lm = load_step_04_module()
    results = []
    print("Step 04 test variations:")
    for index, preset in enumerate(presets, start=1):
        label, panel = run_step_04_single_preset(lm=lm,image_path=image_path,preset=preset)
        print(f"  #{index}: {label}")
        results.append((label, panel))
    print()
    show_variation_pages(title_prefix=f"Step 04 ROI variations | {image_path.name}",results=results)

def load_step_05_module():
    return step05_lib

def build_step_05_presets() -> list[dict]:
    step_config = CONFIG["step_05_valid_hough_lines_in_roi"]
    explicit_presets = step_config.get("test_presets", [])
    if explicit_presets:
        presets = []
        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)
        return presets
    test_dimensions = []

    def walk(node: dict, path: tuple[str, ...] = ()) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, path + (key,))
                continue
            if not key.endswith("_test_values"): continue
            base_key = key.removesuffix("_test_values")
            if base_key not in node: continue
            current_value = copy.deepcopy(node[base_key])
            test_values = [copy.deepcopy(item) for item in value]
            if current_value not in test_values: test_values.insert(0, current_value)
            test_dimensions.append((path + (base_key,), test_values))

    walk(step_config)
    if not test_dimensions: return [{"name": "default", "override": {}}]
    presets = []
    for combination in itertools.product(*(values for _, values in test_dimensions)):
        override = {}
        label_parts = []
        for (path, _), selected_value in zip(test_dimensions, combination):
            current = override
            for key in path[:-1]: current = current.setdefault(key, {})
            current[path[-1]] = copy.deepcopy(selected_value)
            label_parts.append(f"{'.'.join(path)}={selected_value}")
        presets.append({"name": " | ".join(label_parts),"override": override,})

    return presets

def run_step_05_single_preset(lm, image_path: Path, preset: dict,) -> tuple[str, np.ndarray]:
    base_step_config = CONFIG["step_05_valid_hough_lines_in_roi"]
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))
    step_config = deep_merge_dict(base_step_config, override)
    lm.set_step_config(step_config)
    edge_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if edge_image is None: raise ValueError(f"Could not read image: {image_path}")

    roi_dir = get_step_05_roi_dir(step_config)
    roi_path = resolve_existing_image_path(roi_dir, image_path.name)
    roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

    if roi_mask is None: raise ValueError(f"Could not read ROI mask: {roi_path}")

    hough_mask = lm.make_hough_mask(roi_mask)
    row_mask_bounds = lm.build_row_mask_bounds(roi_mask)
    masked_edge = cv2.bitwise_and(edge_image, edge_image, mask=hough_mask)
    raw_lines = lm.detect_hough_lines(masked_edge)
    line_records = [
        lm.build_line_record(record_index, line, roi_mask, row_mask_bounds)
        for record_index, line in enumerate(raw_lines, start=1)
    ]

    valid_overlay = lm.draw_lines(edge_image, line_records, valid_only=True)
    valid_count = sum(1 for record in line_records if bool(record["is_valid"]))
    label = (
        f"{preset_name} | raw={len(line_records)} valid={valid_count} "
        f"thr={step_config['hough_lines_p']['threshold']} "
        f"len={step_config['hough_lines_p']['min_line_length']} "
        f"gap={step_config['hough_lines_p']['max_line_gap']}"
    )
    return label, valid_overlay

def run_step_05_variations(image_path: Path, combo: int | None = None) -> None:
    presets = build_step_05_presets()
    presets = select_combo_items(presets, combo)
    lm = load_step_05_module()
    results = []
    print("Step 05 test variations:")
    for index, preset in enumerate(presets, start=1):
        label, panel = run_step_05_single_preset(lm=lm,image_path=image_path,preset=preset)
        print(f"  #{index}: {label}")
        results.append((f"#{index} | {label}", panel))
    print()
    output_dir = step_image_output_dir(step=5, image_path=image_path)
    save_labeled_results(output_dir=output_dir, results=results)
    show_variation_pages(title_prefix=f"Step 05 valid Hough lines | {image_path.name}", results=results)

def load_step_06_module():
    return step06_lib

def get_step_06_base_config(lm=None) -> dict:
    if lm is None: lm = load_step_06_module()
    step_config = CONFIG.get("step_06_search_central_ruler")
    if step_config is None: return copy.deepcopy(lm.DEFAULT_STEP_CONFIG)
    return lm.deep_merge(lm.DEFAULT_STEP_CONFIG, step_config)

def run_step_06_single_preset(lm,image_path: Path, preset: dict,) -> tuple[str, np.ndarray]:
    base_step_config = get_step_06_base_config(lm)
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))
    step_config = lm.deep_merge(lm.DEFAULT_STEP_CONFIG,deep_merge_dict(base_step_config, override))
    lm.set_step_config(step_config)
    input_root_dir = get_step_06_input_dir(step_config)
    input_json_dir = input_root_dir / step_config.get("input_json_subdir", "valid_lines_json")
    json_path = input_json_dir / f"{image_path.stem}.json"
    if not json_path.exists(): raise FileNotFoundError(f"Step 06 input JSON not found: {json_path}")
    analysis = lm.build_analysis(json_path)
    best_candidate = analysis["best_candidate"]
    label = (
        f"{preset_name} | filt={len(analysis['filtered_lines'])} "
        f"cand={len(analysis['fine_candidates'])} "
        f"sel={best_candidate['selected_fragment_count'] if best_candidate else 0} "
        f"score={best_candidate['score']:.3f}"
        if best_candidate is not None
        else f"{preset_name} | filt={len(analysis['filtered_lines'])} cand=0"
    )
    return label, analysis["overlay"]

def collect_step_06_test_presets( current_value, test_values, path: tuple[str, ...], presets: list[dict]) -> None:
    for test_value in test_values:
        if test_value == current_value: continue
        override = current = {}
        for key in path[:-1]:
            next_level = {}
            current[key] = next_level
            current = next_level
        current[path[-1]] = copy.deepcopy(test_value)
        presets.append({"name": f"{'.'.join(path)}={test_value}", "override": override,})

def build_step_06_presets() -> list[dict]:
    step_config = get_step_06_base_config()
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
            if not key.endswith("_test_values"): continue
            base_key = key.removesuffix("_test_values")
            if base_key not in node: continue
            collect_step_06_test_presets(current_value=node[base_key],test_values=value,path=path + (base_key,),presets=presets)
    walk(step_config)
    return presets

def run_step_06_variations(image_path: Path,workers: int | None = None,combo: int | None = None,) -> None:
    _ = workers
    presets = build_step_06_presets()
    presets = select_combo_items(presets, combo)
    results = []
    print("Step 06 test variations:")
    lm = load_step_06_module()
    for index, preset in enumerate(presets, start=1):
        label, overlay = run_step_06_single_preset(lm=lm,image_path=image_path,preset=preset)
        print(f"  #{index}: {label}")
        results.append((f"#{index} | {label}", overlay))
    print()
    output_dir = step_image_output_dir(step=6, image_path=image_path)
    save_labeled_results(output_dir=output_dir, results=results)
    show_variation_pages(title_prefix=f"Step 06 central ruler search | {image_path.name}", results=results, save_dir=output_dir,)

def load_step_07_module():
    return step07_lib

def load_step_08_module():
    return step08_lib

def build_step_07_presets() -> list[dict]:
    step_config = get_step_07_base_config()
    explicit_presets = step_config.get("test_presets", [])
    if explicit_presets:
        presets = []
        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)
        return presets
    return [{"name": "default", "override": {}}]

def build_step_08_presets() -> list[dict]:
    step_config = get_step_08_base_config()
    explicit_presets = step_config.get("test_presets", [])
    if explicit_presets:
        presets = []
        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)
        return presets
    return [{"name": "default", "override": {}}]

def run_step_07_single_preset(lm, image_path: Path, preset: dict) -> tuple[str, np.ndarray]:
    base_step_config = get_step_07_base_config(lm)
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))
    step_config = lm.context.deep_merge(base_step_config, override)
    lm.set_step_config(step_config)
    lm.ensure_dirs(cleanup=False)
    metadata_path = resolve_step_07_metadata_path(image_path.name, step_config)
    result = lm.process_metadata_file(metadata_path)
    panel_path = PROJECT_ROOT / str(result["comparison_path"] or result["overlay_path"])
    panel = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if panel is None: raise ValueError(f"Could not read Step 07 panel: {panel_path}")
    label = (
        f"{preset_name} | winner={result['winner_label']} "
        f"sym={result['symmetry_percent']:.1f}% "
        f"margin={result['winner_margin_percent']:.1f}%"
    )
    return label, panel

def run_step_08_single_preset(lm, image_path: Path, preset: dict) -> tuple[str, np.ndarray]:
    base_step_config = get_step_08_base_config(lm)
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))
    step_config = lm.context.deep_merge(base_step_config, override)
    lm.set_step_config(step_config)
    lm.ensure_dirs(cleanup=False)
    metadata_path = resolve_step_08_metadata_path(image_path.name, step_config)
    result = lm.process_metadata_file(metadata_path)
    panel_path = PROJECT_ROOT / str(result["comparison_path"] or result["overlay_path"])
    panel = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if panel is None: raise ValueError(f"Could not read Step 08 panel: {panel_path}")
    label = (
        f"{preset_name} | final={result['final_candidate']} "
        f"conf={result['confidence_percent']:.1f}% "
        f"decision={result['decision']}"
    )
    return label, panel

def run_step_07_variations(image_path: Path, combo: int | None = None) -> None:
    presets = build_step_07_presets()
    presets = select_combo_items(presets, combo)
    results = []
    print("Step 07 test variations:")
    lm = load_step_07_module()
    for index, preset in enumerate(presets, start=1):
        label, panel = run_step_07_single_preset(lm=lm, image_path=image_path, preset=preset)
        print(f"  #{index}: {label}")
        results.append((f"#{index} | {label}", panel))
    print()
    output_dir = step_image_output_dir(step=7, image_path=image_path)
    save_labeled_results(output_dir=output_dir, results=results)
    show_variation_pages(title_prefix=f"Step 07 symmetry verification | {image_path.name}", results=results, save_dir=output_dir)

def run_step_08_variations(image_path: Path, combo: int | None = None) -> None:
    presets = build_step_08_presets()
    presets = select_combo_items(presets, combo)
    results = []
    print("Step 08 test variations:")
    lm = load_step_08_module()
    for index, preset in enumerate(presets, start=1):
        label, panel = run_step_08_single_preset(lm=lm, image_path=image_path, preset=preset)
        print(f"  #{index}: {label}")
        results.append((f"#{index} | {label}", panel))
    print()
    output_dir = step_image_output_dir(step=8, image_path=image_path)
    save_labeled_results(output_dir=output_dir, results=results)
    show_variation_pages(title_prefix=f"Step 08 multi validation | {image_path.name}", results=results, save_dir=output_dir)

def load_step_09_module():
    return step09_lib

def build_step_09_presets() -> list[dict]:
    step_config = get_step_09_base_config()
    explicit_presets = step_config.get("test_presets", [])
    if explicit_presets:
        presets = []
        for preset in explicit_presets:
            preset_copy = copy.deepcopy(preset)
            preset_copy.setdefault("override", {})
            presets.append(preset_copy)
        return presets
    return [{"name": "default", "override": {}}]

def run_step_09_single_preset(lm, image_path: Path, preset: dict) -> tuple[str, np.ndarray]:
    base_step_config = get_step_09_base_config(lm)
    preset_name = str(preset.get("name", "unnamed variation"))
    override = copy.deepcopy(preset.get("override", {}))
    step_config = lm.context.deep_merge(base_step_config, override)
    lm.set_step_config(step_config)
    lm.ensure_dirs(cleanup=False)
    metadata_path = resolve_step_09_metadata_path(image_path.name, step_config)
    result = lm.process_metadata_file(metadata_path)
    panel_path = PROJECT_ROOT / str(result["comparison_path"] or result["overlay_path"])
    panel = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if panel is None: raise ValueError(f"Could not read Step 09 panel: {panel_path}")
    angle_text = "none" if result["canting_angle_deg"] is None else f"{result['canting_angle_deg']:+.3f}"
    label = (
        f"{preset_name} | angle={angle_text} "
        f"conf={result['measurement_confidence_percent']:.1f}% "
        f"decision={result['decision']}"
    )
    return label, panel

def run_step_09_variations(image_path: Path, combo: int | None = None) -> None:
    presets = build_step_09_presets()
    presets = select_combo_items(presets, combo)
    results = []
    print("Step 09 test variations:")
    lm = load_step_09_module()
    for index, preset in enumerate(presets, start=1):
        label, panel = run_step_09_single_preset(lm=lm, image_path=image_path, preset=preset)
        print(f"  #{index}: {label}")
        results.append((f"#{index} | {label}", panel))
    print()
    output_dir = step_image_output_dir(step=9, image_path=image_path)
    save_labeled_results(output_dir=output_dir, results=results)
    show_variation_pages(title_prefix=f"Step 09 canting angle | {image_path.name}", results=results, save_dir=output_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visually test all configured parameter variations for one step and one image.")
    parser.add_argument("--step",type=int,required=True,choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],help="Pipeline step to test. Supported: 1, 2, 3, 4, 5, 6, 7, 8, 9")
    parser.add_argument("--image",type=str,default=None,help="Image filename, for example IMG_0502.png.")
    parser.add_argument( "--config",type=str,default=None,help="Optional config path relative to project root, for example config/pipeline_config_step06_test.yaml." )
    parser.add_argument( "--workers",type=int,  default=None,   help="Optional worker count for Step 06 processing. Currently unused.")
    parser.add_argument("--combo",type=int,default=None,help="Optional 1-based variation index. Example: --combo 1 runs only the first variation.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_config(args.config)
    if not args.image: raise ValueError("--image is required.")
    image_path = resolve_image_path(step=args.step,image_name=args.image)

    print()
    print("Running visual test variations")
    print(f"Step:  {args.step}")
    print(f"Image: {image_path}")
    print(f"Config: {args.config or 'config/pipeline_config.yaml'}")
    print("Test files will be saved under data/test.")
    print()

    if args.step == 1: run_step_01_variations(image_path, combo=args.combo)
    elif args.step == 2: run_step_02_variations(image_path, combo=args.combo)
    elif args.step == 3: run_step_03_variations(image_path, combo=args.combo)
    elif args.step == 4: run_step_04_variations(image_path, combo=args.combo)
    elif args.step == 5: run_step_05_variations(image_path, combo=args.combo)
    elif args.step == 6: run_step_06_variations(image_path, workers=args.workers, combo=args.combo)
    elif args.step == 7: run_step_07_variations(image_path, combo=args.combo)
    elif args.step == 8: run_step_08_variations(image_path, combo=args.combo)
    elif args.step == 9: run_step_09_variations(image_path, combo=args.combo)

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
