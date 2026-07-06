from pathlib import Path
import csv
import itertools

import cv2
import numpy as np

from config_loader import deep_merge_dict, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_02_CONFIG = CONFIG["step_02_grayscale_and_blur"]
STEP_03_CONFIG = CONFIG["step_03_edge_detection"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

STEP_02_OUTPUT_DIR = PROCESSED_DIR / STEP_02_CONFIG["output_subdir"]
SELECTED_STEP_02_OUTPUT = str(
    STEP_03_CONFIG.get("selected_input", STEP_02_CONFIG["selected_output"])
).strip()
STEP_03_TEST_INPUT_NAME = str(STEP_02_CONFIG["selected_output"]).strip()

INPUT_DIR = STEP_02_OUTPUT_DIR / SELECTED_STEP_02_OUTPUT
TEST_INPUT_DIR = STEP_02_OUTPUT_DIR / STEP_03_TEST_INPUT_NAME
OUTPUT_DIR = PROCESSED_DIR / STEP_03_CONFIG["output_subdir"]

CLEANED_DIR = OUTPUT_DIR / "cleaned"

CSV_PATH = METADATA_DIR / "processing_03_edge_detection.csv"


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    image_paths = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    return sorted(image_paths)


def load_variant_input_image(
    image_name: str,
    variant: dict,
) -> np.ndarray:
    use_step_03_test_input = bool(variant.get("use_step_03_test_input", False))

    if use_step_03_test_input:
        image_path = TEST_INPUT_DIR / image_name
    else:
        image_path = INPUT_DIR / image_name

    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        raise ValueError(f"Could not read variant input image: {image_path}")

    return image_gray


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])

    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def to_bgr_for_display(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    image = to_bgr_for_display(image).copy()

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
    input_image: np.ndarray,
    raw_edges: np.ndarray,
    cleaned_edges: np.ndarray,
) -> np.ndarray:
    input_display = resize_for_display(to_bgr_for_display(input_image))
    raw_display = resize_for_display(to_bgr_for_display(raw_edges))
    cleaned_display = resize_for_display(to_bgr_for_display(cleaned_edges))

    target_height = min(
        input_display.shape[0],
        raw_display.shape[0],
        cleaned_display.shape[0],
    )

    displays = [
        input_display,
        raw_display,
        cleaned_display,
    ]

    resized_displays = []

    for image in displays:
        height, width = image.shape[:2]
        new_width = int(width * target_height / height)

        resized = cv2.resize(
            image,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        resized_displays.append(resized)

    labeled_images = [
        add_label(resized_displays[0], f"input: {SELECTED_STEP_02_OUTPUT}"),
        add_label(resized_displays[1], "raw canny"),
        add_label(resized_displays[2], "cleaned edges"),
    ]

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)

    combined = labeled_images[0]

    for image in labeled_images[1:]:
        combined = np.hstack([combined, separator, image])

    return combined


def validate_aperture_size(value: int) -> int:
    allowed_values = {3, 5, 7}

    if value not in allowed_values:
        raise ValueError(
            f"Canny aperture_size must be one of {allowed_values}. Got: {value}"
        )

    return value


def validate_odd_kernel_size_pair(value: list[int] | tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values. Got: {value}")

    kernel_width = int(value[0])
    kernel_height = int(value[1])

    if kernel_width == 0 and kernel_height == 0:
        return 0, 0

    if kernel_width < 1 or kernel_height < 1:
        raise ValueError(f"{name} values must be positive or [0, 0]. Got: {value}")

    if kernel_width % 2 == 0 or kernel_height % 2 == 0:
        raise ValueError(f"{name} values must be odd. Got: {value}")

    return kernel_width, kernel_height


def validate_odd_kernel_size(value: int, name: str) -> int:
    value = int(value)

    if value == 0:
        return 0

    if value < 1:
        raise ValueError(f"{name} must be positive or 0. Got: {value}")

    if value % 2 == 0:
        raise ValueError(f"{name} must be odd. Got: {value}")

    return value


def get_canny_mode() -> str:
    canny_config = STEP_03_CONFIG["canny"]
    mode = canny_config.get("mode", "manual")

    return str(mode).strip().lower()


def get_selected_edge_output() -> str:
    selected_output = STEP_03_CONFIG.get("selected_output", "cleaned")

    selected_output = str(selected_output).strip().lower()

    if selected_output not in {"raw", "cleaned"}:
        raise ValueError(
            f"Unsupported step_03 selected_output: {selected_output}. "
            "Supported: raw, cleaned"
        )

    return selected_output


def build_step_03_test_presets(step_config: dict) -> list[dict]:
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

            current_value = node[base_key]
            test_values = list(value)

            if current_value not in test_values:
                test_values.insert(0, current_value)

            test_dimensions.append((path + (base_key,), test_values))

    walk(step_config)

    if not test_dimensions:
        return [{"name": "default", "override": {}}]

    presets = []

    for combination in itertools.product(*(values for _, values in test_dimensions)):
        override = {}
        label_parts = []

        for (path, _), selected_value in zip(test_dimensions, combination):
            current = override

            for key in path[:-1]:
                current = current.setdefault(key, {})

            current[path[-1]] = selected_value
            label_parts.append(f"{'.'.join(path)}={selected_value}")

        presets.append(
            {
                "name": " | ".join(label_parts),
                "override": override,
            }
        )

    return presets


def get_saved_test_variants() -> list[dict]:
    variants = []
    test_presets = build_step_03_test_presets(STEP_03_CONFIG)

    for index, variant in enumerate(STEP_03_CONFIG.get("saved_test_variants", []), start=1):
        variant_name = str(variant.get("name", f"variant_{index:02d}")).strip()
        variant_output_subdir = str(
            variant.get(
                "output_subdir",
                f"{STEP_03_CONFIG['output_subdir']}/{variant_name}",
            )
        ).strip()
        preset_number = variant.get("from_test_preset_number")

        if preset_number is not None:
            preset_index = int(preset_number) - 1

            if preset_index < 0 or preset_index >= len(test_presets):
                raise ValueError(
                    f"Invalid from_test_preset_number={preset_number} for variant '{variant_name}'. "
                    f"Valid range: 1-{len(test_presets)}"
                )

            variant_override = test_presets[preset_index]["override"]
            variant_source = f"test preset #{preset_number}: {test_presets[preset_index]['name']}"
        else:
            variant_override = variant.get("override", {})
            variant_source = "explicit override"

        variant_step_config = deep_merge_dict(STEP_03_CONFIG, variant_override)

        variants.append(
            {
                "name": variant_name,
                "output_subdir": variant_output_subdir,
                "output_dir": PROCESSED_DIR / variant_output_subdir,
                "step_config": variant_step_config,
                "source": variant_source,
                "use_step_03_test_input": bool(variant.get("use_step_03_test_input", False)),
            }
        )

    return variants


def maybe_preprocess_before_canny(image_gray: np.ndarray) -> np.ndarray:
    preprocessing_config = STEP_03_CONFIG.get("preprocessing", {})
    enabled = bool(preprocessing_config.get("enabled", False))

    if not enabled:
        return image_gray

    gaussian_kernel_size = validate_odd_kernel_size(
        int(preprocessing_config.get("gaussian_kernel_size", 3)),
        "preprocessing.gaussian_kernel_size",
    )

    gaussian_sigma_x = float(preprocessing_config.get("gaussian_sigma_x", 0.0))

    if gaussian_kernel_size == 0:
        return image_gray

    return cv2.GaussianBlur(
        image_gray,
        (gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma_x,
    )


def calculate_auto_canny_thresholds(image_gray: np.ndarray) -> tuple[int, int]:
    canny_config = STEP_03_CONFIG["canny"]

    sigma = float(canny_config.get("auto_sigma", 0.33))
    median_intensity = float(np.median(image_gray))

    threshold_1 = int(max(0, (1.0 - sigma) * median_intensity))
    threshold_2 = int(min(255, (1.0 + sigma) * median_intensity))

    if threshold_1 == threshold_2:
        threshold_1 = max(0, threshold_1 - 10)
        threshold_2 = min(255, threshold_2 + 10)

    return threshold_1, threshold_2


def get_manual_canny_thresholds() -> tuple[int, int]:
    canny_config = STEP_03_CONFIG["canny"]

    threshold_1 = int(canny_config["threshold_1"])
    threshold_2 = int(canny_config["threshold_2"])

    if threshold_1 < 0 or threshold_2 < 0:
        raise ValueError(
            f"Canny thresholds must be non-negative. Got: {threshold_1}, {threshold_2}"
        )

    if threshold_1 >= threshold_2:
        raise ValueError(
            f"Canny threshold_1 must be lower than threshold_2. "
            f"Got: {threshold_1}, {threshold_2}"
        )

    return threshold_1, threshold_2


def get_canny_thresholds(image_gray: np.ndarray) -> tuple[int, int]:
    mode = get_canny_mode()

    if mode == "manual":
        return get_manual_canny_thresholds()

    if mode == "auto_median":
        return calculate_auto_canny_thresholds(image_gray)

    raise ValueError(
        f"Unsupported Canny mode: {mode}. Supported: manual, auto_median"
    )


def run_canny(image_gray: np.ndarray) -> tuple[np.ndarray, int, int]:
    canny_config = STEP_03_CONFIG["canny"]

    canny_input = maybe_preprocess_before_canny(image_gray)

    threshold_1, threshold_2 = get_canny_thresholds(canny_input)
    aperture_size = validate_aperture_size(int(canny_config["aperture_size"]))
    use_l2_gradient = bool(canny_config["use_l2_gradient"])

    edges = cv2.Canny(
        image=canny_input,
        threshold1=threshold_1,
        threshold2=threshold_2,
        apertureSize=aperture_size,
        L2gradient=use_l2_gradient,
    )

    return edges, threshold_1, threshold_2


def render_edges_for_config(
    image_gray: np.ndarray,
    step_config: dict,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    global STEP_03_CONFIG

    previous_step_config = STEP_03_CONFIG
    STEP_03_CONFIG = step_config

    try:
        raw_edges, threshold_1, threshold_2 = run_canny(image_gray)
        cleaned_edges = clean_edges(raw_edges)
        selected_output = get_selected_edge_output()
    finally:
        STEP_03_CONFIG = previous_step_config

    return raw_edges, cleaned_edges, threshold_1, threshold_2, selected_output


def make_kernel(kernel_size: tuple[int, int]) -> np.ndarray | None:
    kernel_width, kernel_height = kernel_size

    if kernel_width == 0 and kernel_height == 0:
        return None

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_width, kernel_height),
    )


def clean_edges(raw_edges: np.ndarray) -> np.ndarray:
    postprocessing_config = STEP_03_CONFIG.get("postprocessing", {})
    enabled = bool(postprocessing_config.get("enabled", False))

    if not enabled:
        return raw_edges.copy()

    cleaned = raw_edges.copy()

    dilate_iterations = int(postprocessing_config.get("dilate_iterations", 0))

    if dilate_iterations > 0:
        dilate_kernel_size = validate_odd_kernel_size_pair(
            postprocessing_config.get("dilate_kernel_size", [3, 3]),
            "postprocessing.dilate_kernel_size",
        )

        dilate_kernel = make_kernel(dilate_kernel_size)

        if dilate_kernel is not None:
            cleaned = cv2.dilate(
                cleaned,
                dilate_kernel,
                iterations=dilate_iterations,
            )

    close_kernel_size = validate_odd_kernel_size_pair(
        postprocessing_config.get("close_kernel_size", [0, 0]),
        "postprocessing.close_kernel_size",
    )

    close_kernel = make_kernel(close_kernel_size)

    if close_kernel is not None:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

    open_kernel_size = validate_odd_kernel_size_pair(
        postprocessing_config.get("open_kernel_size", [0, 0]),
        "postprocessing.open_kernel_size",
    )

    open_kernel = make_kernel(open_kernel_size)

    if open_kernel is not None:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            open_kernel,
        )

    erode_iterations = int(postprocessing_config.get("erode_iterations", 0))

    if erode_iterations > 0:
        erode_kernel_size = validate_odd_kernel_size_pair(
            postprocessing_config.get("erode_kernel_size", [3, 3]),
            "postprocessing.erode_kernel_size",
        )

        erode_kernel = make_kernel(erode_kernel_size)

        if erode_kernel is not None:
            cleaned = cv2.erode(
                cleaned,
                erode_kernel,
                iterations=erode_iterations,
            )

    return cleaned


def get_postprocessing_metadata() -> dict:
    postprocessing_config = STEP_03_CONFIG.get("postprocessing", {})

    return {
        "postprocessing_enabled": bool(postprocessing_config.get("enabled", False)),
        "dilate_iterations": int(postprocessing_config.get("dilate_iterations", 0)),
        "dilate_kernel_size": str(postprocessing_config.get("dilate_kernel_size", [3, 3])),
        "close_kernel_size": str(postprocessing_config.get("close_kernel_size", [0, 0])),
        "open_kernel_size": str(postprocessing_config.get("open_kernel_size", [0, 0])),
        "erode_iterations": int(postprocessing_config.get("erode_iterations", 0)),
        "erode_kernel_size": str(postprocessing_config.get("erode_kernel_size", [3, 3])),
    }


def get_preprocessing_metadata() -> dict:
    preprocessing_config = STEP_03_CONFIG.get("preprocessing", {})

    return {
        "preprocessing_enabled": bool(preprocessing_config.get("enabled", False)),
        "preprocessing_gaussian_kernel_size": int(
            preprocessing_config.get("gaussian_kernel_size", 3)
        ),
        "preprocessing_gaussian_sigma_x": float(
            preprocessing_config.get("gaussian_sigma_x", 0.0)
        ),
    }


def save_metadata(rows: list[dict]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "cleaned_output_file",
        "roi_output_file",
        "width",
        "height",
        "processing_step",
        "input_from_step_02",
        "canny_mode",
        "threshold_1",
        "threshold_2",
        "aperture_size",
        "use_l2_gradient",
        "auto_sigma",
        "preprocessing_enabled",
        "preprocessing_gaussian_kernel_size",
        "preprocessing_gaussian_sigma_x",
        "postprocessing_enabled",
        "dilate_iterations",
        "dilate_kernel_size",
        "close_kernel_size",
        "open_kernel_size",
        "erode_iterations",
        "erode_kernel_size",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not STEP_03_CONFIG["enabled"]:
        print("Step 03 is disabled in config.")
        return

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    canny_config = STEP_03_CONFIG["canny"]
    canny_mode = get_canny_mode()
    aperture_size = validate_aperture_size(int(canny_config["aperture_size"]))
    use_l2_gradient = bool(canny_config["use_l2_gradient"])
    auto_sigma = float(canny_config.get("auto_sigma", 0.33))
    saved_test_variants = get_saved_test_variants()

    for variant in saved_test_variants:
        variant["output_dir"].mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    print()
    print("Processing step 03: edge detection")
    print(f"Input:  {INPUT_DIR}")
    print(f"Step 03 test input: {TEST_INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Cleaned output: {CLEANED_DIR}")
    print(f"Selected step 2 output: {SELECTED_STEP_02_OUTPUT}")
    print(f"Canny mode: {canny_mode}")
    if saved_test_variants:
        print("Extra saved test variants:")
        for variant in saved_test_variants:
            variant_input = TEST_INPUT_DIR if variant["use_step_03_test_input"] else INPUT_DIR
            print(f"  {variant['name']} -> {variant['output_dir']} ({variant['source']}, input={variant_input})")
    print()
    print("Controls:")
    print("  n / SPACE / ENTER  -> next image")
    print("  q / ESC            -> quit")
    print()

    preprocessing_metadata = get_preprocessing_metadata()
    postprocessing_metadata = get_postprocessing_metadata()

    for index, image_path in enumerate(image_paths, start=1):
        input_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if input_image is None:
            print(f"Could not read image: {image_path}")
            continue

        raw_edges, cleaned_edges, threshold_1, threshold_2, _ = render_edges_for_config(
            input_image,
            STEP_03_CONFIG,
        )

        cleaned_output_path = CLEANED_DIR / image_path.name
        cv2.imwrite(str(cleaned_output_path), cleaned_edges)
        roi_output_file = ""

        for variant in saved_test_variants:
            variant_input_image = load_variant_input_image(image_path.name, variant)
            variant_raw_edges, variant_cleaned_edges, _, _, variant_selected_output = render_edges_for_config(
                variant_input_image,
                variant["step_config"],
            )

            variant_output_path = variant["output_dir"] / image_path.name
            variant_output = (
                variant_raw_edges
                if variant_selected_output == "raw"
                else variant_cleaned_edges
            )

            cv2.imwrite(str(variant_output_path), variant_output)
            if variant["name"] == "roi_edges":
                roi_output_file = str(variant_output_path.relative_to(PROJECT_ROOT))

        height, width = input_image.shape[:2]

        metadata_row = {
            "source_file": str(image_path.relative_to(PROJECT_ROOT)),
            "cleaned_output_file": str(cleaned_output_path.relative_to(PROJECT_ROOT)),
            "roi_output_file": roi_output_file,
            "width": width,
            "height": height,
            "processing_step": "03_edge_detection",
            "input_from_step_02": SELECTED_STEP_02_OUTPUT,
            "canny_mode": canny_mode,
            "threshold_1": threshold_1,
            "threshold_2": threshold_2,
            "aperture_size": aperture_size,
            "use_l2_gradient": use_l2_gradient,
            "auto_sigma": auto_sigma,
        }

        metadata_row.update(preprocessing_metadata)
        metadata_row.update(postprocessing_metadata)

        metadata_rows.append(metadata_row)

        print(
            f"[{index}/{len(image_paths)}] Saved: {image_path.name} | "
            f"thresholds=({threshold_1}, {threshold_2})"
        )

        if DISPLAY_CONFIG["show_windows"]:
            comparison = make_comparison_view(
                input_image,
                raw_edges,
                cleaned_edges,
            )

            title = (
                f"03 Edge detection | "
                f"{index}/{len(image_paths)} | "
                f"{image_path.name}"
            )

            cv2.imshow(title, comparison)

            if DISPLAY_CONFIG["wait_between_images"]:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(500) & 0xFF

            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass

            if key in [ord("q"), 27]:
                print("Stopped by user.")
                break

    save_metadata(metadata_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Cleaned edges saved to: {CLEANED_DIR}")
    for variant in saved_test_variants:
        print(f"Saved test variant {variant['name']}: {variant['output_dir']}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
