from __future__ import annotations

import itertools
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from backend.pipeline.config_loader import deep_merge_dict

from . import context
from .display import make_comparison_view
from .io import load_grayscale_image, relative_project_path, save_metadata

def validate_aperture_size(value: int) -> int:
    allowed_values = {3, 5, 7}
    if value not in allowed_values: raise ValueError(f"Canny aperture_size must be one of {allowed_values}. Got: {value}")
    return value

def validate_odd_kernel_size_pair(value: list[int] | tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2: raise ValueError(f"{name} must contain exactly two values. Got: {value}")
    kernel_width = int(value[0])
    kernel_height = int(value[1])
    if kernel_width == 0 and kernel_height == 0: return 0, 0
    if kernel_width < 1 or kernel_height < 1: raise ValueError(f"{name} values must be positive or [0, 0]. Got: {value}")
    if kernel_width % 2 == 0 or kernel_height % 2 == 0: raise ValueError(f"{name} values must be odd. Got: {value}")
    return kernel_width, kernel_height

def validate_odd_kernel_size(value: int, name: str) -> int:
    value = int(value)
    if value == 0: return 0
    if value < 1: raise ValueError(f"{name} must be positive or 0. Got: {value}")
    if value % 2 == 0: raise ValueError(f"{name} must be odd. Got: {value}")
    return value

def get_canny_mode(step_config: dict | None = None) -> str:
    config = context.STEP_03_CONFIG if step_config is None else step_config
    mode = config["canny"].get("mode", "manual")
    return str(mode).strip().lower()

def get_selected_edge_output(step_config: dict | None = None) -> str:
    config = context.STEP_03_CONFIG if step_config is None else step_config
    selected_output = str(config.get("selected_output", "cleaned")).strip().lower()
    if selected_output not in {"raw", "cleaned"}:
        raise ValueError(f"Unsupported step_03 selected_output: {selected_output}. Supported: raw, cleaned")
    return selected_output

def build_step_03_test_presets(step_config: dict) -> list[dict]:
    test_dimensions: list[tuple[tuple[str, ...], list[Any]]] = []

    def walk(node: dict, path: tuple[str, ...] = ()) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, path + (key,))
                continue
            if not key.endswith("_test_values"): continue
            base_key = key.removesuffix("_test_values")
            if base_key not in node: continue
            current_value = node[base_key]
            test_values = list(value)
            if current_value not in test_values: test_values.insert(0, current_value)
            test_dimensions.append((path + (base_key,), test_values))

    walk(step_config)
    if not test_dimensions: return [{"name": "default", "override": {}}]

    presets = []
    for combination in itertools.product(*(values for _, values in test_dimensions)):
        override: dict[str, Any] = {}
        label_parts = []
        for (path, _), selected_value in zip(test_dimensions, combination):
            current = override
            for key in path[:-1]: current = current.setdefault(key, {})
            current[path[-1]] = selected_value
            label_parts.append(f"{'.'.join(path)}={selected_value}")
        presets.append({"name": " | ".join(label_parts), "override": override})
    return presets

def get_saved_test_variants() -> list[dict]:
    configured_variants = context.STEP_03_CONFIG.get("saved_test_variants", [])
    variants = []
    needs_test_presets = any(variant.get("from_test_preset_number") is not None for variant in configured_variants)
    test_presets = build_step_03_test_presets(context.STEP_03_CONFIG) if needs_test_presets else []

    for index, variant in enumerate(configured_variants, start=1):
        variant_name = str(variant.get("name", f"variant_{index:02d}")).strip()
        variant_output_subdir = str(variant.get("output_subdir", f"{context.STEP_03_CONFIG['output_subdir']}/{variant_name}")).strip()
        preset_number = variant.get("from_test_preset_number")

        if preset_number is not None:
            preset_index = int(preset_number) - 1
            if preset_index < 0 or preset_index >= len(test_presets):
                raise ValueError(f"Invalid from_test_preset_number={preset_number} for variant '{variant_name}'. Valid range: 1-{len(test_presets)}")
            variant_override = test_presets[preset_index]["override"]
            variant_source = f"test preset #{preset_number}: {test_presets[preset_index]['name']}"
        else:
            variant_override = variant.get("override", {})
            variant_source = "explicit override"

        variant_step_config = deep_merge_dict(context.STEP_03_CONFIG, variant_override)
        use_step_03_test_input = bool(variant.get("use_step_03_test_input", False))
        input_dir = context.TEST_INPUT_DIR if use_step_03_test_input else context.INPUT_DIR
        variants.append(
            {
                "name": variant_name,
                "output_subdir": variant_output_subdir,
                "output_dir": context.PROCESSED_DIR / variant_output_subdir,
                "input_dir": input_dir,
                "step_config": variant_step_config,
                "source": variant_source,
                "use_step_03_test_input": use_step_03_test_input,
            }
        )

    return variants

def make_kernel(kernel_size: tuple[int, int]) -> np.ndarray | None:
    kernel_width, kernel_height = kernel_size
    if kernel_width == 0 and kernel_height == 0: return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))

def get_manual_canny_thresholds(step_config: dict | None = None) -> tuple[int, int]:
    config = context.STEP_03_CONFIG if step_config is None else step_config
    canny_config = config["canny"]
    threshold_1 = int(canny_config["threshold_1"])
    threshold_2 = int(canny_config["threshold_2"])
    if threshold_1 < 0 or threshold_2 < 0:
        raise ValueError(f"Canny thresholds must be non-negative. Got: {threshold_1}, {threshold_2}")
    if threshold_1 >= threshold_2:
        raise ValueError(f"Canny threshold_1 must be lower than threshold_2. Got: {threshold_1}, {threshold_2}")
    return threshold_1, threshold_2

def compile_step_config(step_config: dict) -> dict[str, Any]:
    canny_config = step_config["canny"]
    canny_mode = get_canny_mode(step_config)
    if canny_mode not in {"manual", "auto_median"}:
        raise ValueError(f"Unsupported Canny mode: {canny_mode}. Supported: manual, auto_median")

    manual_thresholds = get_manual_canny_thresholds(step_config) if canny_mode == "manual" else None
    preprocessing_config = step_config.get("preprocessing", {})
    preprocessing_enabled = bool(preprocessing_config.get("enabled", False))
    gaussian_kernel_size = int(preprocessing_config.get("gaussian_kernel_size", 3))
    gaussian_sigma_x = float(preprocessing_config.get("gaussian_sigma_x", 0.0))
    if preprocessing_enabled: gaussian_kernel_size = validate_odd_kernel_size(gaussian_kernel_size, "preprocessing.gaussian_kernel_size")

    postprocessing_config = step_config.get("postprocessing", {})
    postprocessing_enabled = bool(postprocessing_config.get("enabled", False))
    dilate_iterations = int(postprocessing_config.get("dilate_iterations", 0))
    erode_iterations = int(postprocessing_config.get("erode_iterations", 0))
    dilate_kernel = None
    close_kernel = None
    open_kernel = None
    erode_kernel = None

    if postprocessing_enabled:
        if dilate_iterations > 0:
            dilate_kernel = make_kernel(validate_odd_kernel_size_pair(postprocessing_config.get("dilate_kernel_size", [3, 3]), "postprocessing.dilate_kernel_size"))
        close_kernel = make_kernel(validate_odd_kernel_size_pair(postprocessing_config.get("close_kernel_size", [0, 0]), "postprocessing.close_kernel_size"))
        open_kernel = make_kernel(validate_odd_kernel_size_pair(postprocessing_config.get("open_kernel_size", [0, 0]), "postprocessing.open_kernel_size"))
        if erode_iterations > 0:
            erode_kernel = make_kernel(validate_odd_kernel_size_pair(postprocessing_config.get("erode_kernel_size", [3, 3]), "postprocessing.erode_kernel_size"))

    morphology_operations = []
    if postprocessing_enabled:
        if dilate_iterations > 0 and dilate_kernel is not None: morphology_operations.append(("dilate", dilate_kernel, dilate_iterations))
        if close_kernel is not None: morphology_operations.append(("close", close_kernel, 1))
        if open_kernel is not None: morphology_operations.append(("open", open_kernel, 1))
        if erode_iterations > 0 and erode_kernel is not None: morphology_operations.append(("erode", erode_kernel, erode_iterations))

    return {
        "step_config": step_config,
        "selected_output": get_selected_edge_output(step_config),
        "canny_mode": canny_mode,
        "manual_thresholds": manual_thresholds,
        "auto_sigma": float(canny_config.get("auto_sigma", 0.33)),
        "aperture_size": validate_aperture_size(int(canny_config["aperture_size"])),
        "use_l2_gradient": bool(canny_config["use_l2_gradient"]),
        "preprocessing_enabled": preprocessing_enabled,
        "gaussian_kernel_size": gaussian_kernel_size,
        "gaussian_sigma_x": gaussian_sigma_x,
        "postprocessing_enabled": postprocessing_enabled,
        "morphology_operations": morphology_operations,
    }

def maybe_preprocess_before_canny(image_gray: np.ndarray, compiled_config: dict[str, Any] | None = None) -> np.ndarray:
    runtime = compile_step_config(context.STEP_03_CONFIG) if compiled_config is None else compiled_config
    if not runtime["preprocessing_enabled"]: return image_gray
    gaussian_kernel_size = runtime["gaussian_kernel_size"]
    if gaussian_kernel_size == 0: return image_gray
    return cv2.GaussianBlur(image_gray, (gaussian_kernel_size, gaussian_kernel_size), runtime["gaussian_sigma_x"])

def calculate_auto_canny_thresholds(image_gray: np.ndarray, step_config: dict | None = None, auto_sigma: float | None = None) -> tuple[int, int]:
    if auto_sigma is None:
        config = context.STEP_03_CONFIG if step_config is None else step_config
        auto_sigma = float(config["canny"].get("auto_sigma", 0.33))

    median_intensity = float(np.median(image_gray))
    threshold_1 = int(max(0, (1.0 - auto_sigma) * median_intensity))
    threshold_2 = int(min(255, (1.0 + auto_sigma) * median_intensity))
    if threshold_1 == threshold_2:
        threshold_1 = max(0, threshold_1 - 10)
        threshold_2 = min(255, threshold_2 + 10)
    return threshold_1, threshold_2

def get_canny_thresholds(image_gray: np.ndarray, compiled_config: dict[str, Any] | None = None) -> tuple[int, int]:
    runtime = compile_step_config(context.STEP_03_CONFIG) if compiled_config is None else compiled_config
    if runtime["canny_mode"] == "manual": return runtime["manual_thresholds"]
    return calculate_auto_canny_thresholds(image_gray, auto_sigma=runtime["auto_sigma"])

def run_canny(image_gray: np.ndarray, compiled_config: dict[str, Any] | None = None) -> tuple[np.ndarray, int, int]:
    runtime = compile_step_config(context.STEP_03_CONFIG) if compiled_config is None else compiled_config
    canny_input = maybe_preprocess_before_canny(image_gray, runtime)
    threshold_1, threshold_2 = get_canny_thresholds(canny_input, runtime)
    edges = cv2.Canny(image=canny_input, threshold1=threshold_1, threshold2=threshold_2, apertureSize=runtime["aperture_size"], L2gradient=runtime["use_l2_gradient"])
    return edges, threshold_1, threshold_2

class EdgeProcessor:
    def __init__(self, step_config: dict):
        self.runtime = compile_step_config(step_config)

    @property
    def selected_output(self) -> str:
        return self.runtime["selected_output"]

    def clean(self, raw_edges: np.ndarray, *, preserve_raw: bool = True) -> np.ndarray:
        operations = self.runtime["morphology_operations"]
        if not self.runtime["postprocessing_enabled"] or not operations: return raw_edges.copy() if preserve_raw else raw_edges
        cleaned = raw_edges.copy() if preserve_raw else raw_edges

        for operation, kernel, iterations in operations:
            if operation == "dilate":
                cv2.dilate(cleaned, kernel, dst=cleaned, iterations=iterations)
            elif operation == "close":
                cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, dst=cleaned)
            elif operation == "open":
                cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, dst=cleaned)
            elif operation == "erode":
                cv2.erode(cleaned, kernel, dst=cleaned, iterations=iterations)
            else:
                raise ValueError(f"Unsupported morphology operation: {operation}")

        return cleaned

    def render(self, image_gray: np.ndarray, *, require_cleaned: bool = True, preserve_raw: bool = True) -> tuple[np.ndarray, np.ndarray, int, int, str]:
        raw_edges, threshold_1, threshold_2 = run_canny(image_gray, self.runtime)
        cleaned_edges = self.clean(raw_edges, preserve_raw=preserve_raw) if require_cleaned or self.selected_output == "cleaned" else raw_edges
        return raw_edges, cleaned_edges, threshold_1, threshold_2, self.selected_output

def clean_edges(raw_edges: np.ndarray, compiled_config: dict[str, Any] | None = None) -> np.ndarray:
    runtime = compile_step_config(context.STEP_03_CONFIG) if compiled_config is None else compiled_config
    processor = EdgeProcessor.__new__(EdgeProcessor)
    processor.runtime = runtime
    return processor.clean(raw_edges, preserve_raw=True)

def render_edges_for_config(image_gray: np.ndarray, step_config: dict) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    return EdgeProcessor(step_config).render(image_gray, require_cleaned=True, preserve_raw=True)

def get_postprocessing_metadata(step_config: dict | None = None) -> dict:
    config = context.STEP_03_CONFIG if step_config is None else step_config
    postprocessing_config = config.get("postprocessing", {})
    return {
        "postprocessing_enabled": bool(postprocessing_config.get("enabled", False)),
        "dilate_iterations": int(postprocessing_config.get("dilate_iterations", 0)),
        "dilate_kernel_size": str(postprocessing_config.get("dilate_kernel_size", [3, 3])),
        "close_kernel_size": str(postprocessing_config.get("close_kernel_size", [0, 0])),
        "open_kernel_size": str(postprocessing_config.get("open_kernel_size", [0, 0])),
        "erode_iterations": int(postprocessing_config.get("erode_iterations", 0)),
        "erode_kernel_size": str(postprocessing_config.get("erode_kernel_size", [3, 3])),
    }

def get_preprocessing_metadata(step_config: dict | None = None) -> dict:
    config = context.STEP_03_CONFIG if step_config is None else step_config
    preprocessing_config = config.get("preprocessing", {})
    return {
        "preprocessing_enabled": bool(preprocessing_config.get("enabled", False)),
        "preprocessing_gaussian_kernel_size": int(preprocessing_config.get("gaussian_kernel_size", 3)),
        "preprocessing_gaussian_sigma_x": float(preprocessing_config.get("gaussian_sigma_x", 0.0)),
    }

def process_images(image_paths: list, *, debug: bool = False) -> None:
    context.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    context.METADATA_DIR.mkdir(parents=True, exist_ok=True)

    cv2.setUseOptimized(True)
    show_windows = debug and bool(context.DISPLAY_CONFIG.get("show_windows", True))
    main_processor = EdgeProcessor(context.STEP_03_CONFIG)
    main_runtime = main_processor.runtime
    canny_mode = main_runtime["canny_mode"]
    aperture_size = main_runtime["aperture_size"]
    use_l2_gradient = main_runtime["use_l2_gradient"]
    auto_sigma = main_runtime["auto_sigma"]
    saved_test_variants = get_saved_test_variants()
    variants_by_input: dict[Any, list[dict]] = {}

    for variant in saved_test_variants:
        variant["output_dir"].mkdir(parents=True, exist_ok=True)
        variant["processor"] = EdgeProcessor(variant["step_config"])
        variants_by_input.setdefault(variant["input_dir"], []).append(variant)

    print()
    print("Processing step 03: edge detection")
    print(f"Input:  {context.INPUT_DIR}")
    print(f"Step 03 test input: {context.TEST_INPUT_DIR}")
    print(f"Output: {context.OUTPUT_DIR}")
    print(f"Cleaned output: {context.CLEANED_DIR}")
    print(f"Selected step 2 output: {context.SELECTED_STEP_02_OUTPUT}")
    print(f"Canny mode: {canny_mode}")
    if saved_test_variants:
        print("Extra saved test variants:")
        for variant in saved_test_variants:
            print(f"  {variant['name']} -> {variant['output_dir']} ({variant['source']}, input={variant['input_dir']})")
    if show_windows:
        print()
        print("Debug controls:")
        print("  n / SPACE / ENTER  -> next image")
        print("  q / ESC            -> quit")
    print()

    preprocessing_metadata = get_preprocessing_metadata()
    postprocessing_metadata = get_postprocessing_metadata()
    total_images = len(image_paths)
    metadata_rows = []

    for index, image_path in enumerate(image_paths, start=1):
        total_started = perf_counter()
        try:
            read_started = perf_counter()
            input_image = load_grayscale_image(image_path)
            read_time_ms = (perf_counter() - read_started) * 1000.0
        except ValueError:
            print(f"Could not read image: {image_path}")
            continue

        height, width = input_image.shape[:2]
        processing_started = perf_counter()
        raw_edges, cleaned_edges, threshold_1, threshold_2, _ = main_processor.render(input_image, require_cleaned=True, preserve_raw=show_windows)
        processing_time_ms = (perf_counter() - processing_started) * 1000.0

        cleaned_output_path = context.CLEANED_DIR / image_path.name
        roi_output_file = ""
        stop_requested = False

        if show_windows:
            comparison = make_comparison_view(input_image, raw_edges, cleaned_edges)
            title = f"03 Edge detection | {index}/{total_images} | {image_path.name}"
            cv2.imshow(title, comparison)
            key = cv2.waitKey(0 if context.DISPLAY_CONFIG["wait_between_images"] else 500) & 0xFF
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass
            stop_requested = key in [ord("q"), 27]
            del comparison

        write_started = perf_counter()
        cv2.imwrite(str(cleaned_output_path), cleaned_edges)
        raw_edges = None
        cleaned_edges = None
        keep_main_input = context.INPUT_DIR in variants_by_input
        if not keep_main_input: input_image = None

        for variant_input_dir, variants in variants_by_input.items():
            if variant_input_dir == context.INPUT_DIR and input_image is not None:
                variant_input_image = input_image
            else:
                variant_input_path = variant_input_dir / image_path.name
                try:
                    variant_input_image = load_grayscale_image(variant_input_path)
                except ValueError as error:
                    raise ValueError(f"Could not read variant input image: {variant_input_path}") from error

            for variant in variants:
                variant_processor: EdgeProcessor = variant["processor"]
                variant_selected_output = variant_processor.selected_output
                variant_raw_edges, variant_cleaned_edges, _, _, _ = variant_processor.render(
                    variant_input_image,
                    require_cleaned=variant_selected_output == "cleaned",
                    preserve_raw=False,
                )
                variant_output_path = variant["output_dir"] / image_path.name
                variant_output = variant_raw_edges if variant_selected_output == "raw" else variant_cleaned_edges
                cv2.imwrite(str(variant_output_path), variant_output)
                if variant["name"] == "roi_edges": roi_output_file = relative_project_path(variant_output_path)
                variant_raw_edges = None
                variant_cleaned_edges = None
                variant_output = None

            if variant_input_dir != context.INPUT_DIR: variant_input_image = None

        input_image = None
        write_time_ms = (perf_counter() - write_started) * 1000.0
        total_time_ms = (perf_counter() - total_started) * 1000.0

        metadata_row = {
            "source_file": relative_project_path(image_path),
            "cleaned_output_file": relative_project_path(cleaned_output_path),
            "roi_output_file": roi_output_file,
            "width": width,
            "height": height,
            "processing_step": "03_edge_detection",
            "input_from_step_02": context.SELECTED_STEP_02_OUTPUT,
            "canny_mode": canny_mode,
            "threshold_1": threshold_1,
            "threshold_2": threshold_2,
            "aperture_size": aperture_size,
            "use_l2_gradient": use_l2_gradient,
            "auto_sigma": auto_sigma,
            "read_time_ms": round(read_time_ms, 3),
            "processing_time_ms": round(processing_time_ms, 3),
            "write_time_ms": round(write_time_ms, 3),
            "total_time_ms": round(total_time_ms, 3),
        }
        metadata_row.update(preprocessing_metadata)
        metadata_row.update(postprocessing_metadata)
        metadata_rows.append(metadata_row)

        print(
            f"[{index}/{total_images}] Saved: {image_path.name} | "
            f"thresholds=({threshold_1}, {threshold_2}) | "
            f"read={read_time_ms:.1f} ms, "
            f"process={processing_time_ms:.1f} ms, "
            f"write={write_time_ms:.1f} ms, "
            f"total={total_time_ms:.1f} ms"
        )

        if stop_requested:
            print("Stopped by user.")
            break

    save_metadata(metadata_rows)
    if show_windows: cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Cleaned edges saved to: {context.CLEANED_DIR}")
    for variant in saved_test_variants:
        print(f"Saved test variant {variant['name']}: {variant['output_dir']}")
    print(f"Metadata saved to: {context.CSV_PATH}")
