from __future__ import annotations

from . import context
from .context import CLEANED_DIR, CSV_PATH, INPUT_DIR, OUTPUT_DIR, PROJECT_ROOT, STEP_03_CONFIG, TEST_INPUT_DIR, set_step_config
from .display import add_label, make_comparison_view, resize_for_display, to_bgr_for_display
from .io import collect_images, load_grayscale_image, load_variant_input_image, relative_project_path, save_metadata
from .processing import EdgeProcessor, build_step_03_test_presets, calculate_auto_canny_thresholds, clean_edges, compile_step_config, get_canny_mode, get_manual_canny_thresholds, get_postprocessing_metadata, get_preprocessing_metadata, get_saved_test_variants, get_selected_edge_output, maybe_preprocess_before_canny, process_images, render_edges_for_config, run_canny, validate_aperture_size, validate_odd_kernel_size, validate_odd_kernel_size_pair

__all__ = [
    "CLEANED_DIR",
    "CSV_PATH",
    "EdgeProcessor",
    "INPUT_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "STEP_03_CONFIG",
    "TEST_INPUT_DIR",
    "add_label",
    "build_step_03_test_presets",
    "calculate_auto_canny_thresholds",
    "clean_edges",
    "collect_images",
    "compile_step_config",
    "context",
    "get_canny_mode",
    "get_manual_canny_thresholds",
    "get_postprocessing_metadata",
    "get_preprocessing_metadata",
    "get_saved_test_variants",
    "get_selected_edge_output",
    "load_grayscale_image",
    "load_variant_input_image",
    "make_comparison_view",
    "maybe_preprocess_before_canny",
    "process_images",
    "relative_project_path",
    "render_edges_for_config",
    "resize_for_display",
    "run_canny",
    "save_metadata",
    "set_step_config",
    "to_bgr_for_display",
    "validate_aperture_size",
    "validate_odd_kernel_size",
    "validate_odd_kernel_size_pair",
]
