from __future__ import annotations

from . import context
from .axis_metrics import build_axis_quality
from .context import (
    PROJECT_ROOT,
    STEP_CONFIG,
    apply_preset,
    ensure_dirs,
    get_step_dirs,
    save_json,
    set_step_config,
)
from .measurement import combine_measurement_confidence, compute_canting, compute_line_angle_uncertainty
from .processing import collect_metadata_files, process_metadata_file, show_image
from .table_line import detect_table_line_with_stability

detect_table_line = detect_table_line_with_stability

__all__ = [
    "PROJECT_ROOT",
    "STEP_CONFIG",
    "apply_preset",
    "build_axis_quality",
    "combine_measurement_confidence",
    "collect_metadata_files",
    "compute_canting",
    "compute_line_angle_uncertainty",
    "context",
    "detect_table_line",
    "ensure_dirs",
    "get_step_dirs",
    "process_metadata_file",
    "save_json",
    "set_step_config",
    "show_image",
]
