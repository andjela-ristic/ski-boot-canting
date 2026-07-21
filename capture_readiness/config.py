from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True, slots=True)
class GuideConfig:
    x_min_ratio: float = 0.20
    x_max_ratio: float = 0.80
    y_min_ratio: float = 0.10
    y_max_ratio: float = 0.90


@dataclass(frozen=True, slots=True)
class QualityConfig:
    # Preview frames are frequently compressed and slightly soft, so these
    # limits reject only clearly unusable frames.
    min_sharpness: float = 8.0
    min_mean_brightness: float = 12.0
    max_mean_brightness: float = 245.0
    max_dark_ratio: float = 0.90
    max_bright_ratio: float = 0.90
    dark_pixel_threshold: int = 8
    bright_pixel_threshold: int = 247


@dataclass(frozen=True, slots=True)
class BootConfig:
    # Geometry limits are intentionally tolerant. The final measurement
    # pipeline remains responsible for precise canting validation.
    min_height_ratio: float = 0.38
    max_height_ratio: float = 0.995
    min_width_ratio: float = 0.14
    max_width_ratio: float = 0.92
    min_area_ratio: float = 0.050
    max_area_ratio: float = 0.90
    max_center_offset_ratio: float = 0.16
    min_bottom_ratio: float = 0.54
    min_side_margin_ratio: float = 0.0
    min_top_margin_ratio: float = 0.0
    min_candidate_score: float = 0.40
    scale_tolerance_ratio: float = 0.055
    canny_low: int = 32
    canny_high: int = 110
    gaussian_kernel: int = 5
    close_kernel: int = 7
    open_kernel: int = 3


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    required: bool = True
    orientation: str = "horizontal"
    exclude_guide_from_search: bool = True
    x_min_ratio: float = 0.00
    x_max_ratio: float = 1.00
    # The table/platform edge in real preview frames can sit above 68%.
    y_min_ratio: float = 0.52
    y_max_ratio: float = 0.99
    max_angle_error_deg: float = 12.0
    min_total_length_ratio: float = 0.14
    min_best_length_ratio: float = 0.075
    min_segment_length_ratio: float = 0.055
    max_line_gap_ratio: float = 0.10
    canny_low: int = 28
    canny_high: int = 105
    hough_threshold: int = 14

    # Projection fallback catches a long table edge even when Hough splits it
    # into weak or short segments.
    projection_fallback_enabled: bool = True
    projection_min_strength: float = 2.6
    projection_min_coverage_ratio: float = 0.12

    # A clearly detected, complete, centered boot may pass the fast preview
    # check if the line detector alone is uncertain. The slower pipeline still
    # performs the precise reference-line validation.
    allow_strong_boot_fallback: bool = True
    strong_boot_fallback_score: float = 0.78


@dataclass(frozen=True, slots=True)
class ReadinessConfig:
    processing_max_width: int = 480
    jpeg_max_bytes: int = 3_000_000
    opencv_threads: int = 1
    success_score_threshold: float = 0.72
    guide: GuideConfig = field(default_factory=GuideConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    boot: BootConfig = field(default_factory=BootConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    values = values or {}
    allowed = cls.__dataclass_fields__.keys()
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> ReadinessConfig:
    if path is None:
        return ReadinessConfig()

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))

    top_level = {
        key: value
        for key, value in data.items()
        if key not in {"guide", "quality", "boot", "reference"}
    }

    allowed = ReadinessConfig.__dataclass_fields__.keys()
    unknown = set(top_level) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown ReadinessConfig fields: {sorted(unknown)}")

    return ReadinessConfig(
        **top_level,
        guide=_merge_dataclass(GuideConfig, data.get("guide")),
        quality=_merge_dataclass(QualityConfig, data.get("quality")),
        boot=_merge_dataclass(BootConfig, data.get("boot")),
        reference=_merge_dataclass(ReferenceConfig, data.get("reference")),
    )
