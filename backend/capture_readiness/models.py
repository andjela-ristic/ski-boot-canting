from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Candidate:
    x: int
    y: int
    width: int
    height: int
    area: int
    score: float
    source: str
    center_offset_ratio: float
    height_ratio: float
    width_ratio: float
    area_ratio: float
    bottom_ratio: float
    touches_top: bool
    touches_left: bool
    touches_right: bool


@dataclass(slots=True)
class ValidationResult:
    success: bool
    score: float
    reason: str | None
    checks: dict[str, bool]
    metrics: dict[str, Any]
    latency_ms: float
    source_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    debug_image_base64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
