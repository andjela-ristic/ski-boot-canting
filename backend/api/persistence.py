from __future__ import annotations

from typing import Any, Protocol

from .contracts import AnalyzeResult


class AnalysisRepository(Protocol):
    def save_analysis(self, result: AnalyzeResult) -> dict[str, Any]:
        ...


class NoopAnalysisRepository:
    def save_analysis(self, result: AnalyzeResult) -> dict[str, Any]:
        return {
            "saved": False,
            "backend": "noop",
            "message": "Persistence is intentionally disabled for now.",
        }
