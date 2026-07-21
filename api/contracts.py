from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AnalyzeRequest:
    image_path: str
    response_mode: str = "json"
    keep_artifacts: bool = False
    include_step_logs: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalyzeRequest":
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")

        image_path = payload.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("Field 'image_path' is required and must be a non-empty string.")

        response_mode = payload.get("response_mode", "json")
        if response_mode not in {"json", "binary"}:
            raise ValueError("Field 'response_mode' must be either 'json' or 'binary'.")

        keep_artifacts = payload.get("keep_artifacts", False)
        if not isinstance(keep_artifacts, bool):
            raise ValueError("Field 'keep_artifacts' must be a boolean.")

        include_step_logs = payload.get("include_step_logs", False)
        if not isinstance(include_step_logs, bool):
            raise ValueError("Field 'include_step_logs' must be a boolean.")

        return cls(
            image_path=image_path.strip(),
            response_mode=response_mode,
            keep_artifacts=keep_artifacts,
            include_step_logs=include_step_logs,
        )


@dataclass(slots=True)
class StepExecutionLog:
    step: str
    script_name: str
    elapsed_ms: float
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "script_name": self.script_name,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(slots=True)
class AnalyzeResult:
    image_name: str
    input_image_path: str
    processing_time_ms: float
    overlay_png_bytes: bytes
    metadata: dict[str, Any]
    artifacts_dir: str | None
    overlay_output_path: str | None
    metadata_output_path: str | None
    step_logs: list[StepExecutionLog]

    def to_json_payload(
        self,
        persistence: dict[str, Any],
        include_step_logs: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_name": self.image_name,
            "input_image_path": self.input_image_path,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "overlay_data_url": (
                "data:image/png;base64,"
                + base64.b64encode(self.overlay_png_bytes).decode("ascii")
            ),
            "artifacts_dir": self.artifacts_dir,
            "overlay_output_path": self.overlay_output_path,
            "metadata_output_path": self.metadata_output_path,
            "persistence": persistence,
        }

        if include_step_logs:
            payload["step_logs"] = [item.to_dict() for item in self.step_logs]

        return payload
