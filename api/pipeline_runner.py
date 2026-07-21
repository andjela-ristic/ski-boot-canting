from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Callable

import cv2
import yaml

from .contracts import AnalyzeResult, StepExecutionLog
from .exceptions import ApiError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
RUNTIME_ROOT = PROJECT_ROOT / "api" / ".runtime"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True, slots=True)
class PipelineStep:
    step: str
    script_name: str
    args_factory: Callable[[str], list[str]]
    expected_outputs_factory: Callable[[Path, str], list[Path]]


def _image_filter_args(image_name: str) -> list[str]:
    return ["--image", image_name]


def _step01_outputs(job_dir: Path, image_name: str) -> list[Path]:
    return [job_dir / "processed" / "01_illumination_normalized" / image_name]


def _step02_outputs(job_dir: Path, image_name: str) -> list[Path]:
    return [
        job_dir / "processed" / "02_grayscale_blur" / "grayscale_lab_l" / image_name,
        job_dir / "processed" / "02_grayscale_blur" / "bilateral_filter" / image_name,
    ]


def _step03_outputs(job_dir: Path, image_name: str) -> list[Path]:
    return [
        job_dir / "processed" / "03_edges" / "cleaned" / image_name,
        job_dir / "processed" / "03_edges" / "roi_edges" / image_name,
    ]


def _step04_outputs(job_dir: Path, image_name: str) -> list[Path]:
    return [job_dir / "processed" / "04_boot_roi_from_edges" / "mask" / image_name]


def _step05_outputs(job_dir: Path, image_name: str) -> list[Path]:
    stem = Path(image_name).stem
    return [
        job_dir / "processed" / "05_valid_hough_lines_in_roi" / "valid_lines_overlay" / image_name,
        job_dir / "processed" / "05_valid_hough_lines_in_roi" / "valid_lines_json" / f"{stem}.json",
    ]


def _step06_outputs(job_dir: Path, image_name: str) -> list[Path]:
    stem = Path(image_name).stem
    return [job_dir / "processed" / "06_search_central_ruler" / "metadata" / f"{stem}_central_ruler.json"]


def _step07_outputs(job_dir: Path, image_name: str) -> list[Path]:
    stem = Path(image_name).stem
    return [job_dir / "processed" / "07_verify_central_ruler_symmetry" / "metadata" / f"{stem}_symmetry.json"]


def _step08_outputs(job_dir: Path, image_name: str) -> list[Path]:
    stem = Path(image_name).stem
    return [job_dir / "processed" / "08_multi_validate_central_ruler" / "metadata" / f"{stem}_multi_validation.json"]


def _step09_outputs(job_dir: Path, image_name: str) -> list[Path]:
    stem = Path(image_name).stem
    return [
        job_dir / "processed" / "09_measure_canting_angle" / "overlay" / image_name,
        job_dir / "processed" / "09_measure_canting_angle" / "metadata" / f"{stem}_canting_angle.json",
    ]


PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        step="01_illumination_normalization",
        script_name="01_illumination_normalization.py",
        args_factory=lambda _image_name: [],
        expected_outputs_factory=_step01_outputs,
    ),
    PipelineStep(
        step="02_grayscale_and_blur",
        script_name="02_grayscale_and_blur.py",
        args_factory=lambda _image_name: [],
        expected_outputs_factory=_step02_outputs,
    ),
    PipelineStep(
        step="03_edge_detection",
        script_name="03_edge_detection.py",
        args_factory=lambda _image_name: [],
        expected_outputs_factory=_step03_outputs,
    ),
    PipelineStep(
        step="04_boot_roi_from_edges",
        script_name="04_detect_boot_roi_from_edges.py",
        args_factory=_image_filter_args,
        expected_outputs_factory=_step04_outputs,
    ),
    PipelineStep(
        step="05_valid_hough_lines_in_roi",
        script_name="05_detect_valid_hough_lines_in_roi.py",
        args_factory=_image_filter_args,
        expected_outputs_factory=_step05_outputs,
    ),
    PipelineStep(
        step="06_search_central_ruler",
        script_name="06_search_central_ruler.py",
        args_factory=_image_filter_args,
        expected_outputs_factory=_step06_outputs,
    ),
    PipelineStep(
        step="07_verify_central_ruler_symmetry",
        script_name="07_verify_central_ruler_symmetry.py",
        args_factory=_image_filter_args,
        expected_outputs_factory=_step07_outputs,
    ),
    PipelineStep(
        step="08_multi_validate_central_ruler",
        script_name="08_multi_validate_central_ruler.py",
        args_factory=_image_filter_args,
        expected_outputs_factory=_step08_outputs,
    ),
    PipelineStep(
        step="09_measure_canting_angle",
        script_name="09_measure_canting_angle.py",
        args_factory=_image_filter_args,
        expected_outputs_factory=_step09_outputs,
    ),
)


@lru_cache(maxsize=1)
def _load_base_config() -> dict[str, Any]:
    loader_path = PIPELINE_DIR / "config_loader.py"
    spec = importlib.util.spec_from_file_location("api_pipeline_config_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ApiError(500, f"Could not load pipeline config helper from {loader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config("config/pipeline_config.yaml")


def _tail_text(text: str, max_chars: int = 6000) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:]


class PipelineRunner:
    def analyze_image(self, image_path: str | Path, keep_artifacts: bool = False) -> AnalyzeResult:
        resolved_image_path = self._resolve_input_path(image_path)
        image_name = resolved_image_path.name
        job_dir = self._make_job_dir()

        try:
            copied_image = self._prepare_job_input(job_dir, resolved_image_path)
            config_path = self._write_job_config(job_dir)

            started = time.perf_counter()
            step_logs = self._run_pipeline(job_dir, copied_image.name, config_path)
            processing_time_ms = (time.perf_counter() - started) * 1000.0

            overlay_path = job_dir / "processed" / "09_measure_canting_angle" / "overlay" / copied_image.name
            metadata_path = (
                job_dir
                / "processed"
                / "09_measure_canting_angle"
                / "metadata"
                / f"{copied_image.stem}_canting_angle.json"
            )

            if not overlay_path.exists():
                raise ApiError(
                    500,
                    "Pipeline finished without producing Step 09 overlay.",
                    details={
                        "expected_overlay_path": str(overlay_path),
                        "step_logs": [item.to_dict() for item in step_logs],
                    },
                )

            if not metadata_path.exists():
                raise ApiError(
                    500,
                    "Pipeline finished without producing Step 09 metadata.",
                    details={
                        "expected_metadata_path": str(metadata_path),
                        "step_logs": [item.to_dict() for item in step_logs],
                    },
                )

            overlay_png_bytes = overlay_path.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            artifacts_dir = str(job_dir) if keep_artifacts else None
            overlay_output_path = str(overlay_path) if keep_artifacts else None
            metadata_output_path = str(metadata_path) if keep_artifacts else None

            return AnalyzeResult(
                image_name=image_name,
                input_image_path=str(resolved_image_path),
                processing_time_ms=processing_time_ms,
                overlay_png_bytes=overlay_png_bytes,
                metadata=metadata,
                artifacts_dir=artifacts_dir,
                overlay_output_path=overlay_output_path,
                metadata_output_path=metadata_output_path,
                step_logs=step_logs,
            )
        finally:
            if not keep_artifacts and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    def _resolve_input_path(self, image_path: str | Path) -> Path:
        path = Path(image_path)
        resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
        resolved = resolved.resolve()

        if not resolved.exists():
            raise ApiError(400, "Input image path does not exist.", details={"image_path": str(resolved)})
        if not resolved.is_file():
            raise ApiError(400, "Input image path must point to a file.", details={"image_path": str(resolved)})
        if resolved.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ApiError(
                400,
                "Only .png, .jpg, and .jpeg inputs are supported by this API.",
                details={"image_path": str(resolved)},
            )

        image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if image is None:
            raise ApiError(
                400,
                "Input image could not be decoded by OpenCV.",
                details={"image_path": str(resolved)},
            )

        return resolved

    def _make_job_dir(self) -> Path:
        job_dir = RUNTIME_ROOT / "jobs" / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir

    def _prepare_job_input(self, job_dir: Path, image_path: Path) -> Path:
        working_dir = job_dir / "working_png"
        metadata_dir = job_dir / "metadata"
        processed_dir = job_dir / "processed"

        working_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        copied_path = working_dir / image_path.name
        shutil.copy2(image_path, copied_path)
        return copied_path

    def _write_job_config(self, job_dir: Path) -> Path:
        config = json.loads(json.dumps(_load_base_config()))

        config["paths"] = dict(config.get("paths", {}))
        config["paths"]["working_png_dir"] = str((job_dir / "working_png").resolve())
        config["paths"]["processed_dir"] = str((job_dir / "processed").resolve())
        config["paths"]["metadata_dir"] = str((job_dir / "metadata").resolve())

        display = dict(config.get("display", {}))
        display["show_windows"] = False
        display["wait_between_images"] = False
        config["display"] = display

        config_path = job_dir / "pipeline_config.api.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

        return config_path

    def _run_pipeline(
        self,
        job_dir: Path,
        image_name: str,
        config_path: Path,
    ) -> list[StepExecutionLog]:
        logs: list[StepExecutionLog] = []
        env = os.environ.copy()
        env["PIPELINE_CONFIG"] = str(config_path)
        env["PYTHONIOENCODING"] = "utf-8"

        python_executable = sys.executable
        if not python_executable:
            raise ApiError(500, "Could not determine the Python executable for pipeline subprocesses.")

        for step in PIPELINE_STEPS:
            command = [
                python_executable,
                str(PIPELINE_DIR / step.script_name),
                *step.args_factory(image_name),
            ]

            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            step_log = StepExecutionLog(
                step=step.step,
                script_name=step.script_name,
                elapsed_ms=elapsed_ms,
                stdout=_tail_text(completed.stdout),
                stderr=_tail_text(completed.stderr),
            )
            logs.append(step_log)

            if completed.returncode != 0:
                raise ApiError(
                    500,
                    f"Pipeline step failed: {step.step}",
                    details={
                        "script_name": step.script_name,
                        "return_code": completed.returncode,
                        "stdout": step_log.stdout,
                        "stderr": step_log.stderr,
                    },
                )

            self._assert_expected_outputs(job_dir, step, image_name, step_log)

        return logs

    def _assert_expected_outputs(
        self,
        job_dir: Path,
        step: PipelineStep,
        image_name: str,
        step_log: StepExecutionLog,
    ) -> None:
        expected_outputs = step.expected_outputs_factory(job_dir, image_name)
        missing = [str(path) for path in expected_outputs if not path.exists()]

        if missing:
            raise ApiError(
                500,
                f"Pipeline step completed but expected outputs are missing: {step.step}",
                details={
                    "script_name": step.script_name,
                    "missing_outputs": missing,
                    "stdout": step_log.stdout,
                    "stderr": step_log.stderr,
                },
            )
