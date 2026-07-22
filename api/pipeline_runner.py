from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import math
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import re
import shutil
import subprocess
import sys
import threading
import time
import tempfile
import uuid
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from capture_readiness import FrameValidator, load_config as load_capture_readiness_config
from capture_readiness.validator import FrameValidationError

from .contracts import (
    AnalyzeResult,
    FailedFrameAnalysisResult,
    FrameAnalysisResult,
    StepExecutionLog,
)
from .exceptions import ApiError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
RUNTIME_ROOT = Path(
    os.environ.get(
        "API_RUNTIME_ROOT",
        str(Path(tempfile.gettempdir()) / "ml-ski-boot-canting-api"),
    )
).expanduser().resolve()
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True, slots=True)
class PipelineStep:
    step: str
    script_name: str
    args_factory: Callable[[str], list[str]]
    expected_outputs_factory: Callable[[Path, str], list[Path]]


class PipelineExecutionCoordinator:
    def __init__(self, max_parallel: int, reserved_high_priority_slots: int) -> None:
        self.max_parallel = max_parallel
        self.reserved_high_priority_slots = reserved_high_priority_slots
        self._active_total = 0
        self._active_low_priority = 0
        self._pending_high_priority = 0
        self._condition = threading.Condition()

    @property
    def low_priority_capacity(self) -> int:
        return max(1, self.max_parallel - self.reserved_high_priority_slots)

    @contextmanager
    def acquire(self, priority: str) -> Any:
        is_high_priority = priority == "high"

        with self._condition:
            if is_high_priority:
                self._pending_high_priority += 1
                try:
                    while self._active_total >= self.max_parallel:
                        self._condition.wait()
                    self._pending_high_priority -= 1
                    self._active_total += 1
                except Exception:
                    self._pending_high_priority -= 1
                    self._condition.notify_all()
                    raise
            else:
                while (
                    self._active_total >= self.max_parallel
                    or self._active_low_priority >= self.low_priority_capacity
                    or self._pending_high_priority > 0
                ):
                    self._condition.wait()
                self._active_total += 1
                self._active_low_priority += 1

        try:
            yield
        finally:
            with self._condition:
                self._active_total -= 1
                if not is_high_priority:
                    self._active_low_priority -= 1
                self._condition.notify_all()


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


@lru_cache(maxsize=1)
def _load_capture_readiness_validator() -> FrameValidator:
    api_config = _load_base_config().get("api", {})
    readiness_config = api_config.get("capture_readiness", {})

    configured_path = readiness_config.get("config_path")
    env_path = os.environ.get("CAPTURE_READINESS_CONFIG", "").strip()
    config_path = env_path or configured_path

    if config_path:
        resolved_config_path = PROJECT_ROOT / config_path
        if Path(config_path).is_absolute():
            resolved_config_path = Path(config_path)
        return FrameValidator(load_capture_readiness_config(resolved_config_path))

    return FrameValidator(load_capture_readiness_config())


@lru_cache(maxsize=1)
def _load_execution_coordinator() -> PipelineExecutionCoordinator:
    api_config = _load_base_config().get("api", {})
    execution_config = api_config.get("execution", {})

    max_parallel = execution_config.get("max_parallel", 2)
    reserved_high_priority_slots = execution_config.get("reserved_high_priority_slots", 1)

    if not isinstance(max_parallel, int) or max_parallel <= 0:
        raise ApiError(500, "Config field 'api.execution.max_parallel' must be a positive integer.")
    if not isinstance(reserved_high_priority_slots, int) or reserved_high_priority_slots < 0:
        raise ApiError(
            500,
            "Config field 'api.execution.reserved_high_priority_slots' must be a non-negative integer.",
        )
    if reserved_high_priority_slots >= max_parallel:
        raise ApiError(
            500,
            "Config field 'api.execution.reserved_high_priority_slots' must be smaller than 'api.execution.max_parallel'.",
        )

    return PipelineExecutionCoordinator(
        max_parallel=max_parallel,
        reserved_high_priority_slots=reserved_high_priority_slots,
    )


class PipelineRunner:
    def analyze_capture_readiness_frame(
        self,
        frame_bytes: bytes,
        include_debug: bool = False,
        guide_scale: float = 1.0,
    ) -> dict[str, Any]:
        try:
            result = _load_capture_readiness_validator().validate_bytes(
                frame_bytes,
                include_debug=include_debug,
                guide_scale=guide_scale,
            )
        except FrameValidationError as exc:
            raise ApiError(422, str(exc)) from exc

        return result.to_dict()

    def analyze_image(
        self,
        image_path: str | Path,
        keep_artifacts: bool = False,
        priority: str = "high",
    ) -> AnalyzeResult:
        resolved_image_path = self._resolve_input_path(image_path)
        job_dir = self._make_job_dir()

        try:
            with _load_execution_coordinator().acquire(priority):
                copied_image = self._prepare_job_input(job_dir, resolved_image_path)
                return self._execute_analyze_job(
                    job_dir=job_dir,
                    working_image=copied_image,
                    input_image_path=str(resolved_image_path),
                    keep_artifacts=keep_artifacts,
                )
        finally:
            if not keep_artifacts and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    def analyze_uploaded_image(
        self,
        image_bytes: bytes,
        image_filename: str,
        keep_artifacts: bool = False,
        priority: str = "high",
    ) -> AnalyzeResult:
        if not image_bytes:
            raise ApiError(400, "Uploaded image is empty.")

        safe_image_name = self._sanitize_image_filename(image_filename)
        job_dir = self._make_job_dir()

        try:
            with _load_execution_coordinator().acquire(priority):
                uploaded_image = self._prepare_uploaded_image(job_dir, safe_image_name, image_bytes)
                return self._execute_analyze_job(
                    job_dir=job_dir,
                    working_image=uploaded_image,
                    input_image_path=f"uploaded://{safe_image_name}",
                    keep_artifacts=keep_artifacts,
                )
        finally:
            if not keep_artifacts and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    def analyze_video_frames(
        self,
        video_path: str | Path,
        keep_artifacts: bool = False,
        include_step_logs: bool = False,
    ) -> dict[str, Any]:
        resolved_video_path = self._resolve_video_path(video_path)
        extraction_dir = self._make_job_dir()

        try:
            return self._analyze_video_frames_job(
                resolved_video_path=resolved_video_path,
                extraction_dir=extraction_dir,
                keep_artifacts=keep_artifacts,
                include_step_logs=include_step_logs,
                requested_frame_count=None,
                input_video_path=str(resolved_video_path),
                video_name=resolved_video_path.name,
            )
        finally:
            if not keep_artifacts and extraction_dir.exists():
                shutil.rmtree(extraction_dir, ignore_errors=True)

    def analyze_uploaded_video(
        self,
        video_bytes: bytes,
        video_filename: str,
        keep_artifacts: bool = False,
        include_step_logs: bool = False,
        requested_frame_count: int | None = None,
    ) -> dict[str, Any]:
        if not video_bytes:
            raise ApiError(400, "Uploaded video is empty.")

        job_dir = self._make_job_dir()

        try:
            safe_video_name = self._sanitize_video_filename(video_filename)
            uploaded_video_path = self._prepare_uploaded_video(job_dir, safe_video_name, video_bytes)
            resolved_video_path = self._resolve_video_path(uploaded_video_path)

            return self._analyze_video_frames_job(
                resolved_video_path=resolved_video_path,
                extraction_dir=job_dir,
                keep_artifacts=keep_artifacts,
                include_step_logs=include_step_logs,
                requested_frame_count=requested_frame_count,
                input_video_path=f"uploaded://{safe_video_name}",
                video_name=safe_video_name,
            )
        finally:
            if not keep_artifacts and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    def _analyze_video_frames_job(
        self,
        resolved_video_path: Path,
        extraction_dir: Path,
        keep_artifacts: bool,
        include_step_logs: bool,
        requested_frame_count: int | None,
        input_video_path: str,
        video_name: str,
    ) -> dict[str, Any]:
        frames_config = self._load_frames_config()
        sample_count = self._resolve_video_sample_count(
            frames_config=frames_config,
            requested_frame_count=requested_frame_count,
        )
        coordinator = _load_execution_coordinator()
        max_workers = min(
            sample_count,
            frames_config["max_workers"],
            coordinator.low_priority_capacity,
        )

        extracted_frame_paths = self._extract_sampled_frames(
            resolved_video_path,
            extraction_dir,
            sample_count,
        )

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_results = list(
                executor.map(
                    lambda item: self._analyze_extracted_frame_safe(item, keep_artifacts),
                    extracted_frame_paths,
                )
            )
        processing_time_ms = (time.perf_counter() - started) * 1000.0

        frame_results = sorted(
            [
                item
                for item in future_results
                if isinstance(item, FrameAnalysisResult)
            ],
            key=lambda item: item.frame_index,
        )
        frame_failures = sorted(
            [
                item
                for item in future_results
                if isinstance(item, FailedFrameAnalysisResult)
            ],
            key=lambda item: item.frame_index,
        )
        if not frame_results:
            raise ApiError(
                500,
                "All sampled video frames failed analysis.",
                details={
                    "video_path": str(resolved_video_path),
                    "sample_count": sample_count,
                    "frame_failures": [item.to_dict() for item in frame_failures],
                },
            )

        averaged_metadata = self._average_metadata_tree(
            [item.analysis.metadata for item in frame_results]
        )
        ordered_frame_results = self._order_video_frames(frame_results)
        best_frame = ordered_frame_results[0]

        return {
            "video_name": video_name,
            "video_path": str(resolved_video_path),
            "input_video_path": input_video_path,
            "processing_time_ms": round(processing_time_ms, 2),
            "overlay_data_url": self._png_bytes_to_data_url(
                best_frame.analysis.overlay_png_bytes
            ),
            "overlay_output_path": best_frame.analysis.overlay_output_path,
            "metadata_output_path": best_frame.analysis.metadata_output_path,
            "frame_count": len(frame_results),
            "sampled_frame_count": len(extracted_frame_paths),
            "failed_frame_count": len(frame_failures),
            "frame_sampling": {
                "sample_count": sample_count,
                "max_workers": max_workers,
                "requested_frame_count": requested_frame_count,
            },
            "selected_frame_index": best_frame.frame_index,
            "selected_timestamp_ms": round(best_frame.timestamp_ms, 2),
            "frames": [
                item.to_dict(include_step_logs=include_step_logs)
                for item in ordered_frame_results
            ],
            "frame_failures": [item.to_dict() for item in frame_failures],
            "average_metadata": averaged_metadata,
            "artifacts_dir": str(extraction_dir) if keep_artifacts else None,
        }

    def _resolve_video_sample_count(
        self,
        frames_config: dict[str, Any],
        requested_frame_count: int | None,
    ) -> int:
        sample_count = (
            requested_frame_count
            if requested_frame_count is not None
            else frames_config["sample_count"]
        )
        if not isinstance(sample_count, int) or not 1 <= sample_count <= 30:
            raise ApiError(
                400,
                "Field 'frame_count' must be an integer between 1 and 30.",
            )
        return sample_count

    def _choose_best_video_frame(
        self,
        frame_results: list[FrameAnalysisResult],
    ) -> FrameAnalysisResult:
        if not frame_results:
            raise ApiError(500, "Video analysis did not produce any frame results.")

        return self._order_video_frames(frame_results)[0]

    def _order_video_frames(
        self,
        frame_results: list[FrameAnalysisResult],
    ) -> list[FrameAnalysisResult]:
        if not frame_results:
            return []

        middle_index = (len(frame_results) - 1) / 2.0
        return sorted(
            frame_results,
            key=lambda item: (
                self._frame_weighted_priority(item),
                self._frame_absolute_canting_distance(item),
                -self._frame_measurement_confidence(item),
                -self._frame_table_line_quality(item),
                -self._frame_axis_quality(item),
                -self._frame_table_line_stability(item),
                self._frame_uncertainty_95_deg(item),
                abs(item.frame_index - middle_index),
                item.frame_index,
            ),
        )

    def _frame_weighted_priority(self, frame_result: FrameAnalysisResult) -> float:
        distance = self._frame_absolute_canting_distance(frame_result)
        if not math.isfinite(distance):
            return math.inf
        confidence = self._frame_measurement_confidence(frame_result)
        return 0.75 * distance - 0.25 * confidence

    def _frame_absolute_canting_distance(self, frame_result: FrameAnalysisResult) -> float:
        metadata = frame_result.analysis.metadata
        angle_measurement = metadata.get("angle_measurement", {})
        value = angle_measurement.get("absolute_canting_angle_deg")
        if value is None:
            signed = angle_measurement.get("canting_angle_deg")
            try:
                value = abs(float(signed))
            except (TypeError, ValueError):
                return math.inf

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return math.inf
        if not math.isfinite(numeric):
            return math.inf
        return max(0.0, numeric)

    def _frame_measurement_confidence(self, frame_result: FrameAnalysisResult) -> float:
        metadata = frame_result.analysis.metadata
        measurement = metadata.get("measurement_validation", {})
        return self._score_value(measurement.get("measurement_confidence_score"))

    def _frame_table_line_quality(self, frame_result: FrameAnalysisResult) -> float:
        metadata = frame_result.analysis.metadata
        table_line = metadata.get("table_line", {})
        return self._score_value(
            table_line.get("table_line_quality_score", table_line.get("score"))
        )

    def _frame_axis_quality(self, frame_result: FrameAnalysisResult) -> float:
        metadata = frame_result.analysis.metadata
        axis_validation = metadata.get("axis_validation", {})
        return self._score_value(axis_validation.get("axis_quality_score"))

    def _frame_table_line_stability(self, frame_result: FrameAnalysisResult) -> float:
        metadata = frame_result.analysis.metadata
        table_line_stability = metadata.get("table_line_stability", {})
        return self._score_value(table_line_stability.get("score"))

    def _frame_uncertainty_95_deg(self, frame_result: FrameAnalysisResult) -> float:
        metadata = frame_result.analysis.metadata
        uncertainty = metadata.get("angle_uncertainty", {})
        value = uncertainty.get("table_angle_uncertainty_95_deg")
        if value is None:
            return math.inf
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return math.inf
        if not math.isfinite(numeric):
            return math.inf
        return max(0.0, numeric)

    def _score_value(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(numeric):
            return 0.0
        return numeric

    def _resolve_input_path(self, image_path: str | Path) -> Path:
        resolved = self._resolve_existing_path(image_path)

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

    def _resolve_video_path(self, video_path: str | Path) -> Path:
        resolved = self._resolve_existing_path(video_path)

        if not resolved.exists():
            raise ApiError(400, "Input video path does not exist.", details={"video_path": str(resolved)})
        if not resolved.is_file():
            raise ApiError(400, "Input video path must point to a file.", details={"video_path": str(resolved)})
        if resolved.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ApiError(
                400,
                "Only .mp4, .mov, .avi, .mkv, .m4v, and .webm inputs are supported by this API.",
                details={"video_path": str(resolved)},
            )

        capture = cv2.VideoCapture(str(resolved))
        try:
            if not capture.isOpened():
                raise ApiError(
                    400,
                    "Input video could not be opened by OpenCV.",
                    details={"video_path": str(resolved)},
                )
        finally:
            capture.release()

        return resolved

    def _resolve_existing_path(self, raw_path: str | Path) -> Path:
        path_text = str(raw_path).strip()
        candidates: list[Path] = []

        direct_path = Path(path_text)
        candidates.append(direct_path if direct_path.is_absolute() else (PROJECT_ROOT / direct_path))

        for translated in self._translate_windows_host_path(path_text):
            candidates.append(translated)

        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            resolved_key = str(resolved)
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            if resolved.exists():
                return resolved

        return candidates[0].resolve()

    def _translate_windows_host_path(self, path_text: str) -> list[Path]:
        embedded_windows_path = self._extract_embedded_windows_path(path_text)
        if embedded_windows_path is None:
            return []

        windows_path = PureWindowsPath(embedded_windows_path)
        repo_name = PROJECT_ROOT.name.lower()
        lowered_parts = [part.lower() for part in windows_path.parts]
        translated: list[Path] = []

        if repo_name in lowered_parts:
            repo_index = lowered_parts.index(repo_name)
            repo_relative_parts = windows_path.parts[repo_index + 1 :]
            if repo_relative_parts:
                translated.append(PROJECT_ROOT.joinpath(*repo_relative_parts))

        if "data" in lowered_parts:
            data_index = lowered_parts.index("data")
            data_relative_parts = windows_path.parts[data_index + 1 :]
            translated.append(PROJECT_ROOT / "data" / Path(*data_relative_parts))

        return translated

    def _extract_embedded_windows_path(self, path_text: str) -> str | None:
        if WINDOWS_ABSOLUTE_PATH_RE.match(path_text):
            return path_text

        match = re.search(r"[A-Za-z]:[\\/].+", path_text)
        if match is not None:
            return match.group(0)

        return None

    def _make_job_dir(self) -> Path:
        job_dir = RUNTIME_ROOT / "jobs" / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir

    def _load_frames_config(self) -> dict[str, int]:
        api_config = _load_base_config().get("api", {})
        frames_config = api_config.get("frames", {})

        sample_count = frames_config.get("sample_count", 6)
        max_workers = frames_config.get("max_workers", 4)

        if not isinstance(sample_count, int) or not 6 <= sample_count <= 10:
            raise ApiError(
                500,
                "Config field 'api.frames.sample_count' must be an integer between 6 and 10.",
            )
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ApiError(
                500,
                "Config field 'api.frames.max_workers' must be a positive integer.",
            )

        return {
            "sample_count": sample_count,
            "max_workers": max_workers,
        }

    def _extract_sampled_frames(
        self,
        video_path: Path,
        job_dir: Path,
        sample_count: int,
    ) -> list[tuple[int, float, Path]]:
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ApiError(400, "Input video could not be opened by OpenCV.", details={"video_path": str(video_path)})

        try:
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if frame_total <= 0:
                raise ApiError(400, "Input video does not contain readable frames.", details={"video_path": str(video_path)})

            sample_indexes = self._build_sample_indexes(frame_total, sample_count)
            extracted: list[tuple[int, float, Path]] = []

            for output_index, frame_number in enumerate(sample_indexes):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ApiError(
                        400,
                        "Failed to decode one of the sampled frames.",
                        details={"video_path": str(video_path), "frame_number": frame_number},
                    )

                timestamp_ms = (
                    (frame_number / fps) * 1000.0
                    if fps > 0
                    else float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                )
                frame_path = frames_dir / f"frame_{output_index:02d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise ApiError(
                        500,
                        "Failed to persist extracted frame to disk.",
                        details={"frame_path": str(frame_path)},
                    )
                extracted.append((output_index, timestamp_ms, frame_path))

            return extracted
        finally:
            capture.release()

    def _build_sample_indexes(self, frame_total: int, sample_count: int) -> list[int]:
        if frame_total <= 0:
            raise ApiError(
                400,
                "Input video does not contain readable frames.",
                details={"frame_total": frame_total, "sample_count": sample_count},
            )

        candidate_frame_total = frame_total
        if frame_total > sample_count:
            # Exclude the terminal frame only when doing so does not reduce the
            # requested sample count. Very short captures should keep every
            # decodable frame in play.
            candidate_frame_total = frame_total - 1

        effective_sample_count = min(candidate_frame_total, sample_count)
        max_sample_index = candidate_frame_total - 1

        if effective_sample_count == 1:
            return [max(0, min(max_sample_index, candidate_frame_total // 2))]

        indexes = []
        for index in range(effective_sample_count):
            # Sample the midpoint of each temporal bucket from the decodable
            # range. The terminal frame is excluded only when there is enough
            # headroom to keep the requested number of samples unchanged.
            bucket_start = (index * candidate_frame_total) / effective_sample_count
            bucket_end = ((index + 1) * candidate_frame_total) / effective_sample_count
            midpoint = (bucket_start + bucket_end) * 0.5
            frame_number = int(
                np.clip(
                    round(midpoint - 0.5),
                    0,
                    max_sample_index,
                )
            )
            indexes.append(frame_number)

        deduplicated: list[int] = []
        for index in indexes:
            if not deduplicated or deduplicated[-1] != index:
                deduplicated.append(index)

        if len(deduplicated) < effective_sample_count:
            remaining_indexes = [
                index for index in range(candidate_frame_total) if index not in set(deduplicated)
            ]
            deduplicated.extend(remaining_indexes[: effective_sample_count - len(deduplicated)])

        return deduplicated

    def _analyze_extracted_frame(
        self,
        item: tuple[int, float, Path],
        keep_artifacts: bool,
    ) -> FrameAnalysisResult:
        frame_index, timestamp_ms, frame_path = item
        analysis = self.analyze_image(
            frame_path,
            keep_artifacts=keep_artifacts,
            priority="low",
        )
        return FrameAnalysisResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            analysis=analysis,
        )

    def _analyze_extracted_frame_safe(
        self,
        item: tuple[int, float, Path],
        keep_artifacts: bool,
    ) -> FrameAnalysisResult | FailedFrameAnalysisResult:
        try:
            return self._analyze_extracted_frame(item, keep_artifacts)
        except ApiError as exc:
            frame_index, timestamp_ms, _frame_path = item
            details = exc.to_dict().get("details")
            return FailedFrameAnalysisResult(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                error=exc.message,
                details=details if isinstance(details, dict) else {},
            )
        except Exception as exc:
            frame_index, timestamp_ms, _frame_path = item
            return FailedFrameAnalysisResult(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                error="Unexpected frame analysis error.",
                details={"message": str(exc)},
            )

    def _average_metadata_tree(self, values: list[dict[str, Any]]) -> dict[str, Any]:
        averaged = self._average_value(values)
        if not isinstance(averaged, dict):
            return {}
        return averaged

    def _average_value(self, values: list[Any]) -> Any:
        normalized = [value for value in values if value is not None]
        if not normalized:
            return None

        if all(self._is_number(value) for value in normalized):
            return round(sum(float(value) for value in normalized) / len(normalized), 6)

        if all(isinstance(value, dict) for value in normalized):
            keys: set[str] = set()
            for value in normalized:
                keys.update(value.keys())

            aggregated: dict[str, Any] = {}
            for key in sorted(keys):
                child = self._average_value(
                    [value.get(key) for value in normalized if key in value]
                )
                if child is not None:
                    aggregated[key] = child
            return aggregated

        if all(isinstance(value, list) for value in normalized):
            lengths = {len(value) for value in normalized}
            if len(lengths) != 1:
                return None

            aggregated_list: list[Any] = []
            for index in range(len(normalized[0])):
                child = self._average_value([value[index] for value in normalized])
                if child is None:
                    return None
                aggregated_list.append(child)
            return aggregated_list

        return None

    def _is_number(self, value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _prepare_job_input(self, job_dir: Path, image_path: Path) -> Path:
        working_dir = self._ensure_job_dirs(job_dir)
        copied_path = working_dir / image_path.name
        shutil.copy2(image_path, copied_path)
        return copied_path

    def _prepare_uploaded_image(self, job_dir: Path, image_name: str, image_bytes: bytes) -> Path:
        working_dir = self._ensure_job_dirs(job_dir)
        image_path = working_dir / image_name
        image_path.write_bytes(image_bytes)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ApiError(
                400,
                "Uploaded image could not be decoded by OpenCV.",
                details={"image_filename": image_name},
            )

        return image_path

    def _prepare_uploaded_video(self, job_dir: Path, video_name: str, video_bytes: bytes) -> Path:
        uploads_dir = job_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        video_path = uploads_dir / video_name
        video_path.write_bytes(video_bytes)
        return video_path

    def _ensure_job_dirs(self, job_dir: Path) -> Path:
        working_dir = job_dir / "working_png"
        metadata_dir = job_dir / "metadata"
        processed_dir = job_dir / "processed"

        working_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        return working_dir

    def _sanitize_image_filename(self, raw_filename: str) -> str:
        filename = Path(str(raw_filename).strip() or "capture.jpg").name
        if not filename:
            filename = "capture.jpg"

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            if suffix:
                raise ApiError(
                    400,
                    "Uploaded image must use one of: .png, .jpg, .jpeg",
                    details={"image_filename": filename},
                )
            filename = f"{filename}.jpg"

        return filename

    def _sanitize_video_filename(self, raw_filename: str) -> str:
        filename = Path(str(raw_filename).strip() or "capture.mp4").name
        if not filename:
            filename = "capture.mp4"

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            if suffix:
                raise ApiError(
                    400,
                    "Uploaded video must use one of: .mp4, .mov, .avi, .mkv, .m4v, .webm",
                    details={"video_filename": filename},
                )
            filename = f"{filename}.mp4"

        return filename

    def _build_uploaded_video_stub_overlay(
        self,
        video_name: str,
        extracted_frame_paths: list[tuple[int, float, Path]],
        clip_duration_ms: int | None = None,
    ) -> Any:
        if not extracted_frame_paths:
            raise ApiError(500, "No extracted frames were available for stub overlay generation.")

        middle_index = len(extracted_frame_paths) // 2
        _, highlight_timestamp_ms, highlight_frame_path = extracted_frame_paths[middle_index]
        frame = cv2.imread(str(highlight_frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ApiError(
                500,
                "Could not load sampled frame for stub overlay generation.",
                details={"frame_path": str(highlight_frame_path)},
            )

        overlay = frame.copy()
        banner = overlay.copy()
        cv2.rectangle(banner, (0, 0), (overlay.shape[1], 170), (8, 24, 32), thickness=-1)
        cv2.rectangle(
            banner,
            (0, overlay.shape[0] - 80),
            (overlay.shape[1], overlay.shape[0]),
            (8, 24, 32),
            thickness=-1,
        )
        overlay = cv2.addWeighted(banner, 0.72, overlay, 0.28, 0.0)

        sampled_count = len(extracted_frame_paths)
        sampled_timestamps = ", ".join(
            f"{timestamp_ms:.0f}ms" for _, timestamp_ms, _ in extracted_frame_paths[:6]
        )
        clip_duration_label = (
            f"{clip_duration_ms} ms"
            if clip_duration_ms is not None
            else "n/a"
        )

        text_color = (245, 245, 245)
        accent_color = (80, 225, 180)
        self._put_overlay_text(overlay, "POST /frames stub response", 24, 38, accent_color, 0.95, 2)
        self._put_overlay_text(overlay, f"video: {video_name}", 24, 72, text_color, 0.68, 2)
        self._put_overlay_text(overlay, f"requested clip: {clip_duration_label}", 24, 100, text_color, 0.68, 2)
        self._put_overlay_text(overlay, f"sampled frames: {sampled_count}", 24, 128, text_color, 0.68, 2)
        self._put_overlay_text(
            overlay,
            f"highlighted frame timestamp: {highlight_timestamp_ms:.0f} ms",
            24,
            156,
            text_color,
            0.68,
            2,
        )

        footer_text = f"sample timestamps: {sampled_timestamps}"
        self._put_overlay_text(
            overlay,
            footer_text,
            24,
            max(overlay.shape[0] - 28, 28),
            text_color,
            0.58,
            1,
        )

        center_x = overlay.shape[1] // 2
        center_y = overlay.shape[0] // 2
        cv2.circle(overlay, (center_x, center_y), 12, accent_color, thickness=3, lineType=cv2.LINE_AA)
        cv2.line(
            overlay,
            (center_x - 28, center_y),
            (center_x + 28, center_y),
            accent_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            overlay,
            (center_x, center_y - 28),
            (center_x, center_y + 28),
            accent_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return overlay

    def _put_overlay_text(
        self,
        image: Any,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font_scale: float,
        thickness: int,
    ) -> None:
        cv2.putText(
            image,
            text,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _encode_png_bytes(self, image: Any) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ApiError(500, "Failed to encode overlay image as PNG.")
        return encoded.tobytes()

    def _png_bytes_to_data_url(self, png_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    def _apply_request_pipeline_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Disable nonessential per-request artifacts without changing results."""
        step06 = dict(config.get("step_06_search_central_ruler", {}))
        step06["persistence"] = {
            **dict(step06.get("persistence", {})),
            "save_overlay": False,
            "save_comparison": False,
            "save_candidate_snapshots": False,
        }
        config["step_06_search_central_ruler"] = step06

        step07 = dict(config.get("step_07_verify_central_ruler_symmetry", {}))
        step07["persistence"] = {
            **dict(step07.get("persistence", {})),
            "save_overlay": False,
            "save_comparison": False,
            "save_candidate_snapshots": False,
            "save_rectified": False,
            "save_evaluation_mask": True,
            "save_evaluation_edge": False,
        }
        config["step_07_verify_central_ruler_symmetry"] = step07

        step08 = dict(config.get("step_08_multi_validate_central_ruler", {}))
        step08["persistence"] = {
            **dict(step08.get("persistence", {})),
            "save_overlay": False,
            "save_comparison": False,
            "save_diagnostic": False,
        }
        config["step_08_multi_validate_central_ruler"] = step08

        step09 = dict(config.get("step_09_measure_canting_angle", {}))
        step09["persistence"] = {
            **dict(step09.get("persistence", {})),
            "save_comparison": False,
            "save_diagnostic": False,
        }
        config["step_09_measure_canting_angle"] = step09
        return config

    def _write_job_config(self, job_dir: Path) -> Path:
        config = json.loads(json.dumps(_load_base_config()))

        config["paths"] = dict(config.get("paths", {}))
        config["paths"]["working_png_dir"] = str((job_dir / "working_png").resolve())
        config["paths"]["processed_dir"] = str((job_dir / "processed").resolve())
        config["paths"]["metadata_dir"] = str((job_dir / "metadata").resolve())
        config = self._apply_request_pipeline_overrides(config)

        display = dict(config.get("display", {}))
        display["show_windows"] = False
        display["wait_between_images"] = False
        config["display"] = display

        config_path = job_dir / "pipeline_config.api.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

        return config_path

    def _execute_analyze_job(
        self,
        *,
        job_dir: Path,
        working_image: Path,
        input_image_path: str,
        keep_artifacts: bool,
    ) -> AnalyzeResult:
        config_path = self._write_job_config(job_dir)

        started = time.perf_counter()
        step_logs = self._run_pipeline(job_dir, working_image.name, config_path)
        processing_time_ms = (time.perf_counter() - started) * 1000.0

        overlay_path = job_dir / "processed" / "09_measure_canting_angle" / "overlay" / working_image.name
        metadata_path = (
            job_dir
            / "processed"
            / "09_measure_canting_angle"
            / "metadata"
            / f"{working_image.stem}_canting_angle.json"
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
            image_name=working_image.name,
            input_image_path=input_image_path,
            processing_time_ms=processing_time_ms,
            overlay_png_bytes=overlay_png_bytes,
            metadata=metadata,
            artifacts_dir=artifacts_dir,
            overlay_output_path=overlay_output_path,
            metadata_output_path=metadata_output_path,
            step_logs=step_logs,
        )

    def _run_pipeline(
        self,
        job_dir: Path,
        image_name: str,
        config_path: Path,
    ) -> list[StepExecutionLog]:
        """Run each step in a clean Python process.

        This intentionally mirrors the normal pipeline execution model. Running
        all scripts through runpy in one process leaks process-global state
        between steps (notably cv2.setNumThreads from Step 04), retains imported
        pipeline modules and large native allocations, and can make Steps 05-09
        dramatically slower than their standalone execution.
        """
        logs: list[StepExecutionLog] = []
        env = os.environ.copy()
        env["PIPELINE_CONFIG"] = str(config_path)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        python_executable = sys.executable
        if not python_executable:
            raise ApiError(500, "Could not determine the Python executable for pipeline subprocesses.")

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                creationflags=creationflags,
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
                        "step_logs": [item.to_dict() for item in logs],
                    },
                )

            self._assert_expected_outputs(job_dir, step, image_name, step_log)

        return logs

    def _parse_worker_payload(self, stdout: str, stderr: str) -> dict[str, Any]:
        raw_payload = stdout.strip()
        if not raw_payload:
            raise ApiError(
                500,
                "Pipeline worker returned no structured output.",
                details={"worker_stderr": _tail_text(stderr)},
            )

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ApiError(
                500,
                "Pipeline worker returned invalid JSON output.",
                details={
                    "worker_stdout": _tail_text(stdout),
                    "worker_stderr": _tail_text(stderr),
                },
            ) from exc

        if not isinstance(payload, dict):
            raise ApiError(
                500,
                "Pipeline worker returned an unexpected payload type.",
                details={"worker_stdout": _tail_text(stdout)},
            )

        return payload

    def _step_logs_from_payload(self, raw_logs: Any) -> list[StepExecutionLog]:
        if not isinstance(raw_logs, list):
            return []

        step_logs: list[StepExecutionLog] = []
        for item in raw_logs:
            if not isinstance(item, dict):
                continue

            step_logs.append(
                StepExecutionLog(
                    step=str(item.get("step", "")),
                    script_name=str(item.get("script_name", "")),
                    elapsed_ms=float(item.get("elapsed_ms", 0.0) or 0.0),
                    stdout=_tail_text(str(item.get("stdout", ""))),
                    stderr=_tail_text(str(item.get("stderr", ""))),
                )
            )

        return step_logs

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
