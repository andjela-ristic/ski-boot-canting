from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from io import BytesIO
from mimetypes import guess_type
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .contracts import (
    AnalyzeRequest,
    FramesRequest,
    UploadedAnalyzeRequest,
    UploadedCaptureReadinessRequest,
    UploadedFramesRequest,
)
from .exceptions import ApiError
from .persistence import AnalysisRepository, NoopAnalysisRepository
from .pipeline_runner import PipelineRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web_app"
CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)


@dataclass(slots=True)
class MultipartPart:
    name: str
    value: str | None = None
    filename: str | None = None
    content_type: str | None = None
    file: BytesIO | None = None


MultipartForm = dict[str, list[MultipartPart]]


class CantingApiHandler(BaseHTTPRequestHandler):
    runner = PipelineRunner()
    repository: AnalysisRepository = NoopAnalysisRepository()
    server_version = "CantingApi/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(int(HTTPStatus.NO_CONTENT))
        self._write_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Origin")
        self._finalize_response()

    def do_GET(self) -> None:
        raw_path = self._normalize_path()
        path = self._normalize_api_path()

        if raw_path in {"/health", "/api/health"}:
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "endpoints": {
                        "app": "GET /",
                        "service": "GET /api",
                        "analyze": "POST /analyze",
                        "capture_readiness": "POST /capture-readiness",
                        "frames": "POST /frames",
                        "health": "GET /health",
                    },
                },
            )
            return

        if raw_path in {"/api", "/api/"}:
            self._write_service_index()
            return

        if not raw_path.startswith("/api") and self._try_write_web_asset(raw_path):
            return

        if path == "/":
            self._write_service_index()
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def do_POST(self) -> None:
        path = self._normalize_api_path()
        if path not in {"/analyze", "/capture-readiness", "/frames"}:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
            return

        try:
            if path == "/analyze":
                content_type = (self.headers.get("Content-Type", "") or "").lower()
                if content_type.startswith("multipart/form-data"):
                    upload_request = self._read_uploaded_analyze_request()
                    result = self.runner.analyze_uploaded_image(
                        image_bytes=upload_request.image_bytes,
                        image_filename=upload_request.image_filename,
                        keep_artifacts=upload_request.keep_artifacts,
                    )
                    persistence_result = self.repository.save_analysis(result)

                    if upload_request.response_mode == "binary":
                        self._write_binary_overlay(result, persistence_result)
                        return

                    self._write_json(
                        HTTPStatus.OK,
                        result.to_json_payload(
                            persistence=persistence_result,
                            include_step_logs=upload_request.include_step_logs,
                        ),
                    )
                    return

                payload = self._read_json_body()
                request = AnalyzeRequest.from_dict(payload)
                result = self.runner.analyze_image(
                    image_path=request.image_path,
                    keep_artifacts=request.keep_artifacts,
                )
                persistence_result = self.repository.save_analysis(result)

                if request.response_mode == "binary":
                    self._write_binary_overlay(result, persistence_result)
                    return

                self._write_json(
                    HTTPStatus.OK,
                    result.to_json_payload(
                        persistence=persistence_result,
                        include_step_logs=request.include_step_logs,
                    ),
                )
                return

            if path == "/capture-readiness":
                content_type = (self.headers.get("Content-Type", "") or "").lower()
                if not content_type.startswith("multipart/form-data"):
                    raise ValueError(
                        "Capture readiness requests must use multipart/form-data with field 'frame'."
                    )

                readiness_request = self._read_uploaded_capture_readiness_request()
                result = self.runner.analyze_capture_readiness_frame(
                    frame_bytes=readiness_request.frame_bytes,
                    include_debug=readiness_request.include_debug,
                    guide_scale=readiness_request.guide_scale,
                )
                self._write_json(HTTPStatus.OK, result)
                return

            content_type = (self.headers.get("Content-Type", "") or "").lower()
            if content_type.startswith("multipart/form-data"):
                upload_request = self._read_uploaded_frames_request()
                result = self.runner.analyze_uploaded_video(
                    video_bytes=upload_request.video_bytes,
                    video_filename=upload_request.video_filename,
                    keep_artifacts=upload_request.keep_artifacts,
                    include_step_logs=upload_request.include_step_logs,
                    requested_frame_count=upload_request.frame_count,
                )
                self._write_json(HTTPStatus.OK, result)
                return

            payload = self._read_json_body()
            request = FramesRequest.from_dict(payload)
            result = self.runner.analyze_video_frames(
                video_path=request.video_path,
                keep_artifacts=request.keep_artifacts,
                include_step_logs=request.include_step_logs,
            )
            self._write_json(HTTPStatus.OK, result)
        except json.JSONDecodeError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Request body must be valid JSON."},
            )
        except ValueError as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(exc)},
            )
        except ApiError as exc:
            self._write_json(exc.status_code, exc.to_dict())
        except Exception as exc:  # pragma: no cover - defensive catch for server use
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "Unexpected server error.",
                    "details": {"message": str(exc)},
                },
            )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _normalize_path(self) -> str:
        path = urlparse(self.path).path or "/"
        if path != "/" and path.endswith("/"):
            return path[:-1]
        return path

    def _normalize_api_path(self) -> str:
        path = self._normalize_path()
        if path.startswith("/api/"):
            normalized = path[4:]
            return normalized if normalized else "/"
        return path

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is required.")

        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _read_multipart_form(self) -> MultipartForm:
        content_type = self.headers.get("Content-Type", "") or ""
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("Request must use multipart/form-data.")

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length <= 0:
            raise ValueError("Multipart request body is required.")

        raw_body = self.rfile.read(content_length)
        synthetic_message = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + raw_body
        message = BytesParser(policy=email_default_policy).parsebytes(synthetic_message)
        if not message.is_multipart():
            raise ValueError("Malformed multipart/form-data body.")

        form: MultipartForm = {}
        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            if disposition != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue

            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename is not None:
                item = MultipartPart(
                    name=str(name),
                    filename=Path(filename).name,
                    content_type=part.get_content_type(),
                    file=BytesIO(payload),
                )
            else:
                charset = part.get_content_charset() or "utf-8"
                try:
                    value = payload.decode(charset)
                except (LookupError, UnicodeDecodeError):
                    value = payload.decode("utf-8", errors="replace")
                item = MultipartPart(name=str(name), value=value)

            form.setdefault(str(name), []).append(item)

        return form

    def _read_uploaded_frames_request(self) -> UploadedFramesRequest:
        form = self._read_multipart_form()

        video_field = self._first_form_part(form, "video")
        if video_field is None:
            raise ValueError("Multipart field 'video' is required.")
        if video_field.file is None:
            raise ValueError("Multipart field 'video' must contain a file.")

        video_bytes = video_field.file.read()
        filename = Path(video_field.filename or "capture.mp4").name

        return UploadedFramesRequest(
            video_filename=filename or "capture.mp4",
            video_bytes=video_bytes,
            keep_artifacts=self._parse_form_bool(form, "keep_artifacts", default=False),
            include_step_logs=self._parse_form_bool(form, "include_step_logs", default=False),
            frame_count=self._parse_form_int(form, "frame_count", default=None, minimum=1, maximum=30),
            clip_duration_ms=self._parse_form_int(form, "clip_duration_ms", default=None, minimum=1),
        )

    def _read_uploaded_analyze_request(self) -> UploadedAnalyzeRequest:
        form = self._read_multipart_form()

        image_field = self._first_form_part(form, "image")
        if image_field is None:
            raise ValueError("Multipart field 'image' is required.")
        if image_field.file is None:
            raise ValueError("Multipart field 'image' must contain a file.")

        image_bytes = image_field.file.read()
        filename = Path(image_field.filename or "capture.jpg").name

        response_mode = (self._extract_form_value(form, "response_mode") or "json").strip().lower()
        if response_mode not in {"json", "binary"}:
            raise ValueError("Multipart field 'response_mode' must be either 'json' or 'binary'.")

        return UploadedAnalyzeRequest(
            image_filename=filename or "capture.jpg",
            image_bytes=image_bytes,
            response_mode=response_mode,
            keep_artifacts=self._parse_form_bool(form, "keep_artifacts", default=False),
            include_step_logs=self._parse_form_bool(form, "include_step_logs", default=False),
        )

    def _read_uploaded_capture_readiness_request(self) -> UploadedCaptureReadinessRequest:
        form = self._read_multipart_form()

        frame_field = self._first_form_part(form, "frame")
        if frame_field is None:
            raise ValueError("Multipart field 'frame' is required.")
        if frame_field.file is None:
            raise ValueError("Multipart field 'frame' must contain a file.")

        frame_bytes = frame_field.file.read()
        filename = Path(frame_field.filename or "preview.jpg").name

        return UploadedCaptureReadinessRequest(
            frame_filename=filename or "preview.jpg",
            frame_bytes=frame_bytes,
            include_debug=self._parse_query_bool("debug", default=False),
            guide_scale=self._parse_form_float(
                form,
                "guide_scale",
                default=1.0,
                minimum=0.90,
                maximum=1.20,
            ),
        )

    def _parse_form_bool(self, form: MultipartForm, field_name: str, default: bool) -> bool:
        value = self._extract_form_value(form, field_name)
        if value is None or value == "":
            return default

        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Multipart field '{field_name}' must be a boolean.")

    def _parse_form_int(
        self,
        form: MultipartForm,
        field_name: str,
        default: int | None,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int | None:
        value = self._extract_form_value(form, field_name)
        if value is None or value == "":
            return default

        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"Multipart field '{field_name}' must be an integer.") from exc

        if minimum is not None and parsed < minimum:
            raise ValueError(f"Multipart field '{field_name}' must be >= {minimum}.")
        if maximum is not None and parsed > maximum:
            raise ValueError(f"Multipart field '{field_name}' must be <= {maximum}.")
        return parsed

    @staticmethod
    def _first_form_part(form: MultipartForm, field_name: str) -> MultipartPart | None:
        parts = form.get(field_name)
        return parts[0] if parts else None

    def _extract_form_value(self, form: MultipartForm, field_name: str) -> str | None:
        field = self._first_form_part(form, field_name)
        if field is None or field.value is None:
            return None
        return field.value

    def _parse_form_float(
        self,
        form: MultipartForm,
        field_name: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        value = self._extract_form_value(form, field_name)
        if value is None or value == "":
            return default

        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"Multipart field '{field_name}' must be a number.") from exc

        if minimum is not None and parsed < minimum:
            raise ValueError(f"Multipart field '{field_name}' must be >= {minimum}.")
        if maximum is not None and parsed > maximum:
            raise ValueError(f"Multipart field '{field_name}' must be <= {maximum}.")
        return parsed

    def _parse_query_bool(self, field_name: str, default: bool) -> bool:
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        raw_values = query.get(field_name)
        if not raw_values:
            return default

        normalized = raw_values[0].strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Query parameter '{field_name}' must be a boolean.")

    def _write_service_index(self) -> None:
        self._write_json(
            HTTPStatus.OK,
            {
                "service": "ski-boot-canting-api",
                "frontend": "GET /",
                "endpoints": {
                    "analyze": "POST /analyze",
                    "capture_readiness": "POST /capture-readiness",
                    "frames": "POST /frames",
                    "health": "GET /health",
                },
            },
        )

    def _try_write_web_asset(self, request_path: str) -> bool:
        if not WEB_ROOT.exists():
            return False

        candidate = self._resolve_web_asset_path(request_path)
        if candidate is None or not candidate.exists() or not candidate.is_file():
            return False

        body = candidate.read_bytes()
        content_type = self._guess_content_type(candidate)
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # During tunnel testing the frontend changes frequently. Safari and
        # Cloudflare must not keep an older readiness JS/CSS bundle for an hour.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        return self._finalize_response(body)

    def _resolve_web_asset_path(self, request_path: str) -> Path | None:
        if request_path == "/favicon.ico":
            request_path = "/favicon.png"

        relative_path = "index.html" if request_path == "/" else request_path.lstrip("/")
        web_root = WEB_ROOT.resolve()

        candidate = (web_root / relative_path).resolve()
        try:
            candidate.relative_to(web_root)
        except ValueError:
            return None

        if candidate.is_dir():
            candidate = (candidate / "index.html").resolve()

        if candidate.exists():
            return candidate

        if "." not in Path(relative_path).name:
            fallback = (web_root / "index.html").resolve()
            if fallback.exists():
                return fallback

        return None

    def _guess_content_type(self, path: Path) -> str:
        override = {
            ".css": "text/css; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".webmanifest": "application/manifest+json; charset=utf-8",
        }
        if path.suffix.lower() in override:
            return override[path.suffix.lower()]

        guessed, _ = guess_type(path.name)
        return guessed or "application/octet-stream"

    def _write_json(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self._write_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self._finalize_response(body)

    def _write_binary_overlay(
        self,
        result,
        persistence_result: dict[str, Any],
    ) -> None:
        body = result.overlay_png_bytes
        self.send_response(int(HTTPStatus.OK))
        self._write_cors_headers()
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Processing-Time-Ms", f"{result.processing_time_ms:.2f}")
        self.send_header("X-Image-Name", result.image_name)
        self.send_header("X-Input-Image-Path", result.input_image_path)
        self.send_header("X-Persistence-Saved", str(bool(persistence_result.get("saved", False))).lower())
        if result.artifacts_dir is not None:
            self.send_header("X-Artifacts-Dir", result.artifacts_dir)
        if result.overlay_output_path is not None:
            self.send_header("X-Overlay-Output-Path", result.overlay_output_path)
        if result.metadata_output_path is not None:
            self.send_header("X-Metadata-Output-Path", result.metadata_output_path)
        self._finalize_response(body)

    def _finalize_response(self, body: bytes = b"") -> bool:
        try:
            self.end_headers()
            if body:
                self.wfile.write(body)
        except CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return False
        return True

    def _write_cors_headers(self) -> None:
        allow_origin = os.environ.get("API_CORS_ALLOW_ORIGIN", "*").strip() or "*"
        self.send_header("Access-Control-Allow-Origin", allow_origin)
        self.send_header("Vary", "Origin")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local canting pipeline HTTP API.",
    )
    parser.add_argument("--host", type=str, default=os.environ.get("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8000")))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CantingApiHandler)
    print(f"Canting API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
