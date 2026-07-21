from __future__ import annotations

import argparse
import cgi
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .contracts import AnalyzeRequest, FramesRequest, UploadedFramesRequest
from .exceptions import ApiError
from .persistence import AnalysisRepository, NoopAnalysisRepository
from .pipeline_runner import PipelineRunner


class CantingApiHandler(BaseHTTPRequestHandler):
    runner = PipelineRunner()
    repository: AnalysisRepository = NoopAnalysisRepository()
    server_version = "CantingApi/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(int(HTTPStatus.NO_CONTENT))
        self._write_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Origin")
        self.end_headers()

    def do_GET(self) -> None:
        path = self._normalize_api_path()

        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "endpoints": {
                        "analyze": "POST /analyze",
                        "frames": "POST /frames",
                        "health": "GET /health",
                    },
                },
            )
            return

        if path == "/":
            self._write_json(
                HTTPStatus.OK,
                {
                    "service": "ski-boot-canting-api",
                    "endpoints": {
                        "analyze": "POST /analyze",
                        "frames": "POST /frames",
                        "health": "GET /health",
                    },
                },
            )
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def do_POST(self) -> None:
        path = self._normalize_api_path()
        if path not in {"/analyze", "/frames"}:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
            return

        try:
            if path == "/analyze":
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

            content_type = (self.headers.get("Content-Type", "") or "").lower()
            if content_type.startswith("multipart/form-data"):
                upload_request = self._read_uploaded_frames_request()
                result = self.runner.analyze_uploaded_video_stub(
                    video_bytes=upload_request.video_bytes,
                    video_filename=upload_request.video_filename,
                    keep_artifacts=upload_request.keep_artifacts,
                    requested_frame_count=upload_request.frame_count,
                    clip_duration_ms=upload_request.clip_duration_ms,
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

    def _read_uploaded_frames_request(self) -> UploadedFramesRequest:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )

        video_field = form["video"] if "video" in form else None
        if video_field is None:
            raise ValueError("Multipart field 'video' is required.")

        if isinstance(video_field, list):
            video_field = video_field[0]

        video_file = getattr(video_field, "file", None)
        if video_file is None:
            raise ValueError("Multipart field 'video' must contain a file.")

        video_bytes = video_file.read()
        filename = Path(getattr(video_field, "filename", "") or "capture.mp4").name

        return UploadedFramesRequest(
            video_filename=filename or "capture.mp4",
            video_bytes=video_bytes,
            keep_artifacts=self._parse_form_bool(form, "keep_artifacts", default=False),
            include_step_logs=self._parse_form_bool(form, "include_step_logs", default=False),
            frame_count=self._parse_form_int(form, "frame_count", default=None, minimum=1, maximum=30),
            clip_duration_ms=self._parse_form_int(form, "clip_duration_ms", default=None, minimum=1),
        )

    def _parse_form_bool(self, form: cgi.FieldStorage, field_name: str, default: bool) -> bool:
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
        form: cgi.FieldStorage,
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

    def _extract_form_value(self, form: cgi.FieldStorage, field_name: str) -> str | None:
        if field_name not in form:
            return None

        field = form[field_name]
        if isinstance(field, list):
            field = field[0]

        value = getattr(field, "value", None)
        if value is None:
            return None
        return str(value)

    def _write_json(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self._write_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        self.send_header("X-Persistence-Saved", str(bool(persistence_result.get("saved", False))).lower())
        if result.artifacts_dir is not None:
            self.send_header("X-Artifacts-Dir", result.artifacts_dir)
        self.end_headers()
        self.wfile.write(body)

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
