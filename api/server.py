from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.parse import urlparse

from .contracts import AnalyzeRequest
from .exceptions import ApiError
from .persistence import AnalysisRepository, NoopAnalysisRepository
from .pipeline_runner import PipelineRunner


class CantingApiHandler(BaseHTTPRequestHandler):
    runner = PipelineRunner()
    repository: AnalysisRepository = NoopAnalysisRepository()
    server_version = "CantingApi/0.1"

    def do_GET(self) -> None:
        path = self._normalize_path()

        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "endpoints": {
                        "analyze": "POST /analyze",
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
                        "health": "GET /health",
                    },
                },
            )
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def do_POST(self) -> None:
        path = self._normalize_path()
        if path != "/analyze":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
            return

        try:
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

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is required.")

        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _write_json(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
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
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Processing-Time-Ms", f"{result.processing_time_ms:.2f}")
        self.send_header("X-Image-Name", result.image_name)
        self.send_header("X-Persistence-Saved", str(bool(persistence_result.get("saved", False))).lower())
        if result.artifacts_dir is not None:
            self.send_header("X-Artifacts-Dir", result.artifacts_dir)
        self.end_headers()
        self.wfile.write(body)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local canting pipeline HTTP API.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
