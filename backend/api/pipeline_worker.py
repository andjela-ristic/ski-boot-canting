from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any, Iterator

from .pipeline_runner import PIPELINE_DIR, PIPELINE_STEPS, PROJECT_ROOT, _tail_text


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all pipeline steps inside a single warm worker process.",
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--image", type=str, required=True)
    return parser


@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _temporary_sys_path(*paths: Path) -> Iterator[None]:
    inserted: list[str] = []

    for path in reversed(paths):
        text = str(path)
        if text in sys.path:
            continue
        sys.path.insert(0, text)
        inserted.append(text)

    try:
        yield
    finally:
        for text in inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _coerce_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code in (None, False):
        return 0
    if isinstance(code, int):
        return code
    return 1


def _run_step(step, image_name: str) -> dict[str, Any]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    script_path = PIPELINE_DIR / step.script_name
    previous_argv = sys.argv[:]
    started = time.perf_counter()
    return_code = 0

    try:
        sys.argv = [str(script_path), *step.args_factory(image_name)]
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                runpy.run_path(str(script_path), run_name="__main__")
            except SystemExit as exc:
                return_code = _coerce_exit_code(exc)
    finally:
        sys.argv = previous_argv

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "step": step.step,
        "script_name": step.script_name,
        "elapsed_ms": elapsed_ms,
        "stdout": _tail_text(stdout_buffer.getvalue()),
        "stderr": _tail_text(stderr_buffer.getvalue()),
        "return_code": return_code,
    }


def _missing_outputs(job_dir: Path, step, image_name: str) -> list[str]:
    return [
        str(path)
        for path in step.expected_outputs_factory(job_dir, image_name)
        if not path.exists()
    ]


def main() -> int:
    args = build_argument_parser().parse_args()
    logs: list[dict[str, Any]] = []

    try:
        with _temporary_cwd(PROJECT_ROOT), _temporary_sys_path(PIPELINE_DIR, PROJECT_ROOT):
            for step in PIPELINE_STEPS:
                step_log = _run_step(step, args.image)
                logs.append(step_log)

                if step_log["return_code"] != 0:
                    payload = {
                        "status": "error",
                        "message": f"Pipeline step failed: {step.step}",
                        "details": {
                            "script_name": step.script_name,
                            "return_code": step_log["return_code"],
                            "stdout": step_log["stdout"],
                            "stderr": step_log["stderr"],
                        },
                        "step_logs": logs,
                    }
                    print(json.dumps(payload, ensure_ascii=False))
                    return 1

                missing_outputs = _missing_outputs(args.job_dir, step, args.image)
                if missing_outputs:
                    payload = {
                        "status": "error",
                        "message": f"Pipeline step completed but expected outputs are missing: {step.step}",
                        "details": {
                            "script_name": step.script_name,
                            "missing_outputs": missing_outputs,
                            "stdout": step_log["stdout"],
                            "stderr": step_log["stderr"],
                        },
                        "step_logs": logs,
                    }
                    print(json.dumps(payload, ensure_ascii=False))
                    return 1
    except Exception as exc:
        payload = {
            "status": "error",
            "message": "Pipeline worker crashed before completion.",
            "details": {"message": str(exc)},
            "step_logs": logs,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "step_logs": logs,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
