from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.lib import step_07_verify_central_ruler_symmetry as step07_lib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 07: verify Step 06 central-axis candidates using segmented bidirectional mirror symmetry.")
    parser.add_argument("--image", type=str, default=None, help="Optional image-name filter, for example IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional processed output-subdir override.")
    parser.add_argument("--segments", type=int, default=None, help="Override the number of vertical verification segments. Default: 12.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Maximum number of Step 06 top candidates to verify.")
    parser.add_argument("--debug", action="store_true", help="Show saved comparison windows while processing.")
    parser.add_argument("--show", action="store_true", help="Compatibility alias for --debug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_windows = bool(args.debug or args.show)

    step_config = step07_lib.apply_preset(step07_lib.context.STEP_CONFIG, args.preset)
    if args.output_subdir:
        step_config["output_subdir"] = args.output_subdir
    if args.segments is not None:
        if args.segments < 3:
            raise ValueError("--segments must be at least 3")
        step_config["segment_count"] = int(args.segments)
    if args.candidate_limit is not None:
        if args.candidate_limit < 1:
            raise ValueError("--candidate-limit must be at least 1")
        step_config["candidate_limit"] = int(args.candidate_limit)
    step07_lib.set_step_config(step_config)

    step07_lib.ensure_dirs(cleanup=bool(step07_lib.context.STEP_CONFIG.get("cleanup_output_on_start", True)) and args.image is None)

    metadata_files = step07_lib.collect_metadata_files(args.image, args.limit)
    if not metadata_files:
        print("No Step 06 metadata files found.")
        print(f"Input dir: {step07_lib.get_step_dirs()['input_metadata_dir']}")
        return

    print(f"Step 07 input dir: {step07_lib.get_step_dirs()['input_metadata_dir']}")
    print(f"Step 07 output dir: {step07_lib.get_step_dirs()['output_dir']}")
    print(f"Segments: {step07_lib.context.STEP_CONFIG.get('segment_count', 12)}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found Step 06 metadata files: {len(metadata_files)}")

    summary: list[dict] = []
    for metadata_path in metadata_files:
        print(f"\nProcessing: {metadata_path.name}")
        try:
            result = step07_lib.process_metadata_file(metadata_path)
            summary.append(result)
            process_total_sec = float(result.get("timings_sec", {}).get("process_total", 0.0))
            process_total_ms = process_total_sec * 1000.0
            print(
                f"  candidates={result['candidate_count']} "
                f"winner={result['winner_label']} "
                f"symmetry={result['symmetry_percent']:.2f}% "
                f"margin={result['winner_margin_percent']:.2f}% "
                f"total={process_total_ms:.1f} ms"
            )
            print(f"  overlay: {result['overlay_path']}")
            print(f"  metadata: {result['metadata_path']}")
            if show_windows and result["comparison_path"] is not None:
                step07_lib.show_image(step07_lib.PROJECT_ROOT / result["comparison_path"])
        except Exception as exc:  # keep batch processing alive
            print(f"  ERROR: {exc}")

    summary_path = step07_lib.get_step_dirs()["output_dir"] / "step_07_summary.json"
    step07_lib.save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
