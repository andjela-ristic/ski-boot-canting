from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.pipeline.lib import step_06_search_central_ruler as step06_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 06: search a central ruler hypothesis across ROI and fit the final center axis."
    )
    parser.add_argument("--image", type=str, default=None, help="Optional image name filter, for example IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional processed output subdir override.")
    parser.add_argument(
        "--save-all-candidates",
        action="store_true",
        help="Save snapshots and metadata for all final fine candidates.",
    )
    parser.add_argument(
        "--max-saved-candidates",
        type=int,
        default=None,
        help="Optional override for how many top final candidates are saved.",
    )
    parser.add_argument("--debug", action="store_true", help="Show saved comparison windows while processing.")
    parser.add_argument("--show", action="store_true", help="Compatibility alias for --debug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_windows = bool(args.debug or args.show)

    step_config = step06_lib.apply_preset(step06_lib.context.STEP_CONFIG, args.preset)
    if args.output_subdir:
        step_config["output_subdir"] = args.output_subdir
    if args.save_all_candidates:
        step_config["save_all_final_candidates"] = True
    if args.max_saved_candidates is not None:
        step_config.setdefault("candidate_deduplication", {})
        step_config["candidate_deduplication"]["max_saved_candidates"] = int(args.max_saved_candidates)
    step06_lib.set_step_config(step_config)

    step06_lib.ensure_dirs(
        cleanup=bool(step06_lib.context.STEP_CONFIG.get("cleanup_output_on_start", True)) and args.image is None
    )

    json_files = step06_lib.collect_json_files(args.image, args.limit)
    if not json_files:
        print("No JSON files found.")
        print(f"Input dir: {step06_lib.get_step_dirs()['input_json_dir']}")
        return

    print(f"Step 06 input dir: {step06_lib.get_step_dirs()['input_json_dir']}")
    print(f"Step 06 output dir: {step06_lib.get_step_dirs()['output_dir']}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found JSON files: {len(json_files)}")

    summary = []
    for json_path in json_files:
        print(f"\nProcessing: {json_path.name}")
        try:
            result = step06_lib.process_json_file(json_path)
            summary.append(result)
            score_text = "none" if result["best_score"] is None else f"{result['best_score']:.3f}"
            tilt_text = "none" if result["best_tilt_deg"] is None else f"{result['best_tilt_deg']:.2f}"
            process_total_sec = float(result.get("timings_sec", {}).get("process_total", 0.0))
            process_total_ms = process_total_sec * 1000.0
            print(
                f"  filtered={result['filtered_line_count']} candidates={result['candidate_count']} "
                f"selected={result['selected_fragment_count']} score={score_text} tilt={tilt_text} "
                f"total={process_total_ms:.1f} ms"
            )
            print(f"  overlay: {result['overlay_path']}")
            print(f"  metadata: {result['metadata_path']}")
            if show_windows and result["comparison_path"] is not None:
                step06_lib.show_image(step06_lib.PROJECT_ROOT / result["comparison_path"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = step06_lib.get_step_dirs()["output_dir"] / "step_06_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    step06_lib.save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
