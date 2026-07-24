from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.lib import step_09_measure_canting_angle as step09_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 09: detect the bilateral table-edge reference line and measure "
            "the signed canting angle of the final Step 08 boot axis."
        )
    )
    parser.add_argument("--image", type=str, default=None, help="Optional image-name filter, for example IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--output-subdir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Show saved comparison windows while processing.")
    parser.add_argument("--show", action="store_true", help="Compatibility alias for --debug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_windows = bool(args.debug or args.show)
    step_config = step09_lib.apply_preset(step09_lib.context.STEP_CONFIG, args.preset)
    if args.output_subdir:
        step_config["output_subdir"] = args.output_subdir
    step09_lib.set_step_config(step_config)

    step09_lib.ensure_dirs(
        cleanup=(
            bool(step09_lib.context.STEP_CONFIG.get("cleanup_output_on_start", True))
            and args.image is None
        )
    )
    metadata_files = step09_lib.collect_metadata_files(args.image, args.limit)
    if not metadata_files:
        print("No Step 08 metadata files found.")
        print(f"Input dir: {step09_lib.get_step_dirs()['input_metadata_dir']}")
        return

    print(f"Step 09 input dir: {step09_lib.get_step_dirs()['input_metadata_dir']}")
    print(f"Step 09 cleaned edges: {step09_lib.get_step_dirs()['cleaned_edge_dir']}")
    print(f"Step 09 output dir: {step09_lib.get_step_dirs()['output_dir']}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found Step 08 metadata files: {len(metadata_files)}")

    summary: list[dict] = []
    for metadata_path in metadata_files:
        print(f"\nProcessing: {metadata_path.name}")
        try:
            result = step09_lib.process_metadata_file(metadata_path)
            summary.append(result)
            angle_text = "none" if result["canting_angle_deg"] is None else f"{result['canting_angle_deg']:+.3f} deg"
            table_text = "none" if result["table_line_angle_deg"] is None else f"{result['table_line_angle_deg']:+.3f} deg"
            process_total_sec = float(result.get("timings_sec", {}).get("process_total", 0.0))
            process_total_ms = process_total_sec * 1000.0
            print(
                f"  axis={result['final_axis_candidate']} table={table_text} "
                f"canting={angle_text} confidence={result['measurement_confidence_percent']:.2f}% "
                f"decision={result['decision']} total={process_total_ms:.1f} ms"
            )
            print(f"  overlay: {result['overlay_path']}")
            print(f"  metadata: {result['metadata_path']}")
            if show_windows and result["comparison_path"] is not None:
                step09_lib.show_image(step09_lib.PROJECT_ROOT / result["comparison_path"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = step09_lib.get_step_dirs()["output_dir"] / "step_09_summary.json"
    step09_lib.save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
