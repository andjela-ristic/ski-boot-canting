from __future__ import annotations

import argparse

from lib import step_08_multi_validate_central_ruler_lib as step08_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 08: combine the existing Step 07 symmetry result with independent "
            "geometric validators and estimate final-axis confidence."
        )
    )
    parser.add_argument("--image", type=str, default=None, help="Optional image-name filter, for example IMG_0502.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--output-subdir", type=str, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step08_lib.context.STEP_CONFIG = step08_lib.apply_preset(step08_lib.context.STEP_CONFIG, args.preset)
    if args.output_subdir:
        step08_lib.context.STEP_CONFIG["output_subdir"] = args.output_subdir
    if args.candidate_limit is not None:
        if args.candidate_limit < 1:
            raise ValueError("--candidate-limit must be at least 1")
        step08_lib.context.STEP_CONFIG["candidate_limit"] = int(args.candidate_limit)

    step08_lib.ensure_dirs(
        cleanup=bool(step08_lib.context.STEP_CONFIG.get("cleanup_output_on_start", True)) and args.image is None
    )
    metadata_files = step08_lib.collect_metadata_files(args.image, args.limit)
    if not metadata_files:
        print("No Step 07 metadata files found.")
        print(f"Input dir: {step08_lib.get_step_dirs()['input_metadata_dir']}")
        return

    print(f"Step 08 input dir: {step08_lib.get_step_dirs()['input_metadata_dir']}")
    print(f"Step 08 output dir: {step08_lib.get_step_dirs()['output_dir']}")
    print("Step 07 symmetry is read from metadata and is not recalculated.")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"Found Step 07 metadata files: {len(metadata_files)}")

    summary: list[dict] = []
    for metadata_path in metadata_files:
        print(f"\nProcessing: {metadata_path.name}")
        try:
            result = step08_lib.process_metadata_file(metadata_path)
            summary.append(result)
            print(
                f"  final={result['final_candidate']} step07={result['step_07_candidate']} "
                f"agreement={result['step_07_agrees']} symmetry={result['symmetry_percent']:.2f}% "
                f"multi={result['multi_validation_percent']:.2f}% "
                f"confidence={result['confidence_percent']:.2f}% decision={result['decision']}"
            )
            print(f"  overlay: {result['overlay_path']}")
            print(f"  metadata: {result['metadata_path']}")
            if args.show:
                step08_lib.show_image(step08_lib.PROJECT_ROOT / result["comparison_path"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = step08_lib.get_step_dirs()["output_dir"] / "step_08_summary.json"
    step08_lib.save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
