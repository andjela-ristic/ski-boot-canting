from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.pipeline.lib import step_03_edge_detection as step03_lib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step 03 edge detection.")
    parser.add_argument("--debug", action="store_true", help="Show OpenCV comparison windows while processing images.")
    return parser.parse_args()

def main(*, debug: bool = False) -> None:
    if not step03_lib.context.STEP_03_CONFIG["enabled"]:
        print("Step 03 is disabled in config.")
        return

    image_paths = step03_lib.collect_images()
    if not image_paths:
        print(f"No images found in: {step03_lib.context.INPUT_DIR}")
        return

    step03_lib.process_images(image_paths, debug=debug)

if __name__ == "__main__":
    args = parse_args()
    main(debug=args.debug)
