from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.lib import step_02_grayscale_and_blur as step02_lib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 02 grayscale and blur processing.")
    parser.add_argument("--debug", action="store_true", help="Show comparison windows while processing images.")
    return parser.parse_args()

def main(debug: bool = False) -> None:
    if not step02_lib.STEP_CONFIG["enabled"]:
        print("Step 02 is disabled in config.")
        return
    image_paths = step02_lib.collect_images()
    if not image_paths:
        print(f"No images found in: {step02_lib.INPUT_DIR}")
        return
    step02_lib.run(image_paths, debug=debug)

if __name__ == "__main__":
    args = parse_args()
    main(debug=args.debug)
