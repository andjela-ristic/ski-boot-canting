from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.lib import step_04_boot_roi_from_edges as step04_lib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rough ski boot ROI mask from edge images.")
    parser.add_argument("--image", type=str, default=None, help="Process only one image, for example: IMG_0502 or IMG_0502.png")
    parser.add_argument("--debug",action="store_true",help="Save overlay/component/comparison debug images and open debug windows. Without this flag only the ROI mask is produced.",)
    return parser.parse_args()

def main() -> None:
    if not step04_lib.context.STEP_CONFIG["enabled"]:
        print("Step 04 is disabled in config.")
        return
    args = parse_args()
    image_paths = step04_lib.collect_images(args.image)
    if not image_paths:
        print(f"No images found in: {step04_lib.context.INPUT_DIR}")
        return
    print(f"Selected image filter: {args.image if args.image else 'all'}")
    step04_lib.process_images(image_paths, debug=bool(args.debug))

if __name__ == "__main__":
    main()
