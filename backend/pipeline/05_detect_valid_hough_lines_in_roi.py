from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.lib import step_05_valid_hough_lines_in_roi as step05_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect Hough line segments from cleaned edges and keep only lines valid inside ROI.")
    parser.add_argument("--image", type=str, default=None, help="Process only one image, for example: IMG_0502 or IMG_0502.png")
    parser.add_argument("--debug", action="store_true", help="Show debug windows while processing.")
    parser.add_argument("--show", action="store_true", help="Compatibility alias for --debug.")
    return parser.parse_args()


def main() -> None:
    if not step05_lib.context.STEP_CONFIG["enabled"]:
        print("Step 05 is disabled in config.")
        return

    args = parse_args()
    image_names = step05_lib.collect_images(args.image)
    if not image_names:
        print(f"No matching images found in: {step05_lib.context.EDGE_INPUT_DIR} and {step05_lib.context.ROI_MASK_DIR}")
        return

    step05_lib.process_images(image_names, debug=bool(args.debug or args.show))


if __name__ == "__main__":
    main()
