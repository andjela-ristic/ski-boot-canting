import sys
import argparse
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.pipeline.lib import step_01_illumination_normalization as step01_lib


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize image illumination using CLAHE."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show original and processed images in OpenCV windows.",
    )
    return parser.parse_args()


def main(*, debug: bool = False) -> None:
    image_paths = step01_lib.collect_images()
    if not image_paths:
        print(f"No images found in: {step01_lib.INPUT_DIR}")
        return

    step01_lib.run(image_paths, debug=debug)


if __name__ == "__main__":
    args = parse_arguments()
    main(debug=args.debug)
