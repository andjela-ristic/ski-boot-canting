from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .context import BILATERAL_DIR, CSV_PATH, DISPLAY_CONFIG, GRAYSCALE_LAB_L_DIR, INPUT_DIR, METADATA_DIR, OUTPUT_DIR, STEP_CONFIG
from .display import make_comparison_view
from .io import relative_project_path, save_metadata, write_image

def convert_to_lab_l(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    grayscale_lab_l = cv2.extractChannel(lab, 0)
    del lab
    return grayscale_lab_l

def convert_to_bgr2gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#
#
# def convert_to_ycrcb_y(image_bgr: np.ndarray) -> np.ndarray:
#     ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
#     y_channel = cv2.extractChannel(ycrcb, 0)
#     del ycrcb
#     return y_channel

def build_bilateral(image_bgr: np.ndarray) -> np.ndarray:
    # keep the current BGR2GRAY behavior
    bilateral_config = STEP_CONFIG["bilateral_filter"]
    bilateral = build_bilateral_variant(
        convert_to_bgr2gray(image_bgr),
        diameter=int(bilateral_config["diameter"]),
        sigma_color=float(bilateral_config["sigma_color"]),
        sigma_space=float(bilateral_config["sigma_space"]),
    )
    return bilateral

def build_bilateral_variant(grayscale: np.ndarray, *, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(grayscale, diameter, sigma_color, sigma_space)

# disabled Gaussian processing
#
# def build_gaussian(grayscale: np.ndarray) -> np.ndarray:
#     gaussian_config = STEP_CONFIG["gaussian_blur"]
#     kernel_size = int(gaussian_config["kernel_size"])
#     sigma_x = float(gaussian_config["sigma_x"])
#     return cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), sigma_x)

def process_image(image_path: Path, index: int, image_count: int, *, show_windows: bool, wait_between_images: bool) -> dict[str, object] | None:
    total_started = perf_counter()
    read_started = perf_counter()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    read_time_ms = (perf_counter() - read_started) * 1000.0
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        return None

    height, width = image_bgr.shape[:2]
    processing_started = perf_counter()
    grayscale_lab_l = convert_to_lab_l(image_bgr)
    bilateral = build_bilateral(image_bgr)
    processing_time_ms = (perf_counter() - processing_started) * 1000.0

    grayscale_lab_l_path = GRAYSCALE_LAB_L_DIR / image_path.name
    bilateral_path = BILATERAL_DIR / image_path.name

    write_started = perf_counter()
    write_image(grayscale_lab_l_path, grayscale_lab_l)
    write_image(bilateral_path, bilateral)
    write_time_ms = (perf_counter() - write_started) * 1000.0

    total_time_ms = (perf_counter() - total_started) * 1000.0
    bilateral_config = STEP_CONFIG["bilateral_filter"]

    print(
        f"[{index}/{image_count}] Saved: {image_path.name} | "
        f"read={read_time_ms:.1f} ms, "
        f"process={processing_time_ms:.1f} ms, "
        f"write={write_time_ms:.1f} ms, "
        f"total={total_time_ms:.1f} ms"
    )

    if show_windows:
        comparison = make_comparison_view(image_bgr, grayscale_lab_l, bilateral)
        title = f"02 Grayscale and blur | {index}/{image_count} | {image_path.name}"
        cv2.imshow(title, comparison)
        key = cv2.waitKey(0 if wait_between_images else 500) & 0xFF
        cv2.destroyWindow(title)
        del comparison
        if key in (ord("q"), 27):
            print("Stopped by user.")
            del grayscale_lab_l, bilateral, image_bgr
            return {"stopped": True}

    del grayscale_lab_l, bilateral, image_bgr
    return {
        "source_file": relative_project_path(image_path),
        "grayscale_file": "",
        "grayscale_bgr2gray_file": "",
        "grayscale_lab_l_file": relative_project_path(grayscale_lab_l_path),
        "grayscale_ycrcb_y_file": "",
        "gaussian_file": "",
        "bilateral_file": relative_project_path(bilateral_path),
        "width": width,
        "height": height,
        "processing_step": "02_grayscale_and_blur",
        "grayscale_method": "lab_l",
        "gaussian_kernel_size": "",
        "gaussian_sigma_x": "",
        "bilateral_diameter": int(bilateral_config["diameter"]),
        "bilateral_sigma_color": float(bilateral_config["sigma_color"]),
        "bilateral_sigma_space": float(bilateral_config["sigma_space"]),
        "read_time_ms": round(read_time_ms, 3),
        "processing_time_ms": round(processing_time_ms, 3),
        "write_time_ms": round(write_time_ms, 3),
        "total_time_ms": round(total_time_ms, 3),
        "stopped": False,
    }


def run(image_paths: list[Path], *, debug: bool = False) -> None:
    GRAYSCALE_LAB_L_DIR.mkdir(parents=True, exist_ok=True)
    BILATERAL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # disabled directories stay disabled
    # GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
    # GRAYSCALE_BGR2GRAY_DIR.mkdir(parents=True, exist_ok=True)
    # GRAYSCALE_YCRCB_Y_DIR.mkdir(parents=True, exist_ok=True)
    # GAUSSIAN_DIR.mkdir(parents=True, exist_ok=True)

    show_windows = debug
    wait_between_images = bool(DISPLAY_CONFIG.get("wait_between_images", False))
    image_count = len(image_paths)
    metadata_rows: list[dict[str, object]] = []

    print()
    print("Processing step 02: active grayscale and blur outputs")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("Active outputs: grayscale_lab_l, bilateral_filter")

    if show_windows:
        print()
        print("Controls:")
        print("  n / SPACE / ENTER  -> next image")
        print("  q / ESC            -> quit")

    for index, image_path in enumerate(image_paths, start=1):
        row = process_image(image_path, index, image_count, show_windows=show_windows, wait_between_images=wait_between_images)
        if row is None: continue
        stopped = bool(row.pop("stopped", False))
        metadata_rows.append(row)
        if stopped: break

    save_metadata(metadata_rows)
    if show_windows: cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Grayscale LAB L saved to: {GRAYSCALE_LAB_L_DIR}")
    print(f"Bilateral filter saved to: {BILATERAL_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")
