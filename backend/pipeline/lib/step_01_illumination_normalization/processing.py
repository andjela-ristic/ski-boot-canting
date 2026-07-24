from __future__ import annotations

from time import perf_counter
from pathlib import Path

import cv2
import numpy as np

from .context import CLAHE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE_LABEL, CSV_PATH, DISPLAY_CONFIG, INPUT_DIR, METADATA_DIR, OUTPUT_DIR, PNG_COMPRESSION, STEP_CONFIG
from .display import make_side_by_side
from .io import relative_project_path, save_metadata, save_processed_image


def normalize_illumination_bgr(image_bgr: np.ndarray, *, preserve_input: bool = False) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # use only the L channel
    l_channel = cv2.extractChannel(lab, 0)
    CLAHE.apply(l_channel, l_channel)
    cv2.insertChannel(l_channel, lab, 0)

    if preserve_input: return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # write the result back into the same buffer
    cv2.cvtColor(lab, cv2.COLOR_LAB2BGR, dst=image_bgr)
    return image_bgr


def normalize_illumination_variant(image_bgr: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    normalized_l = clahe.apply(l_channel)
    normalized_lab = cv2.merge([normalized_l, a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def process_image(image_path: Path, index: int, image_count: int, *, show_windows: bool, wait_between_images: bool) -> dict | None:
    total_started = perf_counter()

    read_started = perf_counter()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    read_time_ms = (perf_counter() - read_started) * 1000.0
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        return None

    processing_started = perf_counter()
    normalized_bgr = normalize_illumination_bgr(image_bgr, preserve_input=show_windows)
    processing_time_ms = (perf_counter() - processing_started) * 1000.0

    output_path = OUTPUT_DIR / image_path.name
    write_started = perf_counter()
    save_processed_image(output_path, normalized_bgr)
    write_time_ms = (perf_counter() - write_started) * 1000.0

    total_time_ms = (perf_counter() - total_started) * 1000.0
    height, width = normalized_bgr.shape[:2]

    print(
        f"[{index}/{image_count}] Saved: {output_path.name} | "
        f"read={read_time_ms:.1f} ms, "
        f"process={processing_time_ms:.1f} ms, "
        f"write={write_time_ms:.1f} ms, "
        f"total={total_time_ms:.1f} ms"
    )

    if show_windows:
        comparison = make_side_by_side(image_bgr, normalized_bgr)
        title = f"01 Illumination normalization | {index}/{image_count} | {image_path.name}"
        cv2.imshow(title, comparison)
        key = cv2.waitKey(0 if wait_between_images else 500) & 0xFF
        cv2.destroyWindow(title)
        if key in (ord("q"), 27):
            print("Stopped by user.")
            return {"stopped": True}

    return {
        "source_file": relative_project_path(image_path),
        "output_file": relative_project_path(output_path),
        "width": width,
        "height": height,
        "processing_step": "01_illumination_normalization",
        "method": STEP_CONFIG["method"],
        "clahe_clip_limit": CLAHE_CLIP_LIMIT,
        "clahe_tile_grid_size": CLAHE_TILE_GRID_SIZE_LABEL,
        "read_time_ms": round(read_time_ms, 3),
        "processing_time_ms": round(processing_time_ms, 3),
        "write_time_ms": round(write_time_ms, 3),
        "total_time_ms": round(total_time_ms, 3),
        "stopped": False,
    }


def run(image_paths: list[Path], *, debug: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    show_windows = debug
    wait_between_images = bool(DISPLAY_CONFIG.get("wait_between_images", True))
    image_count = len(image_paths)
    metadata_rows: list[dict] = []

    print()
    print("Processing step 01: illumination normalization")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"PNG compression: {PNG_COMPRESSION}")

    if show_windows:
        print()
        print("Controls:")
        print("  n / SPACE / ENTER  -> next image")
        print("  q / ESC            -> quit")

    for index, image_path in enumerate(image_paths, start=1):
        row = process_image(
            image_path,
            index,
            image_count,
            show_windows=show_windows,
            wait_between_images=wait_between_images,
        )
        if row is None: continue
        stopped = bool(row.pop("stopped", False))
        metadata_rows.append(row)
        if stopped: break

    save_metadata(metadata_rows)
    if show_windows: cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Processed images saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")
