from pathlib import Path
import argparse
import csv
import json
import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()

PATHS = CONFIG["paths"]
STEP = CONFIG["step_06_detect_boot_landmarks"]
DISPLAY = CONFIG.get("display", {})

PROCESSED_DIR = PROJECT_ROOT / PATHS["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS["metadata_dir"]

VISUAL_INPUT_DIR = PROCESSED_DIR / STEP["input_visual_subdir"]
EDGE_INPUT_DIR = PROCESSED_DIR / STEP.get("input_edges_subdir", "03_edges")

OUTPUT_DIR = PROCESSED_DIR / STEP["output_subdir"]
OVERLAY_DIR = OUTPUT_DIR / "circles_overlay"
JSON_DIR = OUTPUT_DIR / "json"
CSV_PATH = METADATA_DIR / "processing_06_detect_boot_landmarks.csv"


def ensure_dirs():
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_images():
    allowed = {".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in VISUAL_INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )


def make_edge(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_k = int(STEP.get("edge_fallback", {}).get("blur_kernel", 5))
    if blur_k % 2 == 0:
        blur_k += 1

    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    t1 = int(STEP.get("edge_fallback", {}).get("threshold_1", 50))
    t2 = int(STEP.get("edge_fallback", {}).get("threshold_2", 150))

    return cv2.Canny(gray, t1, t2, apertureSize=3, L2gradient=True)


def load_or_make_edge(image_path, image_bgr):
    edge_path = EDGE_INPUT_DIR / image_path.name

    if edge_path.exists():
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        if edge is not None:
            if edge.shape[:2] != image_bgr.shape[:2]:
                edge = cv2.resize(
                    edge,
                    (image_bgr.shape[1], image_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            return edge

    return make_edge(image_bgr)


def prepare_detection_input(image_path, image_bgr):
    input_mode = str(STEP.get("detection_input", "edges")).lower()

    if input_mode == "grayscale":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), "grayscale"

    if input_mode != "edges":
        raise ValueError(f"Unsupported detection_input: {input_mode}")

    return load_or_make_edge(image_path, image_bgr), "edges"


def find_circles(detection_input, input_mode):
    hough = STEP.get("hough", {})
    hough_input = detection_input

    if input_mode == "edges":
        dilate_iterations = int(hough.get("edge_dilate_iterations", 0))
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hough_input = cv2.dilate(hough_input, kernel, iterations=dilate_iterations)
    elif input_mode == "grayscale":
        blur_k = int(hough.get("median_blur_kernel", 1))
        if blur_k > 1:
            if blur_k % 2 == 0:
                blur_k += 1
            hough_input = cv2.medianBlur(hough_input, blur_k)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    circles = cv2.HoughCircles(
        hough_input,
        cv2.HOUGH_GRADIENT,
        dp=float(hough.get("dp", 1.2)),
        minDist=float(hough.get("min_dist", 35)),
        param1=float(hough.get("param1", 100)),
        param2=float(hough.get("param2", 18)),
        minRadius=int(hough.get("min_radius", 6)),
        maxRadius=int(hough.get("max_radius", 45)),
    )

    if circles is None:
        return [], hough_input

    return np.round(circles[0]).astype(int).tolist(), hough_input


def draw_circles(image_bgr, circles):
    out = image_bgr.copy()

    for i, (x, y, r) in enumerate(circles, start=1):
        cv2.circle(out, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (x + r + 3, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def resize_for_display(image):
    max_h = int(DISPLAY.get("max_height", 800))
    h, w = image.shape[:2]

    if h <= max_h:
        return image

    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h), interpolation=cv2.INTER_AREA)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_metadata(rows: list[dict]) -> None:
    fieldnames = [
        "source_file",
        "edge_input_file",
        "overlay_file",
        "json_file",
        "width",
        "height",
        "processing_step",
        "input_mode",
        "nonzero_before",
        "nonzero_hough",
        "dp",
        "min_dist",
        "param1",
        "param2",
        "min_radius",
        "max_radius",
        "circle_count",
        "circle_id",
        "x",
        "y",
        "radius",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_image(image_path, show):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot read image: {image_path}")
        return

    detection_input, input_mode = prepare_detection_input(image_path, image)
    circles, hough_input = find_circles(detection_input, input_mode)
    overlay = draw_circles(image, circles)

    nonzero_before = int(cv2.countNonZero(detection_input))
    nonzero_hough = int(cv2.countNonZero(hough_input))
    hough = STEP.get("hough", {})
    overlay_path = OVERLAY_DIR / image_path.name
    json_path = JSON_DIR / f"{image_path.stem}.json"

    cv2.imwrite(str(overlay_path), overlay)
    save_json(
        json_path,
        {
            "source_file": relative(image_path),
            "overlay_file": relative(overlay_path),
            "input_mode": input_mode,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "processing_step": "06_detect_boot_landmarks",
            "nonzero_before": nonzero_before,
            "nonzero_hough": nonzero_hough,
            "hough_parameters": {
                "dp": float(hough.get("dp", 1.2)),
                "min_dist": float(hough.get("min_dist", 35)),
                "param1": float(hough.get("param1", 100)),
                "param2": float(hough.get("param2", 18)),
                "min_radius": int(hough.get("min_radius", 6)),
                "max_radius": int(hough.get("max_radius", 45)),
            },
            "circle_count": len(circles),
            "circles": [
                {
                    "id": index,
                    "x": int(x),
                    "y": int(y),
                    "radius": int(r),
                }
                for index, (x, y, r) in enumerate(circles, start=1)
            ],
        },
    )

    print(
        f"{image_path.name}: input={input_mode} "
        f"nonzero_before={nonzero_before} "
        f"nonzero_hough={nonzero_hough} "
        f"circles={len(circles)} "
        f"radius=({hough['min_radius']},{hough['max_radius']}) "
        f"param2={hough['param2']}"
    )

    if show:
        detection_bgr = cv2.cvtColor(hough_input, cv2.COLOR_GRAY2BGR)
        grid = np.hstack([
            resize_for_display(image),
            resize_for_display(detection_bgr),
            resize_for_display(overlay),
        ])

        cv2.imshow(f"circles | {image_path.name}", grid)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in [ord("q"), 27]:
            raise KeyboardInterrupt

    rows = []

    for index, (x, y, r) in enumerate(circles, start=1):
        rows.append({
            "source_file": relative(image_path),
            "edge_input_file": relative(EDGE_INPUT_DIR / image_path.name),
            "overlay_file": relative(overlay_path),
            "json_file": relative(json_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "processing_step": "06_detect_boot_landmarks",
            "input_mode": input_mode,
            "nonzero_before": nonzero_before,
            "nonzero_hough": nonzero_hough,
            "dp": float(hough.get("dp", 1.2)),
            "min_dist": float(hough.get("min_dist", 35)),
            "param1": float(hough.get("param1", 100)),
            "param2": float(hough.get("param2", 18)),
            "min_radius": int(hough.get("min_radius", 6)),
            "max_radius": int(hough.get("max_radius", 45)),
            "circle_count": len(circles),
            "circle_id": f"circle_{index:03d}",
            "x": int(x),
            "y": int(y),
            "radius": int(r),
        })

    if not rows:
        rows.append({
            "source_file": relative(image_path),
            "edge_input_file": relative(EDGE_INPUT_DIR / image_path.name),
            "overlay_file": relative(overlay_path),
            "json_file": relative(json_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "processing_step": "06_detect_boot_landmarks",
            "input_mode": input_mode,
            "nonzero_before": nonzero_before,
            "nonzero_hough": nonzero_hough,
            "dp": float(hough.get("dp", 1.2)),
            "min_dist": float(hough.get("min_dist", 35)),
            "param1": float(hough.get("param1", 100)),
            "param2": float(hough.get("param2", 18)),
            "min_radius": int(hough.get("min_radius", 6)),
            "max_radius": int(hough.get("max_radius", 45)),
            "circle_count": 0,
            "circle_id": "",
            "x": "",
            "y": "",
            "radius": "",
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    images = collect_images()
    if not images:
        print(f"No images found in: {VISUAL_INPUT_DIR}")
        return

    show = bool(DISPLAY.get("show_windows", True)) and not args.no_show
    metadata_rows = []

    for image_path in images:
        metadata_rows.extend(process_image(image_path, show))

    save_metadata(metadata_rows)


if __name__ == "__main__":
    main()
