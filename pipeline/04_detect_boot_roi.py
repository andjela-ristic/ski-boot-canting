from pathlib import Path
import csv

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_04_CONFIG = CONFIG["step_04_detect_boot_roi"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

EDGES_INPUT_DIR = PROCESSED_DIR / STEP_04_CONFIG["input_edges_subdir"]
VISUAL_INPUT_DIR = PROCESSED_DIR / STEP_04_CONFIG["input_visual_subdir"]

OUTPUT_DIR = PROCESSED_DIR / STEP_04_CONFIG["output_subdir"]

ACTIVITY_MASK_DIR = OUTPUT_DIR / "activity_mask"
BOOT_MASK_DIR = OUTPUT_DIR / "boot_mask"
ROI_MASK_DIR = OUTPUT_DIR / "roi_mask"
MASKED_EDGES_BOOT_DIR = OUTPUT_DIR / "masked_edges_boot"
MASKED_EDGES_ROI_DIR = OUTPUT_DIR / "masked_edges_roi"
OVERLAY_DIR = OUTPUT_DIR / "overlay"

CSV_PATH = METADATA_DIR / "processing_04_detect_boot_roi.csv"


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    image_paths = [
        path
        for path in EDGES_INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    return sorted(image_paths)


def ensure_odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError(f"Kernel size must be positive. Got: {value}")

    if value % 2 == 0:
        raise ValueError(f"Kernel size must be odd. Got: {value}")

    return value


def resize_for_display(image: np.ndarray) -> np.ndarray:
    max_height = int(DISPLAY_CONFIG["max_height"])

    height, width = image.shape[:2]

    if height <= max_height:
        return image

    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    image = to_bgr(image).copy()

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 48),
        (0, 0, 0),
        thickness=-1
    )

    cv2.putText(
        image,
        label,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


def make_grid(images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    prepared = []

    for label, image in images:
        display = resize_for_display(to_bgr(image))
        display = add_label(display, label)
        prepared.append(display)

    target_height = min(image.shape[0] for image in prepared)

    resized = []

    for image in prepared:
        height, width = image.shape[:2]
        new_width = int(width * target_height / height)

        resized_image = cv2.resize(
            image,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA
        )

        resized.append(resized_image)

    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)

    combined = resized[0]

    for image in resized[1:]:
        combined = np.hstack([combined, separator, image])

    return combined


def load_visual_image(image_name: str, edge_image: np.ndarray) -> np.ndarray:
    visual_path = VISUAL_INPUT_DIR / image_name

    if visual_path.exists():
        visual = cv2.imread(str(visual_path))

        if visual is not None:
            return visual

    return cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)


def build_activity_mask(edge_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    config = STEP_05_CONFIG["edge_activity"]

    density_kernel_size = ensure_odd_kernel_size(
        int(config["density_kernel_size"])
    )

    density_threshold = float(config["density_threshold"])

    close_kernel_size = ensure_odd_kernel_size(
        int(config["close_kernel_size"])
    )

    dilate_kernel_size = ensure_odd_kernel_size(
        int(config["dilate_kernel_size"])
    )

    dilate_iterations = int(config["dilate_iterations"])

    edge_binary = (edge_image > 0).astype(np.float32)

    density_map = cv2.blur(
        edge_binary,
        (density_kernel_size, density_kernel_size)
    )

    activity_mask = np.where(
        density_map >= density_threshold,
        255,
        0
    ).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size)
    )

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilate_kernel_size, dilate_kernel_size)
    )

    activity_mask = cv2.morphologyEx(
        activity_mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    activity_mask = cv2.dilate(
        activity_mask,
        dilate_kernel,
        iterations=dilate_iterations
    )

    density_visual = np.clip(density_map * 255 * 8, 0, 255).astype(np.uint8)

    return activity_mask, density_visual


def score_component(
    edge_image: np.ndarray,
    component_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    area: int,
    image_width: int,
    image_height: int
) -> float:
    scoring_config = STEP_05_CONFIG["component_scoring"]

    bbox_area = max(1, w * h)

    edge_pixels_inside = int(np.count_nonzero(
        cv2.bitwise_and(edge_image, edge_image, mask=component_mask)
    ))

    edge_density = edge_pixels_inside / bbox_area

    edge_density_score = min(1.0, edge_density / 0.08)
    area_score = min(1.0, area / (image_width * image_height * 0.22))

    center_x_ratio = (x + w / 2) / image_width
    center_y_ratio = (y + h / 2) / image_height

    preferred_x_ratio = float(scoring_config["preferred_x_ratio"])
    preferred_y_ratio = float(scoring_config["preferred_y_ratio"])

    x_band_ratio = float(scoring_config["x_band_ratio"])
    y_band_ratio = float(scoring_config["y_band_ratio"])

    x_center_score = max(
        0.0,
        1.0 - abs(center_x_ratio - preferred_x_ratio) / x_band_ratio
    )

    y_center_score = max(
        0.0,
        1.0 - abs(center_y_ratio - preferred_y_ratio) / y_band_ratio
    )

    center_score = 0.5 * x_center_score + 0.5 * y_center_score

    aspect_ratio = h / max(1, w)

    if 0.7 <= aspect_ratio <= 4.5:
        shape_score = 1.0
    elif 0.45 <= aspect_ratio < 0.7:
        shape_score = 0.65
    elif 4.5 < aspect_ratio <= 6.0:
        shape_score = 0.55
    else:
        shape_score = 0.25

    score = (
        float(scoring_config["edge_density_weight"]) * edge_density_score +
        float(scoring_config["area_weight"]) * area_score +
        float(scoring_config["center_weight"]) * center_score +
        float(scoring_config["shape_weight"]) * shape_score
    )

    return score


def find_boot_component(
    edge_image: np.ndarray,
    activity_mask: np.ndarray
) -> tuple[dict | None, list[dict]]:
    height, width = edge_image.shape[:2]

    filtering_config = STEP_05_CONFIG["component_filtering"]

    min_area = width * height * float(filtering_config["min_area_ratio"])
    max_area = width * height * float(filtering_config["max_area_ratio"])

    min_width = width * float(filtering_config["min_width_ratio"])
    max_width = width * float(filtering_config["max_width_ratio"])

    min_height = height * float(filtering_config["min_height_ratio"])
    max_height = height * float(filtering_config["max_height_ratio"])

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        activity_mask,
        connectivity=8
    )

    candidates = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue

        if w < min_width or w > max_width:
            continue

        if h < min_height or h > max_height:
            continue

        component_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)

        score = score_component(
            edge_image=edge_image,
            component_mask=component_mask,
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            image_width=width,
            image_height=height
        )

        candidates.append({
            "label_id": label_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "score": score,
            "component_mask": component_mask,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected = candidates[0] if candidates else None

    return selected, candidates


def create_roi_mask(
    image_shape: tuple[int, int],
    selected_component: dict | None
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    height, width = image_shape

    roi_mask = np.zeros((height, width), dtype=np.uint8)

    if selected_component is None:
        return roi_mask, None

    padding_config = STEP_04_CONFIG["roi_padding"]

    x_padding = int(width * float(padding_config["x_padding_ratio"]))
    y_padding = int(height * float(padding_config["y_padding_ratio"]))

    x = selected_component["x"]
    y = selected_component["y"]
    w = selected_component["w"]
    h = selected_component["h"]

    x1 = max(0, x - x_padding)
    y1 = max(0, y - y_padding)
    x2 = min(width - 1, x + w + x_padding)
    y2 = min(height - 1, y + h + y_padding)

    roi_mask[y1:y2 + 1, x1:x2 + 1] = 255

    return roi_mask, (x1, y1, x2, y2)


def create_boot_mask(
    image_shape: tuple[int, int],
    selected_component: dict | None
) -> np.ndarray:
    height, width = image_shape

    boot_mask = np.zeros((height, width), dtype=np.uint8)

    if selected_component is None:
        return boot_mask

    component_mask = selected_component["component_mask"]

    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return component_mask

    largest_contour = max(contours, key=cv2.contourArea)

    if bool(STEP_05_CONFIG["boot_mask"]["use_convex_hull"]):
        hull = cv2.convexHull(largest_contour)

        cv2.drawContours(
            boot_mask,
            [hull],
            contourIdx=-1,
            color=255,
            thickness=-1
        )
    else:
        cv2.drawContours(
            boot_mask,
            [largest_contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

    return boot_mask


def draw_overlay(
    visual_image: np.ndarray,
    activity_mask: np.ndarray,
    selected_component: dict | None,
    candidates: list[dict],
    roi_box: tuple[int, int, int, int] | None
) -> np.ndarray:
    overlay = visual_image.copy()

    show_all_candidate_boxes = bool(
        STEP_05_CONFIG["draw"]["show_all_candidate_boxes"]
    )

    if show_all_candidate_boxes:
        for candidate in candidates:
            x = candidate["x"]
            y = candidate["y"]
            w = candidate["w"]
            h = candidate["h"]

            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                (80, 80, 80),
                1
            )

            cv2.putText(
                overlay,
                f"{candidate['score']:.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 80, 80),
                1,
                cv2.LINE_AA
            )

    if selected_component is not None:
        x = selected_component["x"]
        y = selected_component["y"]
        w = selected_component["w"]
        h = selected_component["h"]

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            3
        )

        cv2.putText(
            overlay,
            f"selected activity {selected_component['score']:.2f}",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4
        )

        cv2.putText(
            overlay,
            "vertical search ROI",
            (x1, min(y2 + 30, overlay.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    activity_colored = cv2.applyColorMap(activity_mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(overlay, 0.75, activity_colored, 0.25, 0)

    return blended


def save_metadata(rows: list[dict]) -> None:
    if not rows:
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "activity_mask_file",
        "boot_mask_file",
        "roi_mask_file",
        "masked_edges_boot_file",
        "masked_edges_roi_file",
        "overlay_file",
        "width",
        "height",
        "processing_step",
        "component_found",
        "component_score",
        "component_x",
        "component_y",
        "component_w",
        "component_h",
        "component_area",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "candidates_count",
        "selected_vertical_mask",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_DIR,
        ACTIVITY_MASK_DIR,
        BOOT_MASK_DIR,
        ROI_MASK_DIR,
        MASKED_EDGES_BOOT_DIR,
        MASKED_EDGES_ROI_DIR,
        OVERLAY_DIR,
        METADATA_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not STEP_05_CONFIG["enabled"]:
        print("Step 05 is disabled in config.")
        return

    ensure_output_dirs()

    image_paths = collect_images()

    if not image_paths:
        print(f"No edge images found in: {EDGES_INPUT_DIR}")
        return

    metadata_rows = []

    print()
    print("Processing step 05: detect boot ROI from edge activity")
    print(f"Input edges:  {EDGES_INPUT_DIR}")
    print(f"Input visual: {VISUAL_INPUT_DIR}")
    print(f"Output:       {OUTPUT_DIR}")
    print()
    print("This step is color-invariant.")
    print("It detects a compact high-edge-activity region and creates a boot ROI.")
    print()
    print("Controls:")
    print("  n / SPACE / ENTER  -> next image")
    print("  q / ESC            -> quit")
    print()

    for index, edge_path in enumerate(image_paths, start=1):
        edge_image = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)

        if edge_image is None:
            print(f"Could not read edge image: {edge_path}")
            continue

        height, width = edge_image.shape[:2]

        visual_image = load_visual_image(edge_path.name, edge_image)

        activity_mask, density_visual = build_activity_mask(edge_image)

        selected_component, candidates = find_boot_component(
            edge_image=edge_image,
            activity_mask=activity_mask
        )

        roi_mask, roi_box = create_roi_mask(
            image_shape=(height, width),
            selected_component=selected_component
        )

        boot_mask = create_boot_mask(
            image_shape=(height, width),
            selected_component=selected_component
        )

        masked_edges_boot = cv2.bitwise_and(
            edge_image,
            edge_image,
            mask=boot_mask
        )

        masked_edges_roi = cv2.bitwise_and(
            edge_image,
            edge_image,
            mask=roi_mask
        )

        overlay = draw_overlay(
            visual_image=visual_image,
            activity_mask=activity_mask,
            selected_component=selected_component,
            candidates=candidates,
            roi_box=roi_box
        )

        activity_mask_path = ACTIVITY_MASK_DIR / edge_path.name
        boot_mask_path = BOOT_MASK_DIR / edge_path.name
        roi_mask_path = ROI_MASK_DIR / edge_path.name
        masked_edges_boot_path = MASKED_EDGES_BOOT_DIR / edge_path.name
        masked_edges_roi_path = MASKED_EDGES_ROI_DIR / edge_path.name
        overlay_path = OVERLAY_DIR / edge_path.name

        cv2.imwrite(str(activity_mask_path), activity_mask)
        cv2.imwrite(str(boot_mask_path), boot_mask)
        cv2.imwrite(str(roi_mask_path), roi_mask)
        cv2.imwrite(str(masked_edges_boot_path), masked_edges_boot)
        cv2.imwrite(str(masked_edges_roi_path), masked_edges_roi)
        cv2.imwrite(str(overlay_path), overlay)

        selected_vertical_mask = STEP_05_CONFIG["boot_mask"]["selected_vertical_mask"]

        if selected_component is None:
            row = {
                "source_file": str(edge_path.relative_to(PROJECT_ROOT)),
                "activity_mask_file": str(activity_mask_path.relative_to(PROJECT_ROOT)),
                "boot_mask_file": str(boot_mask_path.relative_to(PROJECT_ROOT)),
                "roi_mask_file": str(roi_mask_path.relative_to(PROJECT_ROOT)),
                "masked_edges_boot_file": str(masked_edges_boot_path.relative_to(PROJECT_ROOT)),
                "masked_edges_roi_file": str(masked_edges_roi_path.relative_to(PROJECT_ROOT)),
                "overlay_file": str(overlay_path.relative_to(PROJECT_ROOT)),
                "width": width,
                "height": height,
                "processing_step": "04_detect_boot_roi",
                "component_found": False,
                "component_score": "",
                "component_x": "",
                "component_y": "",
                "component_w": "",
                "component_h": "",
                "component_area": "",
                "roi_x1": "",
                "roi_y1": "",
                "roi_x2": "",
                "roi_y2": "",
                "candidates_count": len(candidates),
                "selected_vertical_mask": selected_vertical_mask,
            }
        else:
            if roi_box is None:
                roi_x1 = roi_y1 = roi_x2 = roi_y2 = ""
            else:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_box

            row = {
                "source_file": str(edge_path.relative_to(PROJECT_ROOT)),
                "activity_mask_file": str(activity_mask_path.relative_to(PROJECT_ROOT)),
                "boot_mask_file": str(boot_mask_path.relative_to(PROJECT_ROOT)),
                "roi_mask_file": str(roi_mask_path.relative_to(PROJECT_ROOT)),
                "masked_edges_boot_file": str(masked_edges_boot_path.relative_to(PROJECT_ROOT)),
                "masked_edges_roi_file": str(masked_edges_roi_path.relative_to(PROJECT_ROOT)),
                "overlay_file": str(overlay_path.relative_to(PROJECT_ROOT)),
                "width": width,
                "height": height,
                "processing_step": "04_detect_boot_roi",
                "component_found": True,
                "component_score": round(float(selected_component["score"]), 4),
                "component_x": selected_component["x"],
                "component_y": selected_component["y"],
                "component_w": selected_component["w"],
                "component_h": selected_component["h"],
                "component_area": selected_component["area"],
                "roi_x1": roi_x1,
                "roi_y1": roi_y1,
                "roi_x2": roi_x2,
                "roi_y2": roi_y2,
                "candidates_count": len(candidates),
                "selected_vertical_mask": selected_vertical_mask,
            }

        metadata_rows.append(row)

        print(
            f"[{index}/{len(image_paths)}] Saved: {edge_path.name} | "
            f"component_found={row['component_found']} | "
            f"candidates={len(candidates)}"
        )

        if DISPLAY_CONFIG["show_windows"]:
            grid = make_grid([
                ("visual + selected ROI", overlay),
                ("activity mask", activity_mask),
                ("masked edges ROI", masked_edges_roi),
                ("masked edges boot mask", masked_edges_boot),
            ])

            title = f"05 Detect boot ROI | {index}/{len(image_paths)} | {edge_path.name}"

            cv2.imshow(title, grid)

            if DISPLAY_CONFIG["wait_between_images"]:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(500) & 0xFF

            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass

            if key in [ord("q"), 27]:
                print("Stopped by user.")
                break

    save_metadata(metadata_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Boot ROI outputs saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()

