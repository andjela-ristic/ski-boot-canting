from pathlib import Path
import csv
import json
import math
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config()

PATHS_CONFIG = CONFIG["paths"]
DISPLAY_CONFIG = CONFIG["display"]
STEP_06_CONFIG = CONFIG["step_06_complete_line_fragments"]

PROCESSED_DIR = PROJECT_ROOT / PATHS_CONFIG["processed_dir"]
METADATA_DIR = PROJECT_ROOT / PATHS_CONFIG["metadata_dir"]

LINES_INPUT_DIR = PROCESSED_DIR / STEP_06_CONFIG["input_lines_subdir"]
BOOT_MASK_INPUT_DIR = PROCESSED_DIR / STEP_06_CONFIG["input_boot_mask_subdir"]
VISUAL_INPUT_DIR = PROCESSED_DIR / STEP_06_CONFIG["input_visual_subdir"]

OUTPUT_DIR = PROCESSED_DIR / STEP_06_CONFIG["output_subdir"]
VALID_FRAGMENTS_DIR = OUTPUT_DIR / "valid_fragments"
MERGED_LINES_DIR = OUTPUT_DIR / "merged_lines"
OVERLAY_DIR = OUTPUT_DIR / "overlay"
JSON_DIR = OUTPUT_DIR / "json"

CSV_PATH = METADATA_DIR / "processing_06_complete_line_fragments.csv"


@dataclass
class Fragment:
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    mask_coverage: float
    source_component_area: int

    @property
    def p1(self) -> np.ndarray:
        return np.array([self.x1, self.y1], dtype=np.float32)

    @property
    def p2(self) -> np.ndarray:
        return np.array([self.x2, self.y2], dtype=np.float32)


@dataclass
class Group:
    fragment_ids: set[str]
    points: np.ndarray
    point: np.ndarray
    direction: np.ndarray
    angle_deg: float
    fit_rms: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    max_gap_used: float = 0.0
    max_lateral_offset: float = 0.0


@dataclass
class MergedLine:
    id: str
    source_fragment_ids: list[str]
    fragment_count: int
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    mask_coverage: float
    max_gap_pixels_used: float
    max_lateral_offset_px: float
    fit_rms_px: float
    score: float


# ----------------------------- generic display helpers -----------------------------


def collect_images() -> list[Path]:
    allowed_extensions = {".png", ".jpg", ".jpeg"}

    image_paths = [
        path
        for path in LINES_INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    return sorted(image_paths)


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


# ----------------------------- geometry helpers -----------------------------


def canonical_segment(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> tuple[float, float, float, float]:
    if y1 <= y2:
        return float(x1), float(y1), float(x2), float(y2)

    return float(x2), float(y2), float(x1), float(y1)


def segment_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return float(angle)


def distance_to_vertical_deg(angle_deg: float) -> float:
    return abs(90.0 - abs(angle_deg))


def angle_diff_deg(a: float, b: float) -> float:
    difference = abs(a - b) % 180.0

    if difference > 90.0:
        difference = 180.0 - difference

    return float(difference)


def direction_from_angle(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    direction = np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)
    norm = float(np.linalg.norm(direction))

    if norm == 0:
        return np.array([0.0, 1.0], dtype=np.float32)

    direction = direction / norm

    if direction[1] < 0:
        direction *= -1.0

    return direction


def perpendicular(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0]], dtype=np.float32)


def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if len(points) < 2:
        point = points[0].astype(np.float32) if len(points) == 1 else np.array([0, 0], dtype=np.float32)
        return point, np.array([0.0, 1.0], dtype=np.float32), 90.0, 0.0

    fitted = cv2.fitLine(
        points.astype(np.float32).reshape(-1, 1, 2),
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).flatten()

    vx, vy, x0, y0 = fitted
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)

    if direction[1] < 0:
        direction *= -1.0

    point = np.array([float(x0), float(y0)], dtype=np.float32)
    normal = perpendicular(direction)
    residuals = np.abs((points.astype(np.float32) - point) @ normal)
    rms = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else 0.0
    angle = segment_angle_deg(0, 0, float(direction[0]), float(direction[1]))

    return point, direction, angle, rms


def project_interval(
    points: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray
) -> tuple[float, float]:
    projection = (points.astype(np.float32) - point) @ direction

    return float(np.min(projection)), float(np.max(projection))


def segment_endpoints_from_interval(
    point: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    t_max: float
) -> tuple[np.ndarray, np.ndarray]:
    return point + direction * t_min, point + direction * t_max


def sample_segment_points(
    p1: np.ndarray,
    p2: np.ndarray,
    step_px: float
) -> np.ndarray:
    length = float(np.linalg.norm(p2.astype(np.float32) - p1.astype(np.float32)))
    count = max(2, int(math.ceil(length / max(step_px, 0.5))) + 1)

    xs = np.linspace(float(p1[0]), float(p2[0]), count)
    ys = np.linspace(float(p1[1]), float(p2[1]), count)

    return np.stack([xs, ys], axis=1).astype(np.float32)


def prepare_sampling_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8)

    if radius_px <= 0:
        return binary_mask

    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    return cv2.dilate(binary_mask, kernel, iterations=1)


def mask_coverage_for_points(
    sampling_mask: np.ndarray,
    points: np.ndarray
) -> float:
    if points.size == 0:
        return 0.0

    height, width = sampling_mask.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    xs = rounded[:, 0]
    ys = rounded[:, 1]

    valid = (
        (xs >= 0) &
        (ys >= 0) &
        (xs < width) &
        (ys < height)
    )

    if not np.any(valid):
        return 0.0

    inside = sampling_mask[ys[valid], xs[valid]] > 0

    return float(np.count_nonzero(inside)) / float(len(points))


def mask_coverage_for_segment(
    sampling_mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    step_px: float
) -> float:
    points = sample_segment_points(p1, p2, step_px)

    return mask_coverage_for_points(sampling_mask, points)


def interval_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    if a[1] < b[0]:
        return b[0] - a[1]

    if b[1] < a[0]:
        return a[0] - b[1]

    return 0.0


def connector_points_between_intervals(
    point: np.ndarray,
    direction: np.ndarray,
    a: tuple[float, float],
    b: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray] | None:
    if a[1] < b[0]:
        return point + direction * a[1], point + direction * b[0]

    if b[1] < a[0]:
        return point + direction * b[1], point + direction * a[0]

    return None


def extend_interval_inside_mask(
    sampling_mask: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    t_max: float,
    step_px: float,
    max_extend_px: float
) -> tuple[float, float]:
    max_steps = int(max_extend_px / max(step_px, 0.5))

    if max_steps <= 0:
        return float(t_min), float(t_max)

    offsets = np.arange(1, max_steps + 1, dtype=np.float32) * float(step_px)

    backward_ts = float(t_min) - offsets
    backward_points = point.reshape(1, 2) + backward_ts.reshape(-1, 1) * direction.reshape(1, 2)

    new_min = float(t_min)
    rounded = np.rint(backward_points).astype(np.int32)
    xs = rounded[:, 0]
    ys = rounded[:, 1]
    height, width = sampling_mask.shape[:2]
    valid = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
    hits = np.zeros(len(backward_points), dtype=bool)
    hits[valid] = sampling_mask[ys[valid], xs[valid]] > 0
    miss_indices = np.flatnonzero(~hits)
    if len(miss_indices) == 0:
        new_min = float(backward_ts[-1])
    elif miss_indices[0] > 0:
        new_min = float(backward_ts[miss_indices[0] - 1])

    forward_ts = float(t_max) + offsets
    forward_points = point.reshape(1, 2) + forward_ts.reshape(-1, 1) * direction.reshape(1, 2)
    new_max = float(t_max)
    rounded = np.rint(forward_points).astype(np.int32)
    xs = rounded[:, 0]
    ys = rounded[:, 1]
    height, width = sampling_mask.shape[:2]
    valid = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
    hits = np.zeros(len(forward_points), dtype=bool)
    hits[valid] = sampling_mask[ys[valid], xs[valid]] > 0
    miss_indices = np.flatnonzero(~hits)
    if len(miss_indices) == 0:
        new_max = float(forward_ts[-1])
    elif miss_indices[0] > 0:
        new_max = float(forward_ts[miss_indices[0] - 1])

    return new_min, new_max


def build_group(
    fragment_ids: set[str],
    points: np.ndarray,
    max_gap_used: float = 0.0,
    max_lateral_offset: float = 0.0
) -> Group:
    normalized_points = points.astype(np.float32)
    point, direction, angle_deg, fit_rms = fit_line(normalized_points)

    return Group(
        fragment_ids=set(fragment_ids),
        points=normalized_points,
        point=point,
        direction=direction,
        angle_deg=float(angle_deg),
        fit_rms=float(fit_rms),
        x_min=float(np.min(normalized_points[:, 0])),
        x_max=float(np.max(normalized_points[:, 0])),
        y_min=float(np.min(normalized_points[:, 1])),
        y_max=float(np.max(normalized_points[:, 1])),
        max_gap_used=float(max_gap_used),
        max_lateral_offset=float(max_lateral_offset)
    )


# ----------------------------- input loading -----------------------------


def load_boot_mask(image_name: str) -> np.ndarray | None:
    mask_path = BOOT_MASK_INPUT_DIR / image_name

    if not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    return mask


def load_visual_image(image_name: str, fallback: np.ndarray) -> np.ndarray:
    visual_path = VISUAL_INPUT_DIR / image_name

    if visual_path.exists():
        visual = cv2.imread(str(visual_path))

        if visual is not None:
            return visual

    return fallback.copy()


def extract_green_mask(line_image: np.ndarray) -> np.ndarray:
    config = STEP_06_CONFIG["green_extraction"]

    min_green = int(config["min_green"])
    max_red = int(config["max_red"])
    max_blue = int(config["max_blue"])
    green_dominance = int(config["green_dominance"])

    blue, green, red = cv2.split(line_image)

    green_mask = np.where(
        (green >= min_green) &
        (red <= max_red) &
        (blue <= max_blue) &
        ((green.astype(np.int16) - red.astype(np.int16)) >= green_dominance) &
        ((green.astype(np.int16) - blue.astype(np.int16)) >= green_dominance),
        255,
        0
    ).astype(np.uint8)

    close_kernel_size = int(config["close_kernel_size"])

    if close_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_kernel_size, close_kernel_size)
        )
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    return green_mask


def component_to_fragment(
    component_id: int,
    component_points: np.ndarray,
    sampling_mask: np.ndarray
) -> Fragment | None:
    filtering_config = STEP_06_CONFIG["fragment_filtering"]
    mask_config = STEP_06_CONFIG["mask_validation"]

    min_component_area = int(filtering_config["min_component_area_px"])
    min_length = float(filtering_config["min_fragment_length_px"])
    max_vertical_deviation = float(filtering_config["max_vertical_deviation_deg"])
    sample_step = float(mask_config["sample_step_px"])
    min_mask_coverage = float(mask_config["min_fragment_mask_coverage"])

    if len(component_points) < min_component_area:
        return None

    points_xy = component_points[:, ::-1].astype(np.float32)
    point, direction, angle, _ = fit_line(points_xy)

    if distance_to_vertical_deg(angle) > max_vertical_deviation:
        return None

    t_min, t_max = project_interval(points_xy, point, direction)
    p1, p2 = segment_endpoints_from_interval(point, direction, t_min, t_max)

    x1, y1, x2, y2 = canonical_segment(
        float(p1[0]),
        float(p1[1]),
        float(p2[0]),
        float(p2[1])
    )

    length = segment_length(x1, y1, x2, y2)

    if length < min_length:
        return None

    coverage = mask_coverage_for_segment(
        sampling_mask,
        np.array([x1, y1], dtype=np.float32),
        np.array([x2, y2], dtype=np.float32),
        step_px=sample_step
    )

    if coverage < min_mask_coverage:
        return None

    return Fragment(
        id=f"f{component_id:04d}",
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        length=length,
        angle_deg=segment_angle_deg(x1, y1, x2, y2),
        mask_coverage=coverage,
        source_component_area=int(len(component_points))
    )


def extract_fragments_from_green_lines(
    line_image: np.ndarray,
    sampling_mask: np.ndarray
) -> tuple[list[Fragment], np.ndarray]:
    green_mask = extract_green_mask(line_image)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        green_mask,
        connectivity=8
    )

    fragments = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area <= 0:
            continue

        component_points = np.column_stack(np.where(labels == label_id))
        fragment = component_to_fragment(label_id, component_points, sampling_mask)

        if fragment is not None:
            fragments.append(fragment)

    return fragments, green_mask


# ----------------------------- merge logic -----------------------------


def group_from_fragment(fragment: Fragment) -> Group:
    sample_step = float(STEP_06_CONFIG["merge"]["fragment_sample_step_px"])
    points = sample_segment_points(fragment.p1, fragment.p2, sample_step)
    points = np.vstack([
        fragment.p1.reshape(1, 2),
        points,
        fragment.p2.reshape(1, 2)
    ]).astype(np.float32)

    return build_group(
        fragment_ids={fragment.id},
        points=points
    )


def compatibility_with_angle(
    first: Group,
    second: Group,
    sampling_mask: np.ndarray,
    candidate_angle: float,
    all_points: np.ndarray,
    centroid: np.ndarray,
    merged_fit_rms: float
) -> tuple[bool, float, float, float]:
    merge_config = STEP_06_CONFIG["merge"]

    max_gap = float(merge_config["max_projected_gap_px"])
    max_lateral = float(merge_config["max_lateral_offset_px"])
    max_rms = float(merge_config["max_merged_fit_rms_px"])
    min_connector_coverage = float(merge_config["min_connector_mask_coverage"])
    sample_step = float(STEP_06_CONFIG["mask_validation"]["sample_step_px"])

    direction = direction_from_angle(candidate_angle)
    normal = perpendicular(direction)

    first_distances = np.abs((first.points - centroid) @ normal)
    second_distances = np.abs((second.points - centroid) @ normal)

    lateral_offset = float(max(
        np.percentile(first_distances, 90),
        np.percentile(second_distances, 90)
    ))

    if lateral_offset > max_lateral:
        return False, 0.0, lateral_offset, 0.0

    first_interval = project_interval(first.points, centroid, direction)
    second_interval = project_interval(second.points, centroid, direction)
    gap = interval_gap(first_interval, second_interval)

    if gap > max_gap:
        return False, gap, lateral_offset, 0.0

    connector = connector_points_between_intervals(
        centroid,
        direction,
        first_interval,
        second_interval
    )

    connector_coverage = 1.0

    if connector is not None and gap > 0:
        connector_coverage = mask_coverage_for_segment(
            sampling_mask,
            connector[0],
            connector[1],
            step_px=sample_step
        )

        if connector_coverage < min_connector_coverage:
            return False, gap, lateral_offset, connector_coverage

    if merged_fit_rms > max_rms:
        return False, gap, lateral_offset, connector_coverage

    return True, gap, lateral_offset, connector_coverage


def are_groups_compatible(
    first: Group,
    second: Group,
    sampling_mask: np.ndarray
) -> tuple[bool, float, float]:
    merge_config = STEP_06_CONFIG["merge"]

    first_angle = first.angle_deg
    second_angle = second.angle_deg

    max_angle_diff = float(merge_config["max_angle_diff_deg"])
    rotation_probe = float(merge_config["small_rotation_probe_deg"])
    rotation_step = float(merge_config["small_rotation_step_deg"])

    if angle_diff_deg(first_angle, second_angle) > max_angle_diff + rotation_probe:
        return False, 0.0, 0.0

    all_points = np.vstack([first.points, second.points]).astype(np.float32)
    centroid = np.mean(all_points, axis=0)
    _, _, _, merged_fit_rms = fit_line(all_points)

    if merged_fit_rms > float(merge_config["max_merged_fit_rms_px"]):
        return False, 0.0, 0.0

    candidate_angles = [
        (first_angle + second_angle) / 2.0,
        first_angle,
        second_angle,
    ]

    if rotation_probe > 0:
        rotations = np.arange(
            -rotation_probe,
            rotation_probe + 0.001,
            max(rotation_step, 0.25)
        )

        candidate_angles.extend(float(first_angle + rotation) for rotation in rotations)
        candidate_angles.extend(float(second_angle + rotation) for rotation in rotations)

    best: tuple[bool, float, float, float] | None = None

    for candidate_angle in candidate_angles:
        candidate = compatibility_with_angle(
            first,
            second,
            sampling_mask,
            candidate_angle,
            all_points,
            centroid,
            merged_fit_rms
        )

        if not candidate[0]:
            continue

        if best is None or (candidate[1] + candidate[2]) < (best[1] + best[2]):
            best = candidate

    if best is None:
        return False, 0.0, 0.0

    return True, best[1], best[2]


def merge_groups(groups: list[Group], sampling_mask: np.ndarray) -> list[Group]:
    merge_config = STEP_06_CONFIG["merge"]
    gap_weight = float(merge_config["gap_cost_weight"])
    lateral_weight = float(merge_config["lateral_cost_weight"])

    changed = True

    while changed:
        changed = False
        best_pair: tuple[int, int, float, float] | None = None
        best_cost = float("inf")

        for first_index in range(len(groups)):
            first_group = groups[first_index]
            for second_index in range(first_index + 1, len(groups)):
                second_group = groups[second_index]
                compatible, gap, lateral = are_groups_compatible(
                    first_group,
                    second_group,
                    sampling_mask
                )

                if not compatible:
                    continue

                cost = gap * gap_weight + lateral * lateral_weight

                if cost < best_cost:
                    best_cost = cost
                    best_pair = (first_index, second_index, gap, lateral)

        if best_pair is None:
            break

        first_index, second_index, gap, lateral = best_pair
        first = groups[first_index]
        second = groups[second_index]

        merged = build_group(
            fragment_ids=set(first.fragment_ids) | set(second.fragment_ids),
            points=np.vstack([first.points, second.points]).astype(np.float32),
            max_gap_used=max(first.max_gap_used, second.max_gap_used, gap),
            max_lateral_offset=max(first.max_lateral_offset, second.max_lateral_offset, lateral)
        )

        groups = [
            group
            for index, group in enumerate(groups)
            if index not in (first_index, second_index)
        ] + [merged]

        changed = True

    return groups


def group_to_merged_line(
    index: int,
    group: Group,
    sampling_mask: np.ndarray
) -> MergedLine:
    output_config = STEP_06_CONFIG["output_filtering"]
    merge_config = STEP_06_CONFIG["merge"]
    sample_step = float(STEP_06_CONFIG["mask_validation"]["sample_step_px"])

    point = group.point
    direction = group.direction
    rms = group.fit_rms
    t_min, t_max = project_interval(group.points, point, direction)

    if bool(output_config["extend_output_to_mask"]):
        t_min, t_max = extend_interval_inside_mask(
            sampling_mask,
            point,
            direction,
            t_min,
            t_max,
            step_px=float(output_config["output_extend_step_px"]),
            max_extend_px=float(output_config["max_output_extend_px"])
        )

    p1, p2 = segment_endpoints_from_interval(point, direction, t_min, t_max)
    x1, y1, x2, y2 = canonical_segment(
        float(p1[0]),
        float(p1[1]),
        float(p2[0]),
        float(p2[1])
    )

    length = segment_length(x1, y1, x2, y2)
    coverage = mask_coverage_for_segment(
        sampling_mask,
        np.array([x1, y1], dtype=np.float32),
        np.array([x2, y2], dtype=np.float32),
        step_px=sample_step
    )

    score = (
        length *
        coverage *
        (1.0 + 0.20 * max(0, len(group.fragment_ids) - 1)) /
        (
            1.0 +
            rms +
            float(merge_config["gap_score_penalty"]) * group.max_gap_used +
            float(merge_config["lateral_score_penalty"]) * group.max_lateral_offset
        )
    )

    return MergedLine(
        id=f"m{index:03d}",
        source_fragment_ids=sorted(group.fragment_ids),
        fragment_count=len(group.fragment_ids),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        length=length,
        angle_deg=segment_angle_deg(x1, y1, x2, y2),
        mask_coverage=coverage,
        max_gap_pixels_used=float(group.max_gap_used),
        max_lateral_offset_px=float(group.max_lateral_offset),
        fit_rms_px=float(rms),
        score=float(score)
    )


# ----------------------------- outputs -----------------------------


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_DIR,
        VALID_FRAGMENTS_DIR,
        MERGED_LINES_DIR,
        OVERLAY_DIR,
        JSON_DIR,
        METADATA_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def draw_valid_fragments(
    visual_image: np.ndarray,
    fragments: list[Fragment]
) -> np.ndarray:
    output = visual_image.copy()

    for fragment in fragments:
        p1 = (int(round(fragment.x1)), int(round(fragment.y1)))
        p2 = (int(round(fragment.x2)), int(round(fragment.y2)))

        cv2.line(output, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(output, p1, 3, (0, 255, 255), -1)
        cv2.circle(output, p2, 3, (0, 255, 255), -1)

        cv2.putText(
            output,
            fragment.id,
            (p1[0] + 4, max(15, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


MERGED_LINE_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (180, 105, 255),
]


def merged_line_color(index: int) -> tuple[int, int, int]:
    return MERGED_LINE_COLORS[index % len(MERGED_LINE_COLORS)]


def draw_merged_lines(
    visual_image: np.ndarray,
    merged_lines: list[MergedLine]
) -> np.ndarray:
    output = visual_image.copy()

    for index, line in enumerate(merged_lines):
        color = merged_line_color(index)
        p1 = (int(round(line.x1)), int(round(line.y1)))
        p2 = (int(round(line.x2)), int(round(line.y2)))

        cv2.line(output, p1, p2, color, 3, cv2.LINE_AA)

        cv2.putText(
            output,
            f"{line.id} n={line.fragment_count}",
            (p1[0] + 4, max(15, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return output


def draw_debug_overlay(
    visual_image: np.ndarray,
    boot_mask: np.ndarray,
    fragments: list[Fragment],
    merged_lines: list[MergedLine]
) -> np.ndarray:
    output = cv2.convertScaleAbs(visual_image, alpha=0.5, beta=0)
    inside_mask = boot_mask > 0

    tinted = np.zeros_like(output)
    tinted[:, :, 1] = 110
    tinted[:, :, 2] = 185
    output[inside_mask] = cv2.addWeighted(
        visual_image[inside_mask],
        0.35,
        tinted[inside_mask],
        0.65,
        0
    )

    contours, _ = cv2.findContours(boot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (30, 220, 255), 3, cv2.LINE_AA)

    fragment_by_id = {fragment.id: fragment for fragment in fragments}

    for index, line in enumerate(merged_lines):
        color = merged_line_color(index)

        for fragment_id in line.source_fragment_ids:
            fragment = fragment_by_id.get(fragment_id)

            if fragment is None:
                continue

            p1 = (int(round(fragment.x1)), int(round(fragment.y1)))
            p2 = (int(round(fragment.x2)), int(round(fragment.y2)))

            cv2.line(output, p1, p2, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)
            cv2.circle(output, p1, 3, color, -1)
            cv2.circle(output, p2, 3, color, -1)

        p1 = (int(round(line.x1)), int(round(line.y1)))
        p2 = (int(round(line.x2)), int(round(line.y2)))

        cv2.line(output, p1, p2, (255, 255, 255), 8, cv2.LINE_AA)
        cv2.line(output, p1, p2, color, 4, cv2.LINE_AA)

        cv2.putText(
            output,
            f"{line.id} <- {line.fragment_count}",
            (p1[0] + 6, max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return output


def save_json(
    path: Path,
    data: dict[str, Any]
) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_metadata(rows: list[dict]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "line_input_file",
        "boot_mask_file",
        "valid_fragments_file",
        "merged_lines_file",
        "overlay_file",
        "json_file",
        "width",
        "height",
        "processing_step",
        "green_components_count",
        "valid_fragments_count",
        "merged_lines_count",
        "line_id",
        "fragment_count",
        "source_fragment_ids",
        "x1",
        "y1",
        "x2",
        "y2",
        "length",
        "angle_deg",
        "mask_coverage",
        "max_gap_pixels_used",
        "max_lateral_offset_px",
        "fit_rms_px",
        "score",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def make_empty_metadata_row(
    line_image_path: Path,
    boot_mask_path: Path | None,
    valid_fragments_path: Path,
    merged_lines_path: Path,
    overlay_path: Path,
    json_path: Path,
    width: int,
    height: int,
    green_components_count: int,
    valid_fragments_count: int,
    merged_lines_count: int
) -> dict:
    return {
        "source_file": line_image_path.name,
        "line_input_file": relative(line_image_path),
        "boot_mask_file": relative(boot_mask_path) if boot_mask_path is not None else "",
        "valid_fragments_file": relative(valid_fragments_path),
        "merged_lines_file": relative(merged_lines_path),
        "overlay_file": relative(overlay_path),
        "json_file": relative(json_path),
        "width": width,
        "height": height,
        "processing_step": "06_complete_line_fragments",
        "green_components_count": green_components_count,
        "valid_fragments_count": valid_fragments_count,
        "merged_lines_count": merged_lines_count,
        "line_id": "",
        "fragment_count": "",
        "source_fragment_ids": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "length": "",
        "angle_deg": "",
        "mask_coverage": "",
        "max_gap_pixels_used": "",
        "max_lateral_offset_px": "",
        "fit_rms_px": "",
        "score": "",
    }


def make_line_metadata_row(
    line_image_path: Path,
    boot_mask_path: Path,
    valid_fragments_path: Path,
    merged_lines_path: Path,
    overlay_path: Path,
    json_path: Path,
    width: int,
    height: int,
    green_components_count: int,
    valid_fragments_count: int,
    merged_lines_count: int,
    merged_line: MergedLine
) -> dict:
    return {
        "source_file": line_image_path.name,
        "line_input_file": relative(line_image_path),
        "boot_mask_file": relative(boot_mask_path),
        "valid_fragments_file": relative(valid_fragments_path),
        "merged_lines_file": relative(merged_lines_path),
        "overlay_file": relative(overlay_path),
        "json_file": relative(json_path),
        "width": width,
        "height": height,
        "processing_step": "06_complete_line_fragments",
        "green_components_count": green_components_count,
        "valid_fragments_count": valid_fragments_count,
        "merged_lines_count": merged_lines_count,
        "line_id": merged_line.id,
        "fragment_count": merged_line.fragment_count,
        "source_fragment_ids": "|".join(merged_line.source_fragment_ids),
        "x1": round(float(merged_line.x1), 3),
        "y1": round(float(merged_line.y1), 3),
        "x2": round(float(merged_line.x2), 3),
        "y2": round(float(merged_line.y2), 3),
        "length": round(float(merged_line.length), 3),
        "angle_deg": round(float(merged_line.angle_deg), 3),
        "mask_coverage": round(float(merged_line.mask_coverage), 4),
        "max_gap_pixels_used": round(float(merged_line.max_gap_pixels_used), 3),
        "max_lateral_offset_px": round(float(merged_line.max_lateral_offset_px), 3),
        "fit_rms_px": round(float(merged_line.fit_rms_px), 3),
        "score": round(float(merged_line.score), 4),
    }


# ----------------------------- main -----------------------------


def process_image(line_image_path: Path) -> tuple[list[dict], bool]:
    line_image = cv2.imread(str(line_image_path))

    if line_image is None:
        print(f"Could not read line image: {line_image_path}")
        return [], False

    height, width = line_image.shape[:2]
    boot_mask_path = BOOT_MASK_INPUT_DIR / line_image_path.name
    boot_mask = load_boot_mask(line_image_path.name)

    valid_fragments_path = VALID_FRAGMENTS_DIR / line_image_path.name
    merged_lines_path = MERGED_LINES_DIR / line_image_path.name
    overlay_path = OVERLAY_DIR / line_image_path.name
    json_path = JSON_DIR / f"{line_image_path.stem}.json"

    if boot_mask is None:
        blank = np.zeros((height, width), dtype=np.uint8)
        visual = load_visual_image(line_image_path.name, line_image)
        cv2.imwrite(str(valid_fragments_path), visual)
        cv2.imwrite(str(merged_lines_path), visual)
        cv2.imwrite(str(overlay_path), visual)

        save_json(json_path, {
            "source_file": line_image_path.name,
            "processing_step": "06_complete_line_fragments",
            "error": "missing_boot_mask",
            "valid_fragments": [],
            "merged_lines": []
        })

        row = make_empty_metadata_row(
            line_image_path=line_image_path,
            boot_mask_path=None,
            valid_fragments_path=valid_fragments_path,
            merged_lines_path=merged_lines_path,
            overlay_path=overlay_path,
            json_path=json_path,
            width=width,
            height=height,
            green_components_count=0,
            valid_fragments_count=0,
            merged_lines_count=0
        )

        return [row], False

    if boot_mask.shape[:2] != line_image.shape[:2]:
        boot_mask = cv2.resize(
            boot_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    sample_radius = int(STEP_06_CONFIG["mask_validation"]["sample_radius_px"])
    sampling_mask = prepare_sampling_mask(boot_mask, sample_radius)

    visual_image = load_visual_image(line_image_path.name, line_image)

    if visual_image.shape[:2] != line_image.shape[:2]:
        visual_image = cv2.resize(
            visual_image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    fragments, green_mask = extract_fragments_from_green_lines(
        line_image=line_image,
        sampling_mask=sampling_mask
    )

    green_components_count = max(
        0,
        cv2.connectedComponents(green_mask, connectivity=8)[0] - 1
    )

    groups = [group_from_fragment(fragment) for fragment in fragments]
    merged_groups = merge_groups(groups, sampling_mask)

    merged_lines = [
        group_to_merged_line(index, group, sampling_mask)
        for index, group in enumerate(merged_groups, start=1)
    ]

    min_output_length = float(STEP_06_CONFIG["output_filtering"]["min_output_line_length_px"])
    merged_lines = [line for line in merged_lines if line.length >= min_output_length]
    merged_lines.sort(key=lambda line: (line.x1 + line.x2) / 2.0)

    merged_lines = [
        MergedLine(**{**asdict(line), "id": f"m{index:03d}"})
        for index, line in enumerate(merged_lines, start=1)
    ]

    valid_fragments_visual = draw_valid_fragments(visual_image, fragments)
    merged_lines_visual = draw_merged_lines(visual_image, merged_lines)
    overlay = draw_debug_overlay(
        visual_image=visual_image,
        boot_mask=boot_mask,
        fragments=fragments,
        merged_lines=merged_lines
    )

    cv2.imwrite(str(valid_fragments_path), valid_fragments_visual)
    cv2.imwrite(str(merged_lines_path), merged_lines_visual)
    cv2.imwrite(str(overlay_path), overlay)

    save_json(json_path, {
        "source_file": line_image_path.name,
        "processing_step": "06_complete_line_fragments",
        "description": "All vertical step-4 green fragments inside the boot mask are merged when they are compatible. No center line is selected here.",
        "inputs": {
            "line_input_file": relative(line_image_path),
            "boot_mask_file": relative(boot_mask_path),
        },
        "outputs": {
            "valid_fragments_file": relative(valid_fragments_path),
            "merged_lines_file": relative(merged_lines_path),
            "overlay_file": relative(overlay_path),
        },
        "counts": {
            "green_components": green_components_count,
            "valid_fragments_inside_mask": len(fragments),
            "merged_lines": len(merged_lines),
        },
        "valid_fragments": [asdict(fragment) for fragment in fragments],
        "merged_lines": [asdict(line) for line in merged_lines],
    })

    rows = []

    if not merged_lines:
        rows.append(make_empty_metadata_row(
            line_image_path=line_image_path,
            boot_mask_path=boot_mask_path,
            valid_fragments_path=valid_fragments_path,
            merged_lines_path=merged_lines_path,
            overlay_path=overlay_path,
            json_path=json_path,
            width=width,
            height=height,
            green_components_count=green_components_count,
            valid_fragments_count=len(fragments),
            merged_lines_count=0
        ))
    else:
        for merged_line in merged_lines:
            rows.append(make_line_metadata_row(
                line_image_path=line_image_path,
                boot_mask_path=boot_mask_path,
                valid_fragments_path=valid_fragments_path,
                merged_lines_path=merged_lines_path,
                overlay_path=overlay_path,
                json_path=json_path,
                width=width,
                height=height,
                green_components_count=green_components_count,
                valid_fragments_count=len(fragments),
                merged_lines_count=len(merged_lines),
                merged_line=merged_line
            ))

    if DISPLAY_CONFIG["show_windows"]:
        grid = make_grid([
            ("step 4 green lines", line_image),
            ("boot mask", boot_mask),
            ("valid fragments", valid_fragments_visual),
            ("debug overlay", overlay),
            ("merged lines", merged_lines_visual),
        ])

        title = f"06 Complete line fragments | {line_image_path.name}"
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
            return rows, True

    return rows, False


def main() -> None:
    if not STEP_06_CONFIG["enabled"]:
        print("Step 06 is disabled in config.")
        return

    ensure_output_dirs()

    image_paths = collect_images()

    if not image_paths:
        print(f"No step-4 line images found in: {LINES_INPUT_DIR}")
        return

    metadata_rows = []

    print()
    print("Processing step 06: complete vertical line fragments inside boot mask")
    print(f"Input green lines: {LINES_INPUT_DIR}")
    print(f"Input boot masks:  {BOOT_MASK_INPUT_DIR}")
    print(f"Input visual:      {VISUAL_INPUT_DIR}")
    print(f"Output:            {OUTPUT_DIR}")
    print()
    print("This step does not choose the final center/canting line.")
    print("It only merges compatible vertical fragments and keeps fragment_count metadata.")
    print()
    print("Controls:")
    print("  n / SPACE / ENTER  -> next image")
    print("  q / ESC            -> quit")
    print()

    for index, line_image_path in enumerate(image_paths, start=1):
        rows, should_stop = process_image(line_image_path)
        metadata_rows.extend(rows)

        merged_count = rows[0]["merged_lines_count"] if rows else 0
        valid_count = rows[0]["valid_fragments_count"] if rows else 0

        print(
            f"[{index}/{len(image_paths)}] Saved: {line_image_path.name} | "
            f"valid_fragments={valid_count} | merged_lines={merged_count}"
        )

        if should_stop:
            print("Stopped by user.")
            break

    save_metadata(metadata_rows)

    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Step 06 outputs saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
