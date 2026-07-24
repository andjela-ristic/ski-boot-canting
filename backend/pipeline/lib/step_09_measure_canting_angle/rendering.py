from __future__ import annotations

import cv2
import numpy as np

from . import context
from .geometry import axis_x


def load_background(image_name: str, cleaned_edge: np.ndarray) -> tuple[np.ndarray, str]:
    working = context.WORKING_PNG_DIR / image_name
    image = context.load_color(working)
    if image is not None:
        return image, context.relative_project_path(working)
    return cv2.cvtColor(cleaned_edge, cv2.COLOR_GRAY2BGR), context.relative_project_path(context.get_step_dirs()["cleaned_edge_dir"] / image_name)


def _draw_text_block(image: np.ndarray, lines: list[str], x: int = 24, y: int = 34) -> None:
    font_scale = float(context.STEP_CONFIG.get("drawing", {}).get("font_scale", 0.62))
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = max(24, int(round(34 * font_scale / 0.62)))
    widths = [cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in lines]
    box_width = max(widths, default=100) + 26
    box_height = line_height * len(lines) + 18
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 12, y - 26), (x - 12 + box_width, y - 26 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (x, y + index * line_height), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(
    background: np.ndarray,
    edge: np.ndarray,
    exclusion_mask: np.ndarray,
    axis_candidate: dict,
    y_min: int,
    y_max: int,
    table_result: dict,
    canting: dict | None,
    axis_quality: dict,
    confidence: dict | None,
) -> np.ndarray:
    image = background.copy()
    height, width = image.shape[:2]
    mask_boundary = cv2.morphologyEx((exclusion_mask > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    image[mask_boundary > 0] = (180, 80, 180)

    x_top = int(round(axis_x(axis_candidate, y_min)))
    x_bottom = int(round(axis_x(axis_candidate, y_max)))
    cv2.line(
        image,
        (x_top, int(y_min)),
        (x_bottom, int(y_max)),
        (0, 255, 0),
        int(context.STEP_CONFIG.get("drawing", {}).get("axis_thickness", 4)),
        cv2.LINE_AA,
    )

    table_line = table_result.get("winner")
    if table_line is not None:
        slope = float(table_line["slope"])
        intercept = float(table_line["intercept"])
        y_left = int(round(intercept))
        y_right = int(round(slope * (width - 1) + intercept))
        cv2.line(
            image,
            (0, y_left),
            (width - 1, y_right),
            (255, 180, 0),
            int(context.STEP_CONFIG.get("drawing", {}).get("table_line_thickness", 4)),
            cv2.LINE_AA,
        )
        points = table_line.get("support_points")
        if isinstance(points, np.ndarray) and points.size:
            step = max(1, int(points.shape[0] / 1000))
            for x, y in points[::step]:
                if 0 <= int(x) < width and 0 <= int(y) < height:
                    image[int(y), int(x)] = (0, 255, 255)

        axis_y = int(round(float(table_line["y_at_axis"])))
        axis_center_x = int(round(axis_x(axis_candidate, axis_y)))
        normal_length = max(180, int(round((y_max - y_min) * 0.18)))
        normal = np.asarray(canting["table_up_normal_vector"] if canting else [slope, -1.0], dtype=np.float64)
        normal /= max(1e-9, float(np.linalg.norm(normal)))
        normal_end = (
            int(round(axis_center_x + normal[0] * normal_length)),
            int(round(axis_y + normal[1] * normal_length)),
        )
        cv2.arrowedLine(
            image,
            (axis_center_x, axis_y),
            normal_end,
            (255, 0, 255),
            int(context.STEP_CONFIG.get("drawing", {}).get("normal_thickness", 2)),
            cv2.LINE_AA,
            tipLength=0.08,
        )

    if canting and confidence:
        lines = [
            f"Canting: {canting['canting_angle_deg']:+.3f} deg ({canting['canting_direction']})",
            f"Axis-table angle: {canting['axis_table_angle_deg']:.3f} deg",
            f"Axis quality: {axis_quality['axis_quality_percent']:.1f}%",
            f"Table quality: {table_line.get('table_line_quality_percent', table_line.get('score_percent', 0.0)):.1f}%",
            f"Measurement confidence: {confidence['measurement_confidence_percent']:.1f}%",
            f"Decision: {confidence['decision']}",
        ]
    else:
        lines = [
            f"Axis quality: {axis_quality['axis_quality_percent']:.1f}%",
            "Reference table line not reliable",
        ]
    _draw_text_block(image, lines)
    return image


def draw_diagnostic(edge: np.ndarray, exclusion_mask: np.ndarray, table_result: dict) -> np.ndarray:
    image = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    image[exclusion_mask > 0] = (80, 0, 80)
    table_line = table_result.get("winner")
    if table_line is not None:
        for segment in table_line.get("segments", []):
            cv2.line(
                image,
                (int(round(segment["x1"])), int(round(segment["y1"]))),
                (int(round(segment["x2"])), int(round(segment["y2"]))),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        slope = float(table_line["slope"])
        intercept = float(table_line["intercept"])
        cv2.line(image, (0, int(round(intercept))), (image.shape[1] - 1, int(round(slope * (image.shape[1] - 1) + intercept))), (255, 0, 0), 3, cv2.LINE_AA)
    return image


def create_comparison(overlay: np.ndarray, diagnostic: np.ndarray, metadata: dict) -> np.ndarray:
    target_height = max(overlay.shape[0], diagnostic.shape[0])
    def resize_to_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        scale = target_height / image.shape[0]
        return cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), target_height))
    left = resize_to_height(overlay)
    right = resize_to_height(diagnostic)
    return np.hstack([left, right])
