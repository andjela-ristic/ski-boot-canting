from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Iterable
import base64
import math

import cv2
import numpy as np

from .config import BootConfig, GuideConfig, ReadinessConfig, ReferenceConfig
from .models import Candidate, ValidationResult


class FrameValidationError(ValueError):
    pass


class FrameValidator:
    """
    Stateless, low-latency validator for a single camera preview frame.

    The validator intentionally does not identify boot make/model and does not
    measure canting. It checks whether the scene geometry is suitable for the
    slower measurement pipeline.
    """

    def __init__(self, config: ReadinessConfig | None = None) -> None:
        self.config = config or ReadinessConfig()

        cv2.setUseOptimized(True)
        if self.config.opencv_threads >= 0:
            cv2.setNumThreads(self.config.opencv_threads)

        boot = self.config.boot
        self._boot_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (boot.close_kernel, boot.close_kernel)
        )
        self._boot_open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (boot.open_kernel, boot.open_kernel)
        )
        self._edge_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )

    def validate_bytes(
        self,
        encoded_frame: bytes,
        *,
        include_debug: bool = False,
        guide_scale: float = 1.0,
    ) -> ValidationResult:
        if not encoded_frame:
            raise FrameValidationError("The uploaded frame is empty.")
        if len(encoded_frame) > self.config.jpeg_max_bytes:
            raise FrameValidationError(
                f"Frame is too large ({len(encoded_frame)} bytes); "
                f"maximum is {self.config.jpeg_max_bytes} bytes."
            )

        encoded = np.frombuffer(encoded_frame, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise FrameValidationError("The uploaded data is not a decodable image.")

        return self.validate(
            frame,
            include_debug=include_debug,
            guide_scale=guide_scale,
        )

    def validate(
        self,
        frame: np.ndarray,
        *,
        include_debug: bool = False,
        guide_scale: float = 1.0,
    ) -> ValidationResult:
        started = perf_counter()

        if frame is None or frame.size == 0:
            raise FrameValidationError("Frame is empty.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise FrameValidationError("Frame must be a BGR color image.")

        source_height, source_width = frame.shape[:2]
        processed = self._resize_for_processing(frame)
        height, width = processed.shape[:2]
        active_config, normalized_guide_scale = self._config_for_guide_scale(
            guide_scale
        )

        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        quality_checks, quality_metrics = self._quality_checks(gray)

        guide_rect = self._ratio_rect(
            width,
            height,
            active_config.guide.x_min_ratio,
            active_config.guide.x_max_ratio,
            active_config.guide.y_min_ratio,
            active_config.guide.y_max_ratio,
        )
        gx1, gy1, gx2, gy2 = guide_rect
        guide_gray = gray[gy1:gy2, gx1:gx2]

        candidate = self._find_boot_candidate(guide_gray, active_config)
        reference_found, reference_metrics, reference_segments = (
            self._detect_reference_line(gray, active_config)
        )

        boot_checks, boot_metrics = self._boot_checks(
            candidate,
            guide_gray.shape,
            active_config,
        )
        checks = {
            **quality_checks,
            **boot_checks,
            "reference_line_found": (
                reference_found or not active_config.reference.required
            ),
        }

        score_parts = [
            1.0 if quality_checks["sharpness_ok"] else 0.0,
            1.0 if quality_checks["exposure_ok"] else 0.0,
            candidate.score if candidate is not None else 0.0,
            1.0 if checks["reference_line_found"] else 0.0,
        ]
        score = float(np.clip(np.mean(score_parts), 0.0, 1.0))
        success = (
            score >= active_config.success_score_threshold
            and checks["boot_scale_ok"]
        )
        reason = None if success else self._failure_reason(
            checks,
            candidate,
            active_config.boot,
        )

        latency_ms = (perf_counter() - started) * 1000.0
        debug_image_base64 = None
        if include_debug:
            debug_image_base64 = self._render_debug(
                processed,
                guide_rect,
                candidate,
                reference_segments,
                checks,
                success,
            )

        metrics = {
            **quality_metrics,
            **boot_metrics,
            **reference_metrics,
            "guide_scale": round(normalized_guide_scale, 3),
            "guide_x_min_ratio": round(active_config.guide.x_min_ratio, 5),
            "guide_x_max_ratio": round(active_config.guide.x_max_ratio, 5),
            "guide_y_min_ratio": round(active_config.guide.y_min_ratio, 5),
            "guide_y_max_ratio": round(active_config.guide.y_max_ratio, 5),
            "processing_width": width,
            "processing_height": height,
        }

        return ValidationResult(
            success=success,
            score=round(score, 4),
            reason=reason,
            checks=checks,
            metrics=metrics,
            latency_ms=round(latency_ms, 3),
            source_shape=(source_height, source_width),
            processed_shape=(height, width),
            debug_image_base64=debug_image_base64,
        )

    def _config_for_guide_scale(
        self,
        guide_scale: float,
    ) -> tuple[ReadinessConfig, float]:
        normalized = self._normalize_guide_scale(guide_scale)
        if abs(normalized - 1.0) < 1e-6:
            return self.config, normalized

        return (
            replace(
                self.config,
                guide=self._scaled_guide_config(self.config.guide, normalized),
                boot=self._scaled_boot_config(self.config.boot, normalized),
            ),
            normalized,
        )

    @staticmethod
    def _normalize_guide_scale(raw_value: float | None) -> float:
        if raw_value is None:
            return 1.0

        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return 1.0

        return float(np.clip(numeric, 0.75, 1.20))

    @staticmethod
    def _scaled_guide_config(
        guide: GuideConfig,
        guide_scale: float,
    ) -> GuideConfig:
        center_x = (guide.x_min_ratio + guide.x_max_ratio) * 0.5
        center_y = (guide.y_min_ratio + guide.y_max_ratio) * 0.5

        base_width = guide.x_max_ratio - guide.x_min_ratio
        base_height = guide.y_max_ratio - guide.y_min_ratio

        max_width = min(center_x * 2.0, (1.0 - center_x) * 2.0, 0.98)
        max_height = min(center_y * 2.0, (1.0 - center_y) * 2.0, 0.98)

        width = float(np.clip(base_width * guide_scale, 0.18, max_width))
        height = float(np.clip(base_height * guide_scale, 0.24, max_height))

        return replace(
            guide,
            x_min_ratio=center_x - width * 0.5,
            x_max_ratio=center_x + width * 0.5,
            y_min_ratio=center_y - height * 0.5,
            y_max_ratio=center_y + height * 0.5,
        )

    @staticmethod
    def _scaled_boot_config(
        boot: BootConfig,
        guide_scale: float,
    ) -> BootConfig:
        linear_scale = 1.0 / guide_scale
        area_scale = linear_scale * linear_scale

        min_height_ratio = float(np.clip(boot.min_height_ratio * linear_scale, 0.14, 0.97))
        max_height_ratio = float(np.clip(boot.max_height_ratio * linear_scale, min_height_ratio, 0.999))
        min_width_ratio = float(np.clip(boot.min_width_ratio * linear_scale, 0.08, 0.94))
        max_width_ratio = float(np.clip(boot.max_width_ratio * linear_scale, min_width_ratio, 0.999))
        min_area_ratio = float(np.clip(boot.min_area_ratio * area_scale, 0.01, 0.95))
        max_area_ratio = float(np.clip(boot.max_area_ratio * area_scale, min_area_ratio, 0.99))
        min_bottom_ratio = float(
            np.clip(
                0.5 + (boot.min_bottom_ratio - 0.5) * linear_scale,
                0.5,
                0.98,
            )
        )

        return replace(
            boot,
            min_height_ratio=min_height_ratio,
            max_height_ratio=max_height_ratio,
            min_width_ratio=min_width_ratio,
            max_width_ratio=max_width_ratio,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            min_bottom_ratio=min_bottom_ratio,
            min_side_margin_ratio=float(
                np.clip(boot.min_side_margin_ratio * linear_scale, 0.0, 0.25)
            ),
            min_top_margin_ratio=float(
                np.clip(boot.min_top_margin_ratio * linear_scale, 0.0, 0.25)
            ),
        )

    def _resize_for_processing(self, frame: np.ndarray) -> np.ndarray:
        max_width = self.config.processing_max_width
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame
        scale = max_width / float(width)
        target_height = max(1, int(round(height * scale)))
        return cv2.resize(
            frame,
            (max_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def _quality_checks(
        self, gray: np.ndarray
    ) -> tuple[dict[str, bool], dict[str, float]]:
        quality = self.config.quality

        # Evaluate the central 80% to reduce influence from UI borders.
        height, width = gray.shape
        x1, x2 = int(width * 0.10), int(width * 0.90)
        y1, y2 = int(height * 0.10), int(height * 0.90)
        sample = gray[y1:y2, x1:x2]

        mean_brightness = float(sample.mean())
        dark_ratio = float(np.mean(sample <= quality.dark_pixel_threshold))
        bright_ratio = float(np.mean(sample >= quality.bright_pixel_threshold))

        # CV_32F is sufficient and cheaper than CV_64F for this check.
        sharpness = float(cv2.Laplacian(sample, cv2.CV_32F).var())

        sharpness_ok = sharpness >= quality.min_sharpness
        exposure_ok = (
            quality.min_mean_brightness
            <= mean_brightness
            <= quality.max_mean_brightness
            and dark_ratio <= quality.max_dark_ratio
            and bright_ratio <= quality.max_bright_ratio
        )

        return (
            {
                "sharpness_ok": sharpness_ok,
                "exposure_ok": exposure_ok,
            },
            {
                "sharpness": round(sharpness, 3),
                "mean_brightness": round(mean_brightness, 3),
                "dark_ratio": round(dark_ratio, 5),
                "bright_ratio": round(bright_ratio, 5),
            },
        )

    def _find_boot_candidate(
        self,
        guide_gray: np.ndarray,
        config: ReadinessConfig,
    ) -> Candidate | None:
        boot = config.boot
        if guide_gray.size == 0:
            return None

        blurred = cv2.GaussianBlur(
            guide_gray,
            (boot.gaussian_kernel, boot.gaussian_kernel),
            0,
        )

        otsu_threshold, dark_mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        light_mask = cv2.bitwise_not(dark_mask)

        edges = cv2.Canny(
            blurred,
            boot.canny_low,
            boot.canny_high,
            apertureSize=3,
            L2gradient=False,
        )
        edge_mask = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            self._boot_close_kernel,
            iterations=1,
        )
        edge_mask = cv2.dilate(
            edge_mask,
            self._edge_dilate_kernel,
            iterations=1,
        )

        best: Candidate | None = None
        for source, raw_mask in (
            ("otsu_dark", dark_mask),
            ("otsu_light", light_mask),
            ("edges", edge_mask),
        ):
            if source != "edges":
                mask = cv2.morphologyEx(
                    raw_mask,
                    cv2.MORPH_CLOSE,
                    self._boot_close_kernel,
                    iterations=1,
                )
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    self._boot_open_kernel,
                    iterations=1,
                )
            else:
                mask = raw_mask

            mask = self._remove_reference_like_lines(mask, config.reference)
            candidate = self._best_component(mask, source, boot)
            if candidate is not None and (
                best is None or candidate.score > best.score
            ):
                best = candidate

        return best


    def _remove_reference_like_lines(
        self,
        mask: np.ndarray,
        reference: ReferenceConfig,
    ) -> np.ndarray:
        """Remove very long lines that would otherwise join boot and background."""
        height, width = mask.shape
        orientation = reference.orientation

        if orientation == "horizontal":
            kernel_width = max(15, int(round(width * 0.28)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        elif orientation == "vertical":
            kernel_height = max(15, int(round(height * 0.28)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        else:
            return mask

        long_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.subtract(mask, long_lines)

    def _best_component(
        self,
        mask: np.ndarray,
        source: str,
        boot: BootConfig,
    ) -> Candidate | None:
        guide_height, guide_width = mask.shape
        guide_area = guide_height * guide_width
        if guide_area == 0:
            return None

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
            ltype=cv2.CV_32S,
        )

        best: Candidate | None = None
        expected_center_x = guide_width * 0.5

        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if width <= 0 or height <= 0:
                continue

            height_ratio = height / guide_height
            width_ratio = width / guide_width
            area_ratio = area / guide_area
            bbox_fill = area / float(width * height)
            center_x = x + width * 0.5
            center_offset_ratio = (
                center_x - expected_center_x
            ) / guide_width
            bottom_ratio = (y + height) / guide_height

            # Very small pieces are noise. A full-frame component is background.
            source_min_area = 0.012 if source == "edges" else 0.035
            if area_ratio < source_min_area or area_ratio > 0.90:
                continue
            if height_ratio < 0.28 or width_ratio < 0.10:
                continue

            center_score = max(
                0.0,
                1.0 - abs(center_offset_ratio) / 0.40,
            )
            height_score = self._range_score(
                height_ratio,
                boot.min_height_ratio,
                boot.max_height_ratio,
            )
            width_score = self._range_score(
                width_ratio,
                boot.min_width_ratio,
                boot.max_width_ratio,
            )
            area_score = self._range_score(
                area_ratio,
                boot.min_area_ratio,
                boot.max_area_ratio,
            )
            bottom_score = float(
                np.clip(
                    (bottom_ratio - 0.40) / 0.45,
                    0.0,
                    1.0,
                )
            )

            # Filled threshold components should have meaningful occupancy.
            # Edge components are naturally sparse and use a softer term.
            if source == "edges":
                fill_score = float(np.clip(bbox_fill / 0.16, 0.0, 1.0))
            else:
                fill_score = float(np.clip(bbox_fill / 0.55, 0.0, 1.0))

            score = (
                0.28 * center_score
                + 0.24 * height_score
                + 0.13 * width_score
                + 0.14 * area_score
                + 0.12 * bottom_score
                + 0.09 * fill_score
            )

            touches_top = y <= 1
            touches_left = x <= 1
            touches_right = (x + width) >= (guide_width - 1)

            # A component touching two or more guide borders is almost always
            # the thresholded background, not the boot.
            border_touches = sum((touches_top, touches_left, touches_right))
            if border_touches >= 2:
                continue
            if border_touches == 1:
                score *= 0.82

            candidate = Candidate(
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                score=float(np.clip(score, 0.0, 1.0)),
                source=source,
                center_offset_ratio=float(center_offset_ratio),
                height_ratio=float(height_ratio),
                width_ratio=float(width_ratio),
                area_ratio=float(area_ratio),
                bottom_ratio=float(bottom_ratio),
                touches_top=touches_top,
                touches_left=touches_left,
                touches_right=touches_right,
            )

            if best is None or candidate.score > best.score:
                best = candidate

        return best

    def _boot_checks(
        self,
        candidate: Candidate | None,
        guide_shape: tuple[int, int],
        config: ReadinessConfig,
    ) -> tuple[dict[str, bool], dict[str, float | str | None]]:
        boot = config.boot

        if candidate is None:
            return (
                {
                    "boot_present": False,
                    "boot_centered": False,
                    "boot_scale_ok": False,
                    "boot_complete": False,
                },
                {
                    "candidate_source": None,
                    "candidate_score": 0.0,
                    "center_offset_ratio": None,
                    "height_ratio": None,
                    "width_ratio": None,
                    "area_ratio": None,
                    "bottom_ratio": None,
                },
            )

        guide_height, guide_width = guide_shape
        top_margin_ratio = candidate.y / guide_height
        left_margin_ratio = candidate.x / guide_width
        right_margin_ratio = (
            guide_width - candidate.x - candidate.width
        ) / guide_width

        boot_present = (
            candidate.score >= boot.min_candidate_score
            and candidate.area_ratio >= (
                0.012 if candidate.source == "edges" else boot.min_area_ratio
            )
        )
        boot_centered = (
            abs(candidate.center_offset_ratio)
            <= boot.max_center_offset_ratio
        )
        boot_scale_ok = (
            boot.min_height_ratio
            <= candidate.height_ratio
            <= boot.max_height_ratio
            and boot.min_width_ratio
            <= candidate.width_ratio
            <= boot.max_width_ratio
            and candidate.area_ratio <= boot.max_area_ratio
        )
        boot_complete = (
            top_margin_ratio >= boot.min_top_margin_ratio
            and left_margin_ratio >= boot.min_side_margin_ratio
            and right_margin_ratio >= boot.min_side_margin_ratio
            and candidate.bottom_ratio >= boot.min_bottom_ratio
        )

        return (
            {
                "boot_present": boot_present,
                "boot_centered": boot_centered,
                "boot_scale_ok": boot_scale_ok,
                "boot_complete": boot_complete,
            },
            {
                "candidate_source": candidate.source,
                "candidate_score": round(candidate.score, 4),
                "center_offset_ratio": round(
                    candidate.center_offset_ratio, 5
                ),
                "height_ratio": round(candidate.height_ratio, 5),
                "width_ratio": round(candidate.width_ratio, 5),
                "area_ratio": round(candidate.area_ratio, 5),
                "bottom_ratio": round(candidate.bottom_ratio, 5),
                "top_margin_ratio": round(top_margin_ratio, 5),
                "left_margin_ratio": round(left_margin_ratio, 5),
                "right_margin_ratio": round(right_margin_ratio, 5),
            },
        )

    def _detect_reference_line(
        self,
        gray: np.ndarray,
        config: ReadinessConfig,
    ) -> tuple[bool, dict[str, float | int | str], list[tuple[int, int, int, int]]]:
        reference = config.reference
        height, width = gray.shape

        x1, y1, x2, y2 = self._ratio_rect(
            width,
            height,
            reference.x_min_ratio,
            reference.x_max_ratio,
            reference.y_min_ratio,
            reference.y_max_ratio,
        )
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return (
                False,
                {
                    "reference_orientation": reference.orientation,
                    "reference_segment_count": 0,
                    "reference_total_length_ratio": 0.0,
                    "reference_best_angle_deg": None,
                },
                [],
            )

        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(
            blurred,
            reference.canny_low,
            reference.canny_high,
            apertureSize=3,
            L2gradient=False,
        )

        # The boot itself contains many long straight edges. Reference evidence
        # is therefore collected outside the frontend guide where the physical
        # table/platform line should remain visible on one or both sides.
        guide_x1 = int(round(width * config.guide.x_min_ratio)) - x1
        guide_x2 = int(round(width * config.guide.x_max_ratio)) - x1
        guide_x1 = int(np.clip(guide_x1, 0, edges.shape[1]))
        guide_x2 = int(np.clip(guide_x2, 0, edges.shape[1]))
        if reference.exclude_guide_from_search and guide_x2 > guide_x1:
            edges[:, guide_x1:guide_x2] = 0

        roi_height, roi_width = roi.shape
        basis = roi_width if reference.orientation == "horizontal" else roi_height
        min_line_length = max(
            8,
            int(round(basis * reference.min_segment_length_ratio)),
        )
        max_line_gap = max(
            2,
            int(round(basis * reference.max_line_gap_ratio)),
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=reference.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        accepted: list[tuple[int, int, int, int]] = []
        total_length = 0.0
        best_length = 0.0
        best_angle: float | None = None

        if lines is not None:
            for raw in lines[:, 0, :]:
                lx1, ly1, lx2, ly2 = map(int, raw)
                dx = lx2 - lx1
                dy = ly2 - ly1
                length = math.hypot(dx, dy)
                if length <= 0:
                    continue

                angle = math.degrees(math.atan2(dy, dx))
                normalized = abs(angle) % 180.0

                if reference.orientation == "horizontal":
                    error = min(normalized, abs(180.0 - normalized))
                elif reference.orientation == "vertical":
                    error = abs(90.0 - normalized)
                else:
                    raise FrameValidationError(
                        "reference.orientation must be 'horizontal' or 'vertical'."
                    )

                if error > reference.max_angle_error_deg:
                    continue

                accepted.append(
                    (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1)
                )
                total_length += length

                if length > best_length:
                    best_length = length
                    best_angle = angle

        total_length_ratio = total_length / max(1.0, basis)
        found = total_length_ratio >= reference.min_total_length_ratio

        return (
            found,
            {
                "reference_orientation": reference.orientation,
                "reference_segment_count": len(accepted),
                "reference_total_length_ratio": round(
                    total_length_ratio, 5
                ),
                "reference_best_angle_deg": (
                    round(best_angle, 3)
                    if best_angle is not None
                    else None
                ),
            },
            accepted,
        )

    def _render_debug(
        self,
        frame: np.ndarray,
        guide_rect: tuple[int, int, int, int],
        candidate: Candidate | None,
        reference_segments: Iterable[tuple[int, int, int, int]],
        checks: dict[str, bool],
        success: bool,
    ) -> str:
        debug = frame.copy()
        gx1, gy1, gx2, gy2 = guide_rect

        guide_color = (0, 220, 0) if success else (0, 0, 255)
        cv2.rectangle(
            debug,
            (gx1, gy1),
            (gx2 - 1, gy2 - 1),
            guide_color,
            2,
        )

        if candidate is not None:
            x1 = gx1 + candidate.x
            y1 = gy1 + candidate.y
            x2 = x1 + candidate.width
            y2 = y1 + candidate.height
            cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(
                debug,
                f"{candidate.source} {candidate.score:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

        for x1, y1, x2, y2 in reference_segments:
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

        status = "READY" if success else "NOT READY"
        cv2.putText(
            debug,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            guide_color,
            2,
            cv2.LINE_AA,
        )

        missing = [name for name, passed in checks.items() if not passed]
        if missing:
            cv2.putText(
                debug,
                ", ".join(missing[:3]),
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                guide_color,
                1,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            debug,
            [cv2.IMWRITE_JPEG_QUALITY, 72],
        )
        if not ok:
            return ""
        return base64.b64encode(encoded).decode("ascii")

    @staticmethod
    def _failure_reason(
        checks: dict[str, bool],
        candidate: Candidate | None,
        boot: BootConfig,
    ) -> str | None:
        if not checks["sharpness_ok"]:
            return "frame_blurry"
        if not checks["exposure_ok"]:
            return "invalid_exposure"
        if not checks["reference_line_found"]:
            return "reference_line_missing"
        if not checks["boot_present"]:
            return "boot_not_detected"
        if not checks["boot_centered"]:
            if candidate is not None and candidate.center_offset_ratio < 0:
                return "move_boot_right"
            return "move_boot_left"
        if not checks["boot_scale_ok"]:
            if (
                candidate is not None
                and candidate.height_ratio < boot.min_height_ratio
            ):
                return "move_boot_closer"
            return "move_boot_farther"
        if not checks["boot_complete"]:
            return "boot_cropped_or_too_high"
        return None

    @staticmethod
    def _ratio_rect(
        width: int,
        height: int,
        x_min_ratio: float,
        x_max_ratio: float,
        y_min_ratio: float,
        y_max_ratio: float,
    ) -> tuple[int, int, int, int]:
        x1 = int(np.clip(round(width * x_min_ratio), 0, width - 1))
        x2 = int(np.clip(round(width * x_max_ratio), x1 + 1, width))
        y1 = int(np.clip(round(height * y_min_ratio), 0, height - 1))
        y2 = int(np.clip(round(height * y_max_ratio), y1 + 1, height))
        return x1, y1, x2, y2

    @staticmethod
    def _range_score(value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 1.0

        width = max(high - low, 1e-6)
        if value < low:
            return float(np.clip(1.0 - (low - value) / width, 0.0, 1.0))
        return float(np.clip(1.0 - (value - high) / width, 0.0, 1.0))
