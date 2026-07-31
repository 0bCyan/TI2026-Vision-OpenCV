from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .circle_fit import fit_circle_ransac
from .config import DetectorConfig
from .fallback import SegmentationFallback
from .io_utils import imread_path
from .types import CircleCandidate, DetectionResult

SOURCE_INTENSITY = 1
SOURCE_BACKGROUND = 2


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


class BallDetector:
    """Geometry-first detector for a single steel ball in a fixed pipe ROI."""

    def __init__(
        self,
        config: DetectorConfig,
        background_path: Path | None = None,
        semantic_fallback: SegmentationFallback | None = None,
    ) -> None:
        self.config = config
        self.semantic_fallback = semantic_fallback
        self.background: np.ndarray | None = None
        self._background_raw: np.ndarray | None = None
        self._background_median: float | None = None
        self.last_candidates: tuple[CircleCandidate, ...] = ()
        if background_path and background_path.exists():
            background = imread_path(background_path, cv2.IMREAD_GRAYSCALE)
            if background is not None:
                self.set_background(background)

    def set_background(self, image: np.ndarray) -> None:
        self._background_raw = self._to_gray(image)
        self.background = self._preprocess(self._background_raw)
        self._background_median = self._fast_median(self.background)

    def clear_background(self) -> None:
        self.background = None
        self._background_raw = None
        self._background_median = None

    @property
    def background_raw(self) -> np.ndarray | None:
        return self._background_raw

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        blur_size = _odd_kernel(self.config.threshold.median_blur_size)
        if blur_size <= 1:
            return gray.copy()
        return cv2.GaussianBlur(gray, (blur_size, blur_size), 0.0)

    @staticmethod
    def _fast_median(gray: np.ndarray) -> float:
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
        return float(np.searchsorted(np.cumsum(histogram), gray.size * 0.5))

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"不支持的图像形状: {image.shape}")

    def _build_masks(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        threshold_config = self.config.threshold
        filtered = self._preprocess(gray)
        median = self._fast_median(filtered)
        _, standard_deviation = cv2.meanStdDev(filtered)
        deviation = float(standard_deviation[0, 0])
        contrast = max(threshold_config.min_contrast_delta, deviation * threshold_config.std_gain)
        if threshold_config.polarity == "dark":
            threshold = float(np.clip(median - contrast, threshold_config.threshold_min, threshold_config.threshold_max))
            intensity_mask = cv2.inRange(filtered, 0, round(threshold))
        else:
            threshold = float(np.clip(median + contrast, threshold_config.threshold_min, threshold_config.threshold_max))
            intensity_mask = cv2.inRange(filtered, round(threshold), 255)

        background_mask = np.zeros_like(intensity_mask)
        background_available = (
            threshold_config.background_enabled
            and self.background is not None
            and self.background.shape == gray.shape
        )
        if background_available:
            background = self.background
            aligned = filtered
            if threshold_config.background_brightness_align:
                background_median = (
                    self._background_median
                    if self._background_median is not None
                    else self._fast_median(background)
                )
                shift = background_median - median
                aligned = np.clip(
                    filtered.astype(np.int16) + round(shift),
                    0,
                    255,
                ).astype(np.uint8)
            difference = cv2.absdiff(aligned, background)
            background_mask = cv2.inRange(
                difference,
                int(threshold_config.background_diff_threshold),
                255,
            )

        mode = threshold_config.combine_mode
        if mode == "background" and background_available:
            combined = background_mask
        elif mode == "and" and background_available:
            combined = cv2.bitwise_and(intensity_mask, background_mask)
        elif mode == "or" and background_available:
            combined = cv2.bitwise_or(intensity_mask, background_mask)
        else:
            combined = intensity_mask

        open_size = _odd_kernel(threshold_config.morph_open_size)
        close_size = _odd_kernel(threshold_config.morph_close_size)
        if open_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        if close_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined, background_mask, threshold

    def _refine_from_edges(self, gray: np.ndarray, initial_fit):
        """Refine the threshold contour on original-intensity radial edges."""
        if not self.config.edge_refine_enabled:
            return initial_fit
        margin = int(np.ceil(initial_fit.radius * 1.55))
        x0 = max(0, int(initial_fit.center_x) - margin)
        y0 = max(0, int(initial_fit.center_y) - margin)
        x1 = min(gray.shape[1], int(initial_fit.center_x) + margin + 1)
        y1 = min(gray.shape[0], int(initial_fit.center_y) + margin + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 36:
            return initial_fit
        blurred = cv2.GaussianBlur(patch, (3, 3), 0.8)
        median = float(np.median(blurred))
        lower = int(np.clip(0.45 * median, 10, 180))
        upper = int(np.clip(1.15 * median, lower + 20, 255))
        edges = cv2.Canny(blurred, lower, upper, L2gradient=True)
        ys, xs = np.nonzero(edges)
        if len(xs) < 12:
            return initial_fit
        global_x = xs.astype(np.float64) + x0
        global_y = ys.astype(np.float64) + y0
        dx = global_x - initial_fit.center_x
        dy = global_y - initial_fit.center_y
        distance = np.hypot(dx, dy)
        annulus = max(2.0, self.config.edge_annulus_ratio * initial_fit.radius)
        radial_mask = np.abs(distance - initial_fit.radius) <= annulus
        if np.count_nonzero(radial_mask) < 12:
            return initial_fit
        gradient_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)[ys, xs]
        gradient_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)[ys, xs]
        magnitude = np.hypot(gradient_x, gradient_y)
        alignment = np.abs(gradient_x * dx + gradient_y * dy) / np.maximum(
            1e-6, magnitude * distance
        )
        selected = (
            radial_mask
            & (magnitude >= self.config.edge_min_gradient)
            & (alignment >= self.config.edge_radial_alignment_min)
        )
        if np.count_nonzero(selected) < 12:
            return initial_fit
        edge_points = np.column_stack((global_x[selected], global_y[selected]))
        refined = fit_circle_ransac(
            edge_points,
            radius_min=self.config.radius_min_px,
            radius_max=self.config.radius_max_px,
            iterations=self.config.edge_ransac_iterations,
            residual_threshold=self.config.edge_ransac_residual_px,
            sample_limit=self.config.contour_sample_limit,
        )
        if refined is None:
            return initial_fit
        center_shift = np.hypot(
            refined.center_x - initial_fit.center_x,
            refined.center_y - initial_fit.center_y,
        )
        if center_shift > self.config.edge_center_shift_max_ratio * initial_fit.radius:
            return initial_fit
        if refined.arc_coverage < self.config.arc_coverage_min:
            return initial_fit
        return refined

    def detect(
        self,
        roi_image: np.ndarray,
        predicted_center_px: float | None = None,
    ) -> DetectionResult:
        gray = self._to_gray(roi_image)
        mask, background_mask, threshold = self._build_masks(gray)
        if self.semantic_fallback is not None:
            fallback = self.semantic_fallback.infer(roi_image)
            if fallback is not None and fallback.mask.shape == mask.shape:
                fallback_mask = np.where(fallback.mask > 0, 255, 0).astype(np.uint8)
                mask = cv2.bitwise_or(mask, fallback_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        height, width = gray.shape
        channel_center_y = (
            self.config.channel_center_y_px
            if self.config.channel_center_y_px >= 0
            else height * 0.5
        )
        max_y_error = (
            self.config.max_center_y_error_px
            if self.config.max_center_y_error_px > 0
            else height * 0.45
        )
        candidates: list[CircleCandidate] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.area_min_px or area > self.config.area_max_px:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue
            if (
                x <= self.config.border_margin_px
                or y <= self.config.border_margin_px
                or x + box_width >= width - self.config.border_margin_px
                or y + box_height >= height - self.config.border_margin_px
            ):
                continue
            aspect = float(box_width) / float(box_height)
            if not self.config.aspect_ratio_min <= aspect <= self.config.aspect_ratio_max:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.config.circularity_min:
                continue
            fill_ratio = float(area / (box_width * box_height))
            if fill_ratio < self.config.fill_ratio_min:
                continue

            fit = fit_circle_ransac(
                contour,
                radius_min=self.config.radius_min_px,
                radius_max=self.config.radius_max_px,
                iterations=self.config.ransac_iterations,
                residual_threshold=self.config.ransac_residual_px,
                sample_limit=self.config.contour_sample_limit,
            )
            if fit is None:
                continue
            fit = self._refine_from_edges(gray, fit)
            if fit.rms > self.config.fit_rms_max_px or fit.inlier_ratio < self.config.fit_min_inlier_ratio:
                continue
            if fit.arc_coverage < self.config.arc_coverage_min:
                continue
            radius_error_ratio = abs(fit.radius - self.config.expected_radius_px) / max(
                1e-6, self.config.expected_radius_px
            )
            radius_tolerance = (
                self.config.radius_tolerance_track_ratio
                if predicted_center_px is not None
                else self.config.radius_tolerance_search_ratio
            )
            if radius_error_ratio > radius_tolerance:
                continue
            y_error = abs(fit.center_y - channel_center_y)
            if y_error > max_y_error:
                continue
            prediction_distance = 0.0
            if predicted_center_px is not None:
                prediction_distance = abs(fit.center_x - predicted_center_px)
                if prediction_distance > self.config.max_prediction_distance_px:
                    continue

            source_mask = SOURCE_INTENSITY
            patch = background_mask[y : y + box_height, x : x + box_width]
            if patch.size and np.count_nonzero(patch) > 0.15 * area:
                source_mask |= SOURCE_BACKGROUND

            circularity_quality = np.clip(
                (circularity - self.config.circularity_min)
                / max(1e-6, 1.0 - self.config.circularity_min),
                0.0,
                1.0,
            )
            fill_quality = np.exp(-((fill_ratio - np.pi / 4.0) / 0.30) ** 2)
            fit_quality = np.exp(-((fit.rms / max(0.5, self.config.fit_rms_max_px)) ** 2))
            inlier_quality = np.clip(
                (fit.inlier_ratio - self.config.fit_min_inlier_ratio)
                / max(1e-6, 1.0 - self.config.fit_min_inlier_ratio),
                0.0,
                1.0,
            )
            coverage_quality = np.clip(
                (fit.arc_coverage - self.config.arc_coverage_min)
                / max(1e-6, 1.0 - self.config.arc_coverage_min),
                0.0,
                1.0,
            )
            size_scale = max(3.0, self.config.expected_radius_px * 0.65)
            size_quality = np.exp(-abs(fit.radius - self.config.expected_radius_px) / size_scale)
            y_quality = np.exp(-((y_error / max(1.0, max_y_error)) ** 2))
            if predicted_center_px is None:
                prediction_quality = 0.75
            else:
                prediction_quality = np.exp(
                    -((prediction_distance / max(1.0, self.config.max_prediction_distance_px * 0.45)) ** 2)
                )
            confidence = float(
                0.15 * circularity_quality
                + 0.10 * fill_quality
                + 0.22 * fit_quality
                + 0.13 * inlier_quality
                + 0.10 * coverage_quality
                + 0.12 * size_quality
                + 0.08 * y_quality
                + 0.10 * prediction_quality
            )
            score = float(
                (1.0 - confidence) * 100.0
                + prediction_distance * (0.08 if predicted_center_px is not None else 0.0)
            )
            candidates.append(
                CircleCandidate(
                    center_px=(fit.center_x, fit.center_y),
                    radius_px=fit.radius,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    score=score,
                    circularity=circularity,
                    fill_ratio=fill_ratio,
                    fit_rms_px=fit.rms,
                    inlier_ratio=fit.inlier_ratio,
                    arc_coverage=fit.arc_coverage,
                    area_px=area,
                    bbox=(x, y, box_width, box_height),
                    source_mask=source_mask,
                )
            )

        candidates.sort(key=lambda item: item.score)
        self.last_candidates = tuple(candidates)
        best = candidates[0] if candidates else None
        if best is not None and len(candidates) > 1:
            score_gap = candidates[1].score - best.score
            if score_gap < self.config.ambiguity_score_gap:
                best.confidence *= self.config.ambiguity_confidence_scale
        confidence_minimum = (
            self.config.track_confidence_min
            if predicted_center_px is not None
            else self.config.search_confidence_min
        )
        accepted = best is not None and best.confidence >= confidence_minimum
        if best is None:
            reason = "no_geometrically_valid_candidate"
        elif not accepted:
            reason = "low_or_ambiguous_confidence"
        else:
            reason = "ok"
        return DetectionResult(
            valid=accepted,
            candidate=best,
            threshold_value=threshold,
            candidate_count=len(candidates),
            mask=mask if self.config.return_debug_mask else None,
            reason=reason,
            candidates=tuple(candidates),
        )
