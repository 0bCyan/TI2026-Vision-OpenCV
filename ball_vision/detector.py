from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .circle_fit import CircleFit, fit_circle_ransac
from .config import DetectorConfig
from .fallback import SegmentationFallback
from .io_utils import imread_path
from .types import CircleCandidate, DetectionResult

SOURCE_INTENSITY = 1
SOURCE_BACKGROUND = 2


@dataclass(slots=True)
class _EdgeRefinement:
    fit: CircleFit
    attempted: bool = False
    accepted: bool = False
    support_ratio: float = 0.0
    center_shift_px: float = 0.0
    radius_shift_px: float = 0.0


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
        self.last_search_rect: tuple[int, int, int, int] | None = None
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

    def _build_masks(
        self,
        gray: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
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

        if threshold_config.local_contrast_enabled:
            local_kernel_size = _odd_kernel(
                threshold_config.local_contrast_kernel_size
            )
            local_background = cv2.boxFilter(
                filtered,
                -1,
                (local_kernel_size, local_kernel_size),
                normalize=True,
                borderType=cv2.BORDER_REPLICATE,
            )
            if threshold_config.polarity == "dark":
                local_difference = cv2.subtract(local_background, filtered)
            else:
                local_difference = cv2.subtract(filtered, local_background)
            local_mask = cv2.inRange(
                local_difference,
                int(threshold_config.local_contrast_threshold),
                255,
            )
            # Global intensity keeps the target physically dark/bright, while
            # local contrast separates it from broad pipe shadows and seams.
            intensity_mask = cv2.bitwise_and(intensity_mask, local_mask)

        background_mask = np.zeros_like(intensity_mask)
        background = self.background
        background_available = (
            threshold_config.background_enabled
            and background is not None
            and background.shape == gray.shape
        )
        if background_available:
            assert background is not None
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

    def _refine_from_canny_edges(
        self,
        gray: np.ndarray,
        initial_fit: CircleFit,
        iterations: int,
    ) -> CircleFit:
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
        radial_alignment = (gradient_x * dx + gradient_y * dy) / np.maximum(
            1e-6,
            magnitude * distance,
        )
        # At a dark target's outer boundary intensity rises in the outward
        # radial direction; a bright target has the opposite sign. Keeping
        # the sign rejects internal highlights and the wrong side of double
        # edges instead of allowing both through with abs(alignment).
        polarity_alignment = (
            radial_alignment
            if self.config.threshold.polarity == "dark"
            else -radial_alignment
        )
        selected = (
            radial_mask
            & (magnitude >= self.config.edge_min_gradient)
            & (polarity_alignment >= self.config.edge_radial_alignment_min)
        )
        if np.count_nonzero(selected) < 12:
            return initial_fit
        edge_points = np.column_stack((global_x[selected], global_y[selected]))
        refined = fit_circle_ransac(
            edge_points,
            radius_min=self.config.radius_min_px,
            radius_max=self.config.radius_max_px,
            iterations=iterations,
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

    def _accept_edge_refinement(
        self,
        initial_fit: CircleFit,
        refined_fit: CircleFit | None,
        *,
        attempted: bool,
        support_ratio: float,
        allow_recovery: bool,
        temporal_radius_px: float | None,
    ) -> _EdgeRefinement:
        """Accept a fine fit only when it agrees with the stable coarse contour."""
        if refined_fit is None:
            return _EdgeRefinement(
                fit=initial_fit,
                attempted=attempted,
                support_ratio=float(np.clip(support_ratio, 0.0, 1.0)),
            )
        center_shift = float(
            np.hypot(
                refined_fit.center_x - initial_fit.center_x,
                refined_fit.center_y - initial_fit.center_y,
            )
        )
        radius_shift = float(abs(refined_fit.radius - initial_fit.radius))
        initial_expected_error = abs(
            initial_fit.radius - self.config.expected_radius_px
        ) / max(1e-6, self.config.expected_radius_px)
        refined_expected_error = abs(
            refined_fit.radius - self.config.expected_radius_px
        ) / max(1e-6, self.config.expected_radius_px)
        recovery_allowed = (
            allow_recovery
            and initial_expected_error
            >= self.config.edge_coarse_radius_outlier_ratio
            and refined_expected_error < initial_expected_error
        )
        center_shift_ratio = self.config.edge_center_shift_max_ratio
        radius_shift_ratio = self.config.edge_radius_shift_max_ratio
        if recovery_allowed:
            center_shift_ratio = max(
                center_shift_ratio,
                self.config.edge_recovery_shift_max_ratio,
            )
            radius_shift_ratio = max(
                radius_shift_ratio,
                self.config.edge_recovery_shift_max_ratio,
            )
        center_limit = center_shift_ratio * initial_fit.radius
        radius_limit = radius_shift_ratio * initial_fit.radius
        coarse_consistent = (
            np.isfinite(center_shift)
            and np.isfinite(radius_shift)
            and center_shift <= center_limit
            and radius_shift <= radius_limit
        )
        temporal_consistent = False
        if temporal_radius_px is not None:
            temporal_radius_limit = (
                self.config.radius_continuity_max_ratio
                * self.config.expected_radius_px
            )
            temporal_consistent = (
                abs(refined_fit.radius - temporal_radius_px)
                <= temporal_radius_limit
                and center_shift
                <= max(
                    center_limit,
                    self.config.edge_recovery_shift_max_ratio
                    * initial_fit.radius,
                )
                and radius_shift
                <= max(
                    radius_limit,
                    self.config.edge_recovery_shift_max_ratio
                    * self.config.expected_radius_px,
                )
            )
        accepted = (
            (coarse_consistent or temporal_consistent)
            and refined_fit.arc_coverage >= self.config.arc_coverage_min
        )
        return _EdgeRefinement(
            fit=refined_fit if accepted else initial_fit,
            attempted=attempted,
            accepted=accepted,
            support_ratio=float(np.clip(support_ratio, 0.0, 1.0)),
            center_shift_px=center_shift,
            radius_shift_px=radius_shift,
        )

    def _refine_from_radial_edges(
        self,
        gray: np.ndarray,
        initial_fit: CircleFit,
        iterations: int,
        *,
        allow_recovery: bool,
        temporal_radius_px: float | None,
    ) -> _EdgeRefinement:
        """Fit one polarity-consistent edge per radial spoke around the coarse circle.

        The segmentation contour supplies a strong radius prior.  This prevents
        an internal highlight or shadow ring from winning merely because it has
        a larger gradient than the physical outside boundary.
        """
        config = self.config
        radius = float(initial_fit.radius)
        annulus = max(2.0, config.edge_annulus_ratio * radius)
        step = float(config.edge_radial_step_px)
        radii = np.arange(
            max(2.0, radius - annulus),
            radius + annulus + 0.5 * step,
            step,
            dtype=np.float32,
        )
        if len(radii) < 5:
            return _EdgeRefinement(fit=initial_fit, attempted=True)

        spoke_count = int(config.edge_radial_spokes)
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            spoke_count,
            endpoint=False,
            dtype=np.float32,
        )
        cosine = np.cos(angles)
        sine = np.sin(angles)
        map_x = (
            initial_fit.center_x + cosine[:, None] * radii[None, :]
        ).astype(np.float32)
        map_y = (
            initial_fit.center_y + sine[:, None] * radii[None, :]
        ).astype(np.float32)
        sample_margin = int(np.ceil(float(radii[-1]))) + 2
        patch_x0 = max(0, int(np.floor(initial_fit.center_x)) - sample_margin)
        patch_y0 = max(0, int(np.floor(initial_fit.center_y)) - sample_margin)
        patch_x1 = min(
            gray.shape[1],
            int(np.ceil(initial_fit.center_x)) + sample_margin + 1,
        )
        patch_y1 = min(
            gray.shape[0],
            int(np.ceil(initial_fit.center_y)) + sample_margin + 1,
        )
        patch = gray[patch_y0:patch_y1, patch_x0:patch_x1]
        local_map_x = map_x - patch_x0
        local_map_y = map_y - patch_y0
        inside = (
            (local_map_x >= 1.0)
            & (local_map_x <= patch.shape[1] - 2.0)
            & (local_map_y >= 1.0)
            & (local_map_y <= patch.shape[0] - 2.0)
        )
        blurred = cv2.GaussianBlur(patch, (3, 3), 0.8)
        samples = cv2.remap(
            blurred,
            local_map_x,
            local_map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        response = np.gradient(samples, step, axis=1)
        if config.threshold.polarity == "bright":
            response = -response
        response = np.where(inside, response, -np.inf)
        initial_expected_error = abs(
            radius - config.expected_radius_px
        ) / max(1e-6, config.expected_radius_px)
        prior_sigma = float(config.edge_radius_prior_sigma_px)
        if (
            allow_recovery
            and initial_expected_error >= config.edge_coarse_radius_outlier_ratio
        ):
            # A weak/fragmented segmentation contour can sit inside the true
            # edge. Broaden only this acquisition's prior; acceptance below
            # still requires the fit to move toward the calibrated radius.
            prior_sigma = max(
                prior_sigma,
                float(config.edge_recovery_prior_sigma_px),
            )
        prior_radius = radius
        if temporal_radius_px is not None:
            temporal_weight = config.edge_temporal_radius_weight
            prior_radius = (
                (1.0 - temporal_weight) * radius
                + temporal_weight * float(temporal_radius_px)
            )
        prior_radius = float(np.clip(prior_radius, radii[0], radii[-1]))
        radius_prior = np.exp(
            -0.5 * ((radii - prior_radius) / prior_sigma) ** 2
        )[None, :]
        weighted_response = np.maximum(response, 0.0) * radius_prior
        peak_indices = np.argmax(weighted_response, axis=1)
        spoke_indices = np.arange(spoke_count)
        peak_response = response[spoke_indices, peak_indices]
        selected_radii = radii[peak_indices]
        valid_spokes = (
            np.isfinite(peak_response)
            & (peak_response >= config.edge_radial_gradient_min)
        )
        support_ratio = float(np.count_nonzero(valid_spokes) / spoke_count)
        if support_ratio < config.edge_min_spoke_ratio:
            return self._accept_edge_refinement(
                initial_fit,
                None,
                attempted=True,
                support_ratio=support_ratio,
                allow_recovery=allow_recovery,
                temporal_radius_px=temporal_radius_px,
            )
        edge_points = np.column_stack(
            (
                initial_fit.center_x
                + cosine[valid_spokes] * selected_radii[valid_spokes],
                initial_fit.center_y
                + sine[valid_spokes] * selected_radii[valid_spokes],
            )
        )
        refined = fit_circle_ransac(
            edge_points,
            radius_min=config.radius_min_px,
            radius_max=config.radius_max_px,
            iterations=iterations,
            residual_threshold=config.edge_radial_residual_px,
            sample_limit=max(config.contour_sample_limit, spoke_count),
        )
        return self._accept_edge_refinement(
            initial_fit,
            refined,
            attempted=True,
            support_ratio=support_ratio,
            allow_recovery=allow_recovery,
            temporal_radius_px=temporal_radius_px,
        )

    def _refine_from_edges(
        self,
        gray: np.ndarray,
        initial_fit: CircleFit,
        iterations: int,
        *,
        allow_recovery: bool = True,
        temporal_radius_px: float | None = None,
    ) -> _EdgeRefinement:
        if not self.config.edge_refine_enabled:
            return _EdgeRefinement(fit=initial_fit)
        if self.config.edge_refine_method == "radial":
            return self._refine_from_radial_edges(
                gray,
                initial_fit,
                iterations,
                allow_recovery=allow_recovery,
                temporal_radius_px=temporal_radius_px,
            )
        refined = self._refine_from_canny_edges(gray, initial_fit, iterations)
        support_ratio = refined.arc_coverage if refined is not initial_fit else 0.0
        return self._accept_edge_refinement(
            initial_fit,
            refined if refined is not initial_fit else None,
            attempted=True,
            support_ratio=support_ratio,
            allow_recovery=allow_recovery,
            temporal_radius_px=temporal_radius_px,
        )

    def detect(
        self,
        roi_image: np.ndarray,
        predicted_center_px: float | None = None,
        predicted_center_y_px: float | None = None,
        predicted_radius_px: float | None = None,
    ) -> DetectionResult:
        full_gray = self._to_gray(roi_image)
        full_height, full_width = full_gray.shape
        if predicted_center_px is not None and not np.isfinite(predicted_center_px):
            predicted_center_px = None
        if predicted_center_y_px is not None and not np.isfinite(
            predicted_center_y_px
        ):
            predicted_center_y_px = None
        if predicted_radius_px is not None and not np.isfinite(predicted_radius_px):
            predicted_radius_px = None
        search_x0 = 0
        search_x1 = full_width
        search_y0 = 0
        search_y1 = full_height
        morphology_margin = max(
            self.config.threshold.morph_open_size,
            self.config.threshold.morph_close_size,
        )
        if predicted_center_px is not None and self.config.predictive_search_enabled:
            padding = int(
                np.ceil(
                    self.config.max_prediction_distance_px
                    + self.config.radius_max_px
                    + self.config.border_margin_px
                    + morphology_margin
                )
            )
            proposed_x0 = max(0, int(np.floor(predicted_center_px)) - padding)
            proposed_x1 = min(
                full_width,
                int(np.ceil(predicted_center_px)) + padding + 1,
            )
            minimum_width = int(
                np.ceil(2.0 * self.config.radius_max_px)
                + 2 * self.config.border_margin_px
                + 1
            )
            if proposed_x1 - proposed_x0 >= minimum_width:
                search_x0, search_x1 = proposed_x0, proposed_x1
        if (
            predicted_center_px is not None
            and predicted_center_y_px is not None
            and self.config.predictive_search_enabled
        ):
            y_padding = int(
                np.ceil(
                    max(
                        self.config.predictive_search_y_half_height_px,
                        self.config.radius_max_px
                        + self.config.border_margin_px
                        + morphology_margin,
                    )
                )
            )
            proposed_y0 = max(0, int(np.floor(predicted_center_y_px)) - y_padding)
            proposed_y1 = min(
                full_height,
                int(np.ceil(predicted_center_y_px)) + y_padding + 1,
            )
            minimum_height = int(
                np.ceil(2.0 * self.config.radius_max_px)
                + 2 * self.config.border_margin_px
                + 1
            )
            if proposed_y1 - proposed_y0 >= minimum_height:
                search_y0, search_y1 = proposed_y0, proposed_y1

        # Segment the complete ROI once so adaptive intensity statistics,
        # brightness alignment, and morphology are invariant to whether the
        # tracker requested a predictive search window. Cropping before this
        # step made identical pixels acquire different masks as the pipeline
        # switched between global and tracked paths.
        full_mask, full_background_mask, threshold = self._build_masks(full_gray)
        if self.semantic_fallback is not None:
            fallback = self.semantic_fallback.infer(roi_image)
            if fallback is not None and fallback.mask.shape == full_gray.shape:
                fallback_mask = np.where(fallback.mask > 0, 255, 0).astype(np.uint8)
                full_mask = cv2.bitwise_or(full_mask, fallback_mask)

        gray = full_gray[search_y0:search_y1, search_x0:search_x1]
        mask = full_mask[search_y0:search_y1, search_x0:search_x1]
        background_mask = full_background_mask[
            search_y0:search_y1,
            search_x0:search_x1,
        ]
        self.last_search_rect = (
            search_x0,
            search_y0,
            search_x1 - search_x0,
            search_y1 - search_y0,
        )
        local_prediction = (
            predicted_center_px - search_x0
            if predicted_center_px is not None
            else None
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        height, width = gray.shape
        channel_center_y_global = (
            self.config.channel_center_y_px
            if self.config.channel_center_y_px >= 0
            else full_height * 0.5
        )
        channel_center_y = channel_center_y_global - search_y0
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

            initial_fit = fit_circle_ransac(
                contour,
                radius_min=self.config.radius_min_px,
                radius_max=self.config.radius_max_px,
                iterations=(
                    self.config.ransac_iterations_track
                    if local_prediction is not None
                    else self.config.ransac_iterations
                ),
                residual_threshold=self.config.ransac_residual_px,
                sample_limit=self.config.contour_sample_limit,
            )
            if initial_fit is None:
                continue
            refinement = self._refine_from_edges(
                gray,
                initial_fit,
                iterations=(
                    self.config.edge_ransac_iterations_track
                    if local_prediction is not None
                    else self.config.edge_ransac_iterations
                ),
                allow_recovery=local_prediction is None,
                temporal_radius_px=(
                    predicted_radius_px if local_prediction is not None else None
                ),
            )
            fit = refinement.fit
            if predicted_radius_px is not None and local_prediction is not None:
                refined_radius_error = abs(fit.radius - predicted_radius_px)
                initial_radius_error = abs(
                    initial_fit.radius - predicted_radius_px
                )
                temporal_radius_limit = (
                    self.config.radius_continuity_max_ratio
                    * self.config.expected_radius_px
                )
                if (
                    refined_radius_error > temporal_radius_limit
                    and initial_radius_error < refined_radius_error
                ):
                    # Do not discard an otherwise sound contour just because
                    # edge refinement disagrees with the temporal radius. The
                    # coarse contour is the safer fallback and a following
                    # global reacquisition remains available after a miss.
                    fit = initial_fit
                    refinement.fit = initial_fit
                    refinement.accepted = False
            if fit.rms > self.config.fit_rms_max_px or fit.inlier_ratio < self.config.fit_min_inlier_ratio:
                continue
            if fit.arc_coverage < self.config.arc_coverage_min:
                continue
            radius_error_ratio = abs(fit.radius - self.config.expected_radius_px) / max(
                1e-6, self.config.expected_radius_px
            )
            radius_tolerance = (
                self.config.radius_tolerance_track_ratio
                if local_prediction is not None
                else self.config.radius_tolerance_search_ratio
            )
            if radius_error_ratio > radius_tolerance:
                continue
            y_error = abs(fit.center_y - channel_center_y)
            if y_error > max_y_error:
                continue
            prediction_distance = 0.0
            if local_prediction is not None:
                prediction_distance = abs(fit.center_x - local_prediction)
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
            if local_prediction is None:
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
                + prediction_distance * (0.08 if local_prediction is not None else 0.0)
            )
            candidates.append(
                CircleCandidate(
                    center_px=(
                        fit.center_x + search_x0,
                        fit.center_y + search_y0,
                    ),
                    radius_px=fit.radius,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    score=score,
                    circularity=circularity,
                    fill_ratio=fill_ratio,
                    fit_rms_px=fit.rms,
                    inlier_ratio=fit.inlier_ratio,
                    arc_coverage=fit.arc_coverage,
                    area_px=area,
                    bbox=(
                        x + search_x0,
                        y + search_y0,
                        box_width,
                        box_height,
                    ),
                    source_mask=source_mask,
                    initial_center_px=(
                        initial_fit.center_x + search_x0,
                        initial_fit.center_y + search_y0,
                    ),
                    initial_radius_px=initial_fit.radius,
                    edge_refinement_attempted=refinement.attempted,
                    edge_refinement_accepted=refinement.accepted,
                    edge_support_ratio=refinement.support_ratio,
                    edge_center_shift_px=refinement.center_shift_px,
                    edge_radius_shift_px=refinement.radius_shift_px,
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
            if local_prediction is not None
            else self.config.search_confidence_min
        )
        accepted = best is not None and best.confidence >= confidence_minimum
        if best is None:
            reason = "no_geometrically_valid_candidate"
        elif not accepted:
            reason = "low_or_ambiguous_confidence"
        else:
            reason = "ok"
        debug_mask: np.ndarray | None = None
        if self.config.return_debug_mask:
            debug_mask = np.zeros_like(full_gray)
            debug_mask[
                search_y0:search_y1,
                search_x0:search_x1,
            ] = mask
        return DetectionResult(
            valid=accepted,
            candidate=best,
            threshold_value=threshold,
            candidate_count=len(candidates),
            mask=debug_mask,
            reason=reason,
            candidates=tuple(candidates),
        )
