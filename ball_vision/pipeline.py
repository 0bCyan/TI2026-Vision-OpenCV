from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .calibration import CoordinateMapper
from .config import AppConfig
from .detector import BallDetector
from .tracker import BallTracker
from .types import (
    DetectionResult,
    FramePacket,
    RawMeasurement,
    TrackEstimate,
    TrackFlags,
)


@dataclass(slots=True)
class PipelineResult:
    frame: FramePacket
    roi_image: np.ndarray
    roi_rect: tuple[int, int, int, int]
    detection: DetectionResult
    estimate: TrackEstimate
    raw_measurement: RawMeasurement
    predicted_roi_x_px: float | None
    processing_us: int
    frame_age_us: int

    @property
    def raw_global_x_px(self) -> float | None:
        return self.raw_measurement.global_x_px

    @property
    def raw_position_mm(self) -> float | None:
        return self.raw_measurement.position_mm


class VisionPipeline:
    """Pure processing pipeline; it has no camera, GUI, serial, or file I/O loop."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mapper = CoordinateMapper(config.calibration)
        background_path = config.resolve_path(config.detector.threshold.background_path)
        self.detector = BallDetector(config.detector, background_path)
        self.tracker = BallTracker(config.tracker)
        self._force_global_search = False
        self._last_detection_roi_y: float | None = None

    def _local_mm_per_pixel(self, pixel_x: float) -> float:
        minimum_px, maximum_px = self.mapper.pixel_limits
        left_px = max(minimum_px, pixel_x - 0.5)
        right_px = min(maximum_px, pixel_x + 0.5)
        pixel_span = right_px - left_px
        if pixel_span <= 0:
            return float("nan")
        millimeter_span = abs(
            self.mapper.pixel_to_mm(right_px) - self.mapper.pixel_to_mm(left_px)
        )
        return millimeter_span / pixel_span

    def _make_raw_measurement(
        self,
        detection: DetectionResult,
        roi_x: int,
        frame_time_ns: int,
    ) -> RawMeasurement:
        candidate = detection.candidate
        if not detection.valid or candidate is None:
            return RawMeasurement(
                valid=False,
                frame_time_ns=frame_time_ns,
                global_x_px=None,
                position_mm=None,
                sigma_mm=None,
                confidence=0.0,
                near_edge=False,
                reason=detection.reason or "no_measurement",
            )

        global_x_px = float(roi_x + candidate.center_px[0])
        if not self.mapper.contains_pixel(global_x_px):
            detection.valid = False
            detection.reason = "outside_calibrated_pixel_range"
            return RawMeasurement(
                valid=False,
                frame_time_ns=frame_time_ns,
                global_x_px=global_x_px,
                position_mm=None,
                sigma_mm=None,
                confidence=float(candidate.confidence),
                near_edge=False,
                reason=detection.reason,
            )

        position_mm = self.mapper.pixel_to_mm(global_x_px)
        minimum_mm, maximum_mm = self.mapper.millimeter_limits
        edge_distance_mm = min(position_mm - minimum_mm, maximum_mm - position_mm)
        near_edge = edge_distance_mm <= self.config.measurement.near_edge_margin_mm
        center_sigma_px = max(
            float(candidate.fit_rms_px),
            self.config.measurement.minimum_center_sigma_px,
        )
        fit_sigma_mm = center_sigma_px * self._local_mm_per_pixel(global_x_px)
        sigma_mm = float(
            np.hypot(fit_sigma_mm, self.config.measurement.calibration_rms_mm)
        )
        if near_edge:
            sigma_mm *= self.config.measurement.near_edge_multiplier

        finite = all(
            np.isfinite(value)
            for value in (global_x_px, position_mm, sigma_mm, candidate.confidence)
        )
        if not finite or sigma_mm <= 0:
            valid = False
            reason = "non_finite_raw_measurement"
        elif sigma_mm > self.config.measurement.maximum_valid_sigma_mm:
            valid = False
            reason = "sigma_above_maximum"
        else:
            valid = True
            reason = "ok"
        return RawMeasurement(
            valid=valid,
            frame_time_ns=frame_time_ns,
            global_x_px=global_x_px,
            position_mm=position_mm,
            sigma_mm=sigma_mm,
            confidence=float(np.clip(candidate.confidence, 0.0, 1.0)),
            near_edge=near_edge,
            reason=reason,
        )

    def prepare_background_for_frame(self, frame: np.ndarray) -> None:
        background = self.detector.background_raw
        if background is None:
            return
        x, y, width, height = self.config.roi.clamp(frame.shape[1], frame.shape[0])
        if background.shape == frame.shape[:2]:
            self.detector.set_background(background[y : y + height, x : x + width])
        elif background.shape != (height, width):
            message = (
                "背景图尺寸与ROI不一致: "
                f"background={background.shape[::-1]}, roi={(width, height)}"
            )
            self.detector.clear_background()
            if (
                self.config.camera.strict_contract
                or self.config.output.serial_enabled
                or self.config.output.udp_enabled
            ):
                raise RuntimeError(message + "，已拒绝启用正式遥测输出")
            print("[WARN]", message, "，已停用背景差分")

    def process(self, packet: FramePacket) -> PipelineResult:
        start_ns = time.monotonic_ns()
        frame = packet.image
        x, y, width, height = self.config.roi.clamp(frame.shape[1], frame.shape[0])
        roi = frame[y : y + height, x : x + width]

        prediction = self.tracker.predict(packet.frame_time_ns)
        predicted_roi_x: float | None = None
        predicted_roi_y: float | None = None
        prediction_is_fresh = (
            prediction is not None
            and self.tracker.prediction_is_fresh(packet.frame_time_ns)
        )
        if prediction_is_fresh and not self._force_global_search:
            assert prediction is not None
            predicted_global_x = self.mapper.mm_to_pixel(prediction[0])
            predicted_roi_x = predicted_global_x - x
            predicted_roi_y = self._last_detection_roi_y

        detection = self.detector.detect(
            roi,
            predicted_roi_x,
            predicted_roi_y,
        )
        if detection.valid and detection.candidate is not None:
            self._last_detection_roi_y = float(detection.candidate.center_px[1])
        raw_measurement = self._make_raw_measurement(
            detection,
            roi_x=x,
            frame_time_ns=packet.frame_time_ns,
        )
        # A protocol-level high-sigma rejection must not destroy internal
        # temporal association.  Numerical and calibration failures must never
        # enter tracker state, however.
        tracker_position_mm = (
            raw_measurement.position_mm
            if raw_measurement.valid
            or raw_measurement.reason == "sigma_above_maximum"
            else None
        )
        tracker_confidence = (
            raw_measurement.confidence if tracker_position_mm is not None else 0.0
        )

        estimate = self.tracker.update(
            tracker_position_mm,
            tracker_confidence,
            packet.frame_time_ns,
        )
        if prediction_is_fresh:
            # One predictive-window miss is enough to restore the full ROI on
            # subsequent frames. Keep global search until the tracker actually
            # associates a measurement, not merely until any contour appears.
            self._force_global_search = estimate.measurement_mm is None
        else:
            self._force_global_search = False
        minimum_mm, maximum_mm = self.mapper.millimeter_limits
        if estimate.valid and (
            estimate.position_mm <= minimum_mm + self.config.measurement.near_edge_margin_mm
            or estimate.position_mm >= maximum_mm - self.config.measurement.near_edge_margin_mm
        ):
            estimate.flags |= TrackFlags.NEAR_EDGE

        end_ns = time.monotonic_ns()
        return PipelineResult(
            frame=packet,
            roi_image=roi,
            roi_rect=(x, y, width, height),
            detection=detection,
            estimate=estimate,
            raw_measurement=raw_measurement,
            predicted_roi_x_px=predicted_roi_x,
            processing_us=int((end_ns - start_ns) // 1000),
            frame_age_us=max(0, int((end_ns - packet.frame_time_ns) // 1000)),
        )
