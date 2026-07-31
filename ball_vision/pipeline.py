from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .calibration import CoordinateMapper
from .config import AppConfig
from .detector import BallDetector
from .tracker import BallTracker
from .types import DetectionResult, FramePacket, TrackEstimate, TrackFlags


@dataclass(slots=True)
class PipelineResult:
    frame: FramePacket
    roi_image: np.ndarray
    roi_rect: tuple[int, int, int, int]
    detection: DetectionResult
    estimate: TrackEstimate
    raw_global_x_px: float | None
    raw_position_mm: float | None
    predicted_roi_x_px: float | None
    processing_us: int
    frame_age_us: int


class VisionPipeline:
    """Pure processing pipeline; it has no camera, GUI, serial, or file I/O loop."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mapper = CoordinateMapper(config.calibration)
        background_path = config.resolve_path(config.detector.threshold.background_path)
        self.detector = BallDetector(config.detector, background_path)
        self.tracker = BallTracker(config.tracker)

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
        if prediction is not None and self.tracker.prediction_is_fresh(packet.frame_time_ns):
            predicted_global_x = self.mapper.mm_to_pixel(prediction[0])
            predicted_roi_x = predicted_global_x - x

        detection = self.detector.detect(roi, predicted_roi_x)
        raw_global_x: float | None = None
        raw_position_mm: float | None = None
        measurement_confidence = 0.0
        if detection.valid and detection.candidate is not None:
            raw_global_x = x + detection.candidate.center_px[0]
            if self.mapper.contains_pixel(raw_global_x):
                raw_position_mm = self.mapper.pixel_to_mm(raw_global_x)
                measurement_confidence = detection.candidate.confidence
            else:
                detection.valid = False
                detection.reason = "outside_calibrated_pixel_range"

        estimate = self.tracker.update(
            raw_position_mm,
            measurement_confidence,
            packet.frame_time_ns,
        )
        minimum_mm, maximum_mm = self.mapper.millimeter_limits
        if estimate.valid and (
            estimate.position_mm <= minimum_mm + 8.0
            or estimate.position_mm >= maximum_mm - 8.0
        ):
            estimate.flags |= TrackFlags.NEAR_EDGE

        end_ns = time.monotonic_ns()
        return PipelineResult(
            frame=packet,
            roi_image=roi,
            roi_rect=(x, y, width, height),
            detection=detection,
            estimate=estimate,
            raw_global_x_px=raw_global_x,
            raw_position_mm=raw_position_mm,
            predicted_roi_x_px=predicted_roi_x,
            processing_us=int((end_ns - start_ns) // 1000),
            frame_age_us=max(0, int((end_ns - packet.frame_time_ns) // 1000)),
        )
