from __future__ import annotations

import time

import cv2
import numpy as np

from .config import DisplayConfig
from .pipeline import PipelineResult
from .types import TrackFlags


def _state_text(flags: TrackFlags) -> str:
    if flags & TrackFlags.VALID:
        return "REACQUIRED" if flags & TrackFlags.REACQUIRED else "MEASURED"
    if flags & TrackFlags.PREDICTED:
        return "PREDICTED"
    return "LOST"


def render_debug(result: PipelineResult, draw_candidates: bool = True, show_mask: bool = True) -> np.ndarray:
    roi = result.roi_image
    canvas = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi.copy()
    if draw_candidates:
        for candidate in result.detection.candidates[1:]:
            center = tuple(round(value) for value in candidate.center_px)
            cv2.circle(canvas, center, round(candidate.radius_px), (255, 120, 0), 1, cv2.LINE_AA)
    candidate = result.detection.candidate
    if candidate is not None:
        center = tuple(round(value) for value in candidate.center_px)
        cv2.circle(canvas, center, round(candidate.radius_px), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, center, (0, 255, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    if result.predicted_roi_x_px is not None:
        predicted_x = round(result.predicted_roi_x_px)
        cv2.line(canvas, (predicted_x, 0), (predicted_x, canvas.shape[0] - 1), (0, 180, 255), 1)

    estimate = result.estimate
    lines = [
        f"{_state_text(estimate.flags)}  x={estimate.position_mm:+7.2f} mm  v={estimate.velocity_mm_s:+7.1f} mm/s",
        f"conf={estimate.confidence:.3f}  sigma={estimate.position_sigma_mm:.2f} mm  candidates={result.detection.candidate_count}  threshold={result.detection.threshold_value:.1f}",
        f"process={result.processing_us / 1000.0:.2f} ms  age={result.frame_age_us / 1000.0:.2f} ms  seq={result.frame.sequence}",
    ]
    for index, text in enumerate(lines):
        y = 24 + index * 23
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    if show_mask and result.detection.mask is not None:
        mask = cv2.cvtColor(result.detection.mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask, "SEGMENTATION MASK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
        canvas = np.hstack((canvas, mask))
    return canvas


class DebugDisplay:
    def __init__(self, config: DisplayConfig) -> None:
        self.config = config
        self._last_show = 0.0
        self.window_name = "TI H - Ball Vision"

    def show(self, result: PipelineResult) -> int:
        if not self.config.enabled:
            return -1
        now = time.monotonic()
        interval = 1.0 / max(1.0, self.config.max_fps)
        if now - self._last_show < interval:
            return -1
        frame = render_debug(result, self.config.draw_candidates, self.config.show_mask)
        if self.config.scale != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.config.scale,
                fy=self.config.scale,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(self.window_name, frame)
        self._last_show = now
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self.config.enabled:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
