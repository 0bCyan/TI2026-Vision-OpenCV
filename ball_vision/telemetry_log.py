from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

from .config import LoggingConfig
from .pipeline import PipelineResult


class CsvTelemetryLogger:
    FIELDS = (
        "frame_sequence",
        "frame_time_ns",
        "arrival_ns",
        "raw_center_px",
        "raw_position_mm",
        "raw_sigma_mm",
        "raw_measurement_valid",
        "raw_near_edge",
        "raw_reason",
        "filtered_position_mm",
        "velocity_mm_s",
        "position_sigma_mm",
        "track_flags",
        "measurement_valid",
        "confidence",
        "lost_frames",
        "innovation_mm",
        "candidate_count",
        "threshold",
        "radius_px",
        "fit_rms_px",
        "arc_coverage",
        "processing_us",
        "frame_age_us",
        "consumer_skipped_frames",
        "failure_reason",
    )

    def __init__(self, config: LoggingConfig, base_directory: Path) -> None:
        self.enabled = config.enabled
        self.flush_interval = max(0.05, config.flush_interval_seconds)
        self._last_flush = time.monotonic()
        self._handle = None
        self._writer = None
        self.path: Path | None = None
        if not self.enabled:
            return
        directory = Path(config.directory)
        if not directory.is_absolute():
            directory = base_directory / directory
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"vision_{stamp}.csv"
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, result: PipelineResult, skipped_frames: int) -> None:
        if not self.enabled or self._writer is None:
            return
        candidate = result.detection.candidate
        estimate = result.estimate
        measurement = result.raw_measurement
        self._writer.writerow(
            {
                "frame_sequence": result.frame.sequence,
                "frame_time_ns": result.frame.frame_time_ns,
                "arrival_ns": result.frame.arrival_ns,
                "raw_center_px": "" if result.raw_global_x_px is None else f"{result.raw_global_x_px:.4f}",
                "raw_position_mm": "" if result.raw_position_mm is None else f"{result.raw_position_mm:.4f}",
                "raw_sigma_mm": "" if measurement.sigma_mm is None else f"{measurement.sigma_mm:.4f}",
                "raw_measurement_valid": int(measurement.valid),
                "raw_near_edge": int(measurement.near_edge),
                "raw_reason": measurement.reason,
                "filtered_position_mm": f"{estimate.position_mm:.4f}",
                "velocity_mm_s": f"{estimate.velocity_mm_s:.3f}",
                "position_sigma_mm": f"{estimate.position_sigma_mm:.4f}",
                "track_flags": int(estimate.flags),
                "measurement_valid": int(measurement.valid),
                "confidence": f"{estimate.confidence:.5f}",
                "lost_frames": estimate.lost_frames,
                "innovation_mm": "" if estimate.innovation_mm is None else f"{estimate.innovation_mm:.4f}",
                "candidate_count": result.detection.candidate_count,
                "threshold": f"{result.detection.threshold_value:.2f}",
                "radius_px": "" if candidate is None else f"{candidate.radius_px:.3f}",
                "fit_rms_px": "" if candidate is None else f"{candidate.fit_rms_px:.3f}",
                "arc_coverage": "" if candidate is None else f"{candidate.arc_coverage:.4f}",
                "processing_us": result.processing_us,
                "frame_age_us": result.frame_age_us,
                "consumer_skipped_frames": skipped_frames,
                "failure_reason": result.detection.reason,
            }
        )
        now = time.monotonic()
        if now - self._last_flush >= self.flush_interval:
            self._handle.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
