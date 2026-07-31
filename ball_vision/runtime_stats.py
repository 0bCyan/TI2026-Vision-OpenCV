from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProcessingSummary:
    frames: int
    measured_rate: float
    skipped_frames: int
    processing_p50_us: float
    processing_p95_us: float
    processing_p99_us: float
    frame_age_p95_us: float


class RuntimeStats:
    def __init__(self, window: int = 4096) -> None:
        self.processing_us: deque[int] = deque(maxlen=window)
        self.frame_age_us: deque[int] = deque(maxlen=window)
        self.measured: deque[int] = deque(maxlen=window)
        self.frames = 0
        self.skipped_frames = 0

    def add(self, processing_us: int, frame_age_us: int, measured: bool, skipped: int) -> None:
        self.frames += 1
        self.skipped_frames += max(0, int(skipped))
        self.processing_us.append(int(processing_us))
        self.frame_age_us.append(int(frame_age_us))
        self.measured.append(1 if measured else 0)

    def summary(self) -> ProcessingSummary:
        processing = np.asarray(self.processing_us, dtype=np.float64)
        ages = np.asarray(self.frame_age_us, dtype=np.float64)
        return ProcessingSummary(
            frames=self.frames,
            measured_rate=float(np.mean(self.measured)) if self.measured else 0.0,
            skipped_frames=self.skipped_frames,
            processing_p50_us=float(np.percentile(processing, 50)) if len(processing) else 0.0,
            processing_p95_us=float(np.percentile(processing, 95)) if len(processing) else 0.0,
            processing_p99_us=float(np.percentile(processing, 99)) if len(processing) else 0.0,
            frame_age_p95_us=float(np.percentile(ages, 95)) if len(ages) else 0.0,
        )

