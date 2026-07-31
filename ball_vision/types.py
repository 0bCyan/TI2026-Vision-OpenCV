from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

Point = tuple[float, float]


@dataclass(slots=True)
class FramePacket:
    sequence: int
    arrival_ns: int
    frame_time_ns: int
    read_duration_ns: int
    image: np.ndarray


@dataclass(slots=True)
class CircleCandidate:
    center_px: Point
    radius_px: float
    confidence: float
    score: float
    circularity: float
    fill_ratio: float
    fit_rms_px: float
    inlier_ratio: float
    arc_coverage: float
    area_px: float
    bbox: tuple[int, int, int, int]
    source_mask: int


@dataclass(slots=True)
class DetectionResult:
    valid: bool
    candidate: CircleCandidate | None
    threshold_value: float
    candidate_count: int
    mask: np.ndarray | None = None
    reason: str = ""
    candidates: tuple[CircleCandidate, ...] = ()


@dataclass(slots=True, frozen=True)
class RawMeasurement:
    """One image-frame measurement, independent of tracker state."""

    valid: bool
    frame_time_ns: int
    global_x_px: float | None
    position_mm: float | None
    sigma_mm: float | None
    confidence: float
    near_edge: bool
    reason: str


class TrackFlags(IntFlag):
    NONE = 0
    VALID = 1 << 0
    PREDICTED = 1 << 1
    LOST = 1 << 2
    REACQUIRED = 1 << 3
    NEAR_EDGE = 1 << 4


@dataclass(slots=True)
class TrackEstimate:
    position_mm: float
    velocity_mm_s: float
    position_sigma_mm: float
    confidence: float
    flags: TrackFlags
    frame_time_ns: int
    measurement_mm: float | None
    innovation_mm: float | None
    lost_frames: int

    @property
    def valid(self) -> bool:
        return bool(self.flags & TrackFlags.VALID)
