from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class SegmentationFallbackResult:
    mask: np.ndarray
    confidence: float
    source_name: str


class SegmentationFallback(Protocol):
    """Optional coarse-mask provider (ONNX on PC, RKNN on Orange Pi later).

    It must never return a final ball center. The OpenCV geometry stage still
    performs original-resolution edge fitting and confidence checks.
    """

    def infer(self, roi_image: np.ndarray) -> SegmentationFallbackResult | None: ...

