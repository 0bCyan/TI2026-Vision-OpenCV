from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import CalibrationConfig


@dataclass(slots=True)
class CalibrationQuality:
    rms_mm: float
    max_abs_mm: float
    point_count: int


class CoordinateMapper:
    """Maps full-frame horizontal pixel coordinates to the pipe coordinate in mm."""

    def __init__(self, config: CalibrationConfig) -> None:
        points = np.asarray(config.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("标定点必须为 [[pixel, mm], ...]")
        order = np.argsort(points[:, 0])
        self._pixels = points[order, 0]
        self._millimeters = points[order, 1]
        if np.any(np.diff(self._pixels) <= 0):
            raise ValueError("标定像素坐标必须互不相同")
        mm_delta = np.diff(self._millimeters)
        if not (np.all(mm_delta > 0) or np.all(mm_delta < 0)):
            raise ValueError("标定毫米坐标必须严格单调")
        self.mode = config.mode
        self.degree = min(int(config.degree), len(points) - 1)
        self._coefficients = np.polyfit(self._pixels, self._millimeters, self.degree)
        fitted = (
            np.interp(self._pixels, self._pixels, self._millimeters)
            if self.mode == "piecewise_linear"
            else np.polyval(self._coefficients, self._pixels)
        )
        residual = fitted - self._millimeters
        self.quality = CalibrationQuality(
            rms_mm=float(np.sqrt(np.mean(residual * residual))),
            max_abs_mm=float(np.max(np.abs(residual))),
            point_count=len(points),
        )
        dense_px = np.linspace(self._pixels[0], self._pixels[-1], 4096)
        dense_mm = (
            np.interp(dense_px, self._pixels, self._millimeters)
            if self.mode == "piecewise_linear"
            else np.polyval(self._coefficients, dense_px)
        )
        delta = np.diff(dense_mm)
        if not (np.all(delta >= 0) or np.all(delta <= 0)):
            raise ValueError("标定曲线在有效区间内不是单调函数，请降低阶数或检查标定点")
        if dense_mm[0] <= dense_mm[-1]:
            self._inverse_mm = dense_mm
            self._inverse_px = dense_px
        else:
            self._inverse_mm = dense_mm[::-1]
            self._inverse_px = dense_px[::-1]

    @property
    def pixel_limits(self) -> tuple[float, float]:
        return float(self._pixels[0]), float(self._pixels[-1])

    @property
    def millimeter_limits(self) -> tuple[float, float]:
        values = (float(self._millimeters.min()), float(self._millimeters.max()))
        return values

    def contains_pixel(self, pixel_x: float, margin_px: float = 0.0) -> bool:
        margin = max(0.0, float(margin_px))
        pixel = float(pixel_x)
        return bool(self._pixels[0] - margin <= pixel <= self._pixels[-1] + margin)

    def pixel_to_mm(self, pixel_x: float) -> float:
        pixel = float(pixel_x)
        if self.mode == "polynomial":
            return float(np.polyval(self._coefficients, pixel))
        if pixel <= self._pixels[0]:
            index = 0
        elif pixel >= self._pixels[-1]:
            index = len(self._pixels) - 2
        else:
            index = int(np.searchsorted(self._pixels, pixel) - 1)
        x0, x1 = self._pixels[index], self._pixels[index + 1]
        y0, y1 = self._millimeters[index], self._millimeters[index + 1]
        return float(y0 + (pixel - x0) * (y1 - y0) / (x1 - x0))

    def mm_to_pixel(self, position_mm: float) -> float:
        position = float(np.clip(position_mm, self._inverse_mm[0], self._inverse_mm[-1]))
        return float(np.interp(position, self._inverse_mm, self._inverse_px))

    def coefficients(self) -> list[float]:
        return [float(value) for value in self._coefficients]


def fit_calibration_points(
    pixel_values: Sequence[float],
    millimeter_values: Sequence[float],
    degree: int,
) -> CalibrationConfig:
    if len(pixel_values) != len(millimeter_values):
        raise ValueError("像素点和毫米坐标数量不一致")
    points = [[float(px), float(mm)] for px, mm in zip(pixel_values, millimeter_values)]
    config = CalibrationConfig(mode="piecewise_linear", degree=int(degree), points=points)
    CoordinateMapper(config)
    return config
