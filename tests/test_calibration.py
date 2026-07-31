import numpy as np
import pytest

from ball_vision.calibration import CoordinateMapper
from ball_vision.config import CalibrationConfig


def test_piecewise_calibration_and_inverse() -> None:
    config = CalibrationConfig(
        mode="piecewise_linear",
        degree=1,
        points=[[100.0, -120.0], [500.0, 0.0], [920.0, 120.0]],
    )
    mapper = CoordinateMapper(config)
    assert mapper.pixel_to_mm(500.0) == pytest.approx(0.0)
    assert mapper.pixel_to_mm(300.0) == pytest.approx(-60.0)
    assert mapper.contains_pixel(100.0)
    assert mapper.contains_pixel(920.0)
    assert not mapper.contains_pixel(99.9)
    assert mapper.contains_pixel(99.9, margin_px=0.2)
    for position in np.linspace(-120.0, 120.0, 25):
        assert mapper.pixel_to_mm(mapper.mm_to_pixel(position)) == pytest.approx(position, abs=0.08)


def test_reversed_camera_orientation_is_supported() -> None:
    mapper = CoordinateMapper(
        CalibrationConfig(
            mode="piecewise_linear",
            degree=1,
            points=[[100.0, 120.0], [900.0, -120.0]],
        )
    )
    assert mapper.pixel_to_mm(100.0) == pytest.approx(120.0)
    assert mapper.mm_to_pixel(-120.0) == pytest.approx(900.0, abs=0.1)


def test_non_monotonic_calibration_is_rejected() -> None:
    with pytest.raises(ValueError):
        CoordinateMapper(
            CalibrationConfig(
                mode="piecewise_linear",
                degree=1,
                points=[[0.0, -10.0], [10.0, 10.0], [20.0, 0.0]],
            )
        )
