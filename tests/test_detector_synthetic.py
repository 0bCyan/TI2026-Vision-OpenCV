import cv2
import numpy as np
import pytest

from ball_vision.config import DetectorConfig, ThresholdConfig
from ball_vision.detector import BallDetector
from ball_vision.synthetic import make_pipe_background, render_ball


def detector_for_scene(background: np.ndarray) -> BallDetector:
    config = DetectorConfig(
        threshold=ThresholdConfig(
            polarity="dark",
            median_blur_size=3,
            min_contrast_delta=16,
            std_gain=0.65,
            threshold_min=20,
            threshold_max=180,
            background_enabled=True,
            background_diff_threshold=14,
            combine_mode="and",
            morph_open_size=3,
            morph_close_size=3,
        ),
        radius_min_px=15,
        radius_max_px=32,
        expected_radius_px=22,
        area_min_px=500,
        area_max_px=3200,
        circularity_min=0.35,
        fill_ratio_min=0.35,
        channel_center_y_px=80,
        max_center_y_error_px=36,
        ransac_iterations=64,
        fit_rms_max_px=3.0,
        fit_min_inlier_ratio=0.35,
        arc_coverage_min=0.25,
    )
    detector = BallDetector(config)
    detector.set_background(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    return detector


@pytest.mark.parametrize(
    "center,jpeg_quality,highlight",
    [
        ((120.25, 79.6), None, 90.0),
        ((480.50, 80.2), 55, 140.0),
        ((835.75, 78.9), 75, 180.0),
    ],
)
def test_detector_finds_subpixel_ball(center, jpeg_quality, highlight) -> None:
    background = make_pipe_background()
    synthetic = render_ball(
        background,
        center_px=center,
        radius_px=22,
        jpeg_quality=jpeg_quality,
        highlight_strength=highlight,
        noise_sigma=2.0,
    )
    result = detector_for_scene(background).detect(synthetic.image)
    assert result.valid, result.reason
    assert result.candidate is not None
    error = np.hypot(
        result.candidate.center_px[0] - center[0],
        result.candidate.center_px[1] - center[1],
    )
    assert error < 1.2
    assert result.candidate.confidence > 0.50


def test_empty_pipe_has_no_valid_ball() -> None:
    background = make_pipe_background()
    result = detector_for_scene(background).detect(background)
    assert not result.valid


def test_static_dark_fixture_present_in_background_is_rejected() -> None:
    empty = make_pipe_background()
    fixture = render_ball(
        empty,
        center_px=(480.0, 80.0),
        radius_px=22.0,
        highlight_strength=0.0,
        noise_sigma=0.0,
    ).image
    result = detector_for_scene(fixture).detect(fixture)
    assert not result.valid


def test_detector_rejects_circle_with_wrong_physical_radius() -> None:
    background = make_pipe_background()
    frame = render_ball(
        background,
        center_px=(480.0, 80.0),
        radius_px=31.0,
        highlight_strength=90.0,
    ).image
    result = detector_for_scene(background).detect(frame)
    assert not result.valid


def test_partial_occlusion_still_uses_outer_arc_center() -> None:
    background = make_pipe_background()
    center = (480.3, 80.0)
    frame = render_ball(
        background,
        center_px=center,
        radius_px=22,
        occlusion_fraction=0.35,
        jpeg_quality=60,
        highlight_strength=160,
    ).image
    result = detector_for_scene(background).detect(frame)
    assert result.valid
    assert result.candidate is not None
    assert abs(result.candidate.center_px[0] - center[0]) < 1.2
    assert result.candidate.arc_coverage > 0.45
