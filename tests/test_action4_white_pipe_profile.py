from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ball_vision.calibration import CoordinateMapper
from ball_vision.config import AppConfig
from ball_vision.io_utils import imread_path
from ball_vision.pipeline import VisionPipeline
from ball_vision.synthetic import make_pipe_background, render_ball
from ball_vision.types import FramePacket

CONFIG_PATH = Path(__file__).parents[1] / "config" / "action4_white_pipe.yaml"
BACKGROUND_PATH = (
    Path(__file__).parents[1] / "data" / "action4_white_pipe_background.png"
)
ROI_CENTER = (602.5, 41.0)
GLOBAL_CENTER_X = 642.5
ZERO_REFERENCE_PIXEL = 615.889810
TEN_MM_REFERENCE_PIXEL = 648.054020


def _profile() -> AppConfig:
    return AppConfig.load(CONFIG_PATH)


def _frame_with_ball(
    gain: float,
    jpeg_quality: int | None,
    highlight_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    config = _profile()
    background = make_pipe_background(
        width=config.roi.width,
        height=config.roi.height,
        seed=41,
    )
    white_pipe = np.clip(background.astype(np.float32) * gain, 0, 255).astype(
        np.uint8
    )
    roi = render_ball(
        white_pipe,
        center_px=ROI_CENTER,
        radius_px=config.detector.expected_radius_px,
        jpeg_quality=jpeg_quality,
        highlight_strength=highlight_strength,
        noise_sigma=2.0,
    ).image
    frame = np.full(
        (config.camera.height, config.camera.width, 3),
        25,
        dtype=np.uint8,
    )
    x, y, width, height = config.roi.clamp(frame.shape[1], frame.shape[0])
    frame[y : y + height, x : x + width] = roi
    return frame, white_pipe


def _process(
    frame: np.ndarray,
    config: AppConfig | None = None,
    background_roi: np.ndarray | None = None,
):
    config = config or _profile()
    pipeline = VisionPipeline(config)
    if background_roi is not None:
        pipeline.detector.set_background(background_roi)
    frame_time_ns = time.monotonic_ns()
    return pipeline.process(
        FramePacket(
            sequence=1,
            arrival_ns=frame_time_ns,
            frame_time_ns=frame_time_ns,
            read_duration_ns=0,
            image=frame,
        )
    )


def test_action4_profile_is_preview_only() -> None:
    config = _profile()

    assert config.camera.source == 1
    assert config.camera.backend == "dshow"
    assert config.camera.fps == 30.0
    assert config.camera.controls == {}
    assert not config.camera.strict_contract
    assert config.detector.threshold.local_contrast_enabled
    assert config.detector.threshold.local_contrast_kernel_size == 41
    assert config.detector.threshold.local_contrast_threshold == 12
    assert config.detector.threshold.background_enabled
    assert config.detector.threshold.combine_mode == "and"
    assert config.resolve_path(config.detector.threshold.background_path).is_file()
    assert config.detector.ransac_iterations_track == 24
    assert config.detector.edge_ransac_iterations_track == 12
    assert not config.output.serial_enabled
    assert not config.output.udp_enabled


def test_action4_measured_reference_pixel_maps_to_zero_mm() -> None:
    mapper = CoordinateMapper(_profile().calibration)

    assert mapper.pixel_to_mm(ZERO_REFERENCE_PIXEL) == pytest.approx(0.0, abs=1e-9)
    assert mapper.pixel_to_mm(TEN_MM_REFERENCE_PIXEL) == pytest.approx(
        10.0,
        abs=1e-9,
    )
    assert mapper.pixel_to_mm(ZERO_REFERENCE_PIXEL + 1.0) == pytest.approx(
        10.0 / (TEN_MM_REFERENCE_PIXEL - ZERO_REFERENCE_PIXEL),
        rel=1e-9,
    )


@pytest.mark.parametrize(
    "field_name,value,error_text",
    [
        ("local_contrast_kernel_size", 2, "local_contrast_kernel_size"),
        ("local_contrast_threshold", 0, "local_contrast_threshold"),
    ],
)
def test_local_contrast_configuration_rejects_invalid_values(
    field_name: str,
    value: int,
    error_text: str,
) -> None:
    config = _profile()
    setattr(config.detector.threshold, field_name, value)

    with pytest.raises(ValueError, match=error_text):
        config.validate()


@pytest.mark.parametrize(
    "gain,jpeg_quality,highlight_strength",
    [
        (0.55, 45, 90.0),
        (0.72, 70, 130.0),
        (0.90, None, 160.0),
        (1.00, 90, 110.0),
    ],
)
def test_action4_profile_detects_dark_ball_on_white_pipe(
    gain: float,
    jpeg_quality: int | None,
    highlight_strength: float,
) -> None:
    frame, background = _frame_with_ball(gain, jpeg_quality, highlight_strength)
    result = _process(frame, background_roi=background)

    assert result.detection.valid, result.detection.reason
    assert result.detection.candidate_count == 1
    assert result.detection.candidate is not None
    assert abs(result.detection.candidate.center_px[0] - ROI_CENTER[0]) < 1.0
    assert abs(result.detection.candidate.center_px[1] - ROI_CENTER[1]) < 1.0
    assert result.raw_measurement.global_x_px is not None
    assert abs(result.raw_measurement.global_x_px - GLOBAL_CENTER_X) < 1.0


def test_action4_profile_rejects_empty_white_pipe() -> None:
    config = _profile()
    frame = np.full(
        (config.camera.height, config.camera.width, 3),
        25,
        dtype=np.uint8,
    )
    roi = make_pipe_background(
        width=config.roi.width,
        height=config.roi.height,
        seed=41,
    )
    x, y, width, height = config.roi.clamp(frame.shape[1], frame.shape[0])
    frame[y : y + height, x : x + width] = roi

    result = _process(frame, background_roi=roi)

    assert not result.detection.valid
    assert result.detection.candidate_count == 0


def test_action4_captured_empty_pipe_background_matches_roi_and_is_rejected() -> None:
    config = _profile()
    background = imread_path(BACKGROUND_PATH, cv2.IMREAD_GRAYSCALE)

    assert background is not None
    assert background.shape == (config.roi.height, config.roi.width)

    frame = np.full(
        (config.camera.height, config.camera.width, 3),
        25,
        dtype=np.uint8,
    )
    x, y, width, height = config.roi.clamp(frame.shape[1], frame.shape[0])
    frame[y : y + height, x : x + width] = cv2.cvtColor(
        background,
        cv2.COLOR_GRAY2BGR,
    )

    result = _process(frame, config)

    assert not result.detection.valid
    assert result.detection.candidate_count == 0
    assert result.detection.mask is not None
    assert np.count_nonzero(result.detection.mask) == 0


def test_action4_profile_separates_ball_from_wide_pipe_shadow() -> None:
    config = _profile()
    pipe = make_pipe_background(
        width=config.roi.width,
        height=config.roi.height,
        seed=51,
    )
    cv2.rectangle(pipe, (120, 0), (390, config.roi.height - 1), (110, 110, 110), -1)
    empty_pipe = pipe.copy()
    shadow_center = (250.0, ROI_CENTER[1])
    roi = render_ball(
        pipe,
        center_px=shadow_center,
        radius_px=config.detector.expected_radius_px,
        jpeg_quality=65,
        highlight_strength=120.0,
        noise_sigma=2.0,
    ).image
    frame = np.full(
        (config.camera.height, config.camera.width, 3),
        25,
        dtype=np.uint8,
    )
    x, y, width, height = config.roi.clamp(frame.shape[1], frame.shape[0])
    frame[y : y + height, x : x + width] = roi

    result = _process(frame, config, background_roi=empty_pipe)

    assert result.detection.valid, result.detection.reason
    assert result.detection.candidate_count == 1
    assert result.detection.candidate is not None
    assert abs(result.detection.candidate.center_px[0] - shadow_center[0]) < 1.5
