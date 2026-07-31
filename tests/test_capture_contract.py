import numpy as np
import pytest

from ball_vision.capture import LatestFrameCapture
from ball_vision.config import AppConfig, CameraConfig
from ball_vision.types import FramePacket


def make_capture(interval_ms: float, shape: tuple[int, int], read_ms: float = 5.0):
    config = CameraConfig(
        source=0,
        width=1280,
        height=720,
        fps=180.0,
        strict_contract=True,
    )
    capture = LatestFrameCapture(config)
    image = np.zeros(shape, dtype=np.uint8)
    capture._latest = FramePacket(
        sequence=100,
        arrival_ns=2_000_000_000,
        frame_time_ns=2_000_000_000,
        read_duration_ns=int(read_ms * 1e6),
        image=image,
    )
    capture._received = 101
    capture._last_arrival_ns = 2_000_000_000
    capture._intervals_ns.extend([int(interval_ms * 1e6)] * 100)
    capture._read_times_ns.extend([int(read_ms * 1e6)] * 100)
    return capture


def test_capture_contract_accepts_stable_180_hz_stream() -> None:
    capture = make_capture(1000.0 / 180.0, (720, 1280))
    capture.verify_contract()


def test_capture_contract_rejects_wrong_shape_and_bursty_stream() -> None:
    capture = make_capture(20.0, (480, 640), read_ms=12.0)
    capture._read_failures = 1
    with pytest.raises(RuntimeError, match="摄像头合同未通过"):
        capture.verify_contract()


def test_capture_contract_rejects_duplicate_frames_when_fingerprinted() -> None:
    capture = make_capture(1000.0 / 180.0, (720, 1280))
    capture.config.fingerprint_frames = True
    capture._duplicate_fingerprints = 10
    with pytest.raises(RuntimeError, match="重复帧率过高"):
        capture.verify_contract()


def test_external_camera_control_mode_is_mutually_exclusive() -> None:
    config = CameraConfig(
        controls_verified_externally=True,
        controls={"exposure": -8.0},
    )
    app_config = AppConfig(camera=config)
    with pytest.raises(ValueError, match="外部控制模式"):
        app_config.validate()
