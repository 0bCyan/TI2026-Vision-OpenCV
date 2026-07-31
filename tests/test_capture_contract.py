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
    capture._duplicate_window.extend([0] * 90 + [1] * 10)
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


def test_identical_frame_rejection_requires_fingerprints() -> None:
    app_config = AppConfig(
        camera=CameraConfig(
            fingerprint_frames=False,
            reject_identical_frames=True,
        )
    )
    with pytest.raises(ValueError, match="fingerprint_frames"):
        app_config.validate()


def test_enabled_identical_frame_guard_does_not_advance_sequence() -> None:
    config = CameraConfig(
        source=0,
        warmup_seconds=0.0,
        fingerprint_frames=True,
        reject_identical_frames=True,
    )
    capture = LatestFrameCapture(config)
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    class FrozenBackend:
        def __init__(self) -> None:
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls >= 3:
                capture._running = False
            return True, image

    capture._capture = FrozenBackend()
    capture._running = True
    capture._run()

    assert capture.stats().received_frames == 3
    assert capture.stats().duplicate_fingerprints == 2
    assert capture._latest is not None
    assert capture._latest.sequence == 1


def test_capture_thread_exception_becomes_fatal_state() -> None:
    capture = LatestFrameCapture(CameraConfig(source=0, warmup_seconds=0.0))

    class ExplodingBackend:
        def read(self):
            raise RuntimeError("backend exploded")

    capture._capture = ExplodingBackend()
    capture._running = True
    capture._run()

    assert not capture.running
    assert capture._fatal_error is not None
    with pytest.raises(RuntimeError, match="采集线程失败"):
        capture.latest(timeout=0.01)


def test_contract_uses_bounded_recent_failure_window() -> None:
    capture = make_capture(1000.0 / 180.0, (720, 1280))
    capture._read_failures = 100
    capture._read_failure_window.extend([0] * 100)
    capture.verify_contract()

    capture._read_failure_window.append(1)
    capture.config.max_read_failures = 0
    with pytest.raises(RuntimeError, match="最近采集窗口读取失败"):
        capture.verify_contract()
