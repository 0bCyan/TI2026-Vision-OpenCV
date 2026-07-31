from types import SimpleNamespace

import pytest

from ball_vision.app import VisionApplication
from ball_vision.config import (
    POSITION_CONVENTION_UNVERIFIED,
    POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE,
    AppConfig,
    RoiConfig,
)
from ball_vision.runtime_stats import RuntimeStats


def make_ready_config() -> AppConfig:
    config = AppConfig()
    config.roi = RoiConfig(x=10, y=10, width=100, height=50, verified=True)
    config.detector.threshold.background_enabled = False
    config.calibration.coordinate_convention = (
        POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE
    )
    config.calibration.points = [
        [10.0, -20.0],
        [30.0, -10.0],
        [50.0, 0.0],
        [70.0, 10.0],
        [90.0, 20.0],
    ]
    config.camera.controls = {}
    config.camera.controls_verified_externally = True
    return config


def verify_setup(config: AppConfig) -> None:
    application = object.__new__(VisionApplication)
    application.config = config
    application._verify_vision_setup()


def test_production_setup_rejects_unverified_frame_freshness() -> None:
    config = make_ready_config()
    with pytest.raises(RuntimeError, match="可信帧新鲜度"):
        verify_setup(config)


@pytest.mark.parametrize("freshness_mode", ["content", "external"])
def test_production_setup_accepts_explicit_freshness_strategy(
    freshness_mode: str,
) -> None:
    config = make_ready_config()
    if freshness_mode == "content":
        config.camera.fingerprint_frames = True
        config.camera.reject_identical_frames = True
    else:
        config.camera.freshness_verified_externally = True
    verify_setup(config)


def test_production_setup_rejects_unverified_roi() -> None:
    config = make_ready_config()
    config.camera.freshness_verified_externally = True
    config.roi.verified = False
    with pytest.raises(RuntimeError, match="ROI尚未通过实车选择确认"):
        verify_setup(config)


def test_production_setup_rejects_unverified_position_convention() -> None:
    config = make_ready_config()
    config.camera.freshness_verified_externally = True
    config.calibration.coordinate_convention = POSITION_CONVENTION_UNVERIFIED
    with pytest.raises(RuntimeError, match="位置正方向未确认"):
        verify_setup(config)


@pytest.mark.parametrize(
    ("millimeters", "message"),
    [
        ([-20.0, -10.0, -1.0, 10.0, 20.0], "缺少横梁机械中心0 mm"),
        ([0.0, 5.0, 10.0, 15.0, 20.0], "必须同时覆盖车尾负坐标和车头正坐标"),
        ([-126.0, -60.0, 0.0, 60.0, 125.0], "超出MC02允许"),
        ([-20.0, -10.0, 0.0, 10.0, float("nan")], "必须全部是有限数"),
    ],
)
def test_production_setup_rejects_unsafe_calibration_extent(
    millimeters: list[float], message: str
) -> None:
    config = make_ready_config()
    config.camera.freshness_verified_externally = True
    config.calibration.points = [
        [float(index * 10), position]
        for index, position in enumerate(millimeters)
    ]
    with pytest.raises(RuntimeError, match=message):
        verify_setup(config)


def test_production_v4l2_serial_rejects_unstable_camera_path() -> None:
    config = make_ready_config()
    config.camera.backend = "v4l2"
    config.camera.source = "/dev/video0"
    config.camera.freshness_verified_externally = True
    config.output.serial_enabled = True
    with pytest.raises(RuntimeError, match="稳定的/dev/v4l/by-id"):
        verify_setup(config)


def test_production_v4l2_serial_accepts_stable_camera_path() -> None:
    config = make_ready_config()
    config.camera.backend = "v4l2"
    config.camera.source = "/dev/v4l/by-id/usb-camera-video-index0"
    config.camera.freshness_verified_externally = True
    config.output.serial_enabled = True
    verify_setup(config)


def test_production_gstreamer_serial_requires_stable_camera_path() -> None:
    config = make_ready_config()
    config.camera.backend = "gstreamer"
    config.camera.gstreamer_pipeline = "v4l2src device=/dev/video0 ! appsink"
    config.camera.freshness_verified_externally = True
    config.output.serial_enabled = True
    with pytest.raises(RuntimeError, match="正式GStreamer"):
        verify_setup(config)


def test_strict_video_replay_does_not_require_live_device_path() -> None:
    config = make_ready_config()
    config.camera.backend = "auto"
    config.camera.source = "data/recorded-test.mkv"
    config.output.serial_enabled = True
    verify_setup(config)


def test_runtime_measured_rate_tracks_raw_protocol_measurements() -> None:
    application = object.__new__(VisionApplication)
    application.stats = RuntimeStats()
    result = SimpleNamespace(
        processing_us=1000,
        frame_age_us=2000,
        raw_measurement=SimpleNamespace(valid=False),
        estimate=SimpleNamespace(valid=True),
    )

    application._record_runtime_stats(result, skipped=0)

    assert application.stats.summary().measured_rate == 0.0


def test_bench_serial_is_explicit_and_does_not_claim_production_setup() -> None:
    config = AppConfig()
    config.output.serial_enabled = True
    config.output.serial_port = "/dev/null"
    config.detector.threshold.background_enabled = False
    config.display.enabled = False
    config.logging.enabled = False

    application = VisionApplication(config, bench_serial=True)
    try:
        assert not config.camera.strict_contract
        assert not application._production_output
    finally:
        application.close()


def test_bench_serial_requires_serial_output() -> None:
    config = AppConfig()
    with pytest.raises(ValueError, match="启用串口输出"):
        VisionApplication(config, bench_serial=True)
