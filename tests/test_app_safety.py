from types import SimpleNamespace

import pytest

from ball_vision.app import VisionApplication
from ball_vision.config import AppConfig, RoiConfig
from ball_vision.runtime_stats import RuntimeStats


def make_ready_config() -> AppConfig:
    config = AppConfig()
    config.roi = RoiConfig(x=10, y=10, width=100, height=50)
    config.detector.threshold.background_enabled = False
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
