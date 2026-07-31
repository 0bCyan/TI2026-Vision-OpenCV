from __future__ import annotations

from pathlib import Path

import pytest

from ball_vision.config import (
    POSITION_CONVENTION_UNVERIFIED,
    POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE,
    AppConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mc02_serial_contract_accepts_fixed_v2_settings() -> None:
    config = AppConfig()
    config.output.serial_enabled = True
    config.output.serial_port = "/dev/ttyS4"
    config.output.serial_baudrate = 921600
    config.output.max_frame_age_us = 80_000
    config.camera.timestamp_offset_ms = 6.3
    config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("serial_baudrate", 115200, "921600"),
        ("serial_port", "", "serial_port"),
        ("max_frame_age_us", 80_001, "80000"),
    ],
)
def test_mc02_serial_contract_rejects_incompatible_output(
    field: str, value: object, message: str
) -> None:
    config = AppConfig()
    config.output.serial_enabled = True
    config.output.serial_port = "/dev/ttyS4"
    setattr(config.output, field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize("offset", [-0.001, float("nan"), float("inf")])
def test_timestamp_offset_must_be_non_negative_and_finite(offset: float) -> None:
    config = AppConfig()
    config.camera.timestamp_offset_ms = offset
    with pytest.raises(ValueError, match="timestamp_offset_ms"):
        config.validate()


def test_default_config_remains_explicitly_unverified() -> None:
    config = AppConfig.load(PROJECT_ROOT / "config" / "default.yaml")
    assert config.roi.verified is False
    assert config.calibration.coordinate_convention == POSITION_CONVENTION_UNVERIFIED


def test_position_contract_fields_survive_config_round_trip(tmp_path: Path) -> None:
    config = AppConfig()
    config.roi.verified = True
    config.calibration.coordinate_convention = (
        POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE
    )
    path = tmp_path / "production.yaml"

    config.save(path)
    loaded = AppConfig.load(path)

    assert loaded.roi.verified is True
    assert (
        loaded.calibration.coordinate_convention
        == POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE
    )


def test_orange_pi_launcher_requires_explicit_verified_inputs() -> None:
    script = (PROJECT_ROOT / "tools" / "run_orange_pi.sh").read_text(
        encoding="utf-8"
    )

    assert ': "${VISION_CONFIG:?' in script
    assert ': "${VISION_CAMERA_SOURCE:?' in script
    assert '--config "$CONFIG"' in script
    assert '--source "$VISION_CAMERA_SOURCE"' in script
    assert "config/default.yaml is a placeholder" in script
    assert "/dev/v4l/by-id/*)" in script
    assert '--config "$ROOT/config/default.yaml"' not in script


def test_production_config_is_not_tracked() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "config/production.yaml" in gitignore
