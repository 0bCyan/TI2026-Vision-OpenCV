from __future__ import annotations

import pytest

from ball_vision.config import AppConfig


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
