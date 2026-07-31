from __future__ import annotations

import csv
import time
from dataclasses import replace

import numpy as np
import pytest

from ball_vision.config import (
    AppConfig,
    CalibrationConfig,
    LoggingConfig,
    MeasurementConfig,
    RoiConfig,
    TrackerConfig,
)
from ball_vision.pipeline import VisionPipeline
from ball_vision.telemetry_log import CsvTelemetryLogger
from ball_vision.types import CircleCandidate, DetectionResult, FramePacket


def _config(measurement: MeasurementConfig | None = None) -> AppConfig:
    config = AppConfig(
        roi=RoiConfig(x=0, y=0, width=101, height=40),
        calibration=CalibrationConfig(
            mode="piecewise_linear",
            degree=1,
            points=[[0.0, -50.0], [100.0, 50.0]],
        ),
        measurement=measurement
        or MeasurementConfig(
            minimum_center_sigma_px=0.5,
            calibration_rms_mm=0.4,
            near_edge_margin_mm=10.0,
            near_edge_multiplier=2.0,
            maximum_valid_sigma_mm=3.0,
        ),
        tracker=TrackerConfig(confirmation_frames=1),
    )
    config.detector.threshold.background_enabled = False
    config.validate()
    return config


def _candidate(center_x_px: float, fit_rms_px: float) -> CircleCandidate:
    return CircleCandidate(
        center_px=(center_x_px, 20.0),
        radius_px=16.0,
        confidence=0.9,
        score=1.0,
        circularity=0.9,
        fill_ratio=0.75,
        fit_rms_px=fit_rms_px,
        inlier_ratio=0.9,
        arc_coverage=0.9,
        area_px=800.0,
        bbox=(0, 0, 32, 32),
        source_mask=1,
    )


def _process_candidate(config: AppConfig, candidate: CircleCandidate):
    pipeline = VisionPipeline(config)
    pipeline.detector.detect = lambda *_: DetectionResult(
        valid=True,
        candidate=candidate,
        threshold_value=100.0,
        candidate_count=1,
        reason="ok",
    )
    frame_time_ns = time.monotonic_ns()
    return pipeline.process(
        FramePacket(
            sequence=7,
            arrival_ns=frame_time_ns,
            frame_time_ns=frame_time_ns,
            read_duration_ns=0,
            image=np.zeros((40, 101, 3), dtype=np.uint8),
        )
    )


def test_predictive_window_miss_forces_next_frame_global_search() -> None:
    pipeline = VisionPipeline(_config())
    predicted_inputs: list[tuple[float | None, float | None]] = []
    detections = [
        DetectionResult(
            valid=True,
            candidate=_candidate(50.0, 0.5),
            threshold_value=100.0,
            candidate_count=1,
            reason="ok",
        ),
        DetectionResult(
            valid=False,
            candidate=None,
            threshold_value=100.0,
            candidate_count=0,
            reason="missing",
        ),
        DetectionResult(
            valid=False,
            candidate=None,
            threshold_value=100.0,
            candidate_count=0,
            reason="missing",
        ),
    ]

    def detect(
        _image,
        predicted_x=None,
        _predicted_y=None,
        predicted_radius=None,
    ):
        predicted_inputs.append((predicted_x, predicted_radius))
        return detections.pop(0)

    pipeline.detector.detect = detect
    start_ns = time.monotonic_ns()
    for index in range(3):
        frame_time_ns = start_ns + index * 5_000_000
        pipeline.process(
            FramePacket(
                sequence=index + 1,
                arrival_ns=frame_time_ns,
                frame_time_ns=frame_time_ns,
                read_duration_ns=0,
                image=np.zeros((40, 101, 3), dtype=np.uint8),
            )
        )

    assert predicted_inputs[0] == (None, None)
    assert predicted_inputs[1][0] is not None
    assert predicted_inputs[1][1] == pytest.approx(16.0)
    assert predicted_inputs[2] == (None, None)


def test_raw_sigma_combines_fit_and_calibration_noise() -> None:
    result = _process_candidate(_config(), _candidate(50.0, fit_rms_px=0.75))

    measurement = result.raw_measurement
    assert measurement.valid
    assert measurement.position_mm == pytest.approx(0.0)
    assert measurement.sigma_mm == pytest.approx(0.85)
    assert measurement.global_x_px == pytest.approx(50.0)
    assert measurement.frame_time_ns == result.frame.frame_time_ns
    assert not measurement.near_edge
    assert measurement.reason == "ok"
    assert result.raw_position_mm == measurement.position_mm
    assert result.raw_global_x_px == measurement.global_x_px


def test_center_sigma_floor_and_near_edge_multiplier_are_applied() -> None:
    result = _process_candidate(_config(), _candidate(5.0, fit_rms_px=0.1))

    measurement = result.raw_measurement
    assert measurement.valid
    assert measurement.near_edge
    assert measurement.sigma_mm == pytest.approx(np.hypot(0.5, 0.4) * 2.0)


def test_excessive_raw_sigma_is_not_exported_but_tracker_stays_available() -> None:
    measurement_config = MeasurementConfig(
        minimum_center_sigma_px=0.5,
        calibration_rms_mm=0.4,
        near_edge_margin_mm=10.0,
        near_edge_multiplier=2.0,
        maximum_valid_sigma_mm=1.0,
    )
    result = _process_candidate(
        _config(measurement_config),
        _candidate(5.0, fit_rms_px=0.75),
    )

    measurement = result.raw_measurement
    assert not measurement.valid
    assert measurement.reason == "sigma_above_maximum"
    assert measurement.sigma_mm is not None and measurement.sigma_mm > 1.0
    assert result.estimate.valid
    assert result.estimate.measurement_mm == pytest.approx(measurement.position_mm)


def test_non_finite_fit_uncertainty_never_enters_tracker() -> None:
    result = _process_candidate(_config(), _candidate(50.0, fit_rms_px=float("nan")))

    assert not result.raw_measurement.valid
    assert result.raw_measurement.reason == "non_finite_raw_measurement"
    assert not result.estimate.valid
    assert result.estimate.measurement_mm is None


def test_invalid_detection_has_no_raw_numeric_payload() -> None:
    pipeline = VisionPipeline(_config())
    pipeline.detector.detect = lambda *_: DetectionResult(
        valid=False,
        candidate=None,
        threshold_value=100.0,
        candidate_count=0,
        reason="no_geometrically_valid_candidate",
    )
    frame_time_ns = time.monotonic_ns()
    result = pipeline.process(
        FramePacket(
            sequence=8,
            arrival_ns=frame_time_ns,
            frame_time_ns=frame_time_ns,
            read_duration_ns=0,
            image=np.zeros((40, 101, 3), dtype=np.uint8),
        )
    )

    measurement = result.raw_measurement
    assert not measurement.valid
    assert measurement.position_mm is None
    assert measurement.sigma_mm is None
    assert measurement.global_x_px is None
    assert measurement.confidence == 0.0
    assert not measurement.near_edge
    assert measurement.reason == "no_geometrically_valid_candidate"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_center_sigma_px", 0.0),
        ("calibration_rms_mm", 0.0),
        ("near_edge_margin_mm", -0.1),
        ("near_edge_multiplier", 0.99),
        ("maximum_valid_sigma_mm", 0.0),
        ("maximum_valid_sigma_mm", 655.36),
        ("calibration_rms_mm", float("nan")),
    ],
)
def test_measurement_configuration_rejects_invalid_values(field: str, value: float) -> None:
    config = _config()
    config.measurement = replace(config.measurement, **{field: value})
    with pytest.raises(ValueError):
        config.validate()


def test_measurement_configuration_rejects_unusable_near_edge_floor() -> None:
    config = _config()
    config.measurement = replace(
        config.measurement,
        calibration_rms_mm=1.0,
        near_edge_multiplier=2.0,
        maximum_valid_sigma_mm=1.9,
    )
    with pytest.raises(ValueError, match="noise floor"):
        config.validate()


def test_default_yaml_declares_measurement_noise_model() -> None:
    config = AppConfig.load("config/default.yaml")
    assert config.measurement.minimum_center_sigma_px > 0
    assert config.measurement.calibration_rms_mm > 0
    assert config.measurement.near_edge_multiplier >= 1
    assert config.measurement.maximum_valid_sigma_mm > config.measurement.calibration_rms_mm


def test_csv_log_includes_edge_refinement_diagnostics(tmp_path) -> None:
    candidate = _candidate(50.0, fit_rms_px=0.4)
    candidate.initial_center_px = (49.7, 19.8)
    candidate.initial_radius_px = 15.8
    candidate.edge_refinement_attempted = True
    candidate.edge_refinement_accepted = True
    candidate.edge_support_ratio = 0.75
    candidate.edge_center_shift_px = 0.36
    candidate.edge_radius_shift_px = 0.20
    result = _process_candidate(_config(), candidate)
    logger = CsvTelemetryLogger(
        LoggingConfig(enabled=True, directory="."),
        tmp_path,
    )
    assert logger.path is not None

    logger.write(result, skipped_frames=0)
    logger.close()

    with logger.path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["raw_center_y_px"]) == pytest.approx(20.0)
    assert float(row["initial_center_x_px"]) == pytest.approx(49.7)
    assert float(row["initial_center_y_px"]) == pytest.approx(19.8)
    assert float(row["initial_radius_px"]) == pytest.approx(15.8)
    assert row["edge_refinement_attempted"] == "1"
    assert row["edge_refinement_accepted"] == "1"
    assert float(row["edge_support_ratio"]) == pytest.approx(0.75)
    assert float(row["edge_center_shift_px"]) == pytest.approx(0.36)
    assert float(row["edge_radius_shift_px"]) == pytest.approx(0.20)
