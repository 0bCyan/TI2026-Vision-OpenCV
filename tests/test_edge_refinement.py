from __future__ import annotations

import cv2
import numpy as np
import pytest

from ball_vision.circle_fit import CircleFit
from ball_vision.config import AppConfig, DetectorConfig, ThresholdConfig
from ball_vision.detector import BallDetector


def _detector(method: str = "radial") -> BallDetector:
    return BallDetector(
        DetectorConfig(
            threshold=ThresholdConfig(polarity="dark"),
            radius_min_px=10.0,
            radius_max_px=30.0,
            expected_radius_px=20.0,
            edge_refine_method=method,
            edge_annulus_ratio=0.35,
            edge_center_shift_max_ratio=0.10,
            edge_radius_shift_max_ratio=0.10,
            edge_radial_spokes=72,
            edge_radial_step_px=0.35,
            edge_radius_prior_sigma_px=1.0,
            edge_radial_gradient_min=3.0,
            edge_min_spoke_ratio=0.45,
            edge_radial_residual_px=0.8,
            arc_coverage_min=0.25,
        )
    )


def test_radial_refinement_does_not_switch_to_stronger_inner_ring() -> None:
    gray = np.full((128, 128), 220, dtype=np.uint8)
    # The inner 20 -> 160 transition is stronger than the physical outer
    # 160 -> 220 transition and has the same outward gradient polarity.
    cv2.circle(gray, (64, 64), 20, 160, -1, cv2.LINE_AA)
    cv2.circle(gray, (64, 64), 16, 20, -1, cv2.LINE_AA)
    initial = CircleFit(
        center_x=64.2,
        center_y=63.8,
        radius=20.1,
        rms=0.5,
        inlier_ratio=0.9,
        arc_coverage=1.0,
    )

    refinement = _detector()._refine_from_edges(gray, initial, iterations=24)

    assert refinement.attempted
    assert refinement.accepted
    assert refinement.support_ratio > 0.9
    assert refinement.fit.radius == pytest.approx(20.0, abs=0.7)
    assert refinement.fit.radius > 18.5
    assert np.hypot(
        refinement.fit.center_x - 64.0,
        refinement.fit.center_y - 64.0,
    ) < 0.5


def test_canny_fallback_rejects_inconsistent_inner_ring(monkeypatch) -> None:
    detector = _detector(method="canny")
    initial = CircleFit(64.0, 64.0, 20.0, 0.5, 0.9, 1.0)
    inner_ring = CircleFit(67.5, 64.0, 16.5, 0.2, 1.0, 1.0)
    monkeypatch.setattr(
        detector,
        "_refine_from_canny_edges",
        lambda *_args, **_kwargs: inner_ring,
    )

    refinement = detector._refine_from_edges(
        np.zeros((128, 128), dtype=np.uint8),
        initial,
        iterations=24,
    )

    assert refinement.attempted
    assert not refinement.accepted
    assert refinement.fit is initial
    assert refinement.center_shift_px == pytest.approx(3.5)
    assert refinement.radius_shift_px == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("edge_refine_method", "unknown"),
        ("edge_radial_spokes", 23),
        ("edge_radial_step_px", 0.0),
        ("edge_radius_prior_sigma_px", 0.0),
        ("edge_recovery_prior_sigma_px", 0.0),
        ("edge_radial_gradient_min", 0.0),
        ("edge_radial_residual_px", 0.0),
        ("edge_center_shift_max_ratio", 1.01),
        ("edge_radius_shift_max_ratio", -0.01),
        ("edge_min_spoke_ratio", float("nan")),
        ("radius_continuity_max_ratio", 1.01),
        ("edge_coarse_radius_outlier_ratio", 1.01),
        ("edge_recovery_shift_max_ratio", 1.01),
        ("edge_temporal_radius_weight", float("nan")),
    ],
)
def test_edge_refinement_configuration_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    config = AppConfig()
    setattr(config.detector, field, value)

    with pytest.raises((TypeError, ValueError)):
        config.validate()
