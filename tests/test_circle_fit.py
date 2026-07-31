import numpy as np
import pytest

from ball_vision.circle_fit import fit_circle_ransac


def test_ransac_circle_with_partial_arc_and_outliers() -> None:
    rng = np.random.default_rng(42)
    center = np.array([123.4, 56.7])
    radius = 24.0
    angles = np.linspace(-0.8 * np.pi, 0.75 * np.pi, 160)
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    points += rng.normal(0.0, 0.25, points.shape)
    outliers = rng.uniform([75, 20], [175, 95], size=(25, 2))
    fit = fit_circle_ransac(
        np.vstack((points, outliers)),
        radius_min=15,
        radius_max=35,
        iterations=100,
        residual_threshold=1.0,
    )
    assert fit is not None
    assert fit.center_x == pytest.approx(center[0], abs=0.35)
    assert fit.center_y == pytest.approx(center[1], abs=0.35)
    assert fit.radius == pytest.approx(radius, abs=0.35)
    assert fit.arc_coverage > 0.70


def test_circle_fit_rejects_collinear_points() -> None:
    points = np.column_stack((np.arange(30), np.arange(30) * 2.0))
    assert fit_circle_ransac(points, 2, 100, iterations=20) is None


def test_circle_fit_never_returns_radius_outside_contract() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(30):
        points = rng.uniform([0.0, 0.0], [100.0, 70.0], size=(90, 2))
        fit = fit_circle_ransac(
            points,
            radius_min=10.0,
            radius_max=20.0,
            iterations=80,
            residual_threshold=2.5,
        )
        assert fit is None or 10.0 <= fit.radius <= 20.0


def test_opposite_short_arcs_do_not_fake_half_circle_coverage() -> None:
    center = np.array([50.0, 40.0])
    radius = 20.0
    half_width = np.deg2rad(2.5)
    angles = np.concatenate(
        (
            np.linspace(-half_width, half_width, 60),
            np.linspace(np.pi - half_width, np.pi + half_width, 60),
        )
    )
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    fit = fit_circle_ransac(
        points,
        radius_min=18.0,
        radius_max=22.0,
        iterations=120,
        residual_threshold=0.5,
    )
    assert fit is not None
    assert fit.arc_coverage < 0.15
