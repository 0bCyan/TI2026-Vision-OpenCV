from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CircleFit:
    center_x: float
    center_y: float
    radius: float
    rms: float
    inlier_ratio: float
    arc_coverage: float


def _circle_from_three(points: np.ndarray) -> tuple[float, float, float] | None:
    p1, p2, p3 = points
    matrix = np.array(
        [
            [2.0 * (p2[0] - p1[0]), 2.0 * (p2[1] - p1[1])],
            [2.0 * (p3[0] - p1[0]), 2.0 * (p3[1] - p1[1])],
        ],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-7:
        return None
    rhs = np.array(
        [
            p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1],
            p3[0] * p3[0] + p3[1] * p3[1] - p1[0] * p1[0] - p1[1] * p1[1],
        ],
        dtype=np.float64,
    )
    center = np.linalg.solve(matrix, rhs)
    radius = float(np.linalg.norm(p1 - center))
    return float(center[0]), float(center[1]), radius


def _least_squares_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    matrix = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    rhs = points[:, 0] ** 2 + points[:, 1] ** 2
    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center_x, center_y, constant = solution
    radius_sq = constant + center_x * center_x + center_y * center_y
    if radius_sq <= 0:
        return None
    return float(center_x), float(center_y), float(np.sqrt(radius_sq))


def fit_circle_ransac(
    contour_points: np.ndarray,
    radius_min: float,
    radius_max: float,
    iterations: int = 40,
    residual_threshold: float = 2.2,
    sample_limit: int = 240,
) -> CircleFit | None:
    points = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 6:
        return None
    if len(points) > sample_limit:
        indices = np.linspace(0, len(points) - 1, sample_limit, dtype=np.int32)
        points = points[indices]

    # A stable seed makes offline comparisons reproducible without making the
    # sampled triplets depend on thread scheduling.
    seed = int((points[0, 0] * 73856093 + points[0, 1] * 19349663 + len(points)) % (2**32))
    rng = np.random.default_rng(seed)
    iteration_count = max(8, int(iterations))
    # Generate extra triplets once, discard repeated indices, and solve all
    # circumcircles in one NumPy batch. This is materially faster than a
    # Python loop at 180 Hz while retaining deterministic RANSAC behaviour.
    triplets = rng.integers(0, len(points), size=(iteration_count * 3, 3))
    distinct = (
        (triplets[:, 0] != triplets[:, 1])
        & (triplets[:, 0] != triplets[:, 2])
        & (triplets[:, 1] != triplets[:, 2])
    )
    triplets = triplets[distinct][:iteration_count]
    if len(triplets) == 0:
        return None
    samples = points[triplets]
    x1, y1 = samples[:, 0, 0], samples[:, 0, 1]
    x2, y2 = samples[:, 1, 0], samples[:, 1, 1]
    x3, y3 = samples[:, 2, 0], samples[:, 2, 1]
    denominator = 2.0 * (
        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    )
    valid_denominator = np.abs(denominator) > 1e-7
    safe_denominator = np.where(valid_denominator, denominator, 1.0)
    square1 = x1 * x1 + y1 * y1
    square2 = x2 * x2 + y2 * y2
    square3 = x3 * x3 + y3 * y3
    center_x_all = (
        square1 * (y2 - y3)
        + square2 * (y3 - y1)
        + square3 * (y1 - y2)
    ) / safe_denominator
    center_y_all = (
        square1 * (x3 - x2)
        + square2 * (x1 - x3)
        + square3 * (x2 - x1)
    ) / safe_denominator
    radius_all = np.hypot(x1 - center_x_all, y1 - center_y_all)
    valid_model = (
        valid_denominator
        & np.isfinite(radius_all)
        & (radius_all >= radius_min)
        & (radius_all <= radius_max)
    )
    if not np.any(valid_model):
        return None
    center_x_all = center_x_all[valid_model]
    center_y_all = center_y_all[valid_model]
    radius_all = radius_all[valid_model]
    residual_matrix = np.abs(
        np.hypot(
            points[None, :, 0] - center_x_all[:, None],
            points[None, :, 1] - center_y_all[:, None],
        )
        - radius_all[:, None]
    )
    inlier_matrix = residual_matrix <= residual_threshold
    counts = np.count_nonzero(inlier_matrix, axis=1)
    if np.max(counts, initial=0) < 6:
        return None
    mean_inlier_residuals = np.sum(
        np.where(inlier_matrix, residual_matrix, 0.0), axis=1
    ) / np.maximum(1, counts)
    scores = counts - 0.35 * mean_inlier_residuals
    best_index = int(np.argmax(scores))
    best_inliers = inlier_matrix[best_index]
    if np.count_nonzero(best_inliers) < 6:
        return None
    refined = _least_squares_circle(points[best_inliers])
    if refined is None:
        return None
    center_x, center_y, radius = refined
    if not radius_min <= radius <= radius_max:
        return None
    residuals = np.abs(np.hypot(points[:, 0] - center_x, points[:, 1] - center_y) - radius)
    inliers = residuals <= residual_threshold
    if np.count_nonzero(inliers) >= 6:
        second_refinement = _least_squares_circle(points[inliers])
        if second_refinement is not None:
            center_x, center_y, radius = second_refinement
            if (
                not np.isfinite(center_x)
                or not np.isfinite(center_y)
                or not np.isfinite(radius)
                or not radius_min <= radius <= radius_max
            ):
                return None
            residuals = np.abs(np.hypot(points[:, 0] - center_x, points[:, 1] - center_y) - radius)
            inliers = residuals <= residual_threshold
    if np.count_nonzero(inliers) < 6:
        return None
    inlier_residuals = residuals[inliers]
    if len(inlier_residuals) == 0:
        return None
    inlier_points = points[inliers]
    angles = np.mod(
        np.arctan2(inlier_points[:, 1] - center_y, inlier_points[:, 0] - center_x),
        2.0 * np.pi,
    )
    # Occupied angular bins measure actual supported circumference. The old
    # largest-gap formula badly overestimated two tiny arcs 180 degrees apart.
    angular_bin_count = 36
    angular_bins = np.floor(angles * angular_bin_count / (2.0 * np.pi)).astype(np.int32)
    angular_bins = np.clip(angular_bins, 0, angular_bin_count - 1)
    arc_coverage = float(len(np.unique(angular_bins)) / angular_bin_count)
    return CircleFit(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        rms=float(np.sqrt(np.mean(inlier_residuals * inlier_residuals))),
        inlier_ratio=float(np.count_nonzero(inliers) / len(points)),
        arc_coverage=arc_coverage,
    )
