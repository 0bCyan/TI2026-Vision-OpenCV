import numpy as np

from ball_vision.config import TrackerConfig
from ball_vision.tracker import BallTracker
from ball_vision.types import TrackFlags


def test_tracker_state_transitions_and_motion_accuracy() -> None:
    rng = np.random.default_rng(4)
    tracker = BallTracker(
        TrackerConfig(
            process_accel_sigma_mm_s2=1200,
            measurement_sigma_mm=0.5,
            confirmation_frames=2,
            max_prediction_ms=22.0,
            lost_reset_ms=60.0,
        )
    )
    start_ns = 1_000_000_000
    estimates = []
    for frame in range(80):
        time_s = frame / 180.0
        truth = -30.0 + 140.0 * time_s
        measurement = truth + rng.normal(0.0, 0.35)
        if 35 <= frame <= 37:
            measurement = None
        estimate = tracker.update(
            measurement,
            confidence=0.9 if measurement is not None else 0.0,
            frame_time_ns=start_ns + int(time_s * 1e9),
        )
        estimates.append((truth, estimate))
    valid_errors = [abs(truth - estimate.position_mm) for truth, estimate in estimates[10:] if estimate.valid]
    assert np.percentile(valid_errors, 95) < 1.5
    assert estimates[35][1].flags & TrackFlags.PREDICTED
    assert estimates[37][1].flags & TrackFlags.PREDICTED
    assert estimates[38][1].flags & TrackFlags.REACQUIRED


def test_tracker_does_not_report_prediction_as_measurement() -> None:
    tracker = BallTracker(TrackerConfig(confirmation_frames=1, max_prediction_ms=20.0))
    tracker.update(10.0, 1.0, 1_000_000_000)
    predicted = tracker.update(None, 0.0, 1_006_000_000)
    assert predicted.flags & TrackFlags.PREDICTED
    assert not predicted.valid
    assert predicted.measurement_mm is None


def test_tracker_reacquires_large_jump_after_prediction_expires() -> None:
    tracker = BallTracker(
        TrackerConfig(
            confirmation_frames=2,
            max_prediction_ms=20.0,
            lost_reset_ms=80.0,
            gate_min_mm=5.0,
        )
    )
    start_ns = 2_000_000_000
    tracker.update(0.0, 0.95, start_ns)
    initialized = tracker.update(0.2, 0.95, start_ns + 6_000_000)
    assert initialized.valid
    assert not initialized.flags & TrackFlags.REACQUIRED
    lost = tracker.update(None, 0.0, start_ns + 30_000_000)
    assert lost.flags & TrackFlags.LOST
    assert not tracker.prediction_is_fresh(start_ns + 30_000_000)

    first_candidate = tracker.update(100.0, 0.9, start_ns + 36_000_000)
    assert not first_candidate.valid
    reacquired = tracker.update(101.0, 0.9, start_ns + 42_000_000)
    assert reacquired.flags & TrackFlags.REACQUIRED
    assert reacquired.valid
    np.testing.assert_allclose(reacquired.position_mm, 100.5, atol=0.01)


def test_tracker_ignores_duplicate_and_out_of_order_timestamps() -> None:
    tracker = BallTracker(TrackerConfig(confirmation_frames=1))
    first_time = 3_000_000_000
    first = tracker.update(5.0, 0.9, first_time)
    assert first.valid
    next_estimate = tracker.update(6.0, 0.9, first_time + 10_000_000)
    position_before = next_estimate.position_mm

    stale = tracker.update(100.0, 1.0, first_time + 5_000_000)
    assert not stale.valid
    assert stale.flags & TrackFlags.PREDICTED
    assert stale.frame_time_ns == first_time + 10_000_000
    assert tracker.position_mm == position_before
