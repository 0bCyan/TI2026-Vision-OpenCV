from __future__ import annotations

import numpy as np

from .config import TrackerConfig
from .types import TrackEstimate, TrackFlags


class BallTracker:
    """Variable-dt Kalman tracker in physical millimeter coordinates."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._state = np.zeros((2, 1), dtype=np.float64)
        self._covariance = np.eye(2, dtype=np.float64)
        self._initialized = False
        self._last_time_ns = 0
        self._last_update_time_ns = 0
        self._last_measurement_ns = 0
        self._predicted_for_ns = 0
        self._lost_frames = 0
        self._confirmation_count = 0
        self._confirmation_position = 0.0
        self._confirmation_time_ns = 0
        self._ever_initialized = False
        self._was_lost = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def position_mm(self) -> float:
        return float(self._state[0, 0])

    @property
    def velocity_mm_s(self) -> float:
        return float(self._state[1, 0])

    def prediction_is_fresh(self, frame_time_ns: int) -> bool:
        if not self._initialized or not self._last_measurement_ns:
            return False
        age_ns = int(frame_time_ns) - self._last_measurement_ns
        return 0 <= age_ns <= int(self.config.max_prediction_ms * 1e6)

    def reset(self) -> None:
        self._state.fill(0.0)
        self._covariance[:] = np.eye(2)
        self._initialized = False
        self._last_time_ns = 0
        self._last_measurement_ns = 0
        self._predicted_for_ns = 0
        self._lost_frames = 0
        self._confirmation_count = 0
        self._confirmation_position = 0.0
        self._confirmation_time_ns = 0
        self._was_lost = True

    def _initialize(self, position_mm: float, frame_time_ns: int) -> None:
        self._state[:, 0] = (position_mm, 0.0)
        self._covariance = np.diag(
            [
                self.config.initial_position_sigma_mm**2,
                self.config.initial_velocity_sigma_mm_s**2,
            ]
        ).astype(np.float64)
        self._initialized = True
        self._ever_initialized = True
        self._last_time_ns = int(frame_time_ns)
        self._last_measurement_ns = int(frame_time_ns)
        self._predicted_for_ns = int(frame_time_ns)
        self._lost_frames = 0

    def predict(self, frame_time_ns: int) -> tuple[float, float] | None:
        if not self._initialized:
            return None
        frame_time_ns = int(frame_time_ns)
        if frame_time_ns <= self._last_time_ns:
            return self.position_mm, self.velocity_mm_s
        if self._predicted_for_ns == frame_time_ns:
            return self.position_mm, self.velocity_mm_s
        raw_dt = (frame_time_ns - self._last_time_ns) / 1e9
        dt = float(np.clip(raw_dt, self.config.min_dt_seconds, self.config.max_dt_seconds))
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        noise_gain = np.array([[0.5 * dt * dt], [dt]], dtype=np.float64)
        process_noise = (
            self.config.process_accel_sigma_mm_s2**2
            * (noise_gain @ noise_gain.T)
        )
        self._state = transition @ self._state
        self._state[1, 0] = float(
            np.clip(
                self._state[1, 0],
                -self.config.velocity_limit_mm_s,
                self.config.velocity_limit_mm_s,
            )
        )
        self._covariance = transition @ self._covariance @ transition.T + process_noise
        self._last_time_ns = frame_time_ns
        self._predicted_for_ns = frame_time_ns
        return self.position_mm, self.velocity_mm_s

    def _search_confirmation(self, measurement_mm: float, frame_time_ns: int) -> bool:
        if self._confirmation_count == 0:
            self._confirmation_position = measurement_mm
            self._confirmation_time_ns = int(frame_time_ns)
            self._confirmation_count = 1
            return self.config.confirmation_frames <= 1
        elapsed_seconds = (int(frame_time_ns) - self._confirmation_time_ns) / 1e9
        if elapsed_seconds <= 0.0:
            return False
        allowed_move = max(
            self.config.gate_min_mm,
            self.config.velocity_limit_mm_s
            * max(self.config.min_dt_seconds, elapsed_seconds)
            * 1.5,
        )
        if abs(measurement_mm - self._confirmation_position) <= allowed_move:
            self._confirmation_count += 1
            self._confirmation_position = 0.5 * (
                self._confirmation_position + measurement_mm
            )
        else:
            self._confirmation_count = 1
            self._confirmation_position = measurement_mm
        self._confirmation_time_ns = int(frame_time_ns)
        return self._confirmation_count >= self.config.confirmation_frames

    def _stale_estimate(self) -> TrackEstimate:
        return TrackEstimate(
            position_mm=self.position_mm if self._initialized else 0.0,
            velocity_mm_s=self.velocity_mm_s if self._initialized else 0.0,
            position_sigma_mm=(
                float(np.sqrt(max(0.0, self._covariance[0, 0])))
                if self._initialized
                else 999.0
            ),
            confidence=0.0,
            flags=TrackFlags.PREDICTED if self._initialized else TrackFlags.LOST,
            frame_time_ns=self._last_update_time_ns,
            measurement_mm=None,
            innovation_mm=None,
            lost_frames=self._lost_frames,
        )

    def _missing_estimate(self, frame_time_ns: int) -> TrackEstimate:
        self._lost_frames += 1
        missing_ms = (
            max(0.0, (frame_time_ns - self._last_measurement_ns) / 1e6)
            if self._initialized and self._last_measurement_ns
            else float("inf")
        )
        if self._initialized and missing_ms <= self.config.max_prediction_ms:
            covariance_penalty = np.sqrt(max(0.0, self._covariance[0, 0]))
            confidence = float(np.exp(-0.45 * self._lost_frames) * np.exp(-covariance_penalty / 20.0))
            return TrackEstimate(
                position_mm=self.position_mm,
                velocity_mm_s=self.velocity_mm_s,
                position_sigma_mm=float(covariance_penalty),
                confidence=confidence,
                flags=TrackFlags.PREDICTED,
                frame_time_ns=frame_time_ns,
                measurement_mm=None,
                innovation_mm=None,
                lost_frames=self._lost_frames,
            )
        self._was_lost = True
        estimate = TrackEstimate(
            position_mm=self.position_mm if self._initialized else 0.0,
            velocity_mm_s=self.velocity_mm_s if self._initialized else 0.0,
            position_sigma_mm=(
                float(np.sqrt(max(0.0, self._covariance[0, 0])))
                if self._initialized
                else 999.0
            ),
            confidence=0.0,
            flags=TrackFlags.LOST,
            frame_time_ns=frame_time_ns,
            measurement_mm=None,
            innovation_mm=None,
            lost_frames=self._lost_frames,
        )
        if self._initialized and missing_ms >= self.config.lost_reset_ms:
            self.reset()
        return estimate

    def update(
        self,
        measurement_mm: float | None,
        confidence: float,
        frame_time_ns: int,
    ) -> TrackEstimate:
        frame_time_ns = int(frame_time_ns)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if self._last_update_time_ns and frame_time_ns <= self._last_update_time_ns:
            return self._stale_estimate()
        self._last_update_time_ns = frame_time_ns

        if not self._initialized:
            if measurement_mm is None:
                self._confirmation_count = 0
                self._confirmation_time_ns = 0
                return self._missing_estimate(frame_time_ns)
            measurement = float(measurement_mm)
            if not self._search_confirmation(measurement, frame_time_ns):
                return TrackEstimate(
                    position_mm=measurement,
                    velocity_mm_s=0.0,
                    position_sigma_mm=self.config.initial_position_sigma_mm,
                    confidence=confidence * 0.5,
                    flags=TrackFlags.LOST,
                    frame_time_ns=frame_time_ns,
                    measurement_mm=measurement,
                    innovation_mm=None,
                    lost_frames=0,
                )
            reacquired = self._ever_initialized
            self._initialize(self._confirmation_position, frame_time_ns)
            self._confirmation_count = 0
            self._confirmation_time_ns = 0
            self._was_lost = False
            return TrackEstimate(
                position_mm=self.position_mm,
                velocity_mm_s=self.velocity_mm_s,
                position_sigma_mm=float(np.sqrt(max(0.0, self._covariance[0, 0]))),
                confidence=confidence,
                flags=(
                    TrackFlags.VALID | TrackFlags.REACQUIRED
                    if reacquired
                    else TrackFlags.VALID
                ),
                frame_time_ns=frame_time_ns,
                measurement_mm=measurement,
                innovation_mm=measurement - self.position_mm,
                lost_frames=0,
            )

        self.predict(frame_time_ns)
        if measurement_mm is None:
            return self._missing_estimate(frame_time_ns)

        measurement = float(measurement_mm)
        if self._was_lost:
            if not self._search_confirmation(measurement, frame_time_ns):
                return self._missing_estimate(frame_time_ns)
            self._initialize(self._confirmation_position, frame_time_ns)
            self._confirmation_count = 0
            self._confirmation_time_ns = 0
            self._was_lost = False
            return TrackEstimate(
                position_mm=self.position_mm,
                velocity_mm_s=self.velocity_mm_s,
                position_sigma_mm=float(np.sqrt(max(0.0, self._covariance[0, 0]))),
                confidence=confidence,
                flags=TrackFlags.VALID | TrackFlags.REACQUIRED,
                frame_time_ns=frame_time_ns,
                measurement_mm=measurement,
                innovation_mm=measurement - self.position_mm,
                lost_frames=0,
            )
        measurement_sigma = self.config.measurement_sigma_mm / max(0.20, confidence)
        measurement_variance = measurement_sigma * measurement_sigma
        innovation = measurement - self.position_mm
        innovation_variance = float(self._covariance[0, 0] + measurement_variance)
        gate = max(
            self.config.gate_min_mm,
            self.config.gate_sigma * np.sqrt(max(1e-9, innovation_variance)),
        )
        if abs(innovation) > gate:
            return self._missing_estimate(frame_time_ns)

        observation = np.array([[1.0, 0.0]], dtype=np.float64)
        kalman_gain = self._covariance @ observation.T / innovation_variance
        self._state = self._state + kalman_gain * innovation
        identity = np.eye(2, dtype=np.float64)
        kh = kalman_gain @ observation
        # Joseph form preserves a positive semi-definite covariance under
        # finite precision and changing measurement confidence.
        self._covariance = (
            (identity - kh) @ self._covariance @ (identity - kh).T
            + kalman_gain * measurement_variance @ kalman_gain.T
        )
        self._state[1, 0] = float(
            np.clip(
                self._state[1, 0],
                -self.config.velocity_limit_mm_s,
                self.config.velocity_limit_mm_s,
            )
        )
        reacquired = self._was_lost or self._lost_frames > 0
        self._lost_frames = 0
        self._last_measurement_ns = frame_time_ns
        self._was_lost = False
        covariance_penalty = np.sqrt(max(0.0, self._covariance[0, 0]))
        track_confidence = float(confidence * np.exp(-covariance_penalty / 10.0))
        flags = TrackFlags.VALID
        if reacquired:
            flags |= TrackFlags.REACQUIRED
        return TrackEstimate(
            position_mm=self.position_mm,
            velocity_mm_s=self.velocity_mm_s,
            position_sigma_mm=float(covariance_penalty),
            confidence=track_confidence,
            flags=flags,
            frame_time_ns=frame_time_ns,
            measurement_mm=measurement,
            innovation_mm=float(innovation),
            lost_frames=0,
        )
