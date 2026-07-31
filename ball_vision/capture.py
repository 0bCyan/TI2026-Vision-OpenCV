from __future__ import annotations

import threading
import time
import zlib
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import CameraConfig
from .types import FramePacket


@dataclass(slots=True)
class CaptureStats:
    received_frames: int
    read_failures: int
    duplicate_fingerprints: int
    measured_fps: float
    interval_p50_ms: float
    interval_p95_ms: float
    interval_p99_ms: float
    read_p95_ms: float
    max_gap_ms: float
    last_frame_age_ms: float


def _percentile(values: Collection[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.fromiter(values, dtype=np.float64), percentile) / 1e6)


def _backend_id(name: str) -> int:
    normalized = name.strip().lower()
    mapping = {
        "auto": cv2.CAP_ANY,
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
    }
    if normalized not in mapping:
        raise ValueError(f"不支持的摄像头后端: {name}")
    return mapping[normalized]


def _opencv_feature_enabled(feature: str) -> bool:
    prefix = f"{feature}:"
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().upper().startswith("YES")
    return False


_CAMERA_CONTROL_PROPERTIES = {
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "focus": cv2.CAP_PROP_FOCUS,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
}

_CAMERA_CONTROL_PRIORITY = {
    "auto_exposure": 0,
    "auto_wb": 0,
    "autofocus": 0,
    "exposure": 1,
    "gain": 1,
    "wb_temperature": 1,
    "focus": 1,
}


class LatestFrameCapture:
    """Continuously drains a source and exposes only its newest unique frame."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._stats_lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._running = False
        self._fatal_error: BaseException | None = None
        self._sequence = 0
        self._received = 0
        self._read_failures = 0
        self._duplicate_fingerprints = 0
        self._last_fingerprint: int | None = None
        # 512 samples are roughly a 2.8 s sliding window at 180 Hz. A bounded
        # window prevents an initially healthy run from hiding later slowdown.
        self._intervals_ns: deque[int] = deque(maxlen=512)
        self._read_times_ns: deque[int] = deque(maxlen=512)
        self._first_arrival_ns = 0
        self._last_arrival_ns = 0
        self.negotiated: dict[str, Any] = {}
        source_text = str(config.source)
        self.is_live_source = (
            isinstance(config.source, int)
            or source_text.isdigit()
            or source_text.startswith(("/dev/video", "/dev/v4l/"))
            or bool(config.gstreamer_pipeline)
        )

    def _source_and_backend(self) -> tuple[Any, int]:
        if self.config.gstreamer_pipeline:
            return self.config.gstreamer_pipeline, cv2.CAP_GSTREAMER
        if self.config.backend.strip().lower() == "gstreamer":
            raise ValueError("backend=gstreamer 时必须提供 camera.gstreamer_pipeline")
        source = self.config.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if not self.is_live_source:
            return source, cv2.CAP_ANY
        return source, _backend_id(self.config.backend)

    @property
    def running(self) -> bool:
        return self._running

    def _apply_controls(self, capture: cv2.VideoCapture) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        ordered = sorted(
            self.config.controls.items(),
            key=lambda item: _CAMERA_CONTROL_PRIORITY.get(item[0], 2),
        )
        failures: list[str] = []
        for name, requested in ordered:
            property_id = _CAMERA_CONTROL_PROPERTIES[name]
            set_ok = bool(capture.set(property_id, float(requested)))
            actual = float(capture.get(property_id))
            verified = bool(
                set_ok
                and np.isfinite(actual)
                and abs(actual - float(requested)) <= self.config.control_tolerance
            )
            results[name] = {
                "requested": float(requested),
                "actual": actual,
                "set_ok": set_ok,
                "verified": verified,
            }
            if not verified:
                failures.append(
                    f"{name}: requested={requested}, actual={actual}, set_ok={set_ok}"
                )
        if failures and self.config.strict_control_readback:
            raise RuntimeError("摄像头控制项设置/读回失败:\n  - " + "\n  - ".join(failures))
        return results

    def _open(self) -> cv2.VideoCapture:
        source, backend = self._source_and_backend()
        if backend == cv2.CAP_GSTREAMER and not _opencv_feature_enabled("GStreamer"):
            raise RuntimeError(
                "当前OpenCV没有GStreamer支持；香橙派请使用系统/自编译cv2，"
                "不要安装PyPI opencv-python覆盖它"
            )
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频源 {source!r}，后端={self.config.backend}")
        if self.is_live_source and not self.config.gstreamer_pipeline:
            requested_fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
            # Some UVC DirectShow drivers reset the media subtype whenever
            # width/height/FPS changes. Set MJPG last so the final negotiated
            # media type remains compressed instead of silently falling back
            # to bandwidth-limited YUY2.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            capture.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        try:
            control_results = self._apply_controls(capture) if self.is_live_source else {}
        except RuntimeError:
            capture.release()
            raise
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        self.negotiated = {
            "backend": capture.getBackendName() if hasattr(capture, "getBackendName") else self.config.backend,
            "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "fourcc": fourcc,
            "source": source,
            "controls": control_results,
        }
        if self.is_live_source and self.config.strict_contract:
            if (
                self.negotiated["width"] != self.config.width
                or self.negotiated["height"] != self.config.height
            ):
                capture.release()
                raise RuntimeError(
                    "摄像头未协商到要求的分辨率: "
                    f"requested={self.config.width}x{self.config.height}, "
                    f"actual={self.negotiated['width']}x{self.negotiated['height']}"
                )
            actual_fourcc = str(self.negotiated["fourcc"]).rstrip("\x00 ")
            if (
                not self.config.gstreamer_pipeline
                and actual_fourcc.upper() != self.config.fourcc.upper()
            ):
                capture.release()
                raise RuntimeError(
                    "摄像头未协商到要求的像素格式: "
                    f"requested={self.config.fourcc}, actual={actual_fourcc!r}"
                )
        return capture

    def start(self) -> LatestFrameCapture:
        if self._running:
            return self
        self._capture = self._open()
        self._running = True
        self._fatal_error = None
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        warmup_seconds = self.config.warmup_seconds if self.is_live_source else 0.0
        warmup_deadline = time.monotonic() + max(0.0, warmup_seconds)
        replay_start_ns = time.monotonic_ns()
        last_source_frame_time_ns = 0
        reported_source_fps = float(self.negotiated.get("reported_fps", 0.0))
        source_frame_period_ns = int(
            1e9 / max(1.0, reported_source_fps or self.config.fps)
        )
        consecutive_failures = 0
        while self._running:
            capture = self._capture
            if capture is None:
                break
            read_start_ns = time.monotonic_ns()
            ok, image = capture.read()
            read_return_ns = time.monotonic_ns()
            read_duration_ns = read_return_ns - read_start_ns
            if not ok or image is None:
                if time.monotonic() >= warmup_deadline:
                    with self._stats_lock:
                        self._read_failures += 1
                consecutive_failures += 1
                if not self.is_live_source:
                    self._running = False
                    with self._condition:
                        self._condition.notify_all()
                    break
                if consecutive_failures < 8:
                    time.sleep(0.001)
                    continue
                if self.config.controls_verified_externally:
                    self._fatal_error = RuntimeError(
                        "UVC断流且控制项由外部管理；已退出以便监护进程重新设置控制后重启"
                    )
                    self._running = False
                    with self._condition:
                        self._condition.notify_all()
                    break
                try:
                    capture.release()
                    time.sleep(max(0.05, self.config.reconnect_seconds))
                    self._capture = self._open()
                    self._fatal_error = None
                    consecutive_failures = 0
                    continue
                except (RuntimeError, cv2.error) as error:
                    self._fatal_error = error
                    time.sleep(max(0.05, self.config.reconnect_seconds))
                    continue
            consecutive_failures = 0
            source_frame_time_ns = 0
            if not self.is_live_source:
                position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                source_frame_time_ns = replay_start_ns + int(max(0.0, position_ms) * 1e6)
                if source_frame_time_ns <= last_source_frame_time_ns:
                    source_frame_time_ns = last_source_frame_time_ns + source_frame_period_ns
                last_source_frame_time_ns = source_frame_time_ns
                if self.config.file_realtime:
                    delay_ns = source_frame_time_ns - time.monotonic_ns()
                    if delay_ns > 0:
                        time.sleep(delay_ns / 1e9)
            arrival_ns = time.monotonic_ns()
            if time.monotonic() < warmup_deadline:
                continue
            fingerprint: int | None = None
            if self.config.fingerprint_frames:
                thumbnail = cv2.resize(image, (32, 18), interpolation=cv2.INTER_AREA)
                fingerprint = zlib.crc32(thumbnail.tobytes())
            with self._stats_lock:
                if self._last_arrival_ns:
                    self._intervals_ns.append(arrival_ns - self._last_arrival_ns)
                else:
                    self._first_arrival_ns = arrival_ns
                self._last_arrival_ns = arrival_ns
                self._read_times_ns.append(read_duration_ns)
                if fingerprint is not None:
                    if fingerprint == self._last_fingerprint:
                        self._duplicate_fingerprints += 1
                    self._last_fingerprint = fingerprint
                self._received += 1
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            offset_ns = int(self.config.timestamp_offset_ms * 1e6)
            frame_time_ns = source_frame_time_ns or (arrival_ns - offset_ns)
            packet = FramePacket(
                sequence=self._sequence,
                arrival_ns=arrival_ns,
                frame_time_ns=frame_time_ns,
                read_duration_ns=read_duration_ns,
                image=image,
            )
            with self._condition:
                self._latest = packet
                self._condition.notify_all()

    def latest(self, after_sequence: int = -1, timeout: float | None = None) -> FramePacket | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._running and (self._latest is None or self._latest.sequence == after_sequence):
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
            if self._fatal_error and not self._running and (
                self._latest is None or self._latest.sequence == after_sequence
            ):
                raise RuntimeError("采集线程失败") from self._fatal_error
            if not self._running and (
                self._latest is None or self._latest.sequence == after_sequence
            ):
                return None
            return self._latest

    def stats(self) -> CaptureStats:
        now_ns = time.monotonic_ns()
        with self._stats_lock:
            intervals_ns = tuple(self._intervals_ns)
            read_times_ns = tuple(self._read_times_ns)
            received = self._received
            read_failures = self._read_failures
            duplicates = self._duplicate_fingerprints
            last_arrival_ns = self._last_arrival_ns
        interval_sum_ns = sum(intervals_ns)
        measured_fps = len(intervals_ns) * 1e9 / interval_sum_ns if interval_sum_ns else 0.0
        max_gap_ms = max(intervals_ns, default=0) / 1e6
        age_ms = (now_ns - last_arrival_ns) / 1e6 if last_arrival_ns else float("inf")
        return CaptureStats(
            received_frames=received,
            read_failures=read_failures,
            duplicate_fingerprints=duplicates,
            measured_fps=float(measured_fps),
            interval_p50_ms=_percentile(intervals_ns, 50),
            interval_p95_ms=_percentile(intervals_ns, 95),
            interval_p99_ms=_percentile(intervals_ns, 99),
            read_p95_ms=_percentile(read_times_ns, 95),
            max_gap_ms=float(max_gap_ms),
            last_frame_age_ms=float(age_ms),
        )

    def verify_contract(self) -> None:
        if not self.is_live_source:
            return
        stats = self.stats()
        minimum_fps = self.config.fps * self.config.min_measured_fps_ratio
        issues: list[str] = []
        if stats.measured_fps < minimum_fps:
            issues.append(
                f"实测采集帧率不足: {stats.measured_fps:.1f} fps，"
                f"要求至少 {minimum_fps:.1f} fps"
            )
        with self._condition:
            latest = self._latest
        if latest is None or latest.image.shape[:2] != (self.config.height, self.config.width):
            actual_shape = None if latest is None else latest.image.shape[:2][::-1]
            issues.append(
                "首帧实际尺寸不符合合同: "
                f"expected={(self.config.width, self.config.height)}, actual={actual_shape}"
            )
        if stats.interval_p99_ms > self.config.max_interval_p99_ms:
            issues.append(
                f"帧间隔P99过大: {stats.interval_p99_ms:.2f} ms > "
                f"{self.config.max_interval_p99_ms:.2f} ms"
            )
        if stats.max_gap_ms > self.config.max_frame_gap_ms:
            issues.append(
                f"最大采集停顿过大: {stats.max_gap_ms:.2f} ms > "
                f"{self.config.max_frame_gap_ms:.2f} ms"
            )
        if stats.read_p95_ms > self.config.max_read_p95_ms:
            issues.append(
                f"read耗时P95过大: {stats.read_p95_ms:.2f} ms > "
                f"{self.config.max_read_p95_ms:.2f} ms"
            )
        if stats.read_failures > self.config.max_read_failures:
            issues.append(
                f"采集读取失败次数过多: {stats.read_failures} > "
                f"{self.config.max_read_failures}"
            )
        if not issues:
            return
        message = "摄像头合同未通过:\n  - " + "\n  - ".join(issues)
        if self.config.strict_contract:
            raise RuntimeError(message)
        print("[WARN]", message)

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            # In the normal 180 Hz path read() returns promptly, so let the
            # capture thread leave before touching its native handle.
            self._thread.join(timeout=0.25)
        capture = self._capture
        if self._thread and self._thread.is_alive() and capture is not None:
            # Fallback only: some OpenCV backends have no interruptible read.
            # MCU-side telemetry timeout remains the hard safety watchdog.
            capture.release()
            self._thread.join(timeout=2.0)
        elif capture is not None:
            capture.release()
        self._capture = None

    def __enter__(self) -> LatestFrameCapture:
        return self.start()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
