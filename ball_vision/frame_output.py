from __future__ import annotations

import os
import socket
import struct
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import FrameStreamConfig


_FRAME_HEADER = struct.Struct("<IHHQII")
_FRAME_MAGIC = 0x3146564C
_FRAME_VERSION = 1
_FRAME_CODEC_MJPEG = 1


@dataclass(slots=True, frozen=True)
class FramePublisherStats:
    submitted: int
    sent: int
    superseded: int
    encode_errors: int
    transport_errors: int


class LatestMjpegPublisher:
    """Publish a bounded-rate MJPEG preview without blocking vision.

    submit() only replaces one latest-frame reference. Resize, JPEG encode,
    connect, reconnect and socket writes all happen on a low-priority worker.
    """

    def __init__(self, config: FrameStreamConfig) -> None:
        self.config = config
        self._condition = threading.Condition()
        self._latest: tuple[np.ndarray, int] | None = None
        self._socket: socket.socket | None = None
        self._running = bool(config.enabled)
        self._submitted = 0
        self._sent = 0
        self._superseded = 0
        self._encode_errors = 0
        self._transport_errors = 0
        self._last_warning_ns = 0
        self._thread: threading.Thread | None = None
        self.socket_path = os.path.expanduser(os.path.expandvars(config.socket_path))
        if self._running:
            self._thread = threading.Thread(
                target=self._run,
                name="web-frame-output",
                daemon=True,
            )
            self._thread.start()

    def submit(self, image: np.ndarray, capture_time_ns: int) -> None:
        if not self._running:
            return
        with self._condition:
            if not self._running:
                return
            if self._latest is not None:
                self._superseded += 1
            # OpenCV returns a new ndarray for each read. Keeping this reference
            # is enough to preserve the frame while the capture thread advances.
            self._latest = (image, int(capture_time_ns))
            self._submitted += 1
            self._condition.notify()

    def _warn(self, message: str) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_warning_ns >= 5_000_000_000:
            print(f"[WARN] 网页供帧: {message}")
            self._last_warning_ns = now_ns

    def _disconnect(self) -> None:
        connection = self._socket
        self._socket = None
        if connection is None:
            return
        try:
            connection.close()
        except OSError:
            pass

    def _connect(self) -> bool:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.config.send_timeout_seconds)
        try:
            connection.connect(self.socket_path)
        except OSError as error:
            connection.close()
            with self._condition:
                self._transport_errors += 1
            self._warn(f"等待 {self.socket_path}: {error}")
            return False
        self._socket = connection
        print(f"[INFO] 网页供帧已连接: {self.socket_path}")
        return True

    def _encode(self, image: np.ndarray) -> memoryview | None:
        frame = image
        height, width = frame.shape[:2]
        if width > self.config.max_width:
            output_width = self.config.max_width
            output_height = max(1, round(height * output_width / width))
            frame = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not ok:
            return None
        payload = memoryview(encoded).cast("B")
        if len(payload) > self.config.max_frame_bytes:
            self._warn(
                f"JPEG帧 {len(payload)} 字节超过上限 "
                f"{self.config.max_frame_bytes} 字节"
            )
            return None
        return payload

    def _wait_for_work(self, ready_ns: int) -> tuple[np.ndarray, int] | None:
        with self._condition:
            while self._running:
                if self._latest is None:
                    self._condition.wait()
                    continue
                delay_ns = ready_ns - time.monotonic_ns()
                if delay_ns > 0:
                    self._condition.wait(delay_ns / 1e9)
                    continue
                latest = self._latest
                self._latest = None
                return latest
            return None

    def _run(self) -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (AttributeError, OSError):
            pass
        frame_interval_ns = round(1e9 / self.config.max_fps)
        next_frame_ns = 0
        next_connect_ns = 0
        while True:
            ready_ns = max(
                next_frame_ns,
                next_connect_ns if self._socket is None else 0,
            )
            latest = self._wait_for_work(ready_ns)
            if latest is None:
                break
            if self._socket is None:
                # Keep the latest frame replaceable while connect is attempted;
                # do not spend CPU encoding when the web service is absent.
                with self._condition:
                    if self._latest is None:
                        self._latest = latest
                    else:
                        self._superseded += 1
                if not self._connect():
                    next_connect_ns = (
                        time.monotonic_ns()
                        + round(self.config.connect_retry_seconds * 1e9)
                    )
                else:
                    next_connect_ns = 0
                continue

            image, capture_time_ns = latest
            try:
                payload = self._encode(image)
            except cv2.error as error:
                payload = None
                self._warn(f"JPEG编码失败: {error}")
            if payload is None:
                with self._condition:
                    self._encode_errors += 1
                next_frame_ns = time.monotonic_ns() + frame_interval_ns
                continue
            header = _FRAME_HEADER.pack(
                _FRAME_MAGIC,
                _FRAME_VERSION,
                _FRAME_CODEC_MJPEG,
                max(0, capture_time_ns),
                len(payload),
                0,
            )
            try:
                assert self._socket is not None
                self._socket.sendall(header)
                self._socket.sendall(payload)
            except OSError as error:
                with self._condition:
                    self._transport_errors += 1
                self._warn(f"连接中断，等待自动重连: {error}")
                self._disconnect()
                next_connect_ns = (
                    time.monotonic_ns()
                    + round(self.config.connect_retry_seconds * 1e9)
                )
            else:
                with self._condition:
                    self._sent += 1
            next_frame_ns = time.monotonic_ns() + frame_interval_ns
        self._disconnect()

    def stats(self) -> FramePublisherStats:
        with self._condition:
            return FramePublisherStats(
                submitted=self._submitted,
                sent=self._sent,
                superseded=self._superseded,
                encode_errors=self._encode_errors,
                transport_errors=self._transport_errors,
            )

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._condition:
            self._running = False
            self._latest = None
            self._condition.notify_all()
        thread.join(timeout=max(1.0, self.config.send_timeout_seconds + 0.5))
        if thread.is_alive() and self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            thread.join(timeout=0.5)
        self._disconnect()
        self._thread = None
