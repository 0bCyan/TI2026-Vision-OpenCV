import socket
import struct
import time

import cv2
import numpy as np

from ball_vision.config import FrameStreamConfig
from ball_vision.frame_output import LatestMjpegPublisher


HEADER = struct.Struct("<IHHQII")


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise RuntimeError("unexpected publisher disconnect")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def test_latest_mjpeg_publisher_sends_decodable_bounded_frame(tmp_path) -> None:
    socket_path = tmp_path / "frames.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    listener.settimeout(2.0)
    publisher = LatestMjpegPublisher(
        FrameStreamConfig(
            enabled=True,
            socket_path=str(socket_path),
            max_fps=30.0,
            max_width=320,
            jpeg_quality=70,
            connect_retry_seconds=0.01,
            send_timeout_seconds=0.2,
        )
    )
    try:
        image = np.full((360, 640, 3), (15, 90, 180), dtype=np.uint8)
        publisher.submit(image, 123_456_789)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(2.0)
            fields = HEADER.unpack(_receive_exact(connection, HEADER.size))
            magic, version, codec, capture_time_ns, payload_size, reserved = fields
            assert magic == 0x3146564C
            assert version == 1
            assert codec == 1
            assert capture_time_ns == 123_456_789
            assert reserved == 0
            payload = _receive_exact(connection, payload_size)
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[:2] == (180, 320)
        deadline = time.monotonic() + 1.0
        while publisher.stats().sent < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert publisher.stats().sent == 1
    finally:
        publisher.close()
        listener.close()


def test_missing_web_service_never_blocks_submit(tmp_path) -> None:
    publisher = LatestMjpegPublisher(
        FrameStreamConfig(
            enabled=True,
            socket_path=str(tmp_path / "missing.sock"),
            connect_retry_seconds=0.05,
            send_timeout_seconds=0.05,
        )
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    try:
        started = time.monotonic()
        for frame_id in range(100):
            publisher.submit(image, frame_id)
        elapsed = time.monotonic() - started
        assert elapsed < 0.1
        assert publisher.stats().submitted == 100
        assert publisher.stats().superseded > 0
    finally:
        publisher.close()
