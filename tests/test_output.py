from __future__ import annotations

import threading
import time

import pytest

from ball_vision.output import LatestTelemetryPublisher, SerialPublisher


class RecordingPublisher:
    def __init__(self, block_first: bool = False) -> None:
        self.packets: list[bytes] = []
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_first = block_first

    def publish(self, packet: bytes) -> None:
        if self.block_first and not self.packets:
            self.entered.set()
            self.release.wait(timeout=1.0)
        self.packets.append(packet)

    def close(self) -> None:
        self.closed = True


def test_latest_publisher_supersedes_pending_results() -> None:
    target = RecordingPublisher(block_first=True)
    publisher = LatestTelemetryPublisher(target)
    publisher.submit(lambda _: b"first")
    assert target.entered.wait(timeout=1.0)
    publisher.submit(lambda _: b"second")
    publisher.submit(lambda _: b"latest")
    target.release.set()
    deadline = time.monotonic() + 1.0
    while publisher.stats().sent < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    publisher.close()

    assert target.packets == [b"first", b"latest"]
    assert publisher.stats().superseded == 1
    assert target.closed


def test_latest_publisher_passes_send_time_to_factory() -> None:
    target = RecordingPublisher()
    publisher = LatestTelemetryPublisher(target)
    observed: list[int] = []
    before = time.monotonic_ns()
    publisher.submit(lambda send_ns: observed.append(send_ns) or b"packet")
    deadline = time.monotonic() + 1.0
    while not target.packets and time.monotonic() < deadline:
        time.sleep(0.005)
    publisher.close()
    assert observed and observed[0] >= before


def test_serial_publisher_rejects_short_write() -> None:
    publisher = object.__new__(SerialPublisher)

    class ShortSerial:
        def write(self, packet: bytes) -> int:
            return len(packet) - 1

    publisher.serial = ShortSerial()
    with pytest.raises(OSError, match="短写"):
        publisher.publish(b"12345")
