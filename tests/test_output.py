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

    def prepare(self) -> None:
        return

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


def test_latest_publisher_prepares_before_selecting_and_timestamping() -> None:
    class PreparingPublisher(RecordingPublisher):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_entered = threading.Event()
            self.prepare_release = threading.Event()
            self.prepare_calls = 0

        def prepare(self) -> None:
            self.prepare_calls += 1
            if self.prepare_calls == 1:
                self.prepare_entered.set()
                self.prepare_release.wait(timeout=1.0)

    target = PreparingPublisher()
    publisher = LatestTelemetryPublisher(target)
    encoded_at: list[int] = []
    before_prepare_ns = time.monotonic_ns()
    publisher.submit(lambda send_ns: encoded_at.append(send_ns) or b"first")
    assert target.prepare_entered.wait(timeout=1.0)
    publisher.submit(lambda send_ns: encoded_at.append(send_ns) or b"latest")
    release_ns = time.monotonic_ns()
    target.prepare_release.set()
    deadline = time.monotonic() + 1.0
    while publisher.stats().sent < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    publisher.close()

    assert target.packets == [b"latest"]
    assert encoded_at and encoded_at[0] >= release_ns > before_prepare_ns
    assert publisher.stats().superseded == 1


def test_prepare_delay_is_visible_to_stale_packet_guard() -> None:
    class DelayedPublisher(RecordingPublisher):
        def prepare(self) -> None:
            time.sleep(0.03)

    target = DelayedPublisher()
    publisher = LatestTelemetryPublisher(target)
    frame_ns = time.monotonic_ns()

    def reject_stale(send_ns: int) -> bytes:
        if send_ns - frame_ns > 10_000_000:
            raise ValueError("stale")
        return b"incorrectly-fresh"

    publisher.submit(reject_stale)
    deadline = time.monotonic() + 1.0
    while publisher.stats().errors < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    publisher.close()

    assert target.packets == []
    assert publisher.stats().errors == 1


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
        def close(self) -> None:
            return

        def write(self, packet: bytes) -> int:
            return len(packet) - 1

    publisher.serial = ShortSerial()
    publisher.reconnect_seconds = 0.01
    with pytest.raises(OSError, match="短写"):
        publisher.publish(b"12345")
    assert publisher.serial is None


def test_serial_publisher_reopens_after_write_failure(monkeypatch) -> None:
    publisher = object.__new__(SerialPublisher)

    class BrokenSerial:
        def close(self) -> None:
            return

        def write(self, packet: bytes) -> int:
            raise OSError("disconnected")

    class HealthySerial:
        def __init__(self) -> None:
            self.packets: list[bytes] = []

        def close(self) -> None:
            return

        def write(self, packet: bytes) -> int:
            self.packets.append(packet)
            return len(packet)

    healthy = HealthySerial()
    publisher.serial = BrokenSerial()
    publisher.reconnect_seconds = 0.001
    publisher._open = lambda: healthy
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(OSError, match="disconnected"):
        publisher.publish(b"first")
    assert publisher.serial is None
    publisher.publish(b"second")
    assert healthy.packets == [b"second"]
