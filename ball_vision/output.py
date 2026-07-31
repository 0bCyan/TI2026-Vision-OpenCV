from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .config import OutputConfig


class Publisher(Protocol):
    def prepare(self) -> None: ...

    def publish(self, packet: bytes) -> None: ...

    def close(self) -> None: ...


PacketFactory = Callable[[int], bytes]


@dataclass(frozen=True, slots=True)
class PublisherStats:
    submitted: int
    sent: int
    superseded: int
    errors: int
    short_writes: int


class NullPublisher:
    def prepare(self) -> None:
        return

    def publish(self, packet: bytes) -> None:
        return

    def close(self) -> None:
        return


class UdpPublisher:
    def __init__(self, host: str, port: int) -> None:
        self.destination = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def prepare(self) -> None:
        return

    def publish(self, packet: bytes) -> None:
        self.socket.sendto(packet, self.destination)

    def close(self) -> None:
        self.socket.close()


class SerialPublisher:
    def __init__(
        self,
        port: str,
        baudrate: int,
        reconnect_seconds: float = 0.25,
    ) -> None:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("启用串口输出需要安装 pyserial") from error
        self._serial_module = serial
        self.port = port
        self.baudrate = int(baudrate)
        self.reconnect_seconds = max(0.05, float(reconnect_seconds))
        self.serial = None
        self._next_reconnect_at = 0.0

    def _open(self):
        return self._serial_module.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self._serial_module.EIGHTBITS,
            parity=self._serial_module.PARITY_NONE,
            stopbits=self._serial_module.STOPBITS_ONE,
            timeout=0,
            write_timeout=0.02,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def _reset(self) -> None:
        try:
            self.serial.close()
        except OSError:
            pass
        self.serial = None
        self._next_reconnect_at = time.monotonic() + self.reconnect_seconds

    def prepare(self) -> None:
        """Make the serial sink writable before telemetry is timestamped."""

        if self.serial is not None:
            return
        reconnect_at = float(getattr(self, "_next_reconnect_at", 0.0))
        delay = reconnect_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        try:
            self.serial = self._open()
        except OSError as error:
            self._next_reconnect_at = time.monotonic() + self.reconnect_seconds
            raise OSError(f"串口重连失败: {error}") from error
        self._next_reconnect_at = 0.0

    def publish(self, packet: bytes) -> None:
        self.prepare()
        assert self.serial is not None
        try:
            written = self.serial.write(packet)
        except OSError:
            self._reset()
            raise
        if written != len(packet):
            self._reset()
            raise OSError(
                f"串口短写: expected={len(packet)}, actual={written}"
            )

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None


class CompositePublisher:
    def __init__(self, publishers: list[Publisher]) -> None:
        self.publishers = publishers

    def prepare(self) -> None:
        for publisher in self.publishers:
            publisher.prepare()

    def publish(self, packet: bytes) -> None:
        for publisher in self.publishers:
            publisher.publish(packet)

    def close(self) -> None:
        for publisher in self.publishers:
            try:
                publisher.close()
            except OSError as error:
                print("[WARN] 关闭遥测输出失败:", error)


class LatestTelemetryPublisher:
    """Encode and publish only the newest result on one writer thread.

    The factory receives the monotonic timestamp immediately before encoding,
    so FRAME_AGE_US includes any waiting behind the previous serial write.
    """

    def __init__(self, publisher: Publisher) -> None:
        self._publisher = publisher
        self._condition = threading.Condition()
        self._latest: PacketFactory | None = None
        self._running = True
        self._submitted = 0
        self._sent = 0
        self._superseded = 0
        self._errors = 0
        self._short_writes = 0
        self._thread = threading.Thread(
            target=self._run, name="telemetry-output", daemon=True
        )
        self._thread.start()

    def submit(self, packet_factory: PacketFactory) -> None:
        with self._condition:
            if not self._running:
                return
            if self._latest is not None:
                self._superseded += 1
            self._latest = packet_factory
            self._submitted += 1
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._latest is None:
                    self._condition.wait()
                if not self._running:
                    return
            try:
                # Reconnect or otherwise ready the sink while the latest slot
                # remains replaceable. The packet is selected and timestamped
                # only after prepare() returns.
                self._publisher.prepare()
            except OSError as error:
                with self._condition:
                    self._errors += 1
                print("[WARN] 遥测输出准备失败:", error)
                continue
            with self._condition:
                if not self._running:
                    return
                factory = self._latest
                self._latest = None
            if factory is None:
                continue
            try:
                packet = factory(time.monotonic_ns())
                self._publisher.publish(packet)
            except OSError as error:
                with self._condition:
                    self._errors += 1
                    if "短写" in str(error):
                        self._short_writes += 1
                print("[WARN] 遥测输出失败:", error)
            except (TypeError, ValueError) as error:
                with self._condition:
                    self._errors += 1
                print("[WARN] 遥测编码失败:", error)
            else:
                with self._condition:
                    self._sent += 1

    def stats(self) -> PublisherStats:
        with self._condition:
            return PublisherStats(
                submitted=self._submitted,
                sent=self._sent,
                superseded=self._superseded,
                errors=self._errors,
                short_writes=self._short_writes,
            )

    def close(self) -> None:
        with self._condition:
            self._running = False
            self._latest = None
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        self._publisher.close()


def create_publisher(config: OutputConfig) -> Publisher:
    publishers: list[Publisher] = []
    if config.serial_enabled:
        publishers.append(
            SerialPublisher(
                config.serial_port,
                config.serial_baudrate,
                config.serial_reconnect_seconds,
            )
        )
    if config.udp_enabled:
        publishers.append(UdpPublisher(config.udp_host, config.udp_port))
    if not publishers:
        return NullPublisher()
    return CompositePublisher(publishers)
