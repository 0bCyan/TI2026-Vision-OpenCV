from __future__ import annotations

import socket
from typing import Protocol

from .config import OutputConfig


class Publisher(Protocol):
    def publish(self, packet: bytes) -> None: ...

    def close(self) -> None: ...


class NullPublisher:
    def publish(self, packet: bytes) -> None:
        return

    def close(self) -> None:
        return


class UdpPublisher:
    def __init__(self, host: str, port: int) -> None:
        self.destination = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, packet: bytes) -> None:
        self.socket.sendto(packet, self.destination)

    def close(self) -> None:
        self.socket.close()


class SerialPublisher:
    def __init__(self, port: str, baudrate: int) -> None:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("启用串口输出需要安装 pyserial") from error
        self.serial = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            timeout=0,
            write_timeout=0.02,
        )

    def publish(self, packet: bytes) -> None:
        self.serial.write(packet)

    def close(self) -> None:
        self.serial.close()


class CompositePublisher:
    def __init__(self, publishers: list[Publisher]) -> None:
        self.publishers = publishers

    def publish(self, packet: bytes) -> None:
        for publisher in self.publishers:
            publisher.publish(packet)

    def close(self) -> None:
        for publisher in self.publishers:
            try:
                publisher.close()
            except OSError as error:
                print("[WARN] 关闭遥测输出失败:", error)


def create_publisher(config: OutputConfig) -> Publisher:
    publishers: list[Publisher] = []
    if config.serial_enabled:
        publishers.append(SerialPublisher(config.serial_port, config.serial_baudrate))
    if config.udp_enabled:
        publishers.append(UdpPublisher(config.udp_host, config.udp_port))
    if not publishers:
        return NullPublisher()
    return CompositePublisher(publishers)
