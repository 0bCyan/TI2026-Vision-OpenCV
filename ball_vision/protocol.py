from __future__ import annotations

import struct
from dataclasses import dataclass

from .types import TrackEstimate, TrackFlags

HEADER = b"\xAA\x55"
VERSION = 1
_BODY_FORMAT = "<2sBBIIhhBHHH"
_BODY_SIZE = struct.calcsize(_BODY_FORMAT)
PACKET_SIZE = _BODY_SIZE + 2


@dataclass(slots=True)
class DecodedTelemetry:
    version: int
    flags: TrackFlags
    frame_sequence: int
    frame_time_us: int
    position_mm: float
    velocity_mm_s: int
    position_sigma_mm: float
    confidence: float
    frame_age_us: int
    processing_us: int


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_telemetry(
    estimate: TrackEstimate,
    frame_sequence: int,
    frame_age_us: int,
    processing_us: int,
) -> bytes:
    position_tenth_mm = round(estimate.position_mm * 10.0)
    velocity_mm_s = round(estimate.velocity_mm_s)
    confidence_byte = round(max(0.0, min(1.0, estimate.confidence)) * 255.0)
    sigma_hundredth_mm = round(max(0.0, estimate.position_sigma_mm) * 100.0)
    body = struct.pack(
        _BODY_FORMAT,
        HEADER,
        VERSION,
        int(estimate.flags) & 0xFF,
        int(frame_sequence) & 0xFFFFFFFF,
        (int(estimate.frame_time_ns) // 1000) & 0xFFFFFFFF,
        max(-32768, min(32767, position_tenth_mm)),
        max(-32768, min(32767, velocity_mm_s)),
        confidence_byte,
        max(0, min(65535, sigma_hundredth_mm)),
        max(0, min(65535, int(frame_age_us))),
        max(0, min(65535, int(processing_us))),
    )
    return body + struct.pack("<H", crc16_ccitt(body))


def decode_telemetry(packet: bytes) -> DecodedTelemetry:
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"遥测帧长度错误: expected={PACKET_SIZE}, actual={len(packet)}")
    body, received_crc_bytes = packet[:-2], packet[-2:]
    received_crc = struct.unpack("<H", received_crc_bytes)[0]
    calculated_crc = crc16_ccitt(body)
    if received_crc != calculated_crc:
        raise ValueError(
            f"CRC错误: received=0x{received_crc:04X}, calculated=0x{calculated_crc:04X}"
        )
    (
        header,
        version,
        flags,
        sequence,
        frame_time_us,
        position_tenth_mm,
        velocity_mm_s,
        confidence_byte,
        sigma_hundredth_mm,
        frame_age_us,
        processing_us,
    ) = struct.unpack(_BODY_FORMAT, body)
    if header != HEADER:
        raise ValueError("遥测帧头错误")
    return DecodedTelemetry(
        version=version,
        flags=TrackFlags(flags),
        frame_sequence=sequence,
        frame_time_us=frame_time_us,
        position_mm=position_tenth_mm / 10.0,
        velocity_mm_s=velocity_mm_s,
        position_sigma_mm=sigma_hundredth_mm / 100.0,
        confidence=confidence_byte / 255.0,
        frame_age_us=frame_age_us,
        processing_us=processing_us,
    )
