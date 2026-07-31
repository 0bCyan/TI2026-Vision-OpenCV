from __future__ import annotations

import math
import operator
import struct
from dataclasses import dataclass
from enum import IntFlag

HEADER = b"\xaa\x55"
VERSION = 2
PACKET_SIZE = 25

HEADER_OFFSET = 0
VERSION_OFFSET = 2
FLAGS_OFFSET = 3
SESSION_ID_OFFSET = 4
FRAME_ID_OFFSET = 8
POSITION_OFFSET = 12
SIGMA_OFFSET = 14
FRAME_AGE_OFFSET = 16
PROCESSING_OFFSET = 20
CONFIDENCE_OFFSET = 22
CRC_OFFSET = 23

_PAYLOAD_FORMAT = "<BBIIhHIHB"
_PAYLOAD_SIZE = struct.calcsize(_PAYLOAD_FORMAT)
_CRC_FORMAT = "<H"
_UINT8_MAX = (1 << 8) - 1
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
_INT16_MIN = -(1 << 15)
_INT16_MAX = (1 << 15) - 1

assert len(HEADER) + _PAYLOAD_SIZE + struct.calcsize(_CRC_FORMAT) == PACKET_SIZE


class VisionFlags(IntFlag):
    NONE = 0
    MEASUREMENT_VALID = 1 << 0
    PREDICTED = 1 << 1
    LOST = 1 << 2
    REACQUIRED = 1 << 3
    NEAR_EDGE = 1 << 4


_KNOWN_FLAGS_MASK = int(
    VisionFlags.MEASUREMENT_VALID
    | VisionFlags.PREDICTED
    | VisionFlags.LOST
    | VisionFlags.REACQUIRED
    | VisionFlags.NEAR_EDGE
)


@dataclass(frozen=True, slots=True)
class DecodedTelemetry:
    version: int
    flags: VisionFlags
    session_id: int
    frame_id: int
    position_tenth_mm: int
    sigma_hundredth_mm: int
    frame_age_us: int
    processing_us: int
    confidence: int

    @property
    def position_mm(self) -> float:
        return self.position_tenth_mm * 0.1

    @property
    def sigma_mm(self) -> float:
        return self.sigma_hundredth_mm * 0.01

    @property
    def measurement_valid(self) -> bool:
        return bool(self.flags & VisionFlags.MEASUREMENT_VALID)


def crc16_ccitt_false(data: bytes, initial: int = 0xFFFF) -> int:
    """Return CRC-16/CCITT-FALSE (poly 0x1021, no reflection/XOR)."""

    crc = initial & _UINT16_MAX
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & _UINT16_MAX
            else:
                crc = (crc << 1) & _UINT16_MAX
    return crc


# Retain the old public name for callers that only need the CRC primitive.
crc16_ccitt = crc16_ccitt_false


def validate_flags(flags: VisionFlags | int) -> VisionFlags:
    raw = _as_integer("flags", flags)
    if raw < 0 or raw > _UINT8_MAX:
        raise ValueError("flags must fit in uint8")
    if raw & ~_KNOWN_FLAGS_MASK:
        raise ValueError(f"reserved flag bits must be zero: 0x{raw:02X}")

    parsed = VisionFlags(raw)
    valid = bool(parsed & VisionFlags.MEASUREMENT_VALID)
    predicted = bool(parsed & VisionFlags.PREDICTED)
    lost = bool(parsed & VisionFlags.LOST)

    if valid and (predicted or lost):
        raise ValueError("MEASUREMENT_VALID cannot be combined with PREDICTED or LOST")
    if predicted and lost:
        raise ValueError("PREDICTED and LOST are mutually exclusive")
    if parsed & VisionFlags.REACQUIRED and not valid:
        raise ValueError("REACQUIRED requires MEASUREMENT_VALID")
    if parsed & VisionFlags.NEAR_EDGE and not valid:
        raise ValueError("NEAR_EDGE requires MEASUREMENT_VALID")
    return parsed


def encode_telemetry(
    *,
    flags: VisionFlags | int,
    session_id: int,
    frame_id: int,
    position_mm: float | None,
    sigma_mm: float | None,
    frame_age_us: int,
    processing_us: int,
    confidence: int | None,
) -> bytes:
    """Encode one V2 frame.

    Position and sigma describe the current raw image measurement, never a
    tracker prediction. They are forced to zero when MEASUREMENT_VALID is not
    set. processing_us is the only field deliberately saturated on overflow.
    """

    parsed_flags = validate_flags(flags)
    session_id_raw = _require_range("session_id", session_id, 0, _UINT32_MAX)
    frame_id_raw = _require_range("frame_id", frame_id, 0, _UINT32_MAX)
    frame_age_raw = _require_range("frame_age_us", frame_age_us, 0, _UINT32_MAX)
    processing_value = _as_integer("processing_us", processing_us)
    if processing_value < 0:
        raise ValueError("processing_us must be non-negative")
    processing_raw = min(_UINT16_MAX, processing_value)

    if parsed_flags & VisionFlags.MEASUREMENT_VALID:
        position_raw = _quantize("position_mm", position_mm, 10.0)
        sigma_raw = _quantize("sigma_mm", sigma_mm, 100.0)
        confidence_raw = _require_range("confidence", confidence, 0, _UINT8_MAX)
        if not _INT16_MIN <= position_raw <= _INT16_MAX:
            raise ValueError("position_mm is outside the int16 wire range")
        if not 1 <= sigma_raw <= _UINT16_MAX:
            raise ValueError("sigma_mm must quantize to 1..65535")
    else:
        position_raw = 0
        sigma_raw = 0
        confidence_raw = 0

    payload = struct.pack(
        _PAYLOAD_FORMAT,
        VERSION,
        int(parsed_flags),
        session_id_raw,
        frame_id_raw,
        position_raw,
        sigma_raw,
        frame_age_raw,
        processing_raw,
        confidence_raw,
    )
    crc = crc16_ccitt_false(payload)
    return HEADER + payload + struct.pack(_CRC_FORMAT, crc)


def decode_telemetry(packet: bytes) -> DecodedTelemetry:
    if len(packet) != PACKET_SIZE:
        raise ValueError(
            f"telemetry length mismatch: expected={PACKET_SIZE}, actual={len(packet)}"
        )
    if packet[:2] != HEADER:
        raise ValueError("telemetry header mismatch")

    payload = packet[VERSION_OFFSET:CRC_OFFSET]
    received_crc = struct.unpack_from(_CRC_FORMAT, packet, CRC_OFFSET)[0]
    calculated_crc = crc16_ccitt_false(payload)
    if received_crc != calculated_crc:
        raise ValueError(
            f"CRC mismatch: received=0x{received_crc:04X}, "
            f"calculated=0x{calculated_crc:04X}"
        )

    (
        version,
        flags,
        session_id,
        frame_id,
        position_tenth_mm,
        sigma_hundredth_mm,
        frame_age_us,
        processing_us,
        confidence,
    ) = struct.unpack(_PAYLOAD_FORMAT, payload)
    if version != VERSION:
        raise ValueError(f"unsupported telemetry version: {version}")

    parsed_flags = validate_flags(flags)
    return DecodedTelemetry(
        version=version,
        flags=parsed_flags,
        session_id=session_id,
        frame_id=frame_id,
        position_tenth_mm=position_tenth_mm,
        sigma_hundredth_mm=sigma_hundredth_mm,
        frame_age_us=frame_age_us,
        processing_us=processing_us,
        confidence=confidence,
    )


def _as_integer(name: str, value: object) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _require_range(name: str, value: object, minimum: int, maximum: int) -> int:
    integer = _as_integer(name, value)
    if not minimum <= integer <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return integer


def _quantize(name: str, value: float | None, scale: float) -> int:
    if value is None:
        raise ValueError(f"{name} is required for a valid measurement")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    scaled = numeric * scale
    return math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
