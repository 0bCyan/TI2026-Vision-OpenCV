import struct

import pytest

from ball_vision.protocol import (
    CONFIDENCE_OFFSET,
    CRC_OFFSET,
    FLAGS_OFFSET,
    FRAME_AGE_OFFSET,
    FRAME_ID_OFFSET,
    HEADER,
    PACKET_SIZE,
    POSITION_OFFSET,
    PROCESSING_OFFSET,
    SESSION_ID_OFFSET,
    SIGMA_OFFSET,
    VERSION,
    VERSION_OFFSET,
    VisionFlags,
    crc16_ccitt_false,
    decode_telemetry,
    encode_telemetry,
    validate_flags,
)

GOLDEN_PACKET = bytes.fromhex(
    "AA 55 02 01 44 33 22 11 D4 C3 B2 A1 85 FF FA 00 39 30 00 00 29 09 C8 EE 09"
)


def encode_sample(**overrides: object) -> bytes:
    fields: dict[str, object] = {
        "flags": VisionFlags.MEASUREMENT_VALID,
        "session_id": 0x11223344,
        "frame_id": 0xA1B2C3D4,
        "position_mm": -12.3,
        "sigma_mm": 2.50,
        "frame_age_us": 12_345,
        "processing_us": 2_345,
        "confidence": 200,
    }
    fields.update(overrides)
    return encode_telemetry(**fields)  # type: ignore[arg-type]


def test_ccitt_false_check_value() -> None:
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_hard_coded_golden_packet_encode() -> None:
    assert encode_sample() == GOLDEN_PACKET


def test_hard_coded_golden_packet_decode() -> None:
    decoded = decode_telemetry(GOLDEN_PACKET)
    assert decoded.version == 2
    assert decoded.flags == VisionFlags.MEASUREMENT_VALID
    assert decoded.session_id == 0x11223344
    assert decoded.frame_id == 0xA1B2C3D4
    assert decoded.position_tenth_mm == -123
    assert decoded.position_mm == pytest.approx(-12.3)
    assert decoded.sigma_hundredth_mm == 250
    assert decoded.sigma_mm == pytest.approx(2.50)
    assert decoded.frame_age_us == 12_345
    assert decoded.processing_us == 2_345
    assert decoded.confidence == 200


def test_v2_offsets_and_little_endian_layout() -> None:
    packet = GOLDEN_PACKET
    assert len(packet) == PACKET_SIZE == 25
    assert packet[:2] == HEADER
    assert packet[VERSION_OFFSET] == VERSION
    assert packet[FLAGS_OFFSET] == int(VisionFlags.MEASUREMENT_VALID)
    assert struct.unpack_from("<I", packet, SESSION_ID_OFFSET)[0] == 0x11223344
    assert struct.unpack_from("<I", packet, FRAME_ID_OFFSET)[0] == 0xA1B2C3D4
    assert struct.unpack_from("<h", packet, POSITION_OFFSET)[0] == -123
    assert struct.unpack_from("<H", packet, SIGMA_OFFSET)[0] == 250
    assert struct.unpack_from("<I", packet, FRAME_AGE_OFFSET)[0] == 12_345
    assert struct.unpack_from("<H", packet, PROCESSING_OFFSET)[0] == 2_345
    assert packet[CONFIDENCE_OFFSET] == 200
    assert struct.unpack_from("<H", packet, CRC_OFFSET)[0] == 0x09EE


def test_crc_covers_only_bytes_2_through_22() -> None:
    assert crc16_ccitt_false(GOLDEN_PACKET[2:23]) == 0x09EE
    assert crc16_ccitt_false(GOLDEN_PACKET[:23]) != 0x09EE


@pytest.mark.parametrize(
    ("position_mm", "expected_raw"),
    [(-3276.8, -32768), (3276.7, 32767)],
)
def test_position_wire_boundaries(position_mm: float, expected_raw: int) -> None:
    decoded = decode_telemetry(encode_sample(position_mm=position_mm))
    assert decoded.position_tenth_mm == expected_raw


def test_unsigned_boundaries_and_processing_saturation() -> None:
    decoded = decode_telemetry(
        encode_sample(
            session_id=0xFFFFFFFF,
            frame_id=0xFFFFFFFF,
            sigma_mm=655.35,
            frame_age_us=0xFFFFFFFF,
            processing_us=100_000,
            confidence=255,
        )
    )
    assert decoded.session_id == 0xFFFFFFFF
    assert decoded.frame_id == 0xFFFFFFFF
    assert decoded.sigma_hundredth_mm == 0xFFFF
    assert decoded.frame_age_us == 0xFFFFFFFF
    assert decoded.processing_us == 0xFFFF
    assert decoded.confidence == 0xFF


def test_negative_processing_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="processing_us"):
        encode_sample(processing_us=-1)


def test_invalid_measurement_has_deterministic_zero_value_fields() -> None:
    decoded = decode_telemetry(
        encode_sample(
            flags=VisionFlags.PREDICTED,
            position_mm=float("nan"),
            sigma_mm=None,
            confidence=999,
        )
    )
    assert decoded.flags == VisionFlags.PREDICTED
    assert decoded.position_tenth_mm == 0
    assert decoded.sigma_hundredth_mm == 0
    assert decoded.confidence == 0


@pytest.mark.parametrize(
    "flags",
    [
        VisionFlags.NONE,
        VisionFlags.PREDICTED,
        VisionFlags.LOST,
        VisionFlags.MEASUREMENT_VALID,
        VisionFlags.MEASUREMENT_VALID | VisionFlags.REACQUIRED,
        VisionFlags.MEASUREMENT_VALID | VisionFlags.NEAR_EDGE,
        VisionFlags.MEASUREMENT_VALID | VisionFlags.REACQUIRED | VisionFlags.NEAR_EDGE,
    ],
)
def test_legal_flag_combinations(flags: VisionFlags) -> None:
    assert validate_flags(flags) == flags


@pytest.mark.parametrize(
    "flags",
    [
        0x20,
        VisionFlags.MEASUREMENT_VALID | VisionFlags.PREDICTED,
        VisionFlags.MEASUREMENT_VALID | VisionFlags.LOST,
        VisionFlags.PREDICTED | VisionFlags.LOST,
        VisionFlags.REACQUIRED,
        VisionFlags.NEAR_EDGE,
    ],
)
def test_illegal_flag_combinations_are_rejected(flags: VisionFlags | int) -> None:
    with pytest.raises(ValueError):
        encode_sample(flags=flags)


def test_decoder_rejects_illegal_flags_even_with_valid_crc() -> None:
    packet = bytearray(GOLDEN_PACKET)
    packet[FLAGS_OFFSET] = 0x20
    packet[CRC_OFFSET:] = struct.pack("<H", crc16_ccitt_false(packet[2:23]))
    with pytest.raises(ValueError, match="reserved flag"):
        decode_telemetry(bytes(packet))


def test_decoder_rejects_corruption() -> None:
    packet = bytearray(GOLDEN_PACKET)
    packet[FRAME_ID_OFFSET] ^= 0x20
    with pytest.raises(ValueError, match="CRC"):
        decode_telemetry(bytes(packet))


@pytest.mark.parametrize("length", [0, PACKET_SIZE - 1, PACKET_SIZE + 1])
def test_decoder_rejects_wrong_length(length: int) -> None:
    with pytest.raises(ValueError, match="length"):
        decode_telemetry(
            GOLDEN_PACKET[:length] if length <= PACKET_SIZE else GOLDEN_PACKET + b"\0"
        )


def test_decoder_rejects_bad_header() -> None:
    packet = b"\xab\x55" + GOLDEN_PACKET[2:]
    with pytest.raises(ValueError, match="header"):
        decode_telemetry(packet)


def test_decoder_rejects_unsupported_version_with_valid_crc() -> None:
    packet = bytearray(GOLDEN_PACKET)
    packet[VERSION_OFFSET] = 3
    packet[CRC_OFFSET:] = struct.pack("<H", crc16_ccitt_false(packet[2:23]))
    with pytest.raises(ValueError, match="version"):
        decode_telemetry(bytes(packet))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", -1),
        ("session_id", 0x1_0000_0000),
        ("frame_id", -1),
        ("frame_age_us", 0x1_0000_0000),
        ("confidence", 256),
        ("position_mm", float("nan")),
        ("position_mm", 3276.8),
        ("sigma_mm", 0.0),
        ("sigma_mm", float("inf")),
        ("sigma_mm", 655.36),
    ],
)
def test_encoder_rejects_out_of_range_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_sample(**{field: value})
