import pytest

from ball_vision.protocol import PACKET_SIZE, decode_telemetry, encode_telemetry
from ball_vision.types import TrackEstimate, TrackFlags


def sample_estimate() -> TrackEstimate:
    return TrackEstimate(
        position_mm=-42.35,
        velocity_mm_s=318.2,
        position_sigma_mm=0.42,
        confidence=0.876,
        flags=TrackFlags.VALID | TrackFlags.NEAR_EDGE,
        frame_time_ns=123_456_789_000,
        measurement_mm=-42.3,
        innovation_mm=0.2,
        lost_frames=0,
    )


def test_protocol_round_trip() -> None:
    packet = encode_telemetry(sample_estimate(), 0x12345678, 4200, 2100)
    assert len(packet) == PACKET_SIZE
    decoded = decode_telemetry(packet)
    assert decoded.frame_sequence == 0x12345678
    assert decoded.position_mm == pytest.approx(-42.4, abs=0.01)
    assert decoded.velocity_mm_s == 318
    assert decoded.position_sigma_mm == pytest.approx(0.42)
    assert decoded.flags & TrackFlags.VALID
    assert decoded.frame_age_us == 4200


def test_protocol_rejects_corruption() -> None:
    packet = bytearray(encode_telemetry(sample_estimate(), 1, 1, 1))
    packet[8] ^= 0x20
    with pytest.raises(ValueError, match="CRC"):
        decode_telemetry(bytes(packet))
