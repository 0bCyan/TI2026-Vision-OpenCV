from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ball_vision.protocol import decode_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Print UDP telemetry packets")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25060)
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"listening on udp://{args.host}:{args.port}")
    while True:
        packet, address = sock.recvfrom(2048)
        try:
            decoded = decode_telemetry(packet)
            print(
                f"seq={decoded.frame_sequence:10d} flags={int(decoded.flags):02x} "
                f"x={decoded.position_mm:+7.1f}mm v={decoded.velocity_mm_s:+6d}mm/s "
                f"conf={decoded.confidence:.3f} age={decoded.frame_age_us}us "
                f"process={decoded.processing_us}us"
            )
        except ValueError as error:
            print("invalid packet from", address, error)


if __name__ == "__main__":
    main()

