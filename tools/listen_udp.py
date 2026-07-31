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
                f"session={decoded.session_id:08x} frame={decoded.frame_id:10d} "
                f"flags={int(decoded.flags):02x} x={decoded.position_mm:+7.1f}mm "
                f"sigma={decoded.sigma_mm:.2f}mm conf={decoded.confidence}/255 "
                f"age={decoded.frame_age_us}us "
                f"process={decoded.processing_us}us"
            )
        except ValueError as error:
            print("invalid packet from", address, error)


if __name__ == "__main__":
    main()
