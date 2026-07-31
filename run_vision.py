from __future__ import annotations

import argparse
from pathlib import Path

from ball_vision.app import VisionApplication
from ball_vision.config import AppConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TI H steel-ball OpenCV vision")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "default.yaml"),
        help="YAML configuration path",
    )
    parser.add_argument("--source", help="override camera index or video file")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "v4l2", "gstreamer"))
    parser.add_argument("--headless", action="store_true", help="disable debug window")
    parser.add_argument("--strict", action="store_true", help="fail if camera contract is not met")
    parser.add_argument(
        "--serial-port",
        help="enable MC02 V2 output on this serial device (for example /dev/ttyS4)",
    )
    parser.add_argument(
        "--timestamp-offset-ms",
        type=float,
        help="measured camera exposure-to-OpenCV fixed latency compensation",
    )
    parser.add_argument(
        "--bench-serial",
        action="store_true",
        help=(
            "allow serial protocol bring-up with an unverified preview config; "
            "MC02 actuators must remain disabled"
        ),
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="require serial output, strict camera checks and calibrated timestamp offset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.load(args.config)
    if args.source is not None:
        config.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.backend:
        config.camera.backend = args.backend
    if args.headless:
        config.display.enabled = False
    if args.strict:
        config.camera.strict_contract = True
    if args.serial_port is not None:
        config.output.serial_enabled = True
        config.output.serial_port = args.serial_port
    if args.timestamp_offset_ms is not None:
        config.camera.timestamp_offset_ms = args.timestamp_offset_ms
    if args.bench_serial:
        if args.serial_port is None:
            raise SystemExit("[FATAL] --bench-serial requires --serial-port")
        if args.production or args.strict:
            raise SystemExit(
                "[FATAL] --bench-serial cannot be combined with --strict/--production"
            )
    if args.production:
        config.display.enabled = False
        config.camera.strict_contract = True
        if not config.output.serial_enabled:
            raise SystemExit("[FATAL] production mode requires --serial-port or serial_enabled")
        if config.camera.timestamp_offset_ms <= 0.0:
            raise SystemExit(
                "[FATAL] production mode requires a measured --timestamp-offset-ms > 0"
            )
    config.validate()
    try:
        VisionApplication(config, bench_serial=args.bench_serial).run()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[FATAL] {error}") from None


if __name__ == "__main__":
    main()
