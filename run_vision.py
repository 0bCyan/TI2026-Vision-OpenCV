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
    try:
        VisionApplication(config).run()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[FATAL] {error}") from None


if __name__ == "__main__":
    main()
