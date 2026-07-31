from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ball_vision.capture import LatestFrameCapture
from ball_vision.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe actual UVC capture contract")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--backend", default=None, choices=("dshow", "msmf", "auto", "v4l2", "gstreamer"))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce resolution, MJPG, measured FPS and latency contract (default: true)",
    )
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.source is not None:
        config.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.backend:
        config.camera.backend = args.backend
    config.camera.strict_contract = args.strict
    config.camera.fingerprint_frames = True
    capture = LatestFrameCapture(config.camera)
    try:
        capture.start()
        first = capture.latest(timeout=3.0)
        if first is None:
            raise RuntimeError("3秒内未收到图像")
        print("Negotiated:")
        for key, value in capture.negotiated.items():
            print(f"  {key}: {value}")
        deadline = time.monotonic() + max(1.0, args.seconds)
        sequence = first.sequence
        while time.monotonic() < deadline:
            packet = capture.latest(sequence, timeout=0.25)
            if packet is not None:
                sequence = packet.sequence
        stats = capture.stats()
        print("Measured:")
        for name in stats.__dataclass_fields__:
            print(f"  {name}: {getattr(stats, name)}")
        capture.verify_contract()
        print("\nOpenCV GStreamer:", "YES" if "GStreamer:                   YES" in cv2.getBuildInformation() else "check build information")
        if stats.measured_fps < config.camera.fps * config.camera.min_measured_fps_ratio:
            ratio_percent = config.camera.min_measured_fps_ratio * 100.0
            print(f"FAIL: 实测帧率未达到标称值的{ratio_percent:.0f}%")
            raise SystemExit(2)
        print("PASS: 基础采集帧率合同通过；仍需LED实验验证重复帧和光子到结果延迟。")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
