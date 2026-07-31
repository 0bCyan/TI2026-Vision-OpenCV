from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ball_vision.capture import LatestFrameCapture
from ball_vision.config import AppConfig, RoiConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the fixed pipe ROI")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--source")
    parser.add_argument(
        "--backend",
        choices=("dshow", "msmf", "auto", "v4l2", "gstreamer"),
    )
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    if args.source is not None:
        config.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.backend:
        config.camera.backend = args.backend
    with LatestFrameCapture(config.camera) as capture:
        packet = capture.latest(timeout=3.0)
        if packet is None:
            raise RuntimeError("没有收到摄像头画面")
        selection = cv2.selectROI(
            "Select complete pipe ROI, ENTER=save, ESC=cancel",
            packet.image,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyAllWindows()
    x, y, width, height = (int(value) for value in selection)
    if width <= 0 or height <= 0:
        print("已取消，配置未修改")
        return
    config.roi = RoiConfig(x=x, y=y, width=width, height=height)
    config.detector.channel_center_y_px = height * 0.5
    config.detector.max_center_y_error_px = height * 0.45
    config.save()
    print(f"ROI已保存到 {config.config_path}: x={x}, y={y}, w={width}, h={height}")
    print("ROI变化后请重新采集空管背景并重新执行位置标定。")


if __name__ == "__main__":
    main()
