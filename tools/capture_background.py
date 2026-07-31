from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ball_vision.capture import LatestFrameCapture
from ball_vision.config import AppConfig
from ball_vision.io_utils import imwrite_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture an explicit empty-pipe median background")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--source")
    parser.add_argument(
        "--backend",
        choices=("dshow", "msmf", "auto", "v4l2", "gstreamer"),
    )
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    if args.source is not None:
        config.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.backend:
        config.camera.backend = args.backend

    print("请移走钢球，保持水管、挡片、刻度和比赛照明处于正式状态。")
    for remaining in range(max(0, args.countdown), 0, -1):
        print(f"{remaining}...")
        time.sleep(1)
    frames: list[np.ndarray] = []
    with LatestFrameCapture(config.camera) as capture:
        sequence = -1
        while len(frames) < max(16, args.frames):
            packet = capture.latest(sequence, timeout=1.0)
            if packet is None:
                raise RuntimeError("采集背景期间摄像头超时")
            sequence = packet.sequence
            x, y, width, height = config.roi.clamp(packet.image.shape[1], packet.image.shape[0])
            roi = packet.image[y : y + height, x : x + width]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
            frames.append(gray.copy())
            if len(frames) % 16 == 0:
                print(f"已采集 {len(frames)}/{max(16, args.frames)} 帧")
    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    target = config.resolve_path(config.detector.threshold.background_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_path(target, background):
        raise RuntimeError(f"无法保存背景图 {target}")
    print(f"空管中值背景已保存: {target}，尺寸={background.shape[1]}x{background.shape[0]}")


if __name__ == "__main__":
    main()
