from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ball_vision.calibration import CoordinateMapper
from ball_vision.capture import LatestFrameCapture
from ball_vision.config import (
    POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE,
    AppConfig,
    CalibrationConfig,
)
from ball_vision.io_utils import imread_path


def parse_positions(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Click known scale ticks to calibrate pixel -> millimeter")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--source")
    parser.add_argument(
        "--backend",
        choices=("dshow", "msmf", "auto", "v4l2", "gstreamer"),
    )
    parser.add_argument("--image", help="use a saved full-frame image instead of live camera")
    parser.add_argument(
        "--positions-mm",
        default=",".join(str(value) for value in range(-120, 121, 10)),
        help="comma-separated known coordinates in click order",
    )
    parser.add_argument(
        "--confirm-front-positive",
        action="store_true",
        required=True,
        help="confirm 0=center, positive=toward vehicle front, negative=toward rear",
    )
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    if args.image:
        frame = imread_path(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法读取图像 {args.image}")
    else:
        if args.source is not None:
            config.camera.source = int(args.source) if args.source.isdigit() else args.source
        if args.backend:
            config.camera.backend = args.backend
        with LatestFrameCapture(config.camera) as capture:
            packet = capture.latest(timeout=3.0)
            if packet is None:
                raise RuntimeError("没有收到摄像头画面")
            frame = packet.image.copy()

    positions = parse_positions(args.positions_mm)
    if len(positions) < 2:
        raise ValueError("至少需要两个标定位置")
    print("坐标合同：横梁机械中心为0 mm，车头方向为正，车尾方向为负。")
    print("请按提示的物理刻度顺序点击；图像左右方向不代表坐标正负。")
    roi_x, roi_y, roi_width, roi_height = config.roi.clamp(frame.shape[1], frame.shape[0])
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()
    clicked: list[tuple[float, float]] = []
    window = "Click tick: left=accept, right=undo, ESC=cancel"

    def callback(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(positions):
            clicked.append((float(x + roi_x), positions[len(clicked)]))
        elif event == cv2.EVENT_RBUTTONDOWN and clicked:
            clicked.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    while len(clicked) < len(positions):
        display = roi.copy()
        for global_x, position in clicked:
            local_x = round(global_x - roi_x)
            cv2.line(display, (local_x, 0), (local_x, roi_height - 1), (0, 255, 0), 1)
            cv2.putText(display, f"{position:g}", (local_x + 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        next_position = positions[len(clicked)]
        cv2.putText(
            display,
            f"Click scale tick {next_position:+g} mm ({len(clicked)+1}/{len(positions)})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(display, f"Click scale tick {next_position:+g} mm ({len(clicked)+1}/{len(positions)})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            print("已取消，配置未修改")
            return
    cv2.destroyAllWindows()
    clicked.sort(key=lambda item: item[0])
    config.calibration = CalibrationConfig(
        mode="piecewise_linear",
        degree=1,
        points=[[pixel, position] for pixel, position in clicked],
        coordinate_convention=POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE,
    )
    mapper = CoordinateMapper(config.calibration)
    config.save()
    print(f"标定已保存到 {config.config_path}")
    print(f"有效像素范围: {mapper.pixel_limits}，物理范围: {mapper.millimeter_limits}")
    print("下一步应使用错开5mm的独立位置验证，而不是只检查标定点。")


if __name__ == "__main__":
    main()
