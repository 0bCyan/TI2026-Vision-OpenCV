from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ball_vision.config import DetectorConfig, ThresholdConfig
from ball_vision.detector import BallDetector
from ball_vision.synthetic import make_pipe_background, render_ball


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic OpenCV detector benchmark")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=140)
    parser.add_argument("--radius", type=float, default=24.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--combine-mode",
        choices=("and", "or", "background", "intensity"),
        default="and",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    background = make_pipe_background(args.width, args.height)
    config = DetectorConfig(
        threshold=ThresholdConfig(
            median_blur_size=3,
            min_contrast_delta=16,
            std_gain=0.65,
            background_enabled=True,
            background_diff_threshold=14,
            combine_mode=args.combine_mode,
            morph_open_size=3,
            morph_close_size=3,
        ),
        radius_min_px=args.radius * 0.65,
        radius_max_px=args.radius * 1.45,
        expected_radius_px=args.radius,
        area_min_px=np.pi * (args.radius * 0.65) ** 2,
        area_max_px=np.pi * (args.radius * 1.55) ** 2,
        circularity_min=0.33,
        fill_ratio_min=0.34,
        channel_center_y_px=args.height * 0.5,
        max_center_y_error_px=args.height * 0.3,
        ransac_iterations=40,
        fit_rms_max_px=3.0,
        fit_min_inlier_ratio=0.34,
        arc_coverage_min=0.25,
    )
    detector = BallDetector(config)
    detector.set_background(background)

    count = max(20, args.frames)
    frames = []
    truths = []
    for index in range(count):
        phase = index / max(1, count - 1)
        center_x = args.radius * 1.8 + phase * (args.width - args.radius * 3.6)
        center_y = args.height * 0.5 + 2.5 * np.sin(index * 0.13)
        frame = render_ball(
            background,
            (center_x, center_y),
            radius_px=args.radius,
            noise_sigma=2.0,
            highlight_strength=80.0 + 110.0 * (0.5 + 0.5 * np.sin(index * 0.07)),
            jpeg_quality=55 + index % 30,
            seed=index + 100,
        )
        frames.append(frame.image)
        truths.append((center_x, center_y))

    repeat_timings_ms: list[np.ndarray] = []
    errors_px: list[float] = []
    failures = 0
    was_gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(max(1, args.repeats)):
            for image in frames[: min(len(frames), max(0, args.warmup))]:
                detector.detect(image)
            timings_ms: list[float] = []
            for image, truth in zip(frames, truths):
                start_ns = time.perf_counter_ns()
                result = detector.detect(image)
                timings_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
                if not result.valid or result.candidate is None:
                    failures += 1
                    continue
                errors_px.append(
                    float(
                        np.hypot(
                            result.candidate.center_px[0] - truth[0],
                            result.candidate.center_px[1] - truth[1],
                        )
                    )
                )
            repeat_timings_ms.append(np.asarray(timings_ms))
    finally:
        if was_gc_enabled:
            gc.enable()
    timing = np.concatenate(repeat_timings_ms)
    errors = np.asarray(errors_px) if errors_px else np.array([np.inf])
    print("=== Synthetic detector benchmark ===")
    print(f"OpenCV threads : {cv2.getNumThreads()}")
    print(f"ROI            : {args.width}x{args.height}")
    print(f"Mask mode      : {args.combine_mode}")
    print(f"Frames         : {count} x {len(repeat_timings_ms)} runs")
    print(f"Failures       : {failures}")
    for index, run_timing in enumerate(repeat_timings_ms, start=1):
        print(
            f"Run {index} ms       : "
            f"p50={np.percentile(run_timing, 50):.3f}, "
            f"p95={np.percentile(run_timing, 95):.3f}, "
            f"p99={np.percentile(run_timing, 99):.3f}"
        )
    print(f"Processing ms  : p50={np.percentile(timing,50):.3f}, p95={np.percentile(timing,95):.3f}, p99={np.percentile(timing,99):.3f}")
    print(f"Equivalent fps : mean={1000.0/np.mean(timing):.1f}")
    print(f"Center error px: rms={np.sqrt(np.mean(errors*errors)):.3f}, p95={np.percentile(errors,95):.3f}, max={np.max(errors):.3f}")
    print("说明：这是确定性合成ROI算法基准，不包含USB采集和MJPEG解码延迟。")
    if failures or np.percentile(errors, 95) > 1.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
