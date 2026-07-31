from __future__ import annotations

import gc
import time

import cv2

from .capture import LatestFrameCapture
from .config import AppConfig
from .output import create_publisher
from .pipeline import VisionPipeline
from .protocol import encode_telemetry
from .runtime_stats import RuntimeStats
from .telemetry_log import CsvTelemetryLogger
from .visualization import DebugDisplay


class VisionApplication:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if config.output.serial_enabled or config.output.udp_enabled:
            config.camera.strict_contract = True
        self._production_output = (
            config.camera.strict_contract
            or config.output.serial_enabled
            or config.output.udp_enabled
        )
        if self._production_output:
            self._verify_vision_setup()
        cv2.setNumThreads(max(1, config.runtime.opencv_threads))
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(config.runtime.opencl_enabled)
        self._gc_was_enabled = gc.isenabled()
        if config.runtime.disable_python_gc:
            gc.disable()
        self.pipeline = VisionPipeline(config)
        if (
            self._production_output
            and config.detector.threshold.background_enabled
            and self.pipeline.detector.background is None
        ):
            raise RuntimeError("空管背景文件无法读取，已拒绝启用正式遥测输出")
        self.capture = LatestFrameCapture(config.camera)
        self.publisher = create_publisher(config.output)
        self.logger = CsvTelemetryLogger(config.logging, config.base_directory)
        self.display = DebugDisplay(config.display)
        self.stats = RuntimeStats()

    def _verify_vision_setup(self) -> None:
        issues: list[str] = []
        roi = self.config.roi
        if (
            roi.x == 0
            and roi.y == 0
            and roi.width >= self.config.camera.width
            and roi.height >= self.config.camera.height
        ):
            issues.append("ROI仍是全画面占位值，请先运行 tools/select_roi.py")
        if self.config.detector.threshold.background_enabled:
            background_path = self.config.resolve_path(
                self.config.detector.threshold.background_path
            )
            if not background_path.exists():
                issues.append(
                    f"空管背景不存在: {background_path}，请先运行 tools/capture_background.py"
                )
        if len(self.config.calibration.points) < 5:
            issues.append("毫米标定点少于5个，请先运行 tools/calibrate_position.py")
        camera = self.config.camera
        source_text = str(camera.source)
        live_source = (
            isinstance(camera.source, int)
            or source_text.isdigit()
            or source_text.startswith(("/dev/video", "/dev/v4l/"))
            or bool(camera.gstreamer_pipeline)
        )
        if live_source:
            if camera.controls:
                if not camera.strict_control_readback:
                    issues.append("已配置摄像头控制值，但strict_control_readback尚未开启")
            elif not camera.controls_verified_externally:
                issues.append(
                    "尚未证明曝光/增益/白平衡/对焦已锁定；请配置camera.controls，"
                    "或在外部读回验证后设置controls_verified_externally=true"
                )
        if issues:
            raise RuntimeError("正式视觉配置未就绪:\n  - " + "\n  - ".join(issues))

    def _print_contract(self) -> None:
        print("\n=== Vision capture contract ===")
        for key, value in self.capture.negotiated.items():
            print(f"{key:>14}: {value}")
        print(f"{'requested':>14}: {self.config.camera.width}x{self.config.camera.height} @ {self.config.camera.fps:.1f} {self.config.camera.fourcc}")
        print(f"{'opencv':>14}: {cv2.__version__}")
        print("注意：OpenCV到达时间不是UVC真实曝光时间；最终延迟需LED/逻辑分析仪标定。")
        print("===============================\n")

    def run(self) -> None:
        last_sequence = -1
        self.capture.start()
        try:
            first = self.capture.latest(timeout=3.0)
            if first is None:
                raise RuntimeError("启动后3秒内没有收到有效图像")
            self.pipeline.prepare_background_for_frame(first.image)
            self._print_contract()
            if self.capture.is_live_source and self.config.camera.strict_contract:
                time.sleep(2.0)
                self.capture.verify_contract()

            while True:
                packet = self.capture.latest(
                    after_sequence=last_sequence,
                    timeout=self.config.camera.read_timeout_seconds,
                )
                if packet is None:
                    if not self.capture.is_live_source and not self.capture.running:
                        print("录像回放结束")
                        break
                    print("[WARN] 等待新帧超时，控制端应依据frame age进入LOST")
                    continue
                skipped = 0 if last_sequence < 0 else max(0, packet.sequence - last_sequence - 1)
                last_sequence = packet.sequence
                result = self.pipeline.process(packet)
                encoded = encode_telemetry(
                    result.estimate,
                    frame_sequence=packet.sequence,
                    frame_age_us=result.frame_age_us,
                    processing_us=result.processing_us,
                )
                try:
                    self.publisher.publish(encoded)
                except OSError as error:
                    print("[WARN] 遥测输出失败:", error)
                self.logger.write(result, skipped)
                self.stats.add(
                    result.processing_us,
                    result.frame_age_us,
                    result.estimate.valid,
                    skipped,
                )
                key = self.display.show(result)
                if key in (ord("q"), 27):
                    break
        finally:
            self.close()

    def close(self) -> None:
        self.capture.stop()
        self.publisher.close()
        self.logger.close()
        self.display.close()
        if self.config.runtime.disable_python_gc and self._gc_was_enabled:
            gc.enable()
            gc.collect()
        capture_stats = self.capture.stats()
        processing = self.stats.summary()
        print("\n=== Runtime summary ===")
        print(f"capture fps       : {capture_stats.measured_fps:.2f}")
        print(f"capture p95/p99   : {capture_stats.interval_p95_ms:.2f}/{capture_stats.interval_p99_ms:.2f} ms")
        print(f"processing p95/p99: {processing.processing_p95_us / 1000:.3f}/{processing.processing_p99_us / 1000:.3f} ms")
        print(f"frame age p95     : {processing.frame_age_p95_us / 1000:.3f} ms")
        print(f"measured rate     : {processing.measured_rate * 100:.3f}%")
        print(f"consumer skipped  : {processing.skipped_frames}")
        if self.logger.path:
            print(f"CSV log           : {self.logger.path}")
        print("=======================")
