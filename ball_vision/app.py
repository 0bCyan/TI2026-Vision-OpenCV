from __future__ import annotations

import gc
import math
import secrets
import time

import cv2

from .capture import LatestFrameCapture
from .config import (
    POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE,
    AppConfig,
)
from .output import LatestTelemetryPublisher, create_publisher
from .frame_output import LatestMjpegPublisher
from .pipeline import PipelineResult, VisionPipeline
from .protocol import VisionFlags, encode_telemetry
from .runtime_stats import RuntimeStats
from .telemetry_log import CsvTelemetryLogger
from .types import TrackFlags
from .visualization import DebugDisplay


class VisionApplication:
    _MC02_POSITION_LIMIT_MM = 125.0

    def __init__(
        self,
        config: AppConfig,
        *,
        bench_serial: bool = False,
    ) -> None:
        self.config = config
        if bench_serial and not config.output.serial_enabled:
            raise ValueError("bench_serial要求启用串口输出")
        verified_output_required = (
            config.output.udp_enabled
            or (config.output.serial_enabled and not bench_serial)
        )
        if verified_output_required:
            config.camera.strict_contract = True
        self._production_output = (
            config.camera.strict_contract
            or verified_output_required
        )
        if self._production_output:
            self._verify_vision_setup()
        elif bench_serial:
            print(
                "[WARN] BENCH SERIAL已启用：正在发送未通过正式合同的测量，"
                "仅限MC02电机禁能联调，禁止闭环控制"
            )
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
        self.frame_publisher = LatestMjpegPublisher(config.frame_stream)
        self.publisher = LatestTelemetryPublisher(create_publisher(config.output))
        self.session_id = secrets.randbits(32)
        self.logger = CsvTelemetryLogger(config.logging, config.base_directory)
        self.display = DebugDisplay(config.display)
        self.stats = RuntimeStats()

    def _verify_vision_setup(self) -> None:
        issues: list[str] = []
        roi = self.config.roi
        if not roi.verified:
            issues.append("ROI尚未通过实车选择确认，请先运行 tools/select_roi.py")
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
        if (
            self.config.calibration.coordinate_convention
            != POSITION_CONVENTION_VEHICLE_FRONT_POSITIVE
        ):
            issues.append(
                "位置正方向未确认：必须为横梁中心0、车头方向正、车尾方向负"
            )
        millimeter_values = [float(point[1]) for point in self.config.calibration.points]
        if not millimeter_values or not all(
            math.isfinite(value) for value in millimeter_values
        ):
            issues.append("毫米标定坐标必须全部是有限数")
        else:
            if not any(abs(value) <= 1.0e-6 for value in millimeter_values):
                issues.append("毫米标定缺少横梁机械中心0 mm标定点")
            if not (min(millimeter_values) < 0.0 < max(millimeter_values)):
                issues.append("毫米标定必须同时覆盖车尾负坐标和车头正坐标")
            if (
                min(millimeter_values) < -self._MC02_POSITION_LIMIT_MM
                or max(millimeter_values) > self._MC02_POSITION_LIMIT_MM
            ):
                issues.append("毫米标定超出MC02允许的-125..+125 mm范围")
        camera = self.config.camera
        source_text = str(camera.source)
        live_source = (
            isinstance(camera.source, int)
            or source_text.isdigit()
            or source_text.startswith(("/dev/video", "/dev/v4l/"))
            or bool(camera.gstreamer_pipeline)
        )
        if self.config.output.serial_enabled:
            backend = camera.backend.lower()
            if backend == "v4l2" and not source_text.startswith("/dev/v4l/by-id/"):
                issues.append(
                    "正式V4L2串口输出必须使用稳定的/dev/v4l/by-id/...相机路径"
                )
            if (
                backend == "gstreamer"
                and "/dev/v4l/by-id/" not in camera.gstreamer_pipeline
            ):
                issues.append(
                    "正式GStreamer串口输出必须在管线中使用/dev/v4l/by-id/..."
                )
        if live_source:
            if (
                not camera.reject_identical_frames
                and not camera.freshness_verified_externally
            ):
                issues.append(
                    "尚未建立可信帧新鲜度：请在目标相机验证后启用"
                    "reject_identical_frames，或由原生采集服务校验sequence/PTS后设置"
                    "freshness_verified_externally=true"
                )
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
            next_contract_check = time.monotonic() + 1.0
            next_timeout_warning = 0.0

            while True:
                packet = self.capture.latest(
                    after_sequence=last_sequence,
                    timeout=self.config.camera.read_timeout_seconds,
                )
                if packet is None:
                    if not self.capture.is_live_source and not self.capture.running:
                        print("录像回放结束")
                        break
                    now = time.monotonic()
                    if now >= next_timeout_warning:
                        print("[WARN] 等待新帧超时，控制端应依据frame age进入LOST")
                        next_timeout_warning = now + 5.0
                    continue
                skipped = 0 if last_sequence < 0 else max(0, packet.sequence - last_sequence - 1)
                last_sequence = packet.sequence
                self.frame_publisher.submit(packet.image, packet.frame_time_ns)
                result = self.pipeline.process(packet)
                self._submit_telemetry(result)
                if (
                    self.capture.is_live_source
                    and self.config.camera.strict_contract
                    and time.monotonic() >= next_contract_check
                ):
                    self.capture.verify_contract()
                    next_contract_check = time.monotonic() + 1.0
                self.logger.write(result, skipped)
                self._record_runtime_stats(result, skipped)
                key = self.display.show(result)
                if key in (ord("q"), 27):
                    break
        finally:
            self.close()

    def _submit_telemetry(self, result: PipelineResult) -> None:
        measurement = result.raw_measurement
        estimate = result.estimate
        if measurement.valid:
            flags = VisionFlags.MEASUREMENT_VALID
            if measurement.near_edge:
                flags |= VisionFlags.NEAR_EDGE
            if estimate.flags & TrackFlags.REACQUIRED:
                flags |= VisionFlags.REACQUIRED
        elif estimate.flags & TrackFlags.PREDICTED:
            flags = VisionFlags.PREDICTED
        else:
            flags = VisionFlags.LOST

        session_id = self.session_id
        frame_id = result.frame.sequence
        frame_time_ns = result.frame.frame_time_ns
        processing_us = result.processing_us
        position_mm = measurement.position_mm if measurement.valid else None
        sigma_mm = measurement.sigma_mm if measurement.valid else None
        confidence = (
            round(max(0.0, min(1.0, measurement.confidence)) * 255.0)
            if measurement.valid
            else None
        )
        maximum_age_us = self.config.output.max_frame_age_us

        def encode_at(send_time_ns: int) -> bytes:
            frame_age_us = max(0, (send_time_ns - frame_time_ns + 999) // 1000)
            if frame_age_us > maximum_age_us:
                raise ValueError(
                    f"结果过期: frame_age_us={frame_age_us} > {maximum_age_us}"
                )
            return encode_telemetry(
                flags=flags,
                session_id=session_id,
                frame_id=frame_id,
                position_mm=position_mm,
                sigma_mm=sigma_mm,
                frame_age_us=frame_age_us,
                processing_us=processing_us,
                confidence=confidence,
            )

        self.publisher.submit(encode_at)

    def _record_runtime_stats(self, result: PipelineResult, skipped: int) -> None:
        # The public protocol reports only raw current-frame measurements.
        # Tracker validity can legitimately differ during confirmation or a
        # protocol-level high-sigma rejection.
        self.stats.add(
            result.processing_us,
            result.frame_age_us,
            result.raw_measurement.valid,
            skipped,
        )

    def close(self) -> None:
        self.capture.stop()
        self.frame_publisher.close()
        self.publisher.close()
        self.logger.close()
        self.display.close()
        if self.config.runtime.disable_python_gc and self._gc_was_enabled:
            gc.enable()
            gc.collect()
        capture_stats = self.capture.stats()
        processing = self.stats.summary()
        publisher_stats = self.publisher.stats()
        frame_publisher_stats = self.frame_publisher.stats()
        print("\n=== Runtime summary ===")
        print(f"capture fps       : {capture_stats.measured_fps:.2f}")
        print(
            "read failures      : "
            f"{capture_stats.read_failures} total/"
            f"{capture_stats.recent_read_failures} recent"
        )
        print(
            f"unique fps/dup     : {capture_stats.unique_fps:.2f}/"
            f"{capture_stats.duplicate_ratio * 100:.3f}%"
        )
        print(f"capture p95/p99   : {capture_stats.interval_p95_ms:.2f}/{capture_stats.interval_p99_ms:.2f} ms")
        print(f"processing p95/p99: {processing.processing_p95_us / 1000:.3f}/{processing.processing_p99_us / 1000:.3f} ms")
        print(f"frame age p95     : {processing.frame_age_p95_us / 1000:.3f} ms")
        print(f"measured rate     : {processing.measured_rate * 100:.3f}%")
        print(f"consumer skipped  : {processing.skipped_frames}")
        print(
            "telemetry submit/sent/drop/error: "
            f"{publisher_stats.submitted}/{publisher_stats.sent}/"
            f"{publisher_stats.superseded}/{publisher_stats.errors}"
        )
        print(
            "web frames submit/sent/drop/error: "
            f"{frame_publisher_stats.submitted}/{frame_publisher_stats.sent}/"
            f"{frame_publisher_stats.superseded}/"
            f"{frame_publisher_stats.encode_errors + frame_publisher_stats.transport_errors}"
        )
        if self.logger.path:
            print(f"CSV log           : {self.logger.path}")
        print("=======================")
