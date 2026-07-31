from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"配置段 {name!r} 必须是映射")
    return dict(value)


@dataclass(slots=True)
class CameraConfig:
    source: Any = 0
    backend: str = "dshow"
    width: int = 1280
    height: int = 720
    fps: float = 180.0
    fourcc: str = "MJPG"
    buffer_size: int = 1
    gstreamer_pipeline: str = ""
    reconnect_seconds: float = 1.0
    read_timeout_seconds: float = 0.05
    timestamp_offset_ms: float = 0.0
    warmup_seconds: float = 0.5
    strict_contract: bool = False
    min_measured_fps_ratio: float = 0.94
    max_interval_p99_ms: float = 15.0
    max_frame_gap_ms: float = 25.0
    max_read_p95_ms: float = 10.0
    max_read_failures: int = 2
    controls: dict[str, float] = field(default_factory=dict)
    strict_control_readback: bool = False
    controls_verified_externally: bool = False
    control_tolerance: float = 0.5
    file_realtime: bool = True
    fingerprint_frames: bool = False
    reject_identical_frames: bool = False
    freshness_verified_externally: bool = False
    max_duplicate_ratio: float = 0.001


@dataclass(slots=True)
class RoiConfig:
    x: int = 0
    y: int = 0
    width: int = 1280
    height: int = 720

    def clamp(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x = max(0, min(int(self.x), frame_width - 1))
        y = max(0, min(int(self.y), frame_height - 1))
        width = max(1, min(int(self.width), frame_width - x))
        height = max(1, min(int(self.height), frame_height - y))
        return x, y, width, height


@dataclass(slots=True)
class ThresholdConfig:
    polarity: str = "dark"
    median_blur_size: int = 5
    min_contrast_delta: float = 18.0
    std_gain: float = 0.8
    threshold_min: int = 25
    threshold_max: int = 180
    background_enabled: bool = True
    background_path: str = "data/background.png"
    background_diff_threshold: int = 18
    background_brightness_align: bool = True
    combine_mode: str = "and"
    morph_open_size: int = 3
    morph_close_size: int = 5


@dataclass(slots=True)
class DetectorConfig:
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    radius_min_px: float = 14.0
    radius_max_px: float = 36.0
    expected_radius_px: float = 22.0
    radius_tolerance_search_ratio: float = 0.32
    radius_tolerance_track_ratio: float = 0.25
    area_min_px: float = 400.0
    area_max_px: float = 5000.0
    circularity_min: float = 0.40
    fill_ratio_min: float = 0.42
    aspect_ratio_min: float = 0.55
    aspect_ratio_max: float = 1.75
    channel_center_y_px: float = -1.0
    max_center_y_error_px: float = -1.0
    border_margin_px: int = 2
    max_prediction_distance_px: float = 150.0
    ransac_iterations: int = 40
    ransac_residual_px: float = 2.2
    fit_rms_max_px: float = 4.0
    fit_min_inlier_ratio: float = 0.45
    contour_sample_limit: int = 240
    edge_refine_enabled: bool = True
    edge_annulus_ratio: float = 0.35
    edge_min_gradient: float = 18.0
    edge_radial_alignment_min: float = 0.45
    edge_center_shift_max_ratio: float = 0.35
    edge_ransac_iterations: int = 24
    edge_ransac_residual_px: float = 1.6
    arc_coverage_min: float = 0.30
    search_confidence_min: float = 0.65
    track_confidence_min: float = 0.45
    ambiguity_score_gap: float = 8.0
    ambiguity_confidence_scale: float = 0.55
    return_debug_mask: bool = True


@dataclass(slots=True)
class CalibrationConfig:
    mode: str = "piecewise_linear"
    degree: int = 1
    points: list[list[float]] = field(
        default_factory=lambda: [[0.0, -125.0], [1279.0, 125.0]]
    )


@dataclass(slots=True)
class MeasurementConfig:
    minimum_center_sigma_px: float = 0.35
    calibration_rms_mm: float = 0.50
    near_edge_margin_mm: float = 10.0
    near_edge_multiplier: float = 1.5
    maximum_valid_sigma_mm: float = 3.0


@dataclass(slots=True)
class TrackerConfig:
    process_accel_sigma_mm_s2: float = 2500.0
    measurement_sigma_mm: float = 0.7
    initial_position_sigma_mm: float = 3.0
    initial_velocity_sigma_mm_s: float = 250.0
    gate_sigma: float = 5.0
    gate_min_mm: float = 8.0
    confirmation_frames: int = 2
    max_prediction_ms: float = 25.0
    lost_reset_ms: float = 80.0
    min_dt_seconds: float = 0.001
    max_dt_seconds: float = 0.050
    velocity_limit_mm_s: float = 1800.0


@dataclass(slots=True)
class OutputConfig:
    serial_enabled: bool = False
    serial_port: str = "COM3"
    serial_baudrate: int = 921600
    serial_reconnect_seconds: float = 0.25
    udp_enabled: bool = False
    udp_host: str = "127.0.0.1"
    udp_port: int = 25060
    max_frame_age_us: int = 80_000


@dataclass(slots=True)
class DisplayConfig:
    enabled: bool = True
    max_fps: float = 30.0
    scale: float = 0.65
    show_mask: bool = True
    draw_candidates: bool = True


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    directory: str = "logs"
    flush_interval_seconds: float = 1.0


@dataclass(slots=True)
class RuntimeConfig:
    opencv_threads: int = 1
    opencl_enabled: bool = False
    disable_python_gc: bool = True


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    config_path: Path | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        detector_raw = _section(raw, "detector")
        threshold = ThresholdConfig(**dict(detector_raw.pop("threshold", {}) or {}))
        config = cls(
            camera=CameraConfig(**_section(raw, "camera")),
            roi=RoiConfig(**_section(raw, "roi")),
            detector=DetectorConfig(threshold=threshold, **detector_raw),
            calibration=CalibrationConfig(**_section(raw, "calibration")),
            measurement=MeasurementConfig(**_section(raw, "measurement")),
            tracker=TrackerConfig(**_section(raw, "tracker")),
            output=OutputConfig(**_section(raw, "output")),
            display=DisplayConfig(**_section(raw, "display")),
            logging=LoggingConfig(**_section(raw, "logging")),
            runtime=RuntimeConfig(**_section(raw, "runtime")),
            config_path=config_path,
        )
        config.validate()
        return config

    @property
    def base_directory(self) -> Path:
        return self.config_path.parent.parent if self.config_path else Path.cwd()

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.base_directory / path).resolve()

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path).resolve() if path else self.config_path
        if target is None:
            raise ValueError("未指定配置文件保存路径")
        payload = asdict(self)
        payload.pop("config_path", None)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
        self.config_path = target

    def validate(self) -> None:
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ValueError("摄像头分辨率和帧率必须大于0")
        if (
            not math.isfinite(self.camera.timestamp_offset_ms)
            or self.camera.timestamp_offset_ms < 0
        ):
            raise ValueError("camera.timestamp_offset_ms必须是非负有限数")
        if self.camera.buffer_size < 1:
            raise ValueError("camera.buffer_size 必须至少为1")
        if self.camera.read_timeout_seconds <= 0 or self.camera.reconnect_seconds <= 0:
            raise ValueError("摄像头读取超时和重连时间必须大于0")
        if not 0 < self.camera.min_measured_fps_ratio <= 1:
            raise ValueError("camera.min_measured_fps_ratio 必须在(0, 1]内")
        if not 0 <= self.camera.max_duplicate_ratio < 1:
            raise ValueError("camera.max_duplicate_ratio 必须在[0, 1)内")
        if self.camera.reject_identical_frames and not self.camera.fingerprint_frames:
            raise ValueError(
                "camera.reject_identical_frames=true 时必须同时启用 fingerprint_frames"
            )
        if self.camera.controls_verified_externally and self.camera.controls:
            raise ValueError(
                "外部控制模式下camera.controls必须为空，避免OpenCV覆盖v4l2-ctl设置"
            )
        if (
            self.camera.max_interval_p99_ms <= 0
            or self.camera.max_frame_gap_ms <= 0
            or self.camera.max_read_p95_ms <= 0
            or self.camera.max_read_failures < 0
        ):
            raise ValueError("摄像头合同的延迟与失败阈值无效")
        supported_controls = {
            "auto_exposure",
            "exposure",
            "gain",
            "auto_wb",
            "wb_temperature",
            "autofocus",
            "focus",
            "brightness",
            "contrast",
        }
        unknown_controls = set(self.camera.controls) - supported_controls
        if unknown_controls:
            raise ValueError(f"不支持的OpenCV摄像头控制项: {sorted(unknown_controls)}")
        try:
            controls_are_finite = all(
                math.isfinite(float(value)) for value in self.camera.controls.values()
            )
        except (TypeError, ValueError):
            controls_are_finite = False
        if not controls_are_finite:
            raise ValueError("camera.controls 的值必须是有限数值")
        if self.camera.control_tolerance < 0:
            raise ValueError("camera.control_tolerance 不能为负")
        if len(self.camera.fourcc) != 4:
            raise ValueError("camera.fourcc 必须恰好包含4个字符")
        if self.camera.backend.lower() == "gstreamer" and not self.camera.gstreamer_pipeline:
            raise ValueError("backend=gstreamer 时必须配置 camera.gstreamer_pipeline")
        if self.roi.x < 0 or self.roi.y < 0 or self.roi.width <= 0 or self.roi.height <= 0:
            raise ValueError("ROI坐标必须非负且尺寸必须大于0")
        if self.detector.threshold.polarity not in {"dark", "bright"}:
            raise ValueError("detector.threshold.polarity 必须是 dark 或 bright")
        if self.detector.threshold.combine_mode not in {"or", "and", "intensity", "background"}:
            raise ValueError("不支持的二值掩膜组合方式")
        threshold = self.detector.threshold
        if not 0 <= threshold.threshold_min < threshold.threshold_max <= 255:
            raise ValueError("灰度阈值上下限必须满足 0 <= min < max <= 255")
        if not 1 <= threshold.background_diff_threshold <= 255:
            raise ValueError("background_diff_threshold 必须在[1, 255]内")
        if min(
            threshold.median_blur_size,
            threshold.morph_open_size,
            threshold.morph_close_size,
        ) < 1:
            raise ValueError("滤波与形态学核尺寸必须至少为1")
        if self.detector.radius_min_px <= 0 or self.detector.radius_max_px <= self.detector.radius_min_px:
            raise ValueError("钢球半径范围无效")
        if self.detector.expected_radius_px <= 0:
            raise ValueError("detector.expected_radius_px 必须大于0")
        if not self.detector.radius_min_px <= self.detector.expected_radius_px <= self.detector.radius_max_px:
            raise ValueError("expected_radius_px 必须落在钢球半径硬范围内")
        if not 0 < self.detector.radius_tolerance_track_ratio <= 1:
            raise ValueError("detector.radius_tolerance_track_ratio 必须在(0, 1]内")
        if not 0 < self.detector.radius_tolerance_search_ratio <= 1:
            raise ValueError("detector.radius_tolerance_search_ratio 必须在(0, 1]内")
        if self.detector.radius_tolerance_track_ratio > self.detector.radius_tolerance_search_ratio:
            raise ValueError("跟踪半径容差不能宽于搜索半径容差")
        if self.detector.area_min_px <= 0 or self.detector.area_max_px <= self.detector.area_min_px:
            raise ValueError("候选面积范围无效")
        if (
            self.detector.aspect_ratio_min <= 0
            or self.detector.aspect_ratio_max <= self.detector.aspect_ratio_min
        ):
            raise ValueError("候选宽高比范围无效")
        if self.detector.ransac_iterations < 8:
            raise ValueError("ransac_iterations 必须至少为8")
        if self.detector.edge_ransac_iterations < 8:
            raise ValueError("edge_ransac_iterations 必须至少为8")
        if self.detector.contour_sample_limit < 6:
            raise ValueError("contour_sample_limit 必须至少为6")
        if min(
            self.detector.ransac_residual_px,
            self.detector.fit_rms_max_px,
            self.detector.edge_ransac_residual_px,
            self.detector.edge_min_gradient,
            self.detector.max_prediction_distance_px,
        ) <= 0:
            raise ValueError("拟合残差、梯度和预测距离阈值必须大于0")
        unit_interval_values = (
            self.detector.circularity_min,
            self.detector.fill_ratio_min,
            self.detector.fit_min_inlier_ratio,
            self.detector.edge_annulus_ratio,
            self.detector.edge_radial_alignment_min,
            self.detector.edge_center_shift_max_ratio,
            self.detector.arc_coverage_min,
            self.detector.search_confidence_min,
            self.detector.track_confidence_min,
            self.detector.ambiguity_confidence_scale,
        )
        if any(not 0 <= value <= 1 for value in unit_interval_values):
            raise ValueError("检测器的比例与置信度参数必须在[0, 1]内")
        if self.detector.track_confidence_min > self.detector.search_confidence_min:
            raise ValueError("跟踪接收置信度不能高于搜索接收置信度")
        if self.detector.ambiguity_score_gap < 0 or self.detector.border_margin_px < 0:
            raise ValueError("歧义分差和边界余量不能为负")
        points: Sequence[Sequence[float]] = self.calibration.points
        if len(points) < 2 or any(len(point) != 2 for point in points):
            raise ValueError("calibration.points 至少需要两个 [pixel, mm] 标定点")
        if self.calibration.mode == "polynomial" and (
            self.calibration.degree < 1 or self.calibration.degree >= len(points)
        ):
            raise ValueError("标定多项式阶数必须小于标定点数量")
        if self.calibration.mode not in {"piecewise_linear", "polynomial"}:
            raise ValueError("calibration.mode 必须是 piecewise_linear 或 polynomial")
        measurement = self.measurement
        measurement_values = (
            measurement.minimum_center_sigma_px,
            measurement.calibration_rms_mm,
            measurement.near_edge_margin_mm,
            measurement.near_edge_multiplier,
            measurement.maximum_valid_sigma_mm,
        )
        if not all(math.isfinite(value) for value in measurement_values):
            raise ValueError("measurement noise parameters must be finite")
        if measurement.minimum_center_sigma_px <= 0:
            raise ValueError("measurement.minimum_center_sigma_px must be greater than zero")
        if measurement.calibration_rms_mm <= 0:
            raise ValueError("measurement.calibration_rms_mm must be greater than zero")
        if measurement.near_edge_margin_mm < 0:
            raise ValueError("measurement.near_edge_margin_mm cannot be negative")
        if measurement.near_edge_multiplier < 1:
            raise ValueError("measurement.near_edge_multiplier must be at least one")
        if not 0 < measurement.maximum_valid_sigma_mm <= 655.35:
            raise ValueError(
                "measurement.maximum_valid_sigma_mm must be in (0, 655.35]"
            )
        if (
            measurement.calibration_rms_mm * measurement.near_edge_multiplier
            > measurement.maximum_valid_sigma_mm
        ):
            raise ValueError(
                "measurement noise floor exceeds maximum_valid_sigma_mm near an edge"
            )
        millimeter_values = [float(point[1]) for point in points]
        calibration_span_mm = max(millimeter_values) - min(millimeter_values)
        if measurement.near_edge_margin_mm * 2 >= calibration_span_mm:
            raise ValueError(
                "measurement.near_edge_margin_mm must be less than half the calibrated span"
            )
        tracker = self.tracker
        if min(
            tracker.process_accel_sigma_mm_s2,
            tracker.measurement_sigma_mm,
            tracker.initial_position_sigma_mm,
            tracker.initial_velocity_sigma_mm_s,
            tracker.gate_sigma,
            tracker.gate_min_mm,
            tracker.min_dt_seconds,
            tracker.max_dt_seconds,
            tracker.velocity_limit_mm_s,
        ) <= 0:
            raise ValueError("跟踪器噪声、门限、dt和速度上限必须大于0")
        if tracker.min_dt_seconds > tracker.max_dt_seconds:
            raise ValueError("tracker.min_dt_seconds 不能大于 max_dt_seconds")
        if tracker.confirmation_frames < 1:
            raise ValueError("tracker.confirmation_frames 必须至少为1")
        if not 0 < tracker.max_prediction_ms < tracker.lost_reset_ms:
            raise ValueError("必须满足 0 < max_prediction_ms < lost_reset_ms")
        if self.output.udp_port <= 0 or self.output.udp_port > 65535:
            raise ValueError("UDP端口必须在[1, 65535]内")
        if self.output.serial_baudrate != 921600:
            raise ValueError("MC02视觉串口固定使用921600 baud")
        if self.output.serial_enabled and not self.output.serial_port.strip():
            raise ValueError("启用串口输出时output.serial_port不能为空")
        if self.output.serial_reconnect_seconds <= 0:
            raise ValueError("output.serial_reconnect_seconds必须大于0")
        if not 1_000 <= self.output.max_frame_age_us <= 80_000:
            raise ValueError("output.max_frame_age_us 必须在[1000, 80000]内")
        if self.display.max_fps <= 0 or self.display.scale <= 0:
            raise ValueError("显示帧率和缩放必须大于0")
        if self.logging.flush_interval_seconds <= 0:
            raise ValueError("日志刷新间隔必须大于0")
        if self.runtime.opencv_threads < 1:
            raise ValueError("runtime.opencv_threads 必须至少为1")
