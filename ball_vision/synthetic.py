from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class SyntheticFrame:
    image: np.ndarray
    center_px: tuple[float, float]
    radius_px: float


def make_pipe_background(width: int = 960, height: int = 160, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.arange(height, dtype=np.float32)[:, None]
    center = (height - 1) * 0.5
    base = 205.0 - 18.0 * np.exp(-((y - center) / (height * 0.22)) ** 2)
    gray = np.repeat(base, width, axis=1)
    # Fixed pipe edges and legal scale-like marks are deliberate distractors.
    edge_y0 = int(height * 0.18)
    edge_y1 = int(height * 0.82)
    cv2.line(gray, (0, edge_y0), (width - 1, edge_y0), 115, 2)
    cv2.line(gray, (0, edge_y1), (width - 1, edge_y1), 120, 2)
    for x in range(20, width, 38):
        cv2.line(gray, (x, edge_y1 + 2), (x, min(height - 1, edge_y1 + 11)), 95, 1)
    gray += rng.normal(0.0, 1.2, gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def render_ball(
    background: np.ndarray,
    center_px: tuple[float, float],
    radius_px: float = 22.0,
    noise_sigma: float = 1.5,
    highlight_strength: float = 105.0,
    occlusion_fraction: float = 0.0,
    jpeg_quality: int | None = None,
    seed: int = 2,
    supersample: int = 4,
) -> SyntheticFrame:
    rng = np.random.default_rng(seed)
    height, width = background.shape[:2]
    scale = max(2, int(supersample))
    high = cv2.resize(background, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    center_x = center_px[0] * scale
    center_y = center_px[1] * scale
    radius = radius_px * scale
    x0 = max(0, int(np.floor(center_x - radius - 2)))
    x1 = min(width * scale, int(np.ceil(center_x + radius + 3)))
    y0 = max(0, int(np.floor(center_y - radius - 2)))
    y1 = min(height * scale, int(np.ceil(center_y + radius + 3)))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = (xx - center_x) / max(1.0, radius)
    dy = (yy - center_y) / max(1.0, radius)
    distance_sq = dx * dx + dy * dy
    mask = distance_sq <= 1.0
    sphere = 46.0 + 35.0 * (0.35 * dx + 0.65 * dy + 1.0)
    highlight = highlight_strength * np.exp(
        -((dx + 0.32) ** 2 + (dy + 0.35) ** 2) / 0.045
    )
    sphere = np.clip(sphere + highlight, 20, 240)
    patch = high[y0:y1, x0:x1]
    for channel in range(3):
        patch_channel = patch[:, :, channel]
        patch_channel[mask] = sphere[mask]
    if occlusion_fraction > 0:
        cover_width = int(np.clip(occlusion_fraction, 0.0, 0.7) * 2.0 * radius)
        cover_x0 = max(x0, int(center_x + radius - cover_width))
        if cover_x0 < x1:
            original = cv2.resize(background, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
            high[y0:y1, cover_x0:x1] = original[y0:y1, cover_x0:x1]
    low = cv2.resize(high, (width, height), interpolation=cv2.INTER_AREA)
    low += rng.normal(0.0, noise_sigma, low.shape)
    low = np.clip(low, 0, 255).astype(np.uint8)
    if jpeg_quality is not None:
        ok, encoded = cv2.imencode(".jpg", low, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise RuntimeError("synthetic JPEG encode failed")
        low = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return SyntheticFrame(image=low, center_px=center_px, radius_px=radius_px)

