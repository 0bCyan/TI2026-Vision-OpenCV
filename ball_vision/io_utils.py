from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_path(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Unicode-safe image read on Windows and normal read on Linux."""
    path = Path(path)
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def imwrite_path(path: str | Path, image: np.ndarray) -> bool:
    """Unicode-safe image write using the suffix as the codec selector."""
    path = Path(path)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True

