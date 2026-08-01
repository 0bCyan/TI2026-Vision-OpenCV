#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${VISION_PYTHON:-/usr/bin/python3}
CONFIG=${VISION_CONFIG:-$ROOT/config/action4_white_pipe.yaml}
RETRY_SECONDS=${VISION_RETRY_SECONDS:-5}
CAMERA_SOURCE=${VISION_CAMERA_SOURCE:-}

if [ ! -x "$PYTHON" ]; then
    echo "[camera-monitor] Python runtime is not executable: $PYTHON" >&2
    exit 2
fi
if [ ! -f "$CONFIG" ]; then
    echo "[camera-monitor] Vision config does not exist: $CONFIG" >&2
    exit 2
fi
case "$RETRY_SECONDS" in
    ''|*[!0-9.]*|.*|*.*.*)
        echo "[camera-monitor] VISION_RETRY_SECONDS must be a positive number" >&2
        exit 2
        ;;
esac

find_camera() {
    if [ -n "$CAMERA_SOURCE" ]; then
        if [ -c "$CAMERA_SOURCE" ]; then
            printf '%s\n' "$CAMERA_SOURCE"
            return 0
        fi
        return 1
    fi

    # Prefer a stable USB identity, then a stable USB port path.
    for candidate in /dev/v4l/by-id/*-video-index0 \
                     /dev/v4l/by-path/*usb*-video-index0; do
        if [ -c "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    # Some UVC devices expose no by-id/by-path link.  Fall back only to a USB
    # capture node; this intentionally ignores the Orange Pi HDMI receiver.
    for candidate in /dev/video*; do
        [ -c "$candidate" ] || continue
        properties=$(udevadm info --query=property --name="$candidate" 2>/dev/null || true)
        if printf '%s\n' "$properties" | grep -qx 'ID_BUS=usb' &&
           printf '%s\n' "$properties" | grep -q '^ID_V4L_CAPABILITIES=.*:capture:'; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

echo "[camera-monitor] waiting for a USB V4L2 camera; scan interval ${RETRY_SECONDS}s"
scan_count=0
while :; do
    if camera=$(find_camera); then
        echo "[camera-monitor] camera found: $camera"
        exec "$PYTHON" "$ROOT/run_vision.py" \
            --config "$CONFIG" \
            --source "$camera" \
            --backend v4l2 \
            --headless
    fi
    scan_count=$((scan_count + 1))
    if [ "$scan_count" -eq 1 ] || [ $((scan_count % 12)) -eq 0 ]; then
        echo "[camera-monitor] no USB camera; web service remains available"
    fi
    sleep "$RETRY_SECONDS"
done
