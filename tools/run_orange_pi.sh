#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${VISION_PYTHON:-"$ROOT/.venv/bin/python"}

: "${MC02_VISION_UART:?set MC02_VISION_UART to the verified Orange Pi UART device}"
: "${VISION_TIMESTAMP_OFFSET_MS:?set VISION_TIMESTAMP_OFFSET_MS from the LED latency test}"
: "${VISION_CONFIG:?set VISION_CONFIG to a verified production YAML file}"
: "${VISION_CAMERA_SOURCE:?set VISION_CAMERA_SOURCE to /dev/v4l/by-id/...}"

if [ ! -x "$PYTHON" ]; then
    echo "[FATAL] Python runtime is not executable: $PYTHON" >&2
    exit 2
fi
if [ ! -c "$MC02_VISION_UART" ]; then
    echo "[FATAL] UART path is not a character device: $MC02_VISION_UART" >&2
    exit 2
fi

case "$VISION_CONFIG" in
    /*) CONFIG_INPUT=$VISION_CONFIG ;;
    *) CONFIG_INPUT=$ROOT/$VISION_CONFIG ;;
esac
if [ ! -f "$CONFIG_INPUT" ]; then
    echo "[FATAL] Vision config does not exist: $CONFIG_INPUT" >&2
    exit 2
fi
CONFIG=$(readlink -f -- "$CONFIG_INPUT")
DEFAULT_CONFIG=$(readlink -f -- "$ROOT/config/default.yaml")
if [ "$CONFIG" = "$DEFAULT_CONFIG" ]; then
    echo "[FATAL] config/default.yaml is a placeholder; use a verified production config" >&2
    exit 2
fi

case "$VISION_CAMERA_SOURCE" in
    /dev/v4l/by-id/*) ;;
    *)
        echo "[FATAL] Camera must use a stable /dev/v4l/by-id/... path" >&2
        exit 2
        ;;
esac
if [ ! -c "$VISION_CAMERA_SOURCE" ]; then
    echo "[FATAL] Camera path is not a character device: $VISION_CAMERA_SOURCE" >&2
    exit 2
fi

exec "$PYTHON" "$ROOT/run_vision.py" \
    --config "$CONFIG" \
    --source "$VISION_CAMERA_SOURCE" \
    --backend v4l2 \
    --headless \
    --strict \
    --production \
    --serial-port "$MC02_VISION_UART" \
    --timestamp-offset-ms "$VISION_TIMESTAMP_OFFSET_MS"
