#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${VISION_PYTHON:-"$ROOT/.venv/bin/python"}

: "${MC02_VISION_UART:?set MC02_VISION_UART to the verified Orange Pi UART device}"
: "${VISION_TIMESTAMP_OFFSET_MS:?set VISION_TIMESTAMP_OFFSET_MS from the LED latency test}"

if [ ! -x "$PYTHON" ]; then
    echo "[FATAL] Python runtime is not executable: $PYTHON" >&2
    exit 2
fi
if [ ! -c "$MC02_VISION_UART" ]; then
    echo "[FATAL] UART path is not a character device: $MC02_VISION_UART" >&2
    exit 2
fi

exec "$PYTHON" "$ROOT/run_vision.py" \
    --config "$ROOT/config/default.yaml" \
    --backend v4l2 \
    --headless \
    --strict \
    --production \
    --serial-port "$MC02_VISION_UART" \
    --timestamp-offset-ms "$VISION_TIMESTAMP_OFFSET_MS"
