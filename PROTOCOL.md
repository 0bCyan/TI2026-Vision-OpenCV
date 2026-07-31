# Vision telemetry protocol V2

The vision computer sends a unidirectional fixed-length binary stream to the
MC02 controller. There is no ACK or command path from MC02 to vision.

## Wire format

Every frame is exactly 25 bytes. All multi-byte integers use little-endian
byte order.

| Offset | Size | Type | Field | Unit / rule |
|---:|---:|---|---|---|
| 0 | 2 | bytes | `HEADER` | Always `AA 55` |
| 2 | 1 | uint8 | `VERSION` | Always `2` |
| 3 | 1 | uint8 | `FLAGS` | Bits 5..7 must be zero |
| 4 | 4 | uint32 | `SESSION_ID` | New random value on every process start |
| 8 | 4 | uint32 | `FRAME_ID` | Camera frame counter; wrap is allowed |
| 12 | 2 | int16 | `POSITION` | Raw current-frame measurement, 0.1 mm/LSB |
| 14 | 2 | uint16 | `SIGMA` | Raw measurement standard deviation, 0.01 mm/LSB |
| 16 | 4 | uint32 | `FRAME_AGE_US` | Exposure estimate to immediately before send |
| 20 | 2 | uint16 | `PROCESSING_US` | Saturates at 65535 us |
| 22 | 1 | uint8 | `CONFIDENCE` | 0..255 |
| 23 | 2 | uint16 | `CRC16` | Little-endian CRC value |

`POSITION`, `SIGMA`, and `CONFIDENCE` describe only a real measurement from
the current image. A tracker prediction must never be placed in these fields.
When `MEASUREMENT_VALID` is clear, the encoder writes all three fields as zero
and MC02 ignores them.

## Physical position convention

`POSITION` uses the MC02 beam coordinate, independent of image orientation:

- The mechanical center of the beam is exactly `0 mm`.
- Motion from the center toward the vehicle front is positive.
- Motion from the center toward the vehicle rear is negative.
- Production calibration must include the `0 mm` point, cover both signs, and
  remain within the MC02 accepted range of `-125..+125 mm`.

Camera mounting may make positive motion appear leftward or rightward in the
image. The pixel-to-millimeter calibration captures that orientation; neither
side may infer the physical sign from pixel direction.

## Flags

```text
bit0 MEASUREMENT_VALID
bit1 PREDICTED
bit2 LOST
bit3 REACQUIRED
bit4 NEAR_EDGE
bit5..7 reserved, always zero
```

The legal state combinations are:

- `NONE`, `PREDICTED`, or `LOST` when there is no current measurement.
- `MEASUREMENT_VALID`, optionally with `REACQUIRED` and/or `NEAR_EDGE`, when
  the packet carries a current-frame measurement.

`MEASUREMENT_VALID` is mutually exclusive with `PREDICTED` and `LOST`.
`PREDICTED` is mutually exclusive with `LOST`. `REACQUIRED` and `NEAR_EDGE`
both require `MEASUREMENT_VALID`. Initial acquisition is not `REACQUIRED`.

## CRC

The checksum is CRC-16/CCITT-FALSE:

```text
width   = 16
poly    = 0x1021
init    = 0xFFFF
refin   = false
refout  = false
xorout  = 0x0000
check("123456789") = 0x29B1
```

CRC covers bytes 2 through 22 inclusive (`VERSION` through `CONFIDENCE`). It
does not cover the two-byte header or the CRC field itself. The result is
placed at offsets 23..24 in little-endian order.

## Golden frame

Both Python and MC02 tests must decode this hard-coded frame and reproduce it
from the listed values. A test must not generate its own expected bytes with
the implementation under test.

```text
AA 55 02 01 44 33 22 11 D4 C3 B2 A1 85 FF FA 00
39 30 00 00 29 09 C8 EE 09
```

```text
VERSION       = 2
FLAGS         = MEASUREMENT_VALID
SESSION_ID    = 0x11223344
FRAME_ID      = 0xA1B2C3D4
POSITION      = -123  (-12.3 mm)
SIGMA         = 250   (2.50 mm)
FRAME_AGE_US  = 12345
PROCESSING_US = 2345
CONFIDENCE    = 200
CRC16         = 0x09EE
```

At 921600 baud, 8N1, one frame occupies 250 serial bits, approximately
271.3 us. This wire time is not included in `FRAME_AGE_US`; MC02 accounts for
it when reconstructing the capture time from the final-byte receive time.
