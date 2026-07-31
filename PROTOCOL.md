# 视觉遥测协议 v1

固定长度25字节，小端序。帧由`ball_vision.protocol.encode_telemetry()`生成，CRC采用CRC-16/CCITT-FALSE，多项式`0x1021`、初值`0xFFFF`。

| 偏移 | 长度 | 类型 | 内容 |
|---:|---:|---|---|
| 0 | 2 | bytes | 帧头`AA 55` |
| 2 | 1 | uint8 | 协议版本，当前为1 |
| 3 | 1 | uint8 | 状态flags |
| 4 | 4 | uint32 | 帧序号 |
| 8 | 4 | uint32 | `frame_time_us`低32位 |
| 12 | 2 | int16 | 位置，0.1 mm/LSB |
| 14 | 2 | int16 | 速度，1 mm/s/LSB |
| 16 | 1 | uint8 | 置信度，0～255 |
| 17 | 2 | uint16 | 位置标准差，0.01 mm/LSB |
| 19 | 2 | uint16 | 当前帧数据年龄，µs |
| 21 | 2 | uint16 | 视觉处理时间，µs |
| 23 | 2 | uint16 | CRC16 |

Flags：

```text
bit0 VALID
bit1 PREDICTED
bit2 LOST
bit3 REACQUIRED
bit4 NEAR_EDGE
```

控制器只能把`VALID`视为本帧真实测量。`PREDICTED`可用于短时平滑或监控，但不能重置视觉失联计时器。

MCU必须实现独立通信看门狗，不能依赖Python进程主动发送`LOST`：连续约20～30 ms没有收到CRC正确且序号更新的新包，就按视觉失联处理并进入安全策略。OpenCV的部分原生后端可能阻塞在`read()`中，此时应用层无法生成新的LOST包；通信看门狗是硬性安全边界。

`frame_time_us`和MCU不一定共用时钟，主要用于视觉帧间`dt`和回绕安全序号；MCU侧实时补偿优先使用`frame_age_us`与离线LED实验得到的固定延迟。

建议串口波特率921600。每帧25字节、180 Hz时净载荷约4.5 kB/s，带宽余量充足。
