# TI 2026 H题滚球视觉（OpenCV重构版）

这是一套面向“摄像头随摆杆刚性运动、相机与水管相对固定”的低延迟视觉测量工程。它不直接控制电机，只向实时控制器发送本帧钢球原始位置、原始测量不确定度、状态、会话号、帧号和数据年龄；Tracker速度与滤波位置仅供视觉内部关联和调试。

当前版本优先完成纯 OpenCV 主链，并为后续异步 YOLO-Seg/RKNN 粗分割保留接口。神经网络即使加入，也只提供候选Mask，最终位置仍由原始分辨率边缘圆拟合给出。

## 当前验证状态

- Python语法检查通过。
- 94项自动测试通过：圆拟合、反光、JPEG压缩、原始测量sigma、固定暗干扰、标定、跟踪、预测二维搜索窗、漏检全局恢复、采集合约、帧新鲜度、采集线程失效、串口恢复、重连年龄、最新结果发布及V2协议黄金帧/CRC/边界。
- 本机合成跟踪态基准（当前占位ROI尺寸980×260、半径16 px，3×300帧）：0漏检，圆心误差RMS约0.56 px、P95约0.63 px；一次无并发汇总P50约2.05 ms、P95约2.77 ms、P99约3.18 ms。全ROI搜索与Windows调度尾延迟仍更高，应以目标机多轮结果为准。
- 上述只是算法基准，不包含USB采集和MJPEG解码。
- 当前未连接目标720p/180fps外置UVC摄像头，因此尚未验证真实180fps、重复帧、曝光和端到端延迟。不能把标称帧率当成实测结果。

## 数据流

```text
UVC MJPEG采集线程（只保留最新帧）
    ↓
固定水管ROI
    ↓
空管背景差分 + 稳健暗目标阈值
    ↓
连通域几何硬门限
    ↓
轮廓RANSAC圆拟合
    ↓
原始灰度径向边缘二次精修
    ↓
单调分段线性 像素→毫米 标定
    ↓
原始位置 + 圆拟合/标定测量sigma ──────────────→ V2遥测
    ↓
毫米域可变dt Tracker [位置, 速度]（仅ROI、关联和调试）
    ↓
MEASURED / PREDICTED / LOST / REACQUIRED + CSV日志
```

采集、检测和遥测已解耦。采集线程与遥测线程都采用单槽“最新数据优先”，不会在堵塞后补发一串陈旧结果；串口重连会在选择最新结果和计算`FRAME_AGE_US`之前完成，因此重连等待不能伪装成低年龄。像素指纹默认只用于诊断；若目标相机实测证明静止场景仍有稳定的传感器/MJPEG变化，可显式启用`reject_identical_frames`，使相同画面不再推进`FRAME_ID`。更可靠的正式方案仍是原生V4L2/PyGObject采集服务校验sequence/PTS/FID，再设置`freshness_verified_externally: true`。严格模式会拒绝未选择任何新鲜度策略的实时源。

Tracker确认后，检测器只处理预测点附近、仍完整覆盖原有关联门限和钢球外轮廓的二维窗口，并使用较小的跟踪态RANSAC预算。窗口内一旦没有形成可关联测量，下一帧立即恢复全ROI，并持续全局搜索到重新关联；因此优化不会把错误预测长期固化成局部盲区。

正式控制器必须设置独立通信看门狗：约20～30 ms没有收到CRC正确、序号更新的新包就进入视觉失联安全策略。部分OpenCV原生后端无法保证`read()`可被Python超时中断，因此不能只等待视觉程序主动发送`LOST`。

## Windows安装

建议正式统一使用Python 3.10、3.11或3.12；本工程当前也已在Python 3.13上通过测试。
这里的`requirements.txt`包含Windows本机原型所需的PyPI OpenCV；香橙派不要安装它，板端使用独立的`requirements-orange-pi.txt`和带V4L2/GStreamer的系统OpenCV。

```powershell
cd "D:\BaiduNetdiskDownload\26电赛复刻 H车载平衡滚球运动控制系统\03.完赛代码\K230代码\vision_opencv"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

本次工作区已经在上一级目录创建测试环境，也可以直接运行：

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

## 第一次接入摄像头的强制流程

### 1. 验证真实摄像头合同

分别实测DirectShow与MSMF，不能相信`CAP_PROP_FPS`的报告值：

```powershell
.\.venv\Scripts\python.exe tools\probe_camera.py --source 1 --backend dshow --seconds 5 --strict
.\.venv\Scripts\python.exe tools\probe_camera.py --source 1 --backend msmf --seconds 5 --strict
```

把示例中的摄像头索引`1`替换为目标相机的实际索引；后续所有步骤统一使用实测更好的后端。

正式运行建议至少达到：

- 协商格式确为`MJPG 1280×720@180`；
- 实测接收率不低于约170 fps；
- 无20～25 ms以上的频繁停顿；
- 已建立可信帧新鲜度：目标相机验证后的内容冻结保护，或原生sequence/PTS/FID；
- 曝光固定在1～3 ms；
- 关闭自动曝光、自动增益、自动白平衡、自动对焦。

目标摄像头没有接入时，探测失败属于正常现象。

目标相机接入后，把驱动实际采用的控制值写入`camera.controls`。这些值在首次打开和每次重连时都会重新设置、读回并显示；确认后再把`strict_control_readback`设为`true`。DirectShow/MSMF/V4L2对曝光值的单位和手动模式编码不同，必须以目标设备实测为准，不能照抄其他相机参数。

```yaml
camera:
  fingerprint_frames: true
  reject_identical_frames: true
  freshness_verified_externally: false
  controls:
    auto_exposure: 目标驱动的手动模式值
    exposure: 目标驱动对应1至3ms的值
    gain: 实测低噪声值
    auto_wb: 0
    autofocus: 0
  strict_control_readback: true
```

若香橙派启动脚本使用`v4l2-ctl`设置并读回全部控制项，必须把`camera.controls`设为空映射`{}`，再设置`camera.controls_verified_externally: true`，两种控制方式不能同时启用。systemd负责在进程退出后重新执行“设置→读回→启动”；该模式发生UVC断流时程序会退出，不会在未重设曝光的情况下自行恢复。

`reject_identical_frames`不是所有相机都能安全启用：若静止场景会产生逐字节相同的正常帧，它会主动停发遥测并触发MCU看门狗。必须先用目标相机覆盖“空管静止、球静止、低增益、固定曝光”条件验证。使用能够校验原生sequence/PTS的独立采集服务时，应关闭内容拒绝并设置`freshness_verified_externally: true`；仅使用`cv2.VideoCapture`的到达时间不能视为原生PTS证明。

### 2. 选择固定ROI

ROI要覆盖整个水管、两端挡片以及球心的完整有效行程，上下只留约1.3～1.6个球直径余量：

```powershell
.\.venv\Scripts\python.exe tools\select_roi.py --source 1 --backend dshow
```

### 3. 显式采集空管背景

移走钢球，保留正式刻度、挡片、补光和机械位置：

```powershell
.\.venv\Scripts\python.exe tools\capture_background.py --source 1 --backend dshow --frames 96
```

背景是多帧中位数，不会在正常运行中EMA更新，因此钢球长期静止也不会被背景模型“吃掉”。

虽然相机与水管相对固定，摆杆转动后钢球和透明管壁相对赛场灯光的反射仍会变化。国赛装配应在相机旁增加随杆转动的恒流漫射补光与遮光罩，并在最大正/负摆角录制压力数据；背景算法不能代替稳定光学环境。

### 4. 毫米位置标定

按照提示依次点击已知刻度。默认从-120 mm到+120 mm，每10 mm一点：

```powershell
.\.venv\Scripts\python.exe tools\calibrate_position.py --source 1 --backend dshow
```

正式标定建议5～10 mm间隔。随后必须用错开的5 mm位置独立验证，不能只检查标定点。映射采用单调分段线性插值，避免高阶多项式端部发散。

### 5. 调整球像素尺寸

运行主程序，观察调试画面中的拟合圆半径，并修改[default.yaml](config/default.yaml)中的：

```yaml
radius_min_px: 真实半径的约0.65倍
radius_max_px: 真实半径的约1.35倍
expected_radius_px: 正常半径
area_min_px: 对应最小面积
area_max_px: 对应最大面积
```

不要直接复制K230的固定16～52 px参数。

### 6. 启动

`--strict`会拒绝全画面占位ROI、缺失的空管背景、少于5点的毫米标定、未验证的实时源帧新鲜度，以及不满足帧率/帧间隔/读取耗时合同的摄像头。

```powershell
.\.venv\Scripts\python.exe run_vision.py --source 1 --backend dshow --strict
```

按`Q`或`Esc`退出。无显示运行：

```powershell
.\.venv\Scripts\python.exe run_vision.py --source 1 --backend dshow --strict --headless
```

处理录像：

```powershell
.\.venv\Scripts\python.exe run_vision.py --source data\test.mkv
```

## 合成测试和性能基准

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\benchmark_synthetic.py --frames 300 --mode search
.\.venv\Scripts\python.exe tools\benchmark_synthetic.py --frames 300 --mode track --max-p95-ms 5
.\.venv\Scripts\python.exe tools\benchmark_synthetic.py --frames 300 --mode pipeline
```

`search`测量最坏情况下的全ROI搜索；`track`使用上一帧检测中心验证二维窗口本身；`pipeline`运行真实Kalman关联、首次全局捕获和漏检恢复。`--max-p95-ms`是可选目标机验收门限，不建议把受桌面系统调度影响的单次结果当成最终结论。基准会加入钢球高光、噪声和不同JPEG质量，但不替代真实摄像头、真实管壁反射和端部遮挡数据。

## 输出状态

- `MEASUREMENT_VALID`：当前包包含本帧真实原始测量；
- `PREDICTED`：本帧没测到球，只提供短时预测，绝不能当作有效测量；
- `LOST`：连续丢失，控制器必须进入安全策略；
- `REACQUIRED`：刚完成时空一致确认并重捕；
- `NEAR_EDGE`：球接近标定有效区端部。

视觉输出包括：

```text
SESSION_ID + FRAME_ID
POSITION（原始测量，0.1 mm）
SIGMA（圆拟合和标定组成的原始测量sigma，0.01 mm）
FLAGS + CONFIDENCE
FRAME_AGE_US + PROCESSING_US
CRC16-CCITT-FALSE
```

OpenCV摄像头到达时间只是“解码完成到达应用”的时刻，不是真实曝光时刻。`camera.timestamp_offset_ms`只能补偿实测固定延迟。最终应使用视野内LED和逻辑分析仪测量光子到控制输出的P50/P95/P99。

协议定义见[PROTOCOL.md](PROTOCOL.md)。

## 纯OpenCV验收线与YOLO准入

建议正式视觉目标：

| 指标 | 目标 |
|---|---:|
| 静态全行程RMS | ≤0.5 mm |
| 动态位置P95 | ≤1.0 mm |
| 动态最大误差 | ≤2.0 mm |
| 正常场景真实检测率 | ≥99.95% |
| 正常连续漏检 | ≤1帧 |
| 压力场景连续漏检 | ≤3帧 |
| 无球/固定干扰错误锁定 | 10万帧为0 |
| 唯一帧输入率 | ≥175 Hz |
| 有效测量输出 | ≥160 Hz |
| ROI检测P95 | ≤5 ms |
| 曝光到输出P95 | ≤10～12 ms |
| 曝光到输出P99 | ≤15 ms |
| 连续运行 | ≥60 min无队列增长或内存泄漏 |

只有真实holdout录像表明失败主要集中在强反光、遮挡或相似物歧义，并且补光、曝光、背景与几何约束仍无法解决时，才增加YOLO-Seg。网络必须异步运行，只返回粗Mask；原图圆拟合、毫米标定和Kalman保持不变。接口位于`ball_vision/fallback.py`。

以下问题不能靠YOLO修复：USB缓存、重复帧、运动模糊、标定漂移、支架振动、错误时间戳和控制器延迟。

## 香橙派5 Ultra迁移

算法层、配置、协议和测试不变，只替换采集后端。详细步骤见[ORANGE_PI.md](ORANGE_PI.md)。

## Git回退点

本目录是独立Git仓库。查看阶段历史：

```powershell
git log --oneline --decorate
git tag --list
```

两个固定标签：

```text
baseline-pre-v2        优化前完整基线
vision-v2.0.0-alpha.2  V2原始测量初版
vision-v2.0.0-alpha.3  帧新鲜度、重连年龄与采集异常修复版
vision-v2.1.0-alpha.1  当前二维预测搜索与跟踪态低延迟版
```

只查看旧版本而不移动当前分支：

```powershell
git switch --detach baseline-pre-v2
git switch main
```

需要在旧版本上继续开发时，从标签新建分支：

```powershell
git switch -c experiment-from-baseline baseline-pre-v2
```

## 目录

```text
ball_vision/
  capture.py       最新帧采集与采集统计
  detector.py      全局/预测二维搜索、OpenCV候选、圆拟合和边缘精修
  circle_fit.py    向量化RANSAC圆拟合
  calibration.py   单调像素—毫米映射
  tracker.py       毫米域Kalman与状态机
  pipeline.py      无UI/串口耦合的纯处理流水线
  protocol.py      二进制协议与CRC16
  output.py        串口/UDP发布
  telemetry_log.py CSV完整日志
  fallback.py      后续分割网络粗Mask接口
tools/
  probe_camera.py
  select_roi.py
  capture_background.py
  calibrate_position.py
  benchmark_synthetic.py
tests/
```

## 机械提醒

摄像头随摆杆运动会增加转动惯量，USB线也会形成随角度变化的干扰力矩。摄像头和支架应尽量靠近铰链，使用柔软线材和松弛弯曲段，并在最终装配状态重新标定和重新整定摆杆控制器。
