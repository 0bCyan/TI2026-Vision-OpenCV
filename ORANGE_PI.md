# 香橙派5 Ultra迁移

## 0. 安装不会覆盖GStreamer OpenCV的运行环境

板端不要安装本工程Windows依赖中的`opencv-python`，它通常会覆盖系统`cv2`并丢失GStreamer支持。优先使用板卡镜像/apt提供的OpenCV与NumPy：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-yaml python3-serial v4l-utils \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 python3-gi
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-orange-pi.txt
```

安装后必须确认导入的是预期版本，并查看`GStreamer: YES`；若系统包不带GStreamer，再自行编译OpenCV或改用PyGObject appsink：

```bash
.venv/bin/python -c "import cv2; print(cv2.__file__); print(cv2.getBuildInformation())"
gst-inspect-1.0 v4l2src jpegparse jpegdec videoconvert appsink
```

## 1. 验证UVC真实能力

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --get-fmt-video
v4l2-ctl -d /dev/video0 --get-parm
v4l2-ctl -d /dev/video0 --list-ctrls-menus
lsusb -t
```

必须确认摄像头真实枚举`MJPG 1280×720@180fps`，并运行在USB 3 SuperSpeed链路。正式配置使用`/dev/v4l/by-id/...`，不要依赖会变化的`/dev/video0`。

锁定1～3 ms曝光，关闭自动曝光、自动增益、自动白平衡和自动对焦。控制名称和值以实际`v4l2-ctl --list-ctrls-menus`为准，不要照搬Windows的OpenCV属性值。

正式启动脚本应先执行并读回`v4l2-ctl --set-ctrl ...`；成功后把`camera.controls`设为`{}`并设置`camera.controls_verified_externally: true`，再让systemd在进程退出后重新执行“设置→读回→启动”。外部控制和OpenCV控制互斥，防止程序覆盖已验真的曝光。该模式遇到UVC断流会主动退出，避免自动重连后悄悄恢复自动曝光。

## 2. 第一阶段：V4L2

修改配置：

```yaml
camera:
  source: /dev/v4l/by-id/你的摄像头
  backend: v4l2
  strict_contract: true
```

运行`tools/probe_camera.py`，再完成ROI、空管背景和毫米标定。

## 3. 第二阶段：GStreamer低延迟比较

确定性管线示例：

```text
v4l2src device=/dev/v4l/by-id/... io-mode=mmap do-timestamp=true !
image/jpeg,width=1280,height=720,framerate=180/1 !
queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream !
jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR !
appsink max-buffers=1 drop=true sync=false
```

将整行写入`camera.gstreamer_pipeline`并将`backend`设为`gstreamer`。

注意：PyPI的`opencv-python` Linux wheel通常不带GStreamer，也可能遮蔽系统版本。先检查：

```python
import cv2
print(cv2.getBuildInformation())
```

若`GStreamer: NO`，使用发行版带GStreamer的OpenCV、自行编译，或用PyGObject appsink。需要真实源PTS时也应使用PyGObject或原生V4L2；`cv2.VideoCapture`通常不暴露`GstBuffer.pts`。

可以分别基准`jpegdec`、`mppjpegdec`或`v4l2jpegdec`。RK3588硬解后可能产生NV12→BGR转换和额外队列，所以必须以P95/P99端到端延迟决定，不能默认硬解一定更快。

## 4. 性能原则

- MJPEG必须整帧解码后才能裁ROI，窄ROI只减少后续算法成本。
- 不使用`videorate`，否则可能复制帧伪造180fps。
- 解码前队列必须`max-size-buffers=1 leaky=downstream`。
- appsink必须`max-buffers=1 drop=true sync=false`。
- 显示、无线图传和录像限制在20～30fps，不能放入采集或检测线程。
- 比较`runtime.opencv_threads=1/2/4`的P50/P95/P99，选尾延迟最小的配置。
- 正式运行建议关闭Python循环GC，程序退出时恢复并回收。

## 5. 最终延迟验证

软件计时只能得到“解码后处理时间”。最终用MCU驱动视野内LED翻转，同时记录视觉首次检测到变化的时刻，用逻辑分析仪测曝光到结果的P50/P95/P99。完成后把固定流水线延迟写入`camera.timestamp_offset_ms`，并重新验证控制稳定性。

OpenCV/V4L2/GStreamer经`cv2.VideoCapture`时不保证阻塞的`read()`能被Python超时中断。正式比赛至少保证MCU在20～30 ms没有收到新序号遥测时进入安全状态；若目标镜像上复现原生读取永久卡死，再把采集后端改成可由监护进程强制重启的原生V4L2/PyGObject服务。
