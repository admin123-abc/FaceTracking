Bionic Eye Face Tracking System
这是一个基于 YOLOv8 与 ONNX Runtime 实现的仿生眼视觉伺服追踪系统。本项目专为嵌入式 x86 环境设计，实现了从摄像头采集、实时人脸检测、PID 控制量计算到虚拟仪表盘反馈的全链路逻辑。
项目特性
轻量化部署：通过将模型导出为 ONNX 格式并使用 ONNX Runtime 推理，摆脱了对 PyTorch 等重型框架的依赖，极大地降低了内存占用并提升了在裸机上的启动速度。

仿生视觉伺服：采用位置式 PID 控制算法，模拟生物眼球的跟随机制。通过角度累计与误差补偿，实现丝滑的追踪效果。

实时遥测仪表盘：内置基于 OpenCV 渲染的虚拟电机状态窗口，实时反馈 X/Y 轴舵机的绝对物理角度，方便在无硬件连接阶段进行算法调优。

安全限位机制：代码级实现物理机械保护，确保输出指令始终处于预设的安全角度域内（-45° 至 +45°）。

快速开始
1. 环境准备
推荐在 Ubuntu 20.04+ 环境下运行。为了保持生产环境的纯净，请勿直接上传虚拟环境文件夹。

创建并激活你的虚拟环境：

Bash
python3 -m venv venv_deploy
source venv_deploy/bin/activate
2. 安装依赖
使用本项目提供的清单快速复刻环境：

Bash
pip install -r requirements.txt
3. 模型导出
如果你手头只有 .pt 权重，请在开发机中使用 Ultralytics 工具进行导出（建议开启 simplify 优化）：

Python
from ultralytics import YOLO
model = YOLO("yolov8n-face.pt")
model.export(format="onnx", imgsz=640, simplify=True)
4. 运行追踪
确保摄像头已连接，执行主程序：

Bash
python face_track.py
项目结构
Plaintext
.
├── face_track.py           # 核心逻辑：包含 PID 控制器、推理封装及 UI 渲染
├── yolov8n-face.onnx       # 已导出的轻量化推理模型
├── requirements.txt        # 依赖包清单
├── .gitignore              # 忽略 venv、缓存及大型权重文件
└── README.md               # 项目说明文档
核心算法说明
视觉偏差计算
系统将图像中心定义为原点 (0,0)，计算检测框中心与原点的相对偏差，并归一化至 [-0.5, 0.5] 区间，以确保 PID 参数在不同分辨率下的通用性。

PID 控制逻辑
系统不再使用简单的位置映射，而是通过增量累计的方式控制角度。

P (Proportional)：决定对位置偏差的响应速度。

I (Integral)：消除静态误差。

D (Derivative)：抑制由于快速移动产生的震荡。

待办事项 (TODO)
[ ] 集成 pyserial 通讯协议，实现与下位机（Arduino/STM32）的闭环控制。

[ ] 增加多目标选择逻辑，实现对特定 ID 的持续追踪。

[ ] 在 x86 裸主板上进行 OpenVINO 加速测试。

注意事项
在没有图形界面的裸机上运行时，请务必注释掉代码中所有关于 cv2.imshow 的行，并切换至 opencv-python-headless 版本。
