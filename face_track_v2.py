import cv2
import numpy as np
import onnxruntime as ort
import math
import serial
import time
import argparse
import threading
import queue

class BiologicalFilter:
    # 动态仿生滤波器：用于平滑追踪响应，消除机械震荡
    def __init__(self, base_alpha=0.03, dynamic_factor=0.3):
        self.base_alpha = base_alpha
        self.dynamic_factor = dynamic_factor
        self.current_value = 0.0

    def update(self, target):
        error = target - self.current_value
        error_ratio = min(abs(error) / 80.0, 1.0)
        dynamic_alpha = self.base_alpha + (error_ratio * self.dynamic_factor)
        self.current_value += error * dynamic_alpha
        return self.current_value

class SerialManager:
    """异步并发串口管理器：利用独立队列剥离 I/O 阻塞"""
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=0.01):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # 初始化线程安全的指令队列，限制深度防止指令积压导致云台动作滞后
        self.cmd_queue = queue.Queue(maxsize=3)
        self.running = True
        
        # 启动守护线程，程序退出时它会自动陪葬
        self.worker_thread = threading.Thread(target=self._serial_worker, daemon=True)
        self.worker_thread.start()
        
    def _serial_worker(self):
        """隐藏在暗处的硬件交涉专员"""
        while self.running:
            try:
                # 阻塞等待队列中的新指令，超时设为0.1秒以便定期检查 running 状态
                cmd = self.cmd_queue.get(timeout=0.1)
                
                if not self.ser or not self.ser.is_open:
                    self.connect()
                    
                if self.ser and self.ser.is_open:
                    try:
                        self.ser.write(cmd)
                    except Exception as e:
                        print(f"HW_ERR: 底层总线写入异常 {e}")
                        self.ser.close()
                        self.ser = None
            except queue.Empty:
                continue
            except Exception:
                pass

    def connect(self):
        for attempt in range(self.max_retries):
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                print(f"SYS_INFO: 硬件通讯链路已建立: {self.port}")
                return True
            except serial.SerialException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return False
        
    def write(self, data):
        """对外暴露的非阻塞投递接口"""
        # 如果队列满了，直接把最旧的指令扔掉，塞入最新的，保证云台永远追逐最新坐标
        if self.cmd_queue.full():
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                pass
        self.cmd_queue.put(data)
        return True
            
    def close(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                print("SYS_INFO: 硬件通讯链路已安全切断")
            except Exception:
                pass

class TrackerConfig:
    """跟踪器配置类：集中管理所有参数"""
    def __init__(self):
        # 滤波器参数
        self.filter_base_alpha = 0.15
        self.filter_dynamic_factor = 0.35
        
        # 控制参数
        self.k_pan = 13.8
        self.k_tilt = 8.0
        self.deadband = 0.05
        
        # 前馈控制系数（经验值）
        self.k_ff_pan = 0.3  # 水平方向前馈增益
        self.k_ff_tilt = 0.2  # 垂直方向前馈增益
        
        # 物理限制
        self.limit_x = 40.0
        self.limit_y_up = 15.0
        self.limit_y_down = 30.0
        
        # 机械约束
        self.max_velocity_x = 5.0  # 度/帧
        self.max_velocity_y = 5.0  # 度/帧
        
        # 人脸丢失处理
        self.max_face_lost_frames = 30
        self.center_gain = 0.05
        self.auto_reset_timeout = 10.0  # 10秒自动复位
        
        # 串口配置
        self.serial_port = '/dev/ttyACM0'
        self.serial_baudrate = 115200
        self.serial_timeout = 0.01
        
        # 机械常数
        self.mechanical_constant = 4.1667
        self.servo_center = 500
        self.servo_min = 0
        self.servo_max = 1000
        
        # 相机配置
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 30  # 默认帧率
        self.model_input_size = 640
        
        # 检测阈值
        self.detection_threshold = 0.5
        self.nms_threshold = 0.5
        self.nms_score = 0.45
        
    def update_from_dict(self, config_dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class FaceServoTracker:
    def __init__(self, model_path, config=None):
        # 使用配置类
        self.config = config if config else TrackerConfig()
        
        # 激活 GPU 加速优先级，榨干硬件算力（自动检测可用provider）
        available_providers = ort.get_available_providers()
        print(f"可用推理后端: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("使用 CUDA 加速")
        elif 'OpenVINOExecutionProvider' in available_providers:
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
            print("使用 OpenVINO 加速")
        else:
            providers = ['CPUExecutionProvider']
            print("使用 CPU 推理")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # 使用配置参数初始化滤波器
        self.filter_x = BiologicalFilter(base_alpha=self.config.filter_base_alpha, 
                                         dynamic_factor=self.config.filter_dynamic_factor)
        self.filter_y = BiologicalFilter(base_alpha=self.config.filter_base_alpha, 
                                         dynamic_factor=self.config.filter_dynamic_factor)
        
        # 引入目标角度累加器（闭环控制核心变量）
        self.target_angle_x = 0.0
        self.target_angle_y = 0.0
        
        # 当前平滑后的物理执行角度
        self.current_angle_x = 0.0
        self.current_angle_y = 0.0
    
        # 以下两个变量用于记录上次发送的舵机指令，避免重复下发相同命令导致总线拥堵
        self.last_sent_x = -1
        self.last_sent_y = -1  

        # 使用串口管理器，使用配置参数
        self.serial_mgr = SerialManager(port=self.config.serial_port, 
                                        baudrate=self.config.serial_baudrate, 
                                        timeout=self.config.serial_timeout)
        
        # 人脸丢失计数器
        self.face_lost_counter = 0
        
        # 自动复位相关
        self.last_face_time = None  # 最后检测到人脸的时间
        self.is_resetting = False   # 是否正在复位中
        
        # 机械运动约束
        self.last_angle_x = 0.0
        self.last_angle_y = 0.0
        
        # 多目标追踪策略相关变量
        self.locked_face_id = -1  # 锁定目标的唯一标识
        self.locked_face_center = None  # 锁定目标的中心坐标 (cx, cy)
        self.locked_frame_count = 0  # 已连续锁定的帧数
        self.locked_face_area = 0.0  # 锁定目标的框面积
        self.last_face_centers = []  # 上一帧所有人脸中心列表
        self.target_consistency_threshold = 0.7  # 目标一致性阈值
        
        # 场景A锁定切换相关变量
        self.potential_switch_face_id = -1  # 潜在切换目标的ID
        self.potential_switch_start_time = 0.0  # 潜在切换开始时间
        self.potential_switch_area = 0.0  # 潜在切换目标的面积
        
        # 前馈控制相关变量
        self.last_err_x = 0.0
        self.last_err_y = 0.0
        
        print("跟踪器初始化完成，使用配置:")
        print(f"  - 控制参数: K_PAN={self.config.k_pan}, K_TILT={self.config.k_tilt}")
        print(f"  - 前馈系数: K_FF_PAN={self.config.k_ff_pan}, K_FF_TILT={self.config.k_ff_tilt}")
        print(f"  - 物理限制: X={self.config.limit_x}°, Y=({self.config.limit_y_up}° to {self.config.limit_y_down}°)")
        print(f"  - 机械约束: 最大速度 X={self.config.max_velocity_x}°/帧, Y={self.config.max_velocity_y}°/帧")

    def preprocess(self, frame):
        """
        使用 OpenCV 的 blobFromImage 进行高效预处理
        替代手动 NumPy 操作，利用 C++ 底层加速
        参数说明：
          scalefactor=1.0/255.0: 实现 [0,1] 归一化
          size=(640, 640): 模型输入尺寸
          swapRB=True: BGR 转 RGB
          crop=False: 不裁剪
          ddepth=cv2.CV_32F: 32位浮点输出
        """
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0/255.0,
            size=(640, 640),
            swapRB=True,
            crop=False,
            ddepth=cv2.CV_32F
        )
        return blob

    def postprocess(self, outputs, orig_w, orig_h):
        predictions = np.squeeze(outputs[0]).T
        boxes, scores = [], []
        for pred in predictions:
            score = pred[4]
            if score > 0.5:
                cx, cy, w, h = pred[:4]
                x1 = (cx - w / 2) * (orig_w / 640)
                y1 = (cy - h / 2) * (orig_h / 640)
                boxes.append([int(x1), int(y1), int(w * orig_w / 640), int(h * orig_h / 640)])
                scores.append(float(score))
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)
        return [(boxes[i], scores[i]) for i in indices]

    def find_camera(self):
        """自动检测可用的摄像头"""
        for i in range(10):  # 检查前10个索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"检测到摄像头: 索引 {i}")
                    return i
        print("未检测到摄像头，使用默认索引 0")
        return 0
        
    def handle_face_lost(self):
        """处理人脸丢失情况：基于时间的自动复位"""
        current_time = time.time()
        
        # 检查是否需要自动复位
        if self.last_face_time is not None and not self.is_resetting:
            time_since_last_face = current_time - self.last_face_time
            
            # 如果超过10秒没有检测到人脸，开始复位
            if time_since_last_face > self.config.auto_reset_timeout:
                print(f"自动复位：{time_since_last_face:.1f}秒未检测到人脸，发送归中指令")
                self.is_resetting = True
                
                # 发送归中指令
                center_cmd = "X500Y500\n"
                if self.serial_mgr.write(center_cmd.encode()):
                    print("归中指令已发送")
                
                # 重置目标角度和当前角度
                self.target_angle_x = 0.0
                self.target_angle_y = 0.0
                self.current_angle_x = 0.0
                self.current_angle_y = 0.0
                self.filter_x.current_value = 0.0
                self.filter_y.current_value = 0.0
                self.last_sent_x = -1
                self.last_sent_y = -1
                
                # 重置多目标追踪状态
                self.locked_face_id = -1
                self.locked_face_center = None
                self.locked_frame_count = 0
                self.locked_face_area = 0.0
                self.last_face_centers = []
                
                # 重置场景A锁定切换状态
                self.potential_switch_face_id = -1
                self.potential_switch_start_time = 0.0
                self.potential_switch_area = 0.0
                
                # 重置人脸丢失计数器
                self.face_lost_counter = 0
                return
        
        # 如果正在复位中，不需要执行缓慢归中逻辑
        if self.is_resetting:
            return
            
        # 原始缓慢归中逻辑（用于短时间人脸丢失）
        if self.face_lost_counter > self.config.max_face_lost_frames:
            # 缓慢归中
            center_gain = self.config.center_gain
            self.target_angle_x -= self.target_angle_x * center_gain
            self.target_angle_y -= self.target_angle_y * center_gain
            
            # 更新当前角度
            self.current_angle_x = self.filter_x.update(self.target_angle_x)
            self.current_angle_y = self.filter_y.update(self.target_angle_y)
            
    def apply_velocity_limits(self):
        """应用机械运动速度限制"""
        # 计算角度变化
        delta_x = self.current_angle_x - self.last_angle_x
        delta_y = self.current_angle_y - self.last_angle_y
        
        # 限制速度，使用配置参数
        if abs(delta_x) > self.config.max_velocity_x:
            self.current_angle_x = self.last_angle_x + np.sign(delta_x) * self.config.max_velocity_x
            
        if abs(delta_y) > self.config.max_velocity_y:
            self.current_angle_y = self.last_angle_y + np.sign(delta_y) * self.config.max_velocity_y
            
        # 保存当前角度作为下一帧的参考
        self.last_angle_x = self.current_angle_x
        self.last_angle_y = self.current_angle_y
        
    def render_dashboard(self):
        dash = np.ones((300, 600, 3), dtype=np.uint8) * 40
        centers = [(150, 200), (450, 200)]
        labels = ["Pan (Absolute)", "Tilt (Absolute)"]
        angles = [self.current_angle_x, self.current_angle_y]
        radius = 100
        
        for i in range(2):
            cx, cy = centers[i]
            angle = angles[i]
            
            cv2.ellipse(dash, (cx, cy), (radius, radius), 180, 0, 180, (100, 100, 100), 2)
            cv2.line(dash, (cx - radius - 10, cy), (cx + radius + 10, cy), (100, 100, 100), 1)
            cv2.line(dash, (cx, cy), (cx, cy - radius - 10), (100, 100, 100), 1)
            
            theta = math.radians(angle)
            end_x = int(cx + radius * math.sin(theta))
            end_y = int(cy - radius * math.cos(theta))
            
            intensity = min(abs(angle) / self.config.limit_x, 1.0)
            color = (0, int(255 * (1 - intensity)), int(255 * intensity))
            
            cv2.line(dash, (cx, cy), (end_x, end_y), color, 3)
            cv2.circle(dash, (cx, cy), 8, (255, 255, 255), -1)
            
            text_val = f"{angle:+.1f} deg"
            cv2.putText(dash, labels[i], (cx - 60, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(dash, text_val, (cx - 40, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return dash

    def run_stream(self, camera_index=None):
        # 确定摄像头索引
        if camera_index is None or camera_index < 0:
            # 自动检测摄像头
            camera_index = self.find_camera()
            print(f"自动检测到摄像头: 索引 {camera_index}")
        else:
            # 验证指定的摄像头是否可用
            cap_test = cv2.VideoCapture(camera_index)
            if not cap_test.isOpened():
                print(f"警告: 摄像头索引 {camera_index} 不可用，尝试自动检测...")
                cap_test.release()
                camera_index = self.find_camera()
                print(f"使用自动检测的摄像头: 索引 {camera_index}")
            else:
                cap_test.release()
                print(f"使用指定的摄像头: 索引 {camera_index}")
        
        # 初始化摄像头 - 使用命令行传入的索引，删除硬编码
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头索引 {camera_index}")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        
        # 设置摄像头帧率（针对摄像头索引4的特殊优化）
        if camera_index == 4:
            # Fic QHD CAMERA 支持更高帧率
            target_fps = min(self.config.camera_fps, 60)  # 最高60FPS
            print(f"摄像头索引4 (Fic QHD CAMERA): 尝试设置帧率为 {target_fps}FPS")
            cap.set(cv2.CAP_PROP_FPS, target_fps)
        else:
            # 其他摄像头使用配置的帧率
            cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        
        # 验证分辨率设置是否成功
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"摄像头配置: 分辨率={actual_width:.0f}x{actual_height:.0f}, 帧率={actual_fps:.1f}FPS")
        
        # 如果实际帧率远低于目标帧率，给出警告
        if actual_fps > 0 and self.config.camera_fps > 0:
            fps_ratio = actual_fps / self.config.camera_fps
            if fps_ratio < 0.8:
                print(f"警告: 实际帧率({actual_fps:.1f}FPS)低于目标帧率({self.config.camera_fps}FPS)")
                print(f"      建议降低分辨率或使用更低的帧率设置")

        # 帧率统计
        frame_count = 0
        start_time = time.time()
        
        # 使用配置参数
        K_PAN = self.config.k_pan
        K_TILT = self.config.k_tilt
        DEADBAND = self.config.deadband

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("摄像头读取失败")
                break
                
            h, w = frame.shape[:2]
            
            # 推理
            blob = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: blob})
            results = self.postprocess(outputs, w, h)
            
            # 检查是否检测到人脸
            if results and len(results) > 0:
                # 重置人脸丢失计数器
                self.face_lost_counter = 0
                
                # 更新最后检测到人脸的时间
                self.last_face_time = time.time()
                self.is_resetting = False
                
                # 多目标追踪策略：根据人脸数量选择不同的匹配策略
                face_count = len(results)
                
                # 提取所有人脸的中心坐标和面积
                current_face_centers = []
                face_areas = []
                for face_data in results:
                    box = face_data[0]
                    x, y, bw, bh = box
                    cx, cy = x + bw // 2, y + bh // 2
                    area = bw * bh
                    current_face_centers.append((cx, cy))
                    face_areas.append(area)
                
                # 场景判断：人脸数量决定追踪策略
                if face_count < 3:
                    # 场景 A：人脸数量少于3人，使用智能锁定策略
                    current_time = time.time()
                    
                    if self.locked_face_id == -1:
                        # 如果没有锁定目标，选择面积最大的人脸并锁定
                        selected_idx = face_areas.index(max(face_areas))
                        self.locked_face_id = current_time  # 使用时间戳作为唯一标识
                        self.locked_face_center = current_face_centers[selected_idx]
                        self.locked_face_area = face_areas[selected_idx]
                        self.locked_frame_count = 1
                        print(f"场景A：锁定初始目标，面积={self.locked_face_area:.0f}")
                    else:
                        # 已经锁定目标：检查是否需要切换
                        # 1. 首先找到距离锁定目标最近的人脸（保持追踪连续性）
                        min_distance = float('inf')
                        nearest_idx = 0
                        for idx, (cx, cy) in enumerate(current_face_centers):
                            distance = math.sqrt((cx - self.locked_face_center[0])**2 + 
                                                (cy - self.locked_face_center[1])**2)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_idx = idx
                        
                        # 2. 检查是否有更大的人脸（面积大20%以上）
                        max_area_idx = face_areas.index(max(face_areas))
                        max_area = face_areas[max_area_idx]
                        
                        # 计算面积比例：新人脸面积 / 当前锁定人脸面积
                        if max_area_idx != nearest_idx and max_area > self.locked_face_area * 1.2:
                            # 发现更大的人脸，检查是否已经跟踪了一段时间
                            if self.potential_switch_face_id == max_area_idx:
                                # 同一个潜在目标，检查是否已经跟踪了5秒
                                if current_time - self.potential_switch_start_time >= 5.0:
                                    # 5秒已过，切换锁定目标
                                    selected_idx = max_area_idx
                                    self.locked_face_id = current_time
                                    self.locked_face_center = current_face_centers[selected_idx]
                                    self.locked_face_area = face_areas[selected_idx]
                                    self.locked_frame_count = 1
                                    self.potential_switch_face_id = -1
                                    print(f"场景A：切换锁定目标，新面积={self.locked_face_area:.0f} (+{(self.locked_face_area/face_areas[nearest_idx]-1)*100:.0f}%)")
                                else:
                                    # 继续跟踪潜在目标
                                    selected_idx = nearest_idx
                                    remaining_time = 5.0 - (current_time - self.potential_switch_start_time)
                                    print(f"场景A：跟踪潜在切换目标，剩余{remaining_time:.1f}秒")
                            else:
                                # 新的潜在目标，开始计时
                                self.potential_switch_face_id = max_area_idx
                                self.potential_switch_start_time = current_time
                                self.potential_switch_area = max_area
                                selected_idx = nearest_idx
                                print(f"场景A：发现更大的人脸，开始5秒计时（面积大{(max_area/self.locked_face_area-1)*100:.0f}%）")
                        else:
                            # 没有更大的目标，继续追踪当前锁定目标
                            selected_idx = nearest_idx
                            self.potential_switch_face_id = -1  # 重置潜在切换
                        
                        # 更新锁定目标的中心坐标和面积
                        self.locked_face_center = current_face_centers[selected_idx]
                        self.locked_face_area = face_areas[selected_idx]
                        self.locked_frame_count += 1
                else:
                    # 场景 B：人脸数量大于或等于3人
                    if self.locked_face_id == -1:
                        # 如果当前未锁定目标：寻找离屏幕中心点最近的人脸
                        screen_center_x, screen_center_y = w * 0.5, h * 0.5
                        min_center_distance = float('inf')
                        selected_idx = 0
                        for idx, (cx, cy) in enumerate(current_face_centers):
                            # 计算到屏幕中心的归一化距离
                            norm_distance = math.sqrt(((cx / w) - 0.5)**2 + 
                                                     ((cy / h) - 0.5)**2)
                            if norm_distance < min_center_distance:
                                min_center_distance = norm_distance
                                selected_idx = idx
                        
                        # 锁定该目标
                        self.locked_face_id = time.time()  # 使用时间戳作为唯一标识
                        self.locked_face_center = current_face_centers[selected_idx]
                        self.locked_frame_count = 1
                        print(f"锁定目标：人脸{selected_idx}，中心坐标{self.locked_face_center}")
                    else:
                        # 已经锁定目标：通过距离关联继续跟踪
                        if self.locked_face_center is not None:
                            # 计算当前所有人脸中心与锁定目标的距离
                            min_distance = float('inf')
                            selected_idx = 0
                            for idx, (cx, cy) in enumerate(current_face_centers):
                                distance = math.sqrt((cx - self.locked_face_center[0])**2 + 
                                                    (cy - self.locked_face_center[1])**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    selected_idx = idx
                            
                            # 更新锁定目标的中心坐标
                            self.locked_face_center = current_face_centers[selected_idx]
                            self.locked_frame_count += 1
                
                # 获取选定的人脸信息
                best_face = results[selected_idx]
                (x, y, bw, bh), score = best_face
                face_cx, face_cy = x + bw // 2, y + bh // 2
                max_area = face_areas[selected_idx]
                
                # 保存当前帧的人脸中心供下一帧使用
                self.last_face_centers = current_face_centers
                
                # 核心逻辑替换：计算画面像素的相对误差率 (-0.5 到 0.5 之间)
                err_x = (face_cx / float(w)) - 0.5
                err_y = (face_cy / float(h)) - 0.5
                
                # 计算误差变化率（近似为目标速度）
                if hasattr(self, 'last_err_x'):
                    err_rate_x = err_x - self.last_err_x
                    err_rate_y = err_y - self.last_err_y
                else:
                    err_rate_x = 0.0
                    err_rate_y = 0.0
                
                # 保存当前误差供下一帧使用
                self.last_err_x = err_x
                self.last_err_y = err_y
                
                # 闭环增量叠加 + 前馈控制：基于当前目标角度，按误差比例继续施加偏转
                # 控制量 = Kp * 误差 + Kff * 误差变化率
                if abs(err_x) > DEADBAND:
                    self.target_angle_x += err_x * K_PAN + err_rate_x * self.config.k_ff_pan
                if abs(err_y) > DEADBAND:
                    self.target_angle_y += err_y * K_TILT + err_rate_y * self.config.k_ff_tilt
                
                # 物理干涉截断：对累加后的虚拟目标进行无情限幅，保护打印件
                self.target_angle_x = max(-self.config.limit_x, min(self.config.limit_x, self.target_angle_x))
                self.target_angle_y = max(-self.config.limit_y_up, min(self.config.limit_y_down, self.target_angle_y))
                
                # 送入仿生滤波器平滑，得到当前帧需要执行的真实角度
                self.current_angle_x = self.filter_x.update(self.target_angle_x)
                self.current_angle_y = self.filter_y.update(self.target_angle_y)
                
                # 应用机械运动约束
                self.apply_velocity_limits()
                
                # 物理驱动映射，机械常数保持绝对精准
                MECHANICAL_CONSTANT = self.config.mechanical_constant
                
                servo_x = int(self.config.servo_center + (self.current_angle_x * MECHANICAL_CONSTANT))
                servo_y = int(self.config.servo_center + (self.current_angle_y * MECHANICAL_CONSTANT))
                
                servo_x = max(self.config.servo_min, min(self.config.servo_max, servo_x))
                servo_y = max(self.config.servo_min, min(self.config.servo_max, servo_y))
                
                # 发送指令
                if abs(servo_x - self.last_sent_x) >= 2 or abs(servo_y - self.last_sent_y) >= 2:
                    cmd = f"X{servo_x}Y{servo_y}\n"
                    
                    # 使用串口管理器发送
                    if self.serial_mgr.write(cmd.encode()):
                        self.last_sent_x = servo_x
                        self.last_sent_y = servo_y
                
                # 在画面上绘制人脸框
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
                
                # 显示追踪状态信息
                status_text = f"Faces: {face_count}"
                if self.locked_face_id != -1:
                    status_text += f" | Locked: {self.locked_frame_count}f"
                
                cv2.putText(frame, f"Pan: {self.current_angle_x:+.1f} deg", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Tilt: {self.current_angle_y:+.1f} deg", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 未检测到人脸
                self.face_lost_counter += 1
                self.handle_face_lost()
                
                # 应用机械运动约束（即使人脸丢失）
                self.apply_velocity_limits()
                
                # 显示丢失状态
                cv2.putText(frame, "NO FACE DETECTED", (w//2 - 100, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Lost Counter: {self.face_lost_counter}/{self.config.max_face_lost_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 帧率统计
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Vision Tracking", frame)
            
            # 取消注释下一行可以调出你之前写的炫酷仪表盘
            # cv2.imshow("Dashboard", self.render_dashboard())
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            
        # 资源清理
        cap.release()
        cv2.destroyAllWindows()
        
        # 发送归中指令
        center_cmd = "X500Y500\n"
        self.serial_mgr.write(center_cmd.encode())
        
        # 关闭串口
        self.serial_mgr.close()
        
        # 关闭ONNX session
        self.session = None
        print("程序正常退出")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='仿生眼视觉伺服追踪系统 - 人脸追踪机器人头颅控制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                    # 自动检测摄像头
  %(prog)s --camera 0         # 使用摄像头索引0
  %(prog)s --camera 4         # 使用摄像头索引4（Fic QHD CAMERA）
  %(prog)s --camera -1        # 强制自动检测
  %(prog)s --list-cameras     # 列出可用摄像头

x86主板部署建议:
  %(prog)s --camera 0 --width 640 --height 480
        """
    )
    
    parser.add_argument(
        '--camera', 
        type=int, 
        default=-1,
        help='摄像头设备索引 (-1=自动检测, 0=第一个摄像头, 1=第二个摄像头, 以此类推)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='摄像头分辨率宽度 (默认: 640)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='摄像头分辨率高度 (默认: 480)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='摄像头帧率 (默认: 30)'
    )
    
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='列出系统可用的摄像头设备'
    )
    
    return parser.parse_args()

def list_available_cameras(max_index=10):
    """列出可用的摄像头设备"""
    print("正在扫描可用摄像头...")
    available_cameras = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                print(f"  [{i}] 分辨率: {width}x{height}, FPS: {fps:.1f}")
            cap.release()
        else:
            cap.release()
    
    if available_cameras:
        print(f"\n找到 {len(available_cameras)} 个可用摄像头:")
        for cam in available_cameras:
            print(f"  索引 {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS")
    else:
        print("未找到可用摄像头")
    
    return available_cameras

if __name__ == "__main__":
    args = parse_arguments()
    
    # 如果请求列出摄像头，则列出后退出
    if args.list_cameras:
        list_available_cameras()
        exit(0)
    
    # 创建配置对象
    config = TrackerConfig()
    
    # 应用命令行参数到配置
    config.camera_width = args.width
    config.camera_height = args.height
    config.camera_fps = args.fps
    
    print(f"启动人脸追踪系统")
    print(f"摄像头配置: 索引={args.camera if args.camera >= 0 else '自动检测'}, 分辨率={args.width}x{args.height}, 帧率={args.fps}FPS")
    print("按 'q' 键退出程序")
    print("-" * 50)
    
    # 创建跟踪器并运行
    tracker = FaceServoTracker("yolov8n-face.onnx", config)
    tracker.run_stream(args.camera if args.camera >= 0 else None)
