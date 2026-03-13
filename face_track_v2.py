import cv2
import numpy as np
import onnxruntime as ort
import math
import serial

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

class FaceServoTracker:
    def __init__(self, model_path):
        # 激活 GPU 加速优先级，榨干硬件算力
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        self.filter_x = BiologicalFilter(base_alpha=0.15, dynamic_factor=0.35)
        self.filter_y = BiologicalFilter(base_alpha=0.15, dynamic_factor=0.35)
        
        # 引入目标角度累加器（闭环控制核心变量）
        self.target_angle_x = 0.0
        self.target_angle_y = 0.0
        
        # 当前平滑后的物理执行角度
        self.current_angle_x = 0.0
        self.current_angle_y = 0.0
    
        # 物理结构边界参数
        self.limit_x = 40.0
        self.limit_y_base = 40.0
        self.limit_y_up = 15.0    
        self.limit_y_down = 30.0
        #以下两个变量用于记录上次发送的舵机指令，避免重复下发相同命令导致总线拥堵
        self.last_sent_x = -1
        self.last_sent_y = -1  

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.01)
        except serial.SerialException:
            print("WARNING: 串口未连接或被占用")

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

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
            
            intensity = min(abs(angle) / self.limit_x, 1.0)
            color = (0, int(255 * (1 - intensity)), int(255 * intensity))
            
            cv2.line(dash, (cx, cy), (end_x, end_y), color, 3)
            cv2.circle(dash, (cx, cy), 8, (255, 255, 255), -1)
            
            text_val = f"{angle:+.1f} deg"
            cv2.putText(dash, labels[i], (cx - 60, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(dash, text_val, (cx - 40, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return dash

    def run_stream(self):
        # 锁定索引4，强制降低分辨率以释放总线带宽，拉升推理帧率
        cap = cv2.VideoCapture(4)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 比例控制增益常数 (P增益)
        # 数值决定每次发现误差时，摄像头追赶的速度。如果转动时抖动厉害，将此数值调小。
        K_PAN = 13.8
        K_TILT = 8.0
        DEADBAND = 0.05 #新加的阈值

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            
            blob = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: blob})
            results = self.postprocess(outputs, w, h)
            
            max_area = 0
            
            if results:
                best_face = results[0]
                for face_data in results:
                    box = face_data[0]
                    area = box[2] * box[3]
                    if area > max_area:
                        max_area = area
                        best_face = face_data
                
                (x, y, bw, bh), _ = best_face
                face_cx, face_cy = x + bw // 2, y + bh // 2
                
                # 核心逻辑替换：计算画面像素的相对误差率 (-0.5 到 0.5 之间)
                err_x = (face_cx / float(w)) - 0.5
                err_y = (face_cy / float(h)) - 0.5
                
                # 闭环增量叠加：基于当前目标角度，按误差比例继续施加偏转
                if abs(err_x) > DEADBAND:
                    self.target_angle_x += err_x * K_PAN
                if abs(err_y) > DEADBAND:
                    self.target_angle_y += err_y * K_TILT
                
                # 物理干涉截断：对累加后的虚拟目标进行无情限幅，保护打印件
                self.target_angle_x = max(-self.limit_x, min(self.limit_x, self.target_angle_x))
                self.target_angle_y = max(-self.limit_y_up, min(self.limit_y_down, self.target_angle_y))
                
                # 送入仿生滤波器平滑，得到当前帧需要执行的真实角度
                self.current_angle_x = self.filter_x.update(self.target_angle_x)
                self.current_angle_y = self.filter_y.update(self.target_angle_y)

                # 物理驱动映射，机械常数保持绝对精准
                MECHANICAL_CONSTANT = 4.1667
                
                servo_x = int(500 + (self.current_angle_x * MECHANICAL_CONSTANT))
                servo_y = int(500 + (self.current_angle_y * MECHANICAL_CONSTANT))
                
                servo_x = max(0, min(1000, servo_x))
                servo_y = max(0, min(1000, servo_y))
                if abs(servo_x - self.last_sent_x) >= 2 or abs(servo_y - self.last_sent_y) >= 2:#新加，阻断抽搐
                    cmd = f"X{servo_x}Y{servo_y}\n"
                    
                    if hasattr(self, 'ser'):
                        try:
                            self.ser.write(cmd.encode())

                            self.last_sent_x = servo_x
                            self.last_sent_y = servo_y#同步更新

                        except Exception as e:
                            print(f"串口下发失败: {e}")
                
                '''cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Locked Area: {max_area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)'''
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.imshow("Vision Tracking", frame)
            
            # 取消注释下一行可以调出你之前写的炫酷仪表盘
            # cv2.imshow("Dashboard", self.render_dashboard())
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceServoTracker("yolov8n-face.onnx")
    tracker.run_stream()