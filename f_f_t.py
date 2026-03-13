import cv2
import numpy as np
import onnxruntime as ort
import math
import serial
class BiologicalFilter:#dongtaic filter for smoother tracking response
    def __init__(self, base_alpha=0.03, dynamic_factor=0.3):
        self.base_alpha = base_alpha
        self.dynamic_factor = dynamic_factor
        self.current_value = 0.0

    def update(self, target):
        # 计算当前物理位置与目标位置的绝对误差
        error = target - self.current_value
        
        # 归一化误差比例，假定最大合理追踪偏角为 80 度
        error_ratio = min(abs(error) / 80.0, 1.0)
        
        # 核心算法：误差越大，平滑系数越趋近 1；误差趋近 0 时，系数回归极小的底座值
        dynamic_alpha = self.base_alpha + (error_ratio * self.dynamic_factor)
        
        # 位置更新
        self.current_value += error * dynamic_alpha
        return self.current_value

class FaceServoTracker:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        self.filter_x = BiologicalFilter(base_alpha=0.05, dynamic_factor=0.25)#base越大越快，dynamic越大越能适应大幅度变化但可能引入震荡
        self.filter_y = BiologicalFilter(base_alpha=0.05, dynamic_factor=0.25)
        
        self.current_angle_x = 0.0
        self.current_angle_y = 0.0
    
        self.limit_x = 40.0
        self.limit_y_base = 40.0
        
        # Y轴非对称物理限位
        self.limit_y_up = 15.0    # 仰头极限角度
        self.limit_y_down = 30.0  # 低头极限角度

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.01)
        except serial.SerialException:
            pass
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
            
            intensity = min(abs(angle) / self.limit_max, 1.0)
            color = (0, int(255 * (1 - intensity)), int(255 * intensity))
            
            cv2.line(dash, (cx, cy), (end_x, end_y), color, 3)
            cv2.circle(dash, (cx, cy), 8, (255, 255, 255), -1)
            
            text_val = f"{angle:+.1f} deg"
            cv2.putText(dash, labels[i], (cx - 60, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(dash, text_val, (cx - 40, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return dash

    def run_stream(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            
            blob = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: blob})
            results = self.postprocess(outputs, w, h)
            
            if results:
                # 面积过滤逻辑：遍历所有识别结果，计算边界框面积，锁定最大目标
                best_face = results[0]
                max_area = 0
                
                for face_data in results:
                    # face_data 的结构为 ((x, y, bw, bh), score)
                    box = face_data[0]
                    area = box[2] * box[3]
                    if area > max_area:
                        max_area = area
                        best_face = face_data
                
                # 仅解包最大人脸的数据进行跟踪计算
                (x, y, bw, bh), _ = best_face
                face_cx, face_cy = x + bw // 2, y + bh // 2
                
                # 计算X轴目标角度
                target_angle_x = ((face_cx / w) - 0.5) * 2 * self.limit_x
                raw_target_y = ((face_cy / h) - 0.5) * 2 * 40.0
                
                # 严格的非对称物理防干涉限位：仰视极限 15度，俯视极限 30度
                target_angle_y = max(-self.limit_y_up, min(self.limit_y_down, raw_target_y))
                
                # 经过仿生滤波器平滑处理
                self.current_angle_x = self.filter_x.update(target_angle_x)
                self.current_angle_y = self.filter_y.update(target_angle_y)

                # ---------------------------------------------------------
                # 物理驱动域：基于机械常数的绝对坐标计算
                # ---------------------------------------------------------
                # 机械传动常数: 1000 PWM / 240 Degree = 4.1667
                MECHANICAL_CONSTANT = 4.1667
                
                # 以 500 为绝对基准零点进行线性叠加
                # 注：若实际运行中方向相反，只需将加号改为减号即可翻转相位
                servo_x = int(500 + (self.current_angle_x * MECHANICAL_CONSTANT))
                servo_y = int(500 + (self.current_angle_y * MECHANICAL_CONSTANT))
                
                # 最终总线指令安全兜底，防止溢出 0-1000 量程
                servo_x = max(0, min(1000, servo_x))
                servo_y = max(0, min(1000, servo_y))
                
                cmd = f"X{servo_x}Y{servo_y}\n"
                
                try:
                    self.ser.write(cmd.encode())
                except Exception:
                    pass
                
                # 在画面上标记当前锁定的唯一目标
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
                
                # 可以在画面左上角打印锁定状态，方便调试
                cv2.putText(frame, f"Locked Area: {max_area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Vision Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceServoTracker("yolov8n-face.onnx")
    tracker.run_stream()