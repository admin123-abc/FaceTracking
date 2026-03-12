import cv2
import numpy as np
import onnxruntime as ort
import math

# 严谨的低通滤波器（EMA）：用于消除画面噪点导致的舵机抖动
# 取代了之前复杂的 PID 控制器
class EMAFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.current_value = 0.0

    def update(self, target):
        # 核心逻辑：当前值逐步逼近目标值，alpha 越小越平滑但越迟钝
        self.current_value += (target - self.current_value) * self.alpha
        return self.current_value

class FaceServoTracker:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # 使用低通滤波器代替 PID
        self.filter_x = EMAFilter(alpha=0.25)
        self.filter_y = EMAFilter(alpha=0.25)
        
        self.current_angle_x = 0.0
        self.current_angle_y = 0.0
        
        # 硬限位
        self.limit_max = 70

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
                (x, y, bw, bh), _ = results[0]
                face_cx, face_cy = x + bw // 2, y + bh // 2
                
                # 绝对映射逻辑：将像素坐标直接映射为目标角度
                target_angle_x = ((face_cx / w) - 0.5) * 2 * self.limit_max
                target_angle_y = ((face_cy / h) - 0.5) * 2 * self.limit_max
                
                # 反转 Y 轴，符合常规舵机逻辑
                target_angle_y = -target_angle_y
                
                # 经过滤波器平滑处理，防止舵机发抖
                self.current_angle_x = self.filter_x.update(target_angle_x)
                self.current_angle_y = self.filter_y.update(target_angle_y)
                
                print(f"SERVO_ABS -> X: {self.current_angle_x:+05.1f}°, Y: {self.current_angle_y:+05.1f}°")
                
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (w // 2, h // 2), (face_cx, face_cy), (0, 255, 255), 1)

            dashboard_img = self.render_dashboard()
            
            cv2.imshow("Vision Tracking", frame)
            cv2.imshow("Motor Telemetry", dashboard_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceServoTracker("yolov8n-face.onnx")
    tracker.run_stream()