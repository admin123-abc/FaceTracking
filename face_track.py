import cv2
import numpy as np
import onnxruntime as ort
import time
import math

class PIDController:
    def __init__(self, kp, ki, kd, min_out, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 1e-6

        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / dt

        output = p_term + i_term + d_term
        
        # 限制单次增量的幅度，防止狂暴输出
        output = np.clip(output, self.min_out, self.max_out)
        
        self.prev_error = error
        self.last_time = current_time
        
        return output

class FaceServoTracker:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # 修复点 1：适配归一化误差的全新 PID 参数
        # kp=3.0 意味着最大误差(0.5)时，每帧转动约 1.5 度
        self.pid_x = PIDController(kp=15.0, ki=0.2, kd=0.5, min_out=-5, max_out=5)#max_out=5 限制每帧最大转动 5 度，防止过度反应
        self.pid_y = PIDController(kp=15.0, ki=0.2, kd=0.5, min_out=-5, max_out=5)
        
        self.current_angle_x = 0.0
        self.current_angle_y = 0.0
        
        # 同步为你截图中的 70 度限位
        self.limit_min = -70
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
        labels = ["Pan (X-Axis)", "Tilt (Y-Axis)"]
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
                
                # 修复点 2：将误差归一化到 [-0.5, 0.5]
                error_x = (face_cx / w) - 0.5
                error_y = (face_cy / h) - 0.5
                
                # 计算增量并累加
                self.current_angle_x += self.pid_x.update(error_x)
                self.current_angle_y -= self.pid_y.update(error_y) 
                
                self.current_angle_x = np.clip(self.current_angle_x, self.limit_min, self.limit_max)
                self.current_angle_y = np.clip(self.current_angle_y, self.limit_min, self.limit_max)
                
                # 修复点 3：找回终端的串口指令输出
                print(f"SERVO_CMD -> X: {self.current_angle_x:+05.1f}°, Y: {self.current_angle_y:+05.1f}°")
                
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