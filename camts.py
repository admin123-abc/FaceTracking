import cv2

def test_camera_orientation(index):
    # 初始化摄像头：锁定 Fic QHD CAMERA 的系统索引
    cap = cv2.VideoCapture(index)
    
    # 强制尝试设置 QHD 分辨率以榨干硬件性能
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 580)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        return False

    print(f"成功开启目标摄像头！当前索引: {index}")
    print("操作提示：")
    print("1. 观察画面中的 'UP' 标签，确认摄像头物理正反。")
    print("2. 确认画面是否流畅，按 'q' 退出测试。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # 绘制十字参考线，助你物理对齐骨架中心
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
        
        # 标注物理上方参考
        cv2.putText(frame, "PHYSICAL UP", (w//2 - 100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(f"Fic QHD Test (Index {index})", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    # 根据你的 v4l2-ctl 结果，4 是唯一的正解
    # 如果 4 报错，则尝试 0 这种保底选项
    target_indices = [4, 0]
    
    for i in target_indices:
        if test_camera_orientation(i):
            break