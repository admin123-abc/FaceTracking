import os
from ultralytics import YOLO

# 严谨的模型导出流程：从研发格式(.pt)到部署格式(.onnx)
def export_procedure():
    # 指定手动下载好的权重路径
    model_pt_path = "/home/happyman/vision_deploy/yolov8n-face-lindevs.pt"
    
    if not os.path.exists(model_pt_path):
        print(f"错误：在当前目录未找到 {model_pt_path}，请确认下载位置。")
        return

    # 加载 PyTorch 模型
    # 这一步会解析 .pt 文件的结构
    model = YOLO(model_pt_path)

    # 导出为 ONNX
    # imgsz: 指定输入尺寸，需与后续推理保持一致
    # opset: ONNX 算子版本，12 是目前兼容性较好的版本
    # simplify: 自动剔除冗余节点，显著提升裸机运行效率
    print(f"正在转换 {model_pt_path} ...")
    model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    
    print("导出成功。请检查当前目录下生成的 .onnx 文件。")

if __name__ == "__main__":
    export_procedure()