# 用于YOLO模型评估的代码

## 核心代码

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # =====================================================
    # 1. 基础配置
    # =====================================================
    # 指向训练好的最佳权重文件
    MODEL_PATH = 'runs/detect/exp_001/weights/best.pt'
    DATA_YAML = 'data.yaml'

    # 加载模型
    model = YOLO(MODEL_PATH)
    print(f"🚀 加载模型: {MODEL_PATH}")

    # =====================================================
    # 2. 评估参数 (按需取消注释)
    # 完整参数文档: https://docs.ultralytics.com/modes/val/
    # =====================================================
    print("📊 开始在测试集上评估...")
    metrics = model.val(
        # --- [核心参数] ---
        data=DATA_YAML,  # 数据配置
        split='test',  # 指定评估集合: 'val', 'test', 'train'
        imgsz=640,  # 评估尺寸 (需与训练时一致)
        batch=16,  # 批次大小
        device='0',  # 设备

        # --- [指标控制] ---
        # conf = 0.001,        # 评估置信度阈值 (计算 mAP 时通常设很低，默认 0.001)
        # iou = 0.6,           # NMS 阈值 (重叠多少算同一个)
        # max_det = 300,       # 每张图最大检测数量

        # --- [输出控制] ---
        # save_json = False,   # 是否保存结果为 COCO JSON 格式
        # save_hybrid = False, # 保存标签+预测结果 (用于辅助标注)
        # plots = True,        # 是否绘制 P-R 曲线、混淆矩阵等图表

        # --- [性能优化] ---
        # half = False,        # 使用半精度 (FP16) 评估 (显存更小，速度更快)
        # dnn = False,         # 使用 OpenCV DNN 模块进行 ONNX 推理
    )

    # 3. 打印结果
    print("\n" + "=" * 30)
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")
    print("=" * 30)
```