---
title: 检验模型
date: 2026-01-12
tags:
  - Python
  - 计算机视觉
  - 图像分类
---

# 对训练好的模型进行单张图片预测

## 核心代码：

```python
import os
import torch
import timm
from timm.data import resolve_data_config
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


# ==========================================
# 1. 配置区域 [核心修改区]
# ==========================================
class Config:
    # --- 模型与权重 ---
    MODEL_NAME = "resnet50"  # <--- [必须修改] 需与训练时一致
    NUM_CLASSES = 5  # <--- [必须修改] 类别数
    WEIGHT_PATH = "./results/xxx/best_model.pth"  # <--- [必须修改] 权重路径

    # --- 输入与输出 ---
    IMAGE_PATH = "./test_image.jpg"  # <--- [必须修改] 待预测图片
    # 类别名称 (按训练目录的字母顺序或日志打印的顺序)
    CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # <--- [必须修改]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. 核心逻辑
# ==========================================
def get_transforms(model):
    """获取与模型匹配的预处理"""
    config = resolve_data_config({}, model=model)
    mean = config.get('mean', [0.485, 0.456, 0.406])
    std = config.get('std', [0.229, 0.224, 0.225])
    input_size = config.get('input_size', (3, 224, 224))
    crop_size = input_size[1]

    print(f"[Info] 预处理配置: Size={crop_size}")
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def load_trained_model():
    """加载架构与权重"""
    print(f"[Init] 创建模型: {Config.MODEL_NAME}")
    # 创建空模型
    model = timm.create_model(Config.MODEL_NAME, pretrained=False, num_classes=Config.NUM_CLASSES)

    if not os.path.exists(Config.WEIGHT_PATH):
        raise FileNotFoundError(f"权重文件不存在: {Config.WEIGHT_PATH}")

    print(f"[Load] 加载权重: {Config.WEIGHT_PATH}")
    checkpoint = torch.load(Config.WEIGHT_PATH, map_location=Config.DEVICE)

    # 提取参数 (兼容只保存state_dict或完整checkpoint)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(Config.DEVICE)
    model.eval()
    return model


def predict(model, img_path, transform):
    """推理单张图片"""
    if not os.path.exists(img_path): raise FileNotFoundError(f"图片不存在: {img_path}")

    img_raw = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_raw).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)

    topk_probs, topk_ids = torch.topk(probs, k=min(3, len(Config.CLASS_NAMES)))
    return img_raw, topk_probs.cpu().numpy()[0], topk_ids.cpu().numpy()[0]


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    try:
        model = load_trained_model()
        tf = get_transforms(model)
        img, probs, ids = predict(model, Config.IMAGE_PATH, tf)

        print("\n" + "=" * 30)
        print("       PREDICTION RESULT")
        print("=" * 30)

        top1_name = Config.CLASS_NAMES[ids[0]]
        print(f"🏆 预测结果: {top1_name} ({probs[0] * 100:.2f}%)")

        print("\nTop-3 概率分布:")
        for i in range(len(probs)):
            name = Config.CLASS_NAMES[ids[i]]
            print(f"   {i + 1}. {name:<15} : {probs[i] * 100:.2f}%")

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Pred: {top1_name} ({probs[0] * 100:.1f}%)", color='green', fontsize=14)
        plt.axis('off')

        text = "\n".join([f"{Config.CLASS_NAMES[i]}: {p * 100:.1f}%" for p, i in zip(probs, ids)])
        plt.text(10, 20, text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        plt.show()

    except Exception as e:
        print(f"[Error] {e}")
```