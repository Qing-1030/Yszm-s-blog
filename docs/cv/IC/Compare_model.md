---
title: 对比模型性能
date: 2026-01-17
tags:
  - Python
  - 模型性能
---

# 对比模型性能

## 核心代码：

```python
import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

# 尝试导入 FLOPs 计算工具 (如果没有安装 thop，会自动跳过)
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[提示] 未检测到 'thop' 库，将跳过 FLOPs 计算。(建议: pip install thop)")

# ==========================================
# 1. 全局配置 [核心修改区]
# ==========================================
class Config:
    # --- [必填] 待对比的模型清单 ---
    # 请在下方列表中填写你要PK的模型信息
    # 格式: {'name': '自定义显示名', 'arch': '模型架构名(如resnet50)', 'path': '.pth权重路径'}
    MY_MODELS = [
        {
            'name': 'ResNet50_Run1', 
            'arch': 'resnet50', 
            'path': './results/resnet50_run1/best_model.pth' 
        },
        {
            'name': 'ResNet50_Run2', 
            'arch': 'resnet50', 
            'path': './results/resnet50_run2/best_model.pth'
        },
        # {
        #     'name': 'MobileNetV3', 
        #     'arch': 'mobilenetv3_large_100', 
        #     'path': './results/mobilenet/best_model.pth'
        # }
    ]
    
    # --- [必填] 数据集设置 ---
    USE_CUSTOM_DATASET = True        # [必改] True=自定义文件夹, False=内置
    CUSTOM_DATA_ROOT = "./datasets/Intel Image Classification" # [必改] 数据集路径
    BUILTIN_NAME = "CIFAR10"         # [可选] 内置数据集名称
    DATA_DOWNLOAD_ROOT = "./data"    # [可选] 数据缓存路径
    
    # --- [必填] 模型参数 (需与训练时一致) ---
    NUM_CLASSES = 6                  # [必改] 类别数量 (Intel=6, Cassava=5)
    IMG_SIZE = 224                   # [必改] 图片输入尺寸 (影响速度和FLOPs计算)
    BATCH_SIZE = 32                  # [微调] 批次大小
    
    # --- [可选] 硬件与保存 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "./benchmark_results" # 结果保存目录

# ==========================================
# 2. 核心工具函数
# ==========================================
def load_my_model(info):
    """加载本地训练好的模型权重"""
    print(f"🔹 正在加载: {info['name']} ({info['arch']})...")
    
    # 1. 创建模型骨架
    try:
        model = timm.create_model(info['arch'], pretrained=False, num_classes=Config.NUM_CLASSES)
    except Exception as e:
        print(f"   ❌ 架构名 '{info['arch']}' 错误或不支持: {e}")
        return None
    
    # 2. 检查文件是否存在
    if not os.path.exists(info['path']):
        print(f"   ❌ 找不到权重文件: {info['path']}")
        return None
        
    # 3. 加载权重 (兼容处理 state_dict)
    try:
        checkpoint = torch.load(info['path'], map_location=Config.DEVICE)
        # 有些checkpoint保存整个dict，有些只保存权重，这里做自适应处理
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        
        model.to(Config.DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"   ❌ 权重加载失败 (架构不匹配?): {e}")
        return None

def get_model_size_mb(path):
    """获取模型文件大小 (MB)"""
    return os.path.getsize(path) / (1024 * 1024)

def get_params_count(model):
    """计算参数量 (Million)"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def get_flops(model):
    """计算计算量 (GFLOPs)"""
    if not HAS_THOP: return 0
    input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    try:
        # thop 库用于计算 FLOPs
        flops, params = profile(model, inputs=(input, ), verbose=False)
        return flops / 1e9
    except:
        return 0

def measure_speed(model, repetitions=50):
    """测试推理速度 (FPS & Latency)"""
    input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    
    # 预热 GPU (消除首次运行开销)
    with torch.no_grad():
        for _ in range(10): model(input)
    
    # 正式计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(repetitions): model(input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_latency = (end - start) / repetitions * 1000 # ms
    fps = 1000 / avg_latency
    return avg_latency, fps

def evaluate_accuracy(model, dataloader):
    """在测试集上评估准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="   Eval", leave=False):
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_dataloader():
    """加载测试数据 (优先使用 test 目录，没有则复用 val)"""
    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not Config.USE_CUSTOM_DATASET:
        # 内置数据集逻辑
        try: DatasetClass = getattr(datasets, Config.BUILTIN_NAME)
        except: return None
        ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=False, download=True, transform=tf)
    else:
        # 自定义数据集逻辑
        test_path = os.path.join(Config.CUSTOM_DATA_ROOT, "test")
        val_path = os.path.join(Config.CUSTOM_DATA_ROOT, "val")
        
        if os.path.exists(test_path):
            print(f"[Data] 使用测试集: {test_path}")
            ds = datasets.ImageFolder(test_path, transform=tf)
        elif os.path.exists(val_path):
            print(f"[Data] 未找到独立测试集，复用验证集: {val_path}")
            ds = datasets.ImageFolder(val_path, transform=tf)
        else:
            print(f"❌ 错误: 在 {Config.CUSTOM_DATA_ROOT} 下未找到数据")
            return None
            
    return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

def generate_safe_filename(models_list):
    """自动生成唯一且合法的文件名 (基于模型名称拼接)"""
    names = [m['name'] for m in models_list]
    joined_name = "_vs_".join(names)
    # 替换非法字符并限制长度
    safe_name = re.sub(r'[\\/*?:"<>| ]', '_', joined_name)
    if len(safe_name) > 100: safe_name = safe_name[:100] + "_etc"
    return "Compare_" + safe_name

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    print(f"🚀 开始对比本地模型 (共 {len(Config.MY_MODELS)} 个)...")
    
    # 1. 准备数据
    dataloader = get_dataloader()
    if dataloader is None: exit()
    
    results = []
    
    # 2. 循环评测每个模型
    for info in Config.MY_MODELS:
        model = load_my_model(info)
        if model is None: continue
        
        # 采集指标
        params = get_params_count(model)
        size_mb = get_model_size_mb(info['path'])
        flops = get_flops(model)
        latency, fps = measure_speed(model)
        acc = evaluate_accuracy(model, dataloader)
        
        print(f"   ✅ Acc: {acc:.2f}% | FPS: {fps:.1f} | Params: {params:.2f}M")
        
        results.append({
            "Model": info['name'],
            "Accuracy (%)": acc,
            "Parameters (M)": params,
            "FLOPs (G)": flops,
            "Model Size (MB)": size_mb,
            "Inference Speed (FPS)": fps,
        })
        
    if not results:
        print("❌ 未产生任何有效结果，请检查模型路径。")
        exit()
        
    # 3. 保存 CSV 报告
    base_name = generate_safe_filename(Config.MY_MODELS)
    csv_path = os.path.join(Config.SAVE_DIR, base_name + ".csv")
    png_path = os.path.join(Config.SAVE_DIR, base_name + ".png")
    
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"🏆 详细报告已生成: {csv_path}")
    print("="*50)
    print(df.to_string(index=False))
    df.to_csv(csv_path, index=False)
    
    # 4. 生成可视化图表
    metrics = ["Accuracy (%)", "Parameters (M)", "FLOPs (G)", "Model Size (MB)", "Inference Speed (FPS)"]
    valid_metrics = metrics if HAS_THOP else [m for m in metrics if "FLOPs" not in m]

    plt.figure(figsize=(18, 10))
    for i, metric in enumerate(valid_metrics):
        rows = 2
        cols = (len(valid_metrics) + 1) // 2
        plt.subplot(rows, cols, i+1)
        
        # 绘制柱状图
        sns.barplot(x="Model", y=metric, data=df, palette="viridis")
        plt.title(metric, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 在柱子上标注具体数值
        for index, row in df.iterrows():
             plt.text(index, row[metric], round(row[metric], 2), color='black', ha="center", va="bottom")
             
    plt.tight_layout()
    plt.savefig(png_path)
    print(f"\n📊 可视化图表已保存: {png_path}")
```