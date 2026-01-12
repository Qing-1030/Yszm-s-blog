---
title: 图像分类模型通用模板
date: 2026-01-12
tags:
  - Python
  - 计算机视觉
  - 图像分类
---

# 图像分类模型通用模板

> 摘要：一个可配置的代码模板，用于训练图像分类模型

## 核心代码：

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm
import matplotlib.pyplot as plt

# 尝试导入高级评估库
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] 未安装 sklearn 或 seaborn，跳过混淆矩阵绘制。")

# ==========================================
# 1. 全局配置区域
# ==========================================
class Config:
    # --- 数据集设置 ---
    # True = 使用自定义文件夹 (需包含 train/val 文件夹)
    # False = 使用 PyTorch 标准内置数据集 (自动下载)
    USE_CUSTOM_DATASET = True       # <--- [可微调] 数据集模式开关
    
    # 模式1：自定义文件夹路径
    CUSTOM_DATA_ROOT = "flower_data" # <--- [可微调] 你的数据集文件夹路径
    
    # 模式2：内置数据集名称 (如 CIFAR10, CIFAR100, FashionMNIST)
    BUILTIN_NAME = "CIFAR10"        # <--- [可微调] 内置数据集名称
    DATA_DOWNLOAD_ROOT = "./data"   # <--- [可微调] 数据下载路径
    
    # --- 结果保存 ---
    SAVE_DIR = "./results"          # <--- [可微调] 训练结果/模型保存路径
    
    # --- 模型设置 ---
    # 推荐模型: resnet50, resnet18, efficientnet_b0, inception_v3.tf_in1k
    MODEL_NAME = "resnet50"         # <--- [可微调] 使用的模型名称
    
    # 本地权重路径设置：
    # 1. 填入具体路径 (如 "resnet50.bin") -> 强制加载本地权重 (pretrained=False)
    # 2. 留空 "" -> 自动从网络下载预训练权重 (pretrained=True)
    CHECKPOINT_PATH = ""            # <--- [可微调] 本地预训练权重路径
    
    # 初始类别数 (代码会自动检测真实类别数并覆盖此值)
    NUM_CLASSES = 5                 
    
    # --- 训练超参数 ---
    BATCH_SIZE = 32                 # <--- [可微调] 批次大小 (显存不足可调小)
    EPOCHS = 20                     # <--- [可微调] 训练总轮数
    LR = 1e-4                       # <--- [可微调] 初始学习率
    WEIGHT_DECAY = 1e-4             # <--- [可微调] L2正则化系数
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据预处理与加载
# ==========================================
def get_transforms(model_cfg):
    """根据模型配置自动生成预处理流程"""
    # 自动获取模型所需的输入尺寸
    input_size = model_cfg.get('input_size', (3, 224, 224))
    crop_size = input_size[1] 
    
    # 获取模型特定的均值和方差
    mean = model_cfg.get('mean', [0.485, 0.456, 0.406])
    std = model_cfg.get('std', [0.229, 0.224, 0.225])
    
    print(f"[Info] 预处理配置: Size={crop_size}, Mean={mean}, Std={std}")

    # 训练集增强
    train_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),  # 统一图像尺寸
        transforms.RandomHorizontalFlip(0.5),       # <--- [可微调] 随机水平翻转概率
        transforms.RandomRotation(15),              # <--- [可微调] 随机旋转角度
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # <--- [可微调] 颜色/对比度扰动
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 验证/测试集处理 (仅标准化)
    val_test_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(train_tf, val_tf):
    """根据配置加载内置或自定义数据集"""
    train_ds, val_ds, test_ds = None, None, None
    class_names = []
    
    # --- 分支 1: 使用内置数据集 ---
    if not Config.USE_CUSTOM_DATASET:
        print(f"[Data] 加载内置数据集: {Config.BUILTIN_NAME}")
        try:
            DatasetClass = getattr(datasets, Config.BUILTIN_NAME)
        except AttributeError:
            raise ValueError(f"不支持的数据集: {Config.BUILTIN_NAME}")

        # 加载完整数据集
        full_train_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=True, download=True, transform=train_tf)
        test_ds       = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=False, download=True, transform=val_tf)
        
        # 自动划分验证集 (90% 训练, 10% 验证)
        train_size = int(0.9 * len(full_train_ds)) # <--- [可微调] 验证集比例
        val_size = len(full_train_ds) - train_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
        
        class_names = full_train_ds.classes

    # --- 分支 2: 使用自定义文件夹 ---
    else:
        print(f"[Data] 加载自定义文件夹: {Config.CUSTOM_DATA_ROOT}")
        train_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "train")
        val_dir   = os.path.join(Config.CUSTOM_DATA_ROOT, "val")
        test_dir  = os.path.join(Config.CUSTOM_DATA_ROOT, "test")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"找不到训练目录: {train_dir}")

        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
        
        # 检查是否存在测试集
        if os.path.exists(test_dir):
            test_ds = datasets.ImageFolder(test_dir, transform=val_tf)
        else:
            print("[Info] 未找到 test 文件夹，跳过测试步骤。")
            
        class_names = train_ds.classes

    # 更新全局类别数
    Config.NUM_CLASSES = len(class_names)
    print(f"[Data] 检测到 {Config.NUM_CLASSES} 个类别: {class_names}")
    
    # 创建DataLoader
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2) if test_ds else None
    
    return train_loader, val_loader, test_loader, class_names

# ==========================================
# 3. 训练与验证逻辑
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """训练一个Epoch"""
    model.train()
    total_loss, total_correct = 0.0, 0
    
    bar = tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
    for imgs, labels in bar:
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        bar.set_postfix(loss=loss.item())
        
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, epoch, phase="Val"):
    """验证模型性能"""
    model.eval()
    total_loss, total_correct = 0.0, 0
    
    bar = tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]  ")
    for imgs, labels in bar:
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        bar.set_postfix(loss=loss.item())
        
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

def evaluate_test_set(model, test_loader, class_names):
    """测试集详细评估：生成报告与混淆矩阵"""
    if not test_loader: return
    if not HAS_SKLEARN: return

    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n[Test] 正在进行最终测试集评估...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(Config.DEVICE), labels.to(Config.DEVICE) # 确保数据在同一设备
            imgs = imgs[0] # 解包
            labels = labels[0]

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 打印分类报告
    print("\n" + "="*50)
    print("FINAL TEST REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    save_path = os.path.join(Config.SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"[Info] 混淆矩阵已保存至: {save_path}")

def plot_history(history, save_dir):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Val Acc')
    plt.title('Accuracy'); plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('Loss'); plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    print(f"[Info] 训练曲线已保存至: {save_dir}")

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # --- 1. 获取模型默认配置 ---
    print(f"[Init] 获取 {Config.MODEL_NAME} 默认配置...")
    temp_model = timm.create_model(Config.MODEL_NAME, pretrained=True)
    model_cfg = resolve_data_config({}, model=temp_model)
    del temp_model 
    
    # --- 2. 准备数据 ---
    train_tf, val_test_tf = get_transforms(model_cfg)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(train_tf, val_test_tf)
    
    # --- 3. 初始化模型 (核心修改点) ---
    print(f"[Init] 创建模型: {Config.MODEL_NAME}")
    
    # 判断是否加载本地权重
    if Config.CHECKPOINT_PATH and os.path.exists(Config.CHECKPOINT_PATH):
        print(f"[Load] 加载本地权重: {Config.CHECKPOINT_PATH}")
        model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=False,                       # 关闭自动下载
            checkpoint_path=Config.CHECKPOINT_PATH  # 指定本地路径
        )
    else:
        print(f"[Load] 使用在线预训练权重 (pretrained=True)")
        model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True                         # 开启自动下载
        )
    
    # 重置分类头以匹配当前数据类别
    print(f"[Init] 重置分类头为 {Config.NUM_CLASSES} 类")
    model.reset_classifier(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    
    # --- 4. 训练循环 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    print(f"\n[Start] 开始训练... (设备: {Config.DEVICE})")
    for epoch in range(1, Config.EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc = validate(model, val_loader, criterion, epoch)
        
        # 记录日志
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        
        # 保存最佳模型
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "best_model.pth"))
            print(f" -> 🌟 最佳模型已更新 (Acc: {best_acc:.4f})")
            
    # --- 5. 结果可视化与测试 ---
    plot_history(history, Config.SAVE_DIR)
    
    if test_loader:
        print("[Info] 加载最佳模型进行最终测试...")
        model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, "best_model.pth")))
        evaluate_test_set(model, test_loader, class_names)
        
    print("\n[Done] 全部完成！")
```