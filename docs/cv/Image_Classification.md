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
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查是否安装了高级评估库
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
    USE_CUSTOM_DATASET = True        # <--- [可微调] True=自定义文件夹, False=内置数据集
    CUSTOM_DATA_ROOT = "flower_data" # <--- [可微调] 自定义数据集根目录
    BUILTIN_NAME = "CIFAR10"         # <--- [可微调] 内置数据集名称 (如 CIFAR10, CIFAR100)
    DATA_DOWNLOAD_ROOT = "./data"    # <--- [可微调] 数据集下载缓存路径
    
    # --- 结果保存 ---
    SAVE_DIR_ROOT = "./results"      # <--- [可微调] 结果保存根目录 (会自动生成子文件夹)
    SAVE_DIR = ""                    # (运行时自动生成，无需修改)
    
    # --- 模型设置 ---
    MODEL_NAME = "resnet50"          # <--- [可微调] 模型名称 (timm库支持的名称)
    
    # 本地预训练权重路径
    # "" (空字符串) = 自动下载在线权重
    # "xxx.bin"     = 强制加载本地文件
    CHECKPOINT_PATH = ""             # <--- [可微调] 初始预训练权重路径
    
    # 断点续训路径
    # "" (空字符串) = 从头开始训练
    # "./results/xxx/last.pth" = 从指定断点恢复
    RESUME_PATH = ""                 # <--- [可微调] 断点续训文件路径
    
    NUM_CLASSES = 0                  # (运行时自动检测覆盖)
    
    # --- 训练超参数 ---
    BATCH_SIZE = 32                  # <--- [可微调] 批次大小
    EPOCHS = 20                      # <--- [可微调] 训练总轮数
    LR = 1e-4                        # <--- [可微调] 初始学习率
    WEIGHT_DECAY = 1e-4              # <--- [可微调] 权重衰减 (L2正则化)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据预处理与加载
# ==========================================
def get_transforms(model_cfg):
    # 读取模型对应的默认输入尺寸和均值方差
    input_size = model_cfg.get('input_size', (3, 224, 224))
    crop_size = input_size[1]
    mean = model_cfg.get('mean', [0.485, 0.456, 0.406])
    std = model_cfg.get('std', [0.229, 0.224, 0.225])
    
    print(f"[Info] 预处理参数: Size={crop_size}, Mean={mean}, Std={std}")

    # 训练集增强策略
    train_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(0.5),                  # <--- [可微调] 水平翻转概率
        transforms.RandomRotation(15),                         # <--- [可微调] 旋转角度
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # <--- [可微调] 颜色扰动
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 验证/测试集仅做标准化
    val_test_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tf, val_test_tf

def get_dataloaders(train_tf, val_tf):
    if not Config.USE_CUSTOM_DATASET:
        # --- 加载内置数据集 ---
        print(f"[Data] 加载内置数据集: {Config.BUILTIN_NAME}")
        try:
            DatasetClass = getattr(datasets, Config.BUILTIN_NAME)
        except AttributeError:
            raise ValueError(f"不支持的数据集: {Config.BUILTIN_NAME}")
            
        full_train_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=True, download=True, transform=train_tf)
        test_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=False, download=True, transform=val_tf)
        
        # 划分训练集和验证集 (默认9:1)
        train_size = int(0.9 * len(full_train_ds))             # <--- [可微调] 验证集划分比例
        val_size = len(full_train_ds) - train_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
        
        class_names = full_train_ds.classes
        dataset_name = Config.BUILTIN_NAME
    else:
        # --- 加载自定义文件夹 ---
        print(f"[Data] 加载自定义文件夹: {Config.CUSTOM_DATA_ROOT}")
        train_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "train")
        val_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "val")
        test_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "test")
        
        if not os.path.exists(train_dir): 
            raise FileNotFoundError(f"找不到目录: {train_dir}")
            
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        test_ds = datasets.ImageFolder(test_dir, transform=val_tf) if os.path.exists(test_dir) else None
        
        class_names = train_ds.classes
        dataset_name = os.path.basename(Config.CUSTOM_DATA_ROOT)

    Config.NUM_CLASSES = len(class_names)
    print(f"[Data] 类别数: {Config.NUM_CLASSES} -> {class_names}")
    
    # 构建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2) if test_ds else None
    
    return train_loader, val_loader, test_loader, class_names, dataset_name

# ==========================================
# 3. 核心功能函数
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
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
    if not test_loader or not HAS_SKLEARN: return
    print(f"\n[Test] 执行测试集评估...")
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(imgs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 输出分类报告
    print("\n" + classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(Config.SAVE_DIR, 'confusion_matrix.png'))
    print(f"[Info] 混淆矩阵已保存。")

def plot_history(history, save_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['val_acc'], label='Val')
    plt.legend(); plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.legend(); plt.title('Loss')
    
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    print(f"[Info] 曲线已保存。")

def save_checkpoint(state, is_best, filename='last.pth'):
    """保存断点文件"""
    path = os.path.join(Config.SAVE_DIR, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(Config.SAVE_DIR, 'best_model.pth'))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # --- 1. 获取模型默认配置 ---
    temp_model = timm.create_model(Config.MODEL_NAME, pretrained=True)
    cfg = resolve_data_config({}, model=temp_model)
    del temp_model
    
    # --- 2. 准备数据 & 生成保存目录 ---
    train_tf, val_test_tf = get_transforms(cfg)
    train_loader, val_loader, test_loader, class_names, dataset_name = get_dataloaders(train_tf, val_test_tf)
    
    # 确定保存目录逻辑
    if Config.RESUME_PATH:
        # 如果是断点续训，复用原目录
        Config.SAVE_DIR = os.path.dirname(Config.RESUME_PATH)
        print(f"[Config] 断点续训模式，使用原目录: {Config.SAVE_DIR}")
    else:
        # 如果是新训练，生成 "模型_数据集_时间" 格式的目录
        time_str = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{Config.MODEL_NAME}_{dataset_name}_{time_str}"
        Config.SAVE_DIR = os.path.join(Config.SAVE_DIR_ROOT, run_name)
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        print(f"[Config] 全新训练，保存至: {Config.SAVE_DIR}")

    # --- 3. 初始化模型 ---
    print(f"[Init] 创建模型: {Config.MODEL_NAME}")
    
    # 如果指定了本地权重且不是续训模式，则加载本地文件
    if not Config.RESUME_PATH and Config.CHECKPOINT_PATH and os.path.exists(Config.CHECKPOINT_PATH):
        print(f"[Load] 加载本地初始化权重: {Config.CHECKPOINT_PATH}")
        model = timm.create_model(Config.MODEL_NAME, pretrained=False, checkpoint_path=Config.CHECKPOINT_PATH)
    else:
        # 否则使用在线下载的预训练权重（如果是续训，稍后会被覆盖）
        model = timm.create_model(Config.MODEL_NAME, pretrained=True)
    
    # 重置分类头
    model.reset_classifier(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # --- 4. 断点恢复逻辑 ---
    start_epoch = 1
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if Config.RESUME_PATH and os.path.exists(Config.RESUME_PATH):
        print(f"\n[Resume] 正在恢复断点: {Config.RESUME_PATH}")
        checkpoint = torch.load(Config.RESUME_PATH, map_location=Config.DEVICE)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        
        print(f"[Resume] 恢复成功! 从第 {start_epoch} 轮继续 (当前最佳: {best_acc:.4f})")
    
    # --- 5. 训练循环 ---
    print(f"\n[Start] 开始训练... (设备: {Config.DEVICE})")
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc = validate(model, val_loader, criterion, epoch)
        
        # 更新历史记录
        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        
        # 保存最佳模型
        is_best = v_acc > best_acc
        if is_best: 
            best_acc = v_acc
            print(f" -> 🌟 新的最佳模型 (Acc: {best_acc:.4f})")
        
        # 保存断点 (包含模型、优化器、epoch、history)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'history': history,
        }, is_best, filename='last.pth')

    # --- 6. 收尾工作 ---
    plot_history(history, Config.SAVE_DIR)
    
    if test_loader:
        print("[Info] 加载最佳模型进行最终测试...")
        # 加载 best_model.pth 中的权重
        checkpoint = torch.load(os.path.join(Config.SAVE_DIR, "best_model.pth"), map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        evaluate_test_set(model, test_loader, class_names)
        
    print("\n[Done] 全部完成！")
```