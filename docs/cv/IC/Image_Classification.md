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
import sys
import time
import random
import logging
import numpy as np
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
    print("[Warning] 未安装 sklearn 或 seaborn，将跳过混淆矩阵绘制。")

# ==========================================
# 1. 全局配置 [核心修改区]
# ==========================================
class Config:
    # --- 数据集路径设置 ---
    USE_CUSTOM_DATASET = True        # [必改] True=使用自定义文件夹, False=使用内置(CIFAR10等)
    CUSTOM_DATA_ROOT = "flower_data" # [必改] 自定义数据集根目录 (包含 train/val)
    BUILTIN_NAME = "CIFAR10"         # [可选] 内置数据集名称 (仅当上面为False时生效)
    DATA_DOWNLOAD_ROOT = "./data"    # [可选] 数据下载缓存路径
    
    # --- 结果保存设置 ---
    SAVE_DIR_ROOT = "./results"      # [可选] 训练结果保存根目录
    SAVE_DIR = ""                    # (程序自动生成，无需修改)
    
    # --- 模型与训练设置 ---
    MODEL_NAME = "resnet50"          # [可选] 模型名称 (如 resnet50, efficientnet_b0)
    CHECKPOINT_PATH = ""             # [可选] 预训练权重路径 (空则下载ImageNet权重)
    RESUME_PATH = ""                 # [可选] 断点续训的 .pth 文件路径
    NUM_CLASSES = 0                  # (程序自动识别，无需修改)
    
    # --- 超参数设置 ---
    BATCH_SIZE = 32                  # [微调] 批次大小 (显存不足改小)
    EPOCHS = 50                      # [微调] 训练轮数
    LR = 1e-4                        # [微调] 初始学习率 (微调通常用 1e-4 或 1e-5)
    WEIGHT_DECAY = 1e-4              # [微调] 正则化系数 (抗过拟合用)
    SEED = 42                        # [可选] 随机种子 (保证结果可复现)
    
    # --- 优化策略 ---
    OPTIMIZER_NAME = 'adamw'         # [可选] 优化器: 'adamw', 'adam', 'sgd'
    SCHEDULER_NAME = 'plateau'       # [可选] 学习率策略: 'plateau'(监控), 'cosine'(余弦), 'step'
    
    # --- 早停 (Early Stopping) ---
    EARLY_STOP_PATIENCE = 7          # [可选] 连续多少轮不涨分就停止 (0表示关闭)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 辅助工具 (日志/随机种子/早停)
# ==========================================
def setup_logger(save_dir):
    """配置日志系统：同时输出到控制台和文件"""
    log_format = '%(asctime)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件日志 (UTF-8编码)
    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # 控制台日志
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(stream_handler)
    return logger

def seed_everything(seed):
    """固定所有随机种子以保证实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    """早停控制器：当验证集准确率不再提升时提前终止训练"""
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        if not self.patience or self.patience <= 0: return
        
        # 如果是第一次记录
        if self.best_score is None:
            self.best_score = val_acc
        # 如果当前分数没有明显提升
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # 如果有提升，重置计数器
        else:
            self.best_score = val_acc
            self.counter = 0

def get_optimizer(model):
    """根据配置创建优化器"""
    name = Config.OPTIMIZER_NAME.lower()
    p = model.parameters()
    if name == 'adamw': return optim.AdamW(p, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    elif name == 'sgd': return optim.SGD(p, lr=Config.LR, momentum=0.9, weight_decay=Config.WEIGHT_DECAY)
    else: return optim.Adam(p, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

def get_scheduler(optimizer):
    """根据配置创建学习率调度器"""
    name = Config.SCHEDULER_NAME.lower()
    if name == 'plateau': return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    elif name == 'cosine': return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    elif name == 'step': return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return None

# ==========================================
# 3. 数据加载与处理
# ==========================================
def get_transforms(model_cfg):
    """根据模型默认参数自动生成预处理管线"""
    input_size = model_cfg.get('input_size', (3, 224, 224))
    crop_size = input_size[1]
    mean = model_cfg.get('mean', [0.485, 0.456, 0.406])
    std = model_cfg.get('std', [0.229, 0.224, 0.225])
    
    # 训练集增强：随机裁剪、翻转、旋转、颜色抖动
    train_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 验证集处理：仅调整大小和归一化
    val_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tf, val_tf

def get_dataloaders(train_tf, val_tf, logger):
    """
    智能加载数据集
    优化点：如果自定义数据集中没有 test 目录，自动复用 val 集作为测试集，
    保证后续评估代码不报错。
    """
    # --- 分支1: 内置数据集 (CIFAR10等) ---
    if not Config.USE_CUSTOM_DATASET:
        logger.info(f"[Data] 加载内置数据集: {Config.BUILTIN_NAME}")
        try: DatasetClass = getattr(datasets, Config.BUILTIN_NAME)
        except AttributeError: raise ValueError(f"不支持的数据集: {Config.BUILTIN_NAME}")
        
        full_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=True, download=True, transform=train_tf)
        test_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=False, download=True, transform=val_tf)
        
        # 划分 90% 训练, 10% 验证
        train_sz = int(0.9 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [train_sz, len(full_ds)-train_sz])
        class_names = full_ds.classes

    # --- 分支2: 自定义文件夹数据集 ---
    else:
        logger.info(f"[Data] 加载自定义文件夹: {Config.CUSTOM_DATA_ROOT}")
        train_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "train")
        val_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "val")
        test_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "test")
        
        if not os.path.exists(train_dir): raise FileNotFoundError(f"缺失目录: {train_dir}")
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        
        # [逻辑优化] 检查测试集是否存在
        if os.path.exists(test_dir):
            logger.info("[Data] 发现独立测试集 test/")
            test_ds = datasets.ImageFolder(test_dir, transform=val_tf)
        else:
            logger.info("[Data] 未发现独立测试集，将复用验证集(val)进行最终评估")
            test_ds = val_ds 
            
        class_names = train_ds.classes

    # 更新全局配置
    Config.NUM_CLASSES = len(class_names)
    logger.info(f"[Data] 类别数: {Config.NUM_CLASSES}")
    
    # 创建DataLoader
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_dl, val_dl, test_dl, class_names

# ==========================================
# 4. 训练与验证逻辑
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """训练一个 Epoch"""
    model.train()
    total_loss, total_correct = 0.0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]", leave=False)
    
    for imgs, labels in bar:
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, epoch, phase="Val"):
    """验证模型性能"""
    model.eval()
    total_loss, total_correct = 0.0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]  ", leave=False)
    
    for imgs, labels in bar:
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
            
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        bar.set_postfix(loss=loss.item())
        
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

def evaluate_test_set(model, loader, class_names, logger):
    """训练结束后评估并在日志中绘制混淆矩阵"""
    if not loader or not HAS_SKLEARN: return
    logger.info("[Test] 执行最终评估...")
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # 打印分类报告
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    logger.info("\n" + report)
    
    # 绘制并保存混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(targets, preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(Config.SAVE_DIR, 'confusion_matrix.png'))
    logger.info("[Info] 混淆矩阵已保存")

def plot_history(h, save_dir, logger):
    """绘制训练曲线图"""
    epochs = range(1, len(h['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, h['train_acc'], label='Train')
    plt.plot(epochs, h['val_acc'], label='Val')
    plt.legend(); plt.title('Accuracy')
    
    # Loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, h['train_loss'], label='Train')
    plt.plot(epochs, h['val_loss'], label='Val')
    plt.legend(); plt.title('Loss')
    
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    logger.info("[Info] 训练曲线已保存")

def save_checkpoint(state, is_best, filename='last.pth'):
    """保存模型权重"""
    path = os.path.join(Config.SAVE_DIR, filename)
    torch.save(state, path)
    if is_best: torch.save(state, os.path.join(Config.SAVE_DIR, 'best_model.pth'))

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    seed_everything(Config.SEED)
    
    # 1. 初始化保存目录 (格式: 模型名_数据集名_时间)
    if Config.RESUME_PATH:
        Config.SAVE_DIR = os.path.dirname(Config.RESUME_PATH)
    else:
        if Config.USE_CUSTOM_DATASET:
            ds_name = os.path.basename(Config.CUSTOM_DATA_ROOT)
        else:
            ds_name = Config.BUILTIN_NAME
        run_name = f"{Config.MODEL_NAME}_{ds_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        Config.SAVE_DIR = os.path.join(Config.SAVE_DIR_ROOT, run_name)
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    logger = setup_logger(Config.SAVE_DIR)
    logger.info(f"[Config] 保存目录: {Config.SAVE_DIR}")
    
    # 2. 准备数据
    tmp_model = timm.create_model(Config.MODEL_NAME, pretrained=True)
    cfg = resolve_data_config({}, model=tmp_model)
    del tmp_model # 清理临时模型
    
    train_tf, val_tf = get_transforms(cfg)
    train_dl, val_dl, test_dl, class_names = get_dataloaders(train_tf, val_tf, logger)
    logger.info(f"[Data] 类别列表: {class_names}")
    
    # 3. 初始化模型
    logger.info(f"[Init] 创建模型: {Config.MODEL_NAME}")
    if not Config.RESUME_PATH and Config.CHECKPOINT_PATH and os.path.exists(Config.CHECKPOINT_PATH):
        logger.info(f"[Load] 加载本地预训练权重: {Config.CHECKPOINT_PATH}")
        model = timm.create_model(Config.MODEL_NAME, pretrained=False, num_classes=Config.NUM_CLASSES, checkpoint_path=Config.CHECKPOINT_PATH)
    else:
        model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=Config.NUM_CLASSES)
        
    model.to(Config.DEVICE)
    
    # 4. 优化器与调度器
    logger.info(f"[Init] Opt: {Config.OPTIMIZER_NAME}, Sch: {Config.SCHEDULER_NAME}")
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss() # 如需标签平滑可改为 nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 初始化早停
    early_stop = None
    if Config.EARLY_STOP_PATIENCE and Config.EARLY_STOP_PATIENCE > 0:
        logger.info(f"[Init] 早停开启 (Patience={Config.EARLY_STOP_PATIENCE})")
        early_stop = EarlyStopping(patience=Config.EARLY_STOP_PATIENCE)
    
    # 5. 训练循环
    start_epoch = 1
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # 断点恢复逻辑
    if Config.RESUME_PATH and os.path.exists(Config.RESUME_PATH):
        logger.info(f"[Resume] 恢复断点: {Config.RESUME_PATH}")
        ckpt = torch.load(Config.RESUME_PATH, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        history = ckpt['history']
    
    logger.info("[Start] 开始训练...")
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        v_loss, v_acc = validate(model, val_dl, criterion, epoch)
        
        # 记录历史
        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        
        logger.info(f"Epoch {epoch}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Loss: {t_loss:.4f}")
        
        # 更新学习率
        if scheduler:
            if Config.SCHEDULER_NAME == 'plateau': scheduler.step(v_acc)
            else: scheduler.step()
            
        # 保存最佳模型
        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            logger.info(f" -> 🌟 新的最佳模型 (Acc: {best_acc:.4f})")
            
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'history': history
        }, is_best)
        
        # 早停检测
        if early_stop:
            early_stop(v_acc)
            if early_stop.early_stop:
                logger.info("[Stop] 触发早停")
                break
                
    # 6. 结束评估
    plot_history(history, Config.SAVE_DIR, logger)
    if test_dl:
        # 加载最佳模型进行最终测试
        best_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=Config.DEVICE)
            model.load_state_dict(ckpt['state_dict'])
        evaluate_test_set(model, test_dl, class_names, logger)
        
    logger.info("[Done] 完成")
```