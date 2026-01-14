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

# 检查高级绘图库
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] 未安装 sklearn 或 seaborn，跳过混淆矩阵绘制。")

# ==========================================
# 1. 全局配置 [核心修改区]
# ==========================================
class Config:
    # --- 数据集设置 ---
    USE_CUSTOM_DATASET = True        # <--- [可微调] True=自定义文件夹, False=内置
    CUSTOM_DATA_ROOT = "flower_data" # <--- [可微调] 自定义数据集路径
    BUILTIN_NAME = "CIFAR10"         # <--- [可微调] 内置数据集名称
    DATA_DOWNLOAD_ROOT = "./data"    # <--- [可微调] 下载缓存路径
    
    # --- 结果保存 ---
    SAVE_DIR_ROOT = "./results"      # <--- [可微调] 结果保存根目录
    SAVE_DIR = ""                    # (运行时自动生成)
    
    # --- 模型设置 ---
    MODEL_NAME = "resnet50"          # <--- [可微调] 模型名称 (timm库支持的名称)
    CHECKPOINT_PATH = ""             # <--- [可微调] 初始预训练权重 (迁移学习用)
    RESUME_PATH = ""                 # <--- [可微调] 断点续训文件路径 (.pth)
    NUM_CLASSES = 0                  # (运行时自动覆盖)
    
    # --- 训练超参数 ---
    BATCH_SIZE = 32                  # <--- [可微调] 批次大小
    EPOCHS = 50                      # <--- [可微调] 训练总轮数
    LR = 1e-4                        # <--- [可微调] 初始学习率
    WEIGHT_DECAY = 1e-4              # <--- [可微调] L2正则化系数
    SEED = 42                        # <--- [可微调] 随机种子
    
    # --- 策略选择 ---
    OPTIMIZER_NAME = 'adamw'         # <--- [可微调] 'adamw', 'adam', 'sgd'
    SCHEDULER_NAME = 'plateau'       # <--- [可微调] 'plateau', 'cosine', 'step'
    EARLY_STOP_PATIENCE = 7          # <--- [可微调] 早停耐心轮数 (0=关闭)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 辅助工具
# ==========================================
def setup_logger(save_dir):
    """配置日志：同时输出到文件和控制台"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def seed_everything(seed):
    """固定随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    """早停控制器"""
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        if self.patience <= 0: return
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0

def get_optimizer(model):
    """优化器工厂"""
    name = Config.OPTIMIZER_NAME.lower()
    p = model.parameters()
    if name == 'adamw': return optim.AdamW(p, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    elif name == 'sgd': return optim.SGD(p, lr=Config.LR, momentum=0.9, weight_decay=Config.WEIGHT_DECAY)
    else: return optim.Adam(p, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

def get_scheduler(optimizer):
    """学习率策略工厂"""
    name = Config.SCHEDULER_NAME.lower()
    if name == 'plateau': return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    elif name == 'cosine': return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    elif name == 'step': return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return None

# ==========================================
# 3. 数据加载与处理
# ==========================================
def get_transforms(model_cfg):
    """根据模型配置生成Transforms"""
    input_size = model_cfg.get('input_size', (3, 224, 224))
    crop_size = input_size[1]
    mean = model_cfg.get('mean', [0.485, 0.456, 0.406])
    std = model_cfg.get('std', [0.229, 0.224, 0.225])
    
    train_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tf, val_tf

def get_dataloaders(train_tf, val_tf, logger):
    """加载数据集"""
    if not Config.USE_CUSTOM_DATASET:
        logger.info(f"[Data] 加载内置数据集: {Config.BUILTIN_NAME}")
        try: DatasetClass = getattr(datasets, Config.BUILTIN_NAME)
        except AttributeError: raise ValueError(f"不支持的数据集: {Config.BUILTIN_NAME}")
        
        full_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=True, download=True, transform=train_tf)
        test_ds = DatasetClass(root=Config.DATA_DOWNLOAD_ROOT, train=False, download=True, transform=val_tf)
        
        train_sz = int(0.9 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [train_sz, len(full_ds)-train_sz])
        class_names = full_ds.classes
        ds_name = Config.BUILTIN_NAME
    else:
        logger.info(f"[Data] 加载自定义文件夹: {Config.CUSTOM_DATA_ROOT}")
        train_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "train")
        val_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "val")
        test_dir = os.path.join(Config.CUSTOM_DATA_ROOT, "test")
        
        if not os.path.exists(train_dir): raise FileNotFoundError(f"缺失目录: {train_dir}")
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        test_ds = datasets.ImageFolder(test_dir, transform=val_tf) if os.path.exists(test_dir) else None
        class_names = train_ds.classes
        ds_name = os.path.basename(Config.CUSTOM_DATA_ROOT)

    Config.NUM_CLASSES = len(class_names)
    logger.info(f"[Data] 类别数: {Config.NUM_CLASSES}")
    
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4) if test_ds else None
    
    return train_dl, val_dl, test_dl, class_names, ds_name

# ==========================================
# 4. 训练与验证核心
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
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
    """测试集评估与混淆矩阵绘制"""
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
    
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    logger.info("\n" + report)
    print(report)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(targets, preds), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.savefig(os.path.join(Config.SAVE_DIR, 'confusion_matrix.png'))
    logger.info("[Info] 混淆矩阵已保存")

def plot_history(h, save_dir, logger):
    epochs = range(1, len(h['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(epochs, h['train_acc'], label='Train'); plt.plot(epochs, h['val_acc'], label='Val'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1, 2, 2); plt.plot(epochs, h['train_loss'], label='Train'); plt.plot(epochs, h['val_loss'], label='Val'); plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    logger.info("[Info] 训练曲线已保存")

def save_checkpoint(state, is_best, filename='last.pth'):
    path = os.path.join(Config.SAVE_DIR, filename)
    torch.save(state, path)
    if is_best: torch.save(state, os.path.join(Config.SAVE_DIR, 'best_model.pth'))

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    seed_everything(Config.SEED)
    
    # 1. 确定保存目录
    if Config.RESUME_PATH:
        # 断点续训复用原目录
        Config.SAVE_DIR = os.path.dirname(Config.RESUME_PATH)
    else:
        # 获取数据集名称用于命名
        if Config.USE_CUSTOM_DATASET:
            ds_name = os.path.basename(Config.CUSTOM_DATA_ROOT)
        else:
            ds_name = Config.BUILTIN_NAME
            
        # 格式: 模型名_数据集名_时间戳
        run_name = f"{Config.MODEL_NAME}_{ds_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        Config.SAVE_DIR = os.path.join(Config.SAVE_DIR_ROOT, run_name)
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    logger = setup_logger(Config.SAVE_DIR)
    logger.info(f"[Config] 保存目录: {Config.SAVE_DIR}")
    
    # 2. 准备数据
    tmp_model = timm.create_model(Config.MODEL_NAME, pretrained=True)
    cfg = resolve_data_config({}, model=tmp_model)
    del tmp_model
    
    train_tf, val_tf = get_transforms(cfg)
    train_dl, val_dl, test_dl, class_names, ds_name = get_dataloaders(train_tf, val_tf, logger)
    
    # 3. 初始化模型
    logger.info(f"[Init] 创建模型: {Config.MODEL_NAME}")
    if not Config.RESUME_PATH and Config.CHECKPOINT_PATH and os.path.exists(Config.CHECKPOINT_PATH):
        logger.info(f"[Load] 加载本地初始化权重: {Config.CHECKPOINT_PATH}")
        model = timm.create_model(Config.MODEL_NAME, pretrained=False, checkpoint_path=Config.CHECKPOINT_PATH)
    else:
        model = timm.create_model(Config.MODEL_NAME, pretrained=True)
        
    model.reset_classifier(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    
    # 4. 优化器/调度器/Loss
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=Config.EARLY_STOP_PATIENCE) if Config.EARLY_STOP_PATIENCE > 0 else None
    
    # 5. 断点恢复逻辑
    start_epoch = 1
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if Config.RESUME_PATH and os.path.exists(Config.RESUME_PATH):
        logger.info(f"[Resume] 恢复断点: {Config.RESUME_PATH}")
        ckpt = torch.load(Config.RESUME_PATH, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        history = ckpt['history']
        logger.info(f"[Resume] 从第 {start_epoch} 轮继续 (最佳Acc: {best_acc:.4f})")
    
    # 6. 训练循环
    logger.info("[Start] 开始训练...")
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        v_loss, v_acc = validate(model, val_dl, criterion, epoch)
        
        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        
        logger.info(f"Epoch {epoch}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Loss: {t_loss:.4f}")
        
        # 学习率更新
        if scheduler:
            if Config.SCHEDULER_NAME == 'plateau': scheduler.step(v_acc)
            else: scheduler.step()
            
        # 保存最佳模型
        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            logger.info(f" -> 🌟 新的最佳模型 (Acc: {best_acc:.4f})")
            
        # 保存断点
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
                
    # 7. 收尾
    plot_history(history, Config.SAVE_DIR, logger)
    if test_dl:
        ckpt = torch.load(os.path.join(Config.SAVE_DIR, "best_model.pth"), map_location=Config.DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        evaluate_test_set(model, test_dl, class_names, logger)
        
    logger.info("[Done] 完成")
```