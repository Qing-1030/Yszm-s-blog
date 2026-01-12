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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm
import matplotlib.pyplot as plt

# 尝试导入高级评估绘图库，如果没安装则跳过
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] 未安装 scikit-learn 或 seaborn，将跳过混淆矩阵绘制。建议安装: pip install scikit-learn seaborn")


# ==========================================
# 1. 配置区域 [可微调]
# ==========================================
class Config:
    # 路径设置
    data_root = "flower_data"  # <--- [可修改] 数据集根目录 (需包含 train/val/test 文件夹)
    save_dir = "./results"  # <--- [可修改] 结果保存路径

    # 模型设置
    model_name = "resnet50"  # <--- [可微调] 模型名称 (如 resnet18, efficientnet_b0, mobilenetv3_large_100)
    num_classes = 5  # <--- [可修改] 分类类别数
    pretrained = True  # <--- [可微调] 是否使用在线预训练权重
    checkpoint_path = ""  # <--- [可微调] 本地权重路径 (仅当 pretrained=False 时使用)

    # 训练超参数
    batch_size = 32  # <--- [可微调] 批次大小
    epochs = 20  # <--- [可微调] 训练轮数
    lr = 1e-4  # <--- [可微调] 学习率
    weight_decay = 1e-4  # <--- [可微调] 权重衰减 (L2正则化)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. 准备工作：模型与数据
# ==========================================
def get_model_and_transforms():
    # 1. 创建模型
    model = timm.create_model(
        Config.model_name,
        pretrained=Config.pretrained,
        checkpoint_path=Config.checkpoint_path,
        num_classes=Config.num_classes
    )
    model.to(Config.device)

    # 2. 获取默认配置并打印
    config = resolve_data_config({}, model=model)
    # print(f"[Info] Model Config: {config}")

    # 3. 定义数据增强 [可微调]
    # 训练集：需要增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),  # 或使用 RandomResizedCrop(224)
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集/测试集：不需要增强，只需要标准化
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, train_transform, val_test_transform


# ==========================================
# 3. 核心功能：训练、验证、测试
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, total_correct = 0.0, 0

    bar = tqdm(loader, desc=f"Epoch {epoch}/{Config.epochs} [Train]")
    for imgs, labels in bar:
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)

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

    bar = tqdm(loader, desc=f"Epoch {epoch}/{Config.epochs} [{phase}]  ")
    for imgs, labels in bar:
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        bar.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


def evaluate_test_set(model, test_loader, class_names):
    """
    独立测试函数：输出分类报告和混淆矩阵
    """
    if not HAS_SKLEARN:
        print("[Info] 跳过详细测试报告（缺少sklearn库）")
        return

    model.eval()
    all_preds = []
    all_labels = []

    print(f"\n[Test] 正在进行最终测试集评估...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(Config.device)
            labels = labels.to(Config.device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. 打印分类报告 (Precision, Recall, F1)
    print("\n" + "=" * 50)
    print("FINAL TEST REPORT")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 2. 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    save_path = os.path.join(Config.save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"[Info] 混淆矩阵已保存至: {save_path}")
    # plt.show()


# ==========================================
# 4. 辅助功能：画图
# ==========================================
def plot_history(history, save_dir):
    epochs = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    print(f"[Info] 训练曲线已保存至: {os.path.join(save_dir, 'training_curve.png')}")


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 初始化
    os.makedirs(Config.save_dir, exist_ok=True)
    model, train_tf, val_test_tf = get_model_and_transforms()

    # 加载数据集
    # 假设目录结构为: data_root/train, data_root/val, data_root/test
    train_ds = datasets.ImageFolder(os.path.join(Config.data_root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(Config.data_root, "val"), transform=val_test_tf)

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # 定义优化器与损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

    # ---------------------------
    # Phase 1: 训练循环
    # ---------------------------
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    print(f"\n[Start] 开始训练... (Total Epochs: {Config.epochs})")
    for epoch in range(1, Config.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc = validate(model, val_loader, criterion, epoch, phase="Val")

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"Epoch {epoch}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")

        # 保存最佳模型
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(Config.save_dir, "best_model.pth"))
            print(f" -> 🌟 最佳模型已更新 (Acc: {best_acc:.4f})")

    # 绘制曲线
    plot_history(history, Config.save_dir)

    # ---------------------------
    # Phase 2: 测试集评估
    # ---------------------------
    print("\n" + "=" * 30)
    print("进入测试阶段 (Test Phase)")
    print("=" * 30)

    # 1. 加载测试数据 (注意使用 val_test_tf，不做增强)
    test_dir = os.path.join(Config.data_root, "test")
    if os.path.exists(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=val_test_tf)
        test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4)

        # 2. 必须重新加载最佳权重 (Best Weights)
        best_path = os.path.join(Config.save_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_path))
        print(f"[Info] 已加载最佳权重用于测试: {best_path}")

        # 3. 执行详细评估
        evaluate_test_set(model, test_loader, train_ds.classes)

    else:
        print(f"[Warning] 未找到测试集文件夹 {test_dir}，跳过测试步骤。")

    print("\n[Done] 所有任务完成。")
```