---
title: 数据集划分
date: 2026-01-12
tags:
  - Python
  - 数据集
---

# 数据集划分

> 摘要：一个将数据集以`7:2:1`的比例划分的代码

## 核心代码：

```python
import os
import shutil
import random
from tqdm import tqdm

# ==========================================
# 配置区域
# ==========================================
source_root = "./raw_data"  # <--- [请修改] 你的原始数据集路径
target_root = "./flower_data"  # <--- [请修改] 输出路径
split_ratio = [0.7, 0.2, 0.1]  # <--- [可修改] 训练:验证:测试 比例 (和需为1)


def split_dataset():
    # 检查原始路径
    if not os.path.exists(source_root):
        print(f"错误: 找不到路径 {source_root}")
        return

    # 获取所有类别文件夹
    classes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    print(f"发现类别: {classes}")

    # 创建目标文件夹结构
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

    # 开始划分
    for cls in tqdm(classes, desc="正在划分数据集"):
        cls_path = os.path.join(source_root, cls)
        # 获取该类别下所有图片文件
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # 随机打乱
        random.shuffle(images)

        # 计算切分点
        num_total = len(images)
        num_train = int(num_total * split_ratio[0])
        num_val = int(num_total * split_ratio[1])
        # 剩下的都给测试集，避免因取整丢失数据

        train_imgs = images[:num_train]
        val_imgs = images[num_train: num_train + num_val]
        test_imgs = images[num_train + num_val:]

        # 复制文件函数
        def copy_files(file_list, split_name):
            for file_name in file_list:
                src = os.path.join(cls_path, file_name)
                dst = os.path.join(target_root, split_name, cls, file_name)
                shutil.copy(src, dst)

        # 执行复制
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')

        print(f"类别 [{cls}]: 总数 {num_total} -> Train:{len(train_imgs)} Val:{len(val_imgs)} Test:{len(test_imgs)}")

    print(f"\n[Done] 数据集划分完成！保存在: {target_root}")


if __name__ == "__main__":
    split_dataset()
```