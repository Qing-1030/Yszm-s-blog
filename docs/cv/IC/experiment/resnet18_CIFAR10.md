# 基于 ResNet18 的 CIFAR-10 图像分类

在深度学习的入门与进阶过程中，CIFAR-10 是最经典的“磨刀石”。近期我使用 PyTorch 和 `timm` 库，基于预训练的 **ResNet18** 模型进行了一次完整的训练实验。

令人兴奋的是，仅仅经过 20 个 Epoch 的训练，模型在测试集上就达到了 **96.16%** 的准确率 。以下是我的实验记录与详细分析。

## 1. 实验环境与配置

本次实验采用迁移学习的思路，加载了在 ImageNet 上预训练的权重，这大大加速了收敛过程。

* **模型架构**: ResNet18 (使用 `timm` 库的 `resnet18.a1_in1k` 权重) 

* **数据集**: CIFAR-10 (包含 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck 共10类) 

* **优化器**: Adam 

* **学习率策略**: Cosine (余弦退火) 

* **训练轮数**: 20 Epochs 

* **早停策略**: 关闭 

## 2. 训练过程分析

通过观察训练曲线 和日志数据，可以发现模型学习得非常快：

* **极速收敛**: 在第 1 个 Epoch 结束时，验证集准确率（Val Acc）就已经达到了 **91.18%** 。这得益于预训练权重的强大特征提取能力。

* **稳步提升**: 此后准确率一路攀升，在第 15 个 Epoch 突破了 96% 的大关 (96.32%) 。

* **最佳模型**: 最终在第 19 个 Epoch 达到了训练过程中的最高验证集准确率 **96.36%** 。

从损失曲线 来看，训练集 Loss (蓝色) 持续下降接近于 0，而验证集 Loss (橙色) 在第 8-10 轮左右开始趋于平缓（约 0.02 - 0.03 之间）。虽然训练集准确率最终达到了 99.48% ，略高于验证集，存在轻微的过拟合，但考虑到验证集准确率依然很高，这种程度的拟合是可以接受的。

![训练曲线](resnet18_CIFAR10_training_curve.png)

## 3. 模型评估与表现

训练结束后，我对模型在测试集上进行了最终评估，结果如下：

### 3.1 整体指标

* **Test Accuracy**: **96.16%** 

* **Macro Avg F1-Score**: 0.9616 

### 3.2 混淆矩阵深度解析

混淆矩阵 为我们提供了比准确率更细节的视角。通过观察矩阵，我们可以发现模型在不同类别上的表现差异：

**表现最好的类别 (Top Performers):**

* **Frog (青蛙)**: 准确识别了 980 张，仅有极少数错误。
* **Ship (船)**: 准确识别 977 张。
* **Automobile (汽车)**: 准确识别 973 张。

**容易混淆的类别 (Confusion Pairs):**
正如预期的那样，主要错误集中在外观相似的类别上：

1. **Cat vs Dog (猫狗大战)**:
* 有 **43** 张猫的照片被误认成了狗。
* 有 **42** 张狗的照片被误认成了猫。
这是整个模型最大的失分点，导致 Cat 类的 Recall 仅为 92.30% 。

2. **Automobile vs Truck (机动车混淆)**:
* 有 **17** 张汽车被误认为卡车，**15** 张卡车被误认为汽车。

3. **Airplane vs Bird (背景干扰?)**:
* 有 **11** 张飞机被误认为鸟。这可能是因为两者都常出现在蓝色背景（天空）中。

![混淆矩阵](resnet18_CIFAR10_confusion_matrix.png)

## 4. 总结

本次实验使用 ResNet18 配合余弦退火策略，在 CIFAR-10 上取得了非常令人满意的结果。

* **优点**: 训练效率高，收敛速度快，整体准确率达到了 SOTA 入门级水平 (96%+)。
* **改进空间**: 针对“猫狗混淆”的问题，未来可以尝试引入更强的数据增强（如 CutMix 或 Mixup）来提取更细粒度的特征，或者尝试更大参数量的模型（如 ResNet50 或 EfficientNet）。

## 5.代码

本次实验基于预先构建的图像分类通用框架进行，此处仅对差异化的微调部分（Fine-tuning）进行说明。

```python
class Config:
    # --- 数据集设置 ---
    USE_CUSTOM_DATASET = False  # 设为False，即使用内置数据集
    CUSTOM_DATA_ROOT = ""  # 自定义数据集路径
    BUILTIN_NAME = "CIFAR10"  # 内置数据集名称
    DATA_DOWNLOAD_ROOT = "./data"  # 下载缓存路径

    # --- 结果保存 ---
    SAVE_DIR_ROOT = "./results"  # 结果保存根目录
    SAVE_DIR = ""  # (运行时自动生成)

    # --- 模型设置 ---
    MODEL_NAME = "resnet18"  # 模型名称 (timm库支持的名称)
    CHECKPOINT_PATH = ""  # 初始预训练权重
    RESUME_PATH = ""  # 断点续训文件路径 (.pth)
    NUM_CLASSES = 0  # (运行时自动覆盖)

    # --- 训练超参数 ---
    BATCH_SIZE = 32  # 批次大小
    EPOCHS = 20  # 训练总轮数
    LR = 1e-4  # 初始学习率
    WEIGHT_DECAY = 1e-4  # L2正则化系数
    SEED = 42  # 随机种子

    # --- 策略选择 ---
    OPTIMIZER_NAME = 'adam'  # 使用 'adam' 优化器
    SCHEDULER_NAME = 'cosine'  # 使用 '余弦退火' 学习率调度策略

    # --- 早停设置 ---
    # 0 或 None 表示关闭早停，> 0 表示开启早停的耐心轮数
    EARLY_STOP_PATIENCE = 0  # 早停耐心轮数 (0=关闭)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```