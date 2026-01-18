# 深度学习实战指南：如何科学选择模型、策略与增强？

在深度学习项目中，经常被戏称为“炼丹”，因为超参数众多。但实际上，从模型选择到训练策略，都有一套成熟的**方法论**。本文将总结一套通用的决策逻辑，帮助你告别“盲猜”。

## 1. 核心决策逻辑：三步走

### 第一步：模型选择 (Model Selection)

选择模型主要看**数据量**和**图像分辨率**。

| 场景 | 特征 | 推荐模型 | 训练策略 |
| --- | --- | --- | --- |
| **小数据** | 每类 < 100 张 | **ResNet18**, MobileNet, ShuffleNet | **必须**迁移学习 (Pretrained=True)，冻结部分层 |
| **中等数据** | 每类 100-1000 张 | **ResNet34/50**, EfficientNet-B0 | 迁移学习，全参数微调 (Full Fine-tuning) |
| **大数据** | 每类 > 1000 张 | **ResNet101**, ViT, Swin Transformer | 可从头训练，或微调大模型 |
| **低分辨率** | 32x32 (如 CIFAR) | **CNN** (ResNet, VGG) | 避免使用未经优化的 ViT |
| **高分辨率** | 224x224 以上 | **EfficientNet**, Swin | 模型设计包含了对大感受野的优化 |

### 第二步：优化器与调度器 (Optimizer & Scheduler)

工业界的“二选一”法则：

* **方案 A（省心首选）：AdamW + CosineAnnealing**
* **特点**：收敛快，对超参不敏感，效果通常很稳。
* **适用**：绝大多数任务，Transformer，以及快速验证想法时。
* **初始 LR**：通常设为 `1e-3` 或 `1e-4`。


* **方案 B（刷榜进阶）：SGD + Momentum + ReduceLROnPlateau**
* **特点**：收敛慢，难调，但调好了泛化性能往往略高于 AdamW。
* **适用**：学术复现、竞赛刷榜、CNN 架构。
* **初始 LR**：通常设为 `1e-1` 或 `1e-2`。



### 第三步：数据增强策略 (Data Augmentation Strategy)

增强强度应与**过拟合程度**成正比。

1. **起步期**：仅用 `Resize` + `RandomHorizontalFlip`。确保模型能跑通，观察基准。
2. **发现过拟合**（训练集 >> 验证集）：加入 `ColorJitter`（颜色抖动）、`RandomResizedCrop`（随机裁剪）。
3. **严重过拟合/冲刺高分**：加入 `RandomErasing`（擦除）、`Mixup`、`CutMix` 或 `AutoAugment`。

---

## 2. 数据增强参数详解 (参数速查表)

在使用 `torchvision.transforms` 时，各个参数的具体含义如下：

### A. 几何变换类

| 变换方法 | 关键参数 | 说明 | 推荐值 |
| --- | --- | --- | --- |
| **RandomResizedCrop** | `size` | 输出图像尺寸 | `(224, 224)` 或 `(32, 32)` |
|  | `scale` | 随机裁剪面积占原图的比例 | `(0.08, 1.0)` (标准) 或 `(0.6, 1.0)` (轻度) |
|  | `ratio` | 裁剪区域的长宽比 | `(3/4, 4/3)` |
| **RandomHorizontalFlip** | `p` | 执行翻转的概率 | `0.5` (即一半概率翻转) |
| **RandomRotation** | `degrees` | 旋转角度范围 | `15` (即 -15° 到 +15°) |
|  | `fill` | 旋转后空白区域填充色 | `0` (黑色) |

### B. 颜色变换类

| 变换方法 | 关键参数 | 说明 | 推荐值 |
| --- | --- | --- | --- |
| **ColorJitter** | `brightness` | 亮度抖动因子 | `0.4` |
|  | `contrast` | 对比度抖动因子 | `0.4` |
|  | `saturation` | 饱和度抖动因子 | `0.4` |
|  | `hue` | 色相抖动因子 | `0.1` (不要太大，否则颜色会失真) |
| **RandomGrayscale** | `p` | 转为灰度图的概率 | `0.1` |

### C. 遮挡与擦除类 (抗过拟合神器)

| 变换方法 | 关键参数 | 说明 | 推荐值 |
| --- | --- | --- | --- |
| **RandomErasing** | `p` | 执行擦除的概率 | `0.2` 到 `0.5` |
| (需放在ToTensor之后) | `scale` | 擦除区域面积比例 | `(0.02, 0.33)` |
|  | `ratio` | 擦除区域长宽比 | `(0.3, 3.3)` |
|  | `value` | 填充值 | `'random'` (噪点) 或 `0` (黑块) |

---

## 3. 常用的数据增强配置代码

以下是两套拿来即用的配置，直接复制到代码中即可。

### 配置一：工业级标准增强 (ResNet/ImageNet 标准)

适用于大多数 CNN 训练任务，兼顾多样性和稳定性。

```python
from torchvision import transforms

# 假设输入尺寸
crop_size = 224 # 如果是CIFAR改为32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf_standard = transforms.Compose([
    # 1. 随机裁剪并缩放 (最强几何增强)
    # scale=(0.08, 1.0) 允许裁剪出物体的一小部分，强迫模型学习局部特征
    transforms.RandomResizedCrop((crop_size, crop_size), scale=(0.08, 1.0)),
    
    # 2. 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 3. 颜色抖动 (亮度/对比度/饱和度/色相)
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    
    # 4. 转 Tensor 并 归一化
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    
    # 5. [可选] 随机擦除 (必须在 Normalize 之后)
    transforms.RandomErasing(p=0.25, value='random')
])

```

### 配置二：SOTA 懒人增强 (AutoAugment)

如果你不想手动调参，直接用 Google 搜索出来的最佳策略。

```python
from torchvision import transforms

train_tf_auto = transforms.Compose([
    # 先把图片变成固定大小，方便 AutoAugment 处理
    transforms.Resize((crop_size, crop_size)), 
    
    # 自动增强策略 (使用 ImageNet 或 CIFAR10 策略)
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    
    # 依然可以叠加 RandomErasing
    transforms.RandomErasing(p=0.25)
])

```

---

## 4. 总结：万能起手式

如果你面对一个新任务，不知道从何下手，请直接套用这个配置：

* **模型**：`ResNet50` (Pretrained=True)
* **优化器**：`AdamW` (LR=1e-4)
* **调度器**：`CosineAnnealingLR`
* **增强**：`RandomResizedCrop` + `HorizontalFlip` + `Normalize`
* **Batch Size**：显存允许的最大值

先跑通，再根据**“训练集好于验证集 -> 过拟合”**的原则去加增强或正则化。