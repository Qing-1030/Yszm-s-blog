# 目标检测通用模板

## 一、基于MMDetection

```python
from mmengine import Config
from mmengine.runner import Runner
import os
import ssl
import json
import glob
import matplotlib.pyplot as plt

# 解决SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# =========================================================
# 1. 用户配置区 (USER CONFIGURATION)
# =========================================================

# --- [核心配置] 数据集路径 ---
DATA_ROOT = 'mask_coco/'  # 数据集根目录
CLASSES = ('un_mask', 'mask')  # 类别名称 (需与标注一致)

TRAIN_ANN = 'annotations/train.json'  # 训练集标注
TRAIN_IMG = 'train/'  # 训练集图片路径
VAL_ANN = 'annotations/val.json'  # 验证集标注
VAL_IMG = 'val/'  # 验证集图片路径
TEST_ANN = None  # 测试集标注 (可选, None则复用验证集)
TEST_IMG = None  # 测试集图片 (可选)

# --- [核心配置] 模型基座 ---
# 建议：RTMDet (新一代高性能模型) 或 Faster R-CNN (经典稳定)
BASE_CONFIG = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
PRETRAINED = None  # [可调节] 预训练权重路径 (None=自动下载)

# --- [可调节] 训练超参数 ---
BATCH_SIZE = 8  # 批次大小 (显存允许越大越好)
NUM_EPOCHS = 24  # 总轮数 (建议 12/24/36)
WORK_DIR = './work_dirs/mask_v2'  # 结果输出目录
IMAGE_SCALE = (1333, 800)  # 输入尺寸 (W, H)

# --- [高阶微调] 优化器与调度 ---
OPTIMIZER = 'AdamW'  # 选项: 'AdamW' (推荐), 'SGD'
INIT_LR = 0.0001  # 初始学习率 (AdamW: 1e-4, SGD: 0.02)
SCHEDULER = 'Cosine'  # 选项: 'Cosine' (平滑), 'Step' (阶梯)

# --- [高阶微调] 数据增强开关 ---
ENABLE_AUG = True  # 是否开启增强 (颜色抖动/多尺度)


# =========================================================
# 2. 工具函数：自动绘制训练曲线
# =========================================================
def plot_logs(work_dir):
    """读取最新JSON日志并绘制Loss/mAP曲线"""
    try:
        json_files = glob.glob(os.path.join(work_dir, '*', 'vis_data', '*.json'))
        if not json_files: return
        latest_log = max(json_files, key=os.path.getmtime)

        losses, mAPs, iters, epochs = [], [], [], []
        with open(latest_log, 'r') as f:
            for line in f:
                log = json.loads(line)
                if 'loss' in log:
                    losses.append(log['loss'])
                    iters.append(log['step'])
                if 'coco/bbox_mAP' in log:
                    mAPs.append(log['coco/bbox_mAP'])
                    epochs.append(log.get('epoch', len(mAPs)))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iters, losses, label='Train Loss')
        plt.title('Loss Curve');
        plt.legend();
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if mAPs:
            plt.plot(epochs, mAPs, label='mAP', color='orange', marker='o')
            plt.title(f'Max mAP: {max(mAPs):.4f}');
            plt.legend();
            plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(work_dir, 'training_result.png'))
        print(f"✅ 训练曲线已保存: {os.path.join(work_dir, 'training_result.png')}")
        plt.close()
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


# =========================================================
# 3. 核心配置构建逻辑
# =========================================================
print(f"🚀 Loading Base Config: {BASE_CONFIG}")
cfg = Config.fromfile(BASE_CONFIG)

# --- 全局环境配置 ---
cfg.data_root = DATA_ROOT
cfg.work_dir = WORK_DIR
cfg.metainfo = dict(classes=CLASSES, palette=[(220, 20, 60)] * len(CLASSES))

# --- [重点] 增强型数据管道 (Augmentation Pipeline) ---
# 1. 基础管道
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]

# 2. 注入增强策略
if ENABLE_AUG:
    pipeline.extend([
        # [增强] 光度畸变: 随机调整亮度、对比度、饱和度、色相
        dict(type='PhotoMetricDistortion'),
        # [增强] 多尺度训练: 尺寸在 (W*0.8, H*0.8) 到 (W, H) 之间浮动
        dict(type='RandomResize', scale=IMAGE_SCALE, ratio_range=(0.8, 1.0), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
    ])
else:
    # 仅做基础Resize
    pipeline.extend([
        dict(type='Resize', scale=IMAGE_SCALE, keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
    ])

# 3. 打包数据
pipeline.append(dict(type='PackDetInputs'))

# --- 数据加载器 (DataLoader) ---
# 训练集
cfg.train_dataloader.dataset = dict(
    type='CocoDataset',
    data_root=DATA_ROOT,
    ann_file=TRAIN_ANN,
    data_prefix=dict(img=TRAIN_IMG),
    metainfo=cfg.metainfo,
    pipeline=pipeline,
    backend_args=None
)

# 验证集 pipeline (不做增强，只做Resize)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMAGE_SCALE, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# 验证集
cfg.val_dataloader.dataset = dict(
    type='CocoDataset',
    data_root=DATA_ROOT,
    ann_file=VAL_ANN,
    data_prefix=dict(img=VAL_IMG),
    metainfo=cfg.metainfo,
    pipeline=test_pipeline,
    test_mode=True,
    backend_args=None
)

cfg.val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(DATA_ROOT, VAL_ANN),
    metric='bbox'
)

# --- 智能测试集逻辑 ---
if TEST_ANN and TEST_IMG and os.path.exists(os.path.join(DATA_ROOT, TEST_ANN)):
    print(f"✅ 检测到测试集，将在训练后进行独立测试。")
    cfg.test_dataloader = cfg.val_dataloader.copy()
    cfg.test_dataloader.dataset.ann_file = TEST_ANN
    cfg.test_dataloader.dataset.data_prefix = dict(img=TEST_IMG)
    cfg.test_evaluator = cfg.val_evaluator.copy()
    cfg.test_evaluator.ann_file = os.path.join(DATA_ROOT, TEST_ANN)
else:
    print(f"ℹ️ 未检测到测试集，将复用验证集。")
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

# --- 模型类别自适应 ---
if hasattr(cfg.model, 'roi_head'):
    cfg.model.roi_head.bbox_head.num_classes = len(CLASSES)
elif hasattr(cfg.model, 'bbox_head'):
    cfg.model.bbox_head.num_classes = len(CLASSES)

# --- 优化器配置 (AdamW/SGD) ---
if OPTIMIZER == 'AdamW':
    cfg.optim_wrapper.optimizer = dict(type='AdamW', lr=INIT_LR, weight_decay=0.05)
else:
    cfg.optim_wrapper.optimizer = dict(type='SGD', lr=INIT_LR, momentum=0.9, weight_decay=0.0001)

# --- 学习率调度 (Warmup + Main) ---
cfg.train_cfg.max_epochs = NUM_EPOCHS
cfg.train_cfg.val_interval = max(1, NUM_EPOCHS // 10)  # [可调节] 验证频率

# 热身策略 (前500次迭代)
warmup = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)

# 主调度策略
if SCHEDULER == 'Cosine':
    main_sched = dict(type='CosineAnnealingLR', T_max=NUM_EPOCHS, by_epoch=True, begin=0, end=NUM_EPOCHS,
                      eta_min=INIT_LR * 0.05)
else:
    main_sched = dict(type='MultiStepLR', milestones=[int(NUM_EPOCHS * 0.75), int(NUM_EPOCHS * 0.9)], gamma=0.1,
                      by_epoch=True)

cfg.param_scheduler = [warmup, main_sched]

# --- 钩子与可视化 ---
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint = dict(
    type='CheckpointHook',
    interval=1,
    max_keep_ckpts=2,  # [可调节] 只保留最近2个权重
    save_best='coco/bbox_mAP',  # [可调节] 保存mAP最高的
    rule='greater'
)

# 启用 TensorBoard
cfg.vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
cfg.visualizer = dict(type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer')

# --- 运行参数 ---
cfg.load_from = PRETRAINED
cfg.resume = False
os.makedirs(WORK_DIR, exist_ok=True)
cfg.dump(os.path.join(WORK_DIR, 'final_config.py'))  # 保存完整配置供Debug

if __name__ == '__main__':
    runner = Runner.from_cfg(cfg)
    runner.train()

    print("\n>>> 开始最终测试...")
    runner.test()

    print("\n>>> 正在绘制曲线...")
    plot_logs(WORK_DIR)
```

## 二、基于Ultralytics（YOLO）

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # =====================================================
    # 1. 基础配置 (必填)
    # =====================================================
    MODEL_NAME = 'yolov8n.pt'  # 预训练模型 (yolov8n.pt, yolov8s.pt, yolo11n.pt...)
    DATA_YAML = 'data.yaml'  # 数据集配置文件路径
    PROJECT = 'runs/detect'  # 项目保存根目录
    NAME = 'exp_001'  # 本次实验名称

    # 加载模型
    model = YOLO(MODEL_NAME)

    print("🚀 开始全参数训练...")
    # =====================================================
    # 2. 训练参数 (按需取消注释)
    # 完整参数文档: https://docs.ultralytics.com/modes/train/
    # =====================================================
    results = model.train(
        # --- [核心参数] (建议始终启用) ---
        data=DATA_YAML,  # 数据集配置
        epochs=100,  # 训练轮数 (建议 100-300)
        imgsz=640,  # 输入图片尺寸
        batch=16,  # 批次大小 (-1 为自动适配 AutoBatch)
        device='0',  # 显卡索引 ('0', '0,1', 'cpu')
        workers=8,  # 数据加载线程数 (Windows 若报错请设为 0)
        project=PROJECT,  # 保存根目录
        name=NAME,  # 实验保存目录名
        exist_ok=False,  # 是否覆盖同名文件夹 (False=自动新建 exp2, exp3...)
        pretrained=True,  # 是否加载预训练权重
        optimizer='auto',  # 优化器 ('SGD', 'Adam', 'AdamW', 'auto')
        verbose=True,  # 打印详细日志
        seed=42,  # 随机种子 (固定以复现结果)

        # --- [训练控制] (常用) ---
        # patience = 50,       # 早停机制: 连续 N 轮不涨分则停止
        # save = True,         # 是否保存模型 (True=保存最后和最佳)
        # save_period = -1,    # 每隔 N 轮强制保存一次 (-1=关闭)
        # resume = False,      # 断点续训 (设为 True 需加载 last.pt)
        # amp = True,          # 混合精度训练 (节省显存，推荐开启)
        # val = True,          # 训练过程中是否进行验证
        # cache = False,       # 是否缓存图片到内存 (加速训练，需大内存)

        # --- [超参数与学习率] (进阶) ---
        # lr0 = 0.01,          # 初始学习率 (SGD=0.01, Adam=0.001)
        # lrf = 0.01,          # 最终学习率 (lr0 * lrf)
        # momentum = 0.937,    # 动量
        # weight_decay = 0.0005, # 权重衰减 (防止过拟合)
        # warmup_epochs = 3.0,   # 热身轮数
        # cos_lr = False,        # 使用余弦退火学习率 (有助于提升最终精度)

        # --- [数据增强] (核心提分点 - 0.0~1.0) ---
        # mosaic = 1.0,        # 马赛克增强 (拼接4张图, 对小目标极好, 推荐 1.0)
        # mixup = 0.0,         # 混合增强 (叠图, 复杂场景可试 0.1)
        # hsv_h = 0.015,       # 色调变化 (对颜色敏感的任务如红绿灯，设为 0)
        # hsv_s = 0.7,         # 饱和度变化
        # hsv_v = 0.4,         # 亮度变化 (光照变化大可调高)
        # degrees = 0.0,       # 旋转角度 (+/- deg)
        # translate = 0.1,     # 平移
        # scale = 0.5,         # 缩放 (+/- gain)
        # shear = 0.0,         # 剪切
        # perspective = 0.0,   # 透视变换
        # flipud = 0.0,        # 上下翻转 (俯拍/显微镜可设 0.5)
        # fliplr = 0.5,        # 左右翻转 (区分左右方向的任务设为 0)

        # --- [特殊策略] ---
        # close_mosaic = 10,   # 最后 10 轮关闭 Mosaic (稳定模型分布，提升精度)
    )
    print("✅ 训练结束")
```