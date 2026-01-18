# 图片预测

## 一、基于MMDetection

```python
import os
import glob
from mmdet.apis import DetInferencer

# =========================================================
# 1. 用户配置区 (USER CONFIGURATION)
# =========================================================

# --- [必须修改] 核心路径配置 ---
# 训练生成的配置文件路径 (.py)
CONFIG_FILE = 'my_mask_config.py'

# 训练好的权重文件路径 (.pth)
# 通常在 work_dirs/xxx/ 目录下，建议选择 best_coco_bbox_mAP_epoch_xx.pth
WEIGHTS_FILE = 'work_dirs/mask_detection/epoch_10.pth'

# 输入路径：支持 "单张图片路径" 或 "文件夹路径"
# 示例: 'data/test.jpg' 或 'data/val/'
INPUT_PATH = 'mask_coco/val/'

# --- [可微调] 推理参数设置 ---
# 结果保存目录 (脚本会自动创建)
OUT_DIR = 'outputs_test'

# 置信度阈值 (0.0 ~ 1.0)
# 低于此分数的预测框将被过滤，不会显示在终端或保存的图片上
SCORE_THR = 0.4

# 运行设备 ('cuda:0' 或 'cpu')
DEVICE = 'cuda:0'


# =========================================================
# 2. 核心逻辑 (CORE LOGIC) - 通常无需修改
# =========================================================

def main():
    # 1. 检查输入路径有效性
    files = []
    if os.path.isdir(INPUT_PATH):
        # 若是文件夹，递归获取常见图片格式
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
        files.sort()
        print(f"📂 检测到输入为文件夹，共找到 {len(files)} 张图片。")
    elif os.path.isfile(INPUT_PATH):
        # 若是单张图片
        files = [INPUT_PATH]
        print(f"📄 检测到输入为单张图片。")
    else:
        print(f"❌ 错误：输入路径不存在 -> {INPUT_PATH}")
        return

    if not files:
        print("⚠️ 目录下未找到有效图片文件。")
        return

    # 2. 初始化推理器
    print(f"🚀 初始化模型...\n   Config: {CONFIG_FILE}\n   Weights: {WEIGHTS_FILE}")
    try:
        inferencer = DetInferencer(model=CONFIG_FILE, weights=WEIGHTS_FILE, device=DEVICE)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 自动从元数据获取类别名称 (避免硬编码)
    if hasattr(inferencer.model, 'dataset_meta'):
        class_names = inferencer.model.dataset_meta.get('classes', [])
    else:
        class_names = []
    print(f"✅ 模型加载完毕，类别列表: {class_names}")
    print("-" * 60)

    # 4. 批量推理循环
    for i, img_path in enumerate(files):
        file_name = os.path.basename(img_path)
        print(f"[{i + 1}/{len(files)}] 正在推理: {file_name} ...")

        # 执行推理
        # pred_score_thr: 控制可视化图片的绘制阈值
        # no_save_vis=False: 开启自动保存可视化结果到 out_dir
        result = inferencer(
            img_path,
            out_dir=OUT_DIR,
            pred_score_thr=SCORE_THR,
            no_save_vis=False
        )

        # 5. 解析并过滤结果
        predictions = result['predictions'][0]
        labels = predictions['labels']
        scores = predictions['scores']
        bboxes = predictions['bboxes']

        found_target = False
        for idx, score in enumerate(scores):
            # [核心过滤逻辑] 仅处理大于设定阈值的目标
            if score >= SCORE_THR:
                found_target = True
                label_id = labels[idx]
                # 坐标取整，保留整数像素位
                box = [int(x) for x in bboxes[idx]]

                # 映射类别名称
                name = class_names[label_id] if label_id < len(class_names) else str(label_id)

                print(f"   -> 🎯 目标: {name:<10} | 置信度: {score:.2f} | 坐标: {box}")

        if not found_target:
            print("   (未检测到满足阈值的目标)")

    print("-" * 60)
    print(f"✅ 推理完成！可视化结果已保存至: {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    main()
```

## 二、基于Ultralytics（YOLO）

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # =====================================================
    # 1. 基础配置
    # =====================================================
    MODEL_PATH = 'runs/detect/exp_001/weights/best.pt'
    SOURCE = 'assets/bus.jpg'  # 图片、视频、文件夹、'0'(摄像头)、RTSP流

    # 加载模型
    model = YOLO(MODEL_PATH)

    # =====================================================
    # 2. 推理参数 (按需取消注释)
    # 完整参数文档: https://docs.ultralytics.com/modes/predict/
    # =====================================================
    results = model.predict(
        source=SOURCE,

        # --- [过滤阈值] (决定检测的灵敏度) ---
        conf=0.25,  # 置信度阈值 (低于此值不显示，默认 0.25)
        iou=0.7,  # NMS 阈值 (去除重叠框，默认 0.7)
        # classes = None,      # 过滤器: 仅检测特定类别 (如 [0, 2] 只看人、车)
        # max_det = 300,       # 每张图最大检测数量

        # --- [显示与保存] ---
        save=True,  # 保存预测图片/视频
        show=True,  # 实时弹窗显示结果
        # save_txt = False,    # 保存检测结果为 .txt 文件
        # save_conf = False,   # 保存 txt 时包含置信度
        # save_crop = False,   # 将检测到的物体裁剪并单独保存
        # show_labels = True,  # 图片上显示类别名
        # show_conf = True,    # 图片上显示置信度
        # line_width = None,   # 框的粗细 (None=自动适配)

        # --- [高级设置] ---
        imgsz=640,  # 推理尺寸
        # augment = False,     # TTA 测试时增强 (更准但更慢)
        # visualize = False,   # 可视化特征图 (调试模型用)
        # retina_masks = False,# (分割模型专用) 使用高分辨率掩码
        # stream = False,      # 流式加载 (用于长视频/监控，防内存溢出)
        # vid_stride = 1,      # 视频帧间隔 (每隔 N 帧测一次，加速视频处理)
    )

    print(f"✅ 推理完成，结果已保存")
```