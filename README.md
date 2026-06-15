# H3HOTA Creature Classifier — 训练与检测框架

英雄无敌3:深渊号角（HotA）兵种图像识别模型训练框架。基于 PyTorch + torchvision，支持 CPU/GPU/混合精度训练，专为 Kaggle Notebook 环境优化。

分类模型与锚框检测器联动，实现端到端的战场兵种检测：**锚框生成 → 图像裁剪 → 分类推理 → NMS后处理**。

## 目录

- [快速开始](#快速开始)
- [Kaggle Notebook 训练指南](#kaggle-notebook-训练指南)
- [数据集说明](#数据集说明)
- [数据增强](#数据增强)
- [模型选型](#模型选型)
- [训练参数说明](#训练参数说明)
- [锚框检测系统](#锚框检测系统)
- [常用命令](#常用命令)
- [项目结构](#项目结构)
- [FAQ](#faq)

---

## 快速开始

### 本地 CPU 快速验证（5 epoch）

```bash
python scripts/train.py --data_dir data --model mobilenet_v3_large --epochs 5 --batch_size 4 --device cpu --no_amp
```

### GPU 完整训练

```bash
python scripts/train.py --data_dir data --model mobilenet_v3_large --epochs 50 --batch_size 64
```

### 仅评估已有模型

```bash
python scripts/train.py --evaluate outputs/best_mobilenet_v3_large.pth --data_dir data
```

---

## Kaggle Notebook 训练指南

### 环境说明

| 项目 | 值 |
|------|-----|
| GPU | P100 (16GB 显存) |
| 框架 | PyTorch + torchvision |
| 推荐模型 | **MobileNetV3-Large** (5.5M 参数) |
| 推荐 Batch Size | 64 (约 6GB 显存) |

### Step 1: 上传数据到 Kaggle

将 `training/` 文件夹上传为 Kaggle Dataset，或直接上传到 Notebook 的 input 目录。

### Step 2: Notebook 训练代码

在 Kaggle Notebook 中依次运行:

```python
# ========== Cell 1: 环境检查 ==========
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ========== Cell 2: 安装依赖 ==========
!pip install matplotlib openpyxl -q

# ========== Cell 3: 开始训练 ==========
!python <datasetpath>/heroesagent-training/scripts/train.py \
    --data_dir <datasetpath>/heroesagent-training/data \
    --model mobilenet_v3_large \
    --epochs 10 \
    --batch_size 64 \
    --lr 3e-4 \
    --warmup_epochs 3 \
    --early_stop 10
```

也可以使用 Python 调用方式:

```python
import sys
import argparse
sys.path.insert(0, '<datasetpath>/heroesagent-training/scripts')
from train import train

args = argparse.Namespace(
    model='mobilenet_v3_large',
    data_dir='<datasetpath>/heroesagent-training/data',
    epochs=10, batch_size=64, lr=3e-4,
    weight_decay=1e-4, label_smoothing=0.1,
    clip_grad=1.0, input_size=224,
    seed=42, device='auto', num_workers=2,
    early_stop=10, warmup_epochs=3,
    no_pretrain=False, no_amp=False,
)
history = train(args)
print(f"Best val accuracy: {history['best_val_acc']:.2f}%")
```

### Step 3: 下载训练结果

训练完成后，模型和图表保存在 `outputs/`:

| 文件 | 说明 |
|------|------|
| `outputs/best_{model}.pth` | 最佳模型 checkpoint |
| `outputs/training_curves_{model}.png` | Loss/Accuracy 曲线图 |
| `outputs/confusion_matrix_{model}.png` | 混淆矩阵 |
| `outputs/history_{model}.json` | 训练历史数据 |
| `outputs/eval_{model}.json` | 测试集评估结果 |

打包下载:
```python
import shutil
shutil.make_archive("model_output", "zip", "outputs")
```

### Kaggle P100 显存参考

| 模型 | 推荐 Batch | 预计显存 |
|------|-----------|---------|
| **mobilenet_v3_large** | **64** | **~6GB** |
| mobilenet_v3_small | 128 | ~4GB |
| efficientnet_b0 | 64 | ~8GB |
| efficientnet_b1 | 48 | ~12GB |
| resnet18 | 64 | ~6GB |
| resnet34 | 48 | ~10GB |
| resnet50 | 32 | ~14GB |
| shufflenet_v2_x1_0 | 128 | ~4GB |

---

## 数据集说明

### 当前数据（29类兵种，1845张图片）

图片按兵种编号分文件夹组织: `data/images/{编号}/{编号}_{序号}.png`

- **训练数据**: 29类兵种（编号0-28），约1845张真实游戏截图
- **全部类别**: 模型架构支持191类（creature_index.xlsx中的全部189兵种 + 障碍物 + 空地），当前29类有训练数据
- **分割策略**: 每次训练动态随机分割，训练:验证 = 5:1，每类保证至少1张验证图（仅1张图时全进训练集）
- **不使用预分配**: 训练/验证集在每次训练时重新随机生成，不使用固定CSV分割
- **全部数据用于训练/验证**: 不预留测试集，最大化训练数据利用率
- 图片格式: PNG，统一命名 `{兵种编号}_{序号}.png`
- **当前各兵种图片数量**: 每类约62-79张

| 数据集 | 数量(约) | 比例 |
|--------|----------|------|
| 训练集 | ~1523 | ~83% |
| 验证集 | ~322 | ~17% |
| 测试集 | 无 | 全部数据用于训练/验证 |

### 当前训练效果（MobileNetV3-Large）

| 指标 | 值 |
|------|-----|
| 测试准确率 | 84.93% |
| 最佳验证准确率 | 90.81% |
| 训练设备 | CPU (本地验证) |
| Epochs | 1 (仅验证流程) |

### 添加新数据

将新图片按兵种编号放入对应子文件夹:

```
data/images/
├── 0/   ← 兵种0的图片 (已存在79张)
├── 1/
├── ...
└── 29/  ← 新增兵种29的图片放这里
```

然后重新生成标注:

```bash
python scripts/generate_annotations.py
```

### 图片命名规范

使用 `rename_images.py` 将非规范命名的图片（如QQ截图）重命名为标准格式:

```bash
python scripts/rename_images.py
```

规范格式: `{兵种编号}_{序号}.png`，如 `0_0.png`

---

## 数据增强

训练时通过 `dataset.py` 中 `get_default_transforms` 自动应用在线增强:

- `RandomResizedCrop` (scale 0.7-1.0)
- `RandomHorizontalFlip` (50%，不包含垂直翻转 — 兵种不会倒立)
- `ColorJitter` (亮度/对比度/饱和度 轻微抖动)
- `RandomErasing` (随机遮挡, 20%)

验证/测试时仅做 `Resize + Normalize`（ImageNet标准均值/方差）。

---

## 模型选型

所有模型基于 torchvision，使用 ImageNet 预训练权重。

| 模型 | 参数量 | Top-1 | 推理(CPU) | 推荐场景 |
|------|--------|-------|----------|---------|
| **mobilenet_v3_large** | **5.5M** | **75.2%** | **~18ms** | **首选**: 精度/效率最佳平衡 |
| efficientnet_b0 | 5.3M | 77.1% | ~22ms | 精度略高，显存需求大 |
| mobilenet_v3_small | 2.5M | 67.4% | ~10ms | 极致轻量，CPU优先 |
| shufflenet_v2_x1_0 | 2.3M | 69.4% | ~8ms | ARM/移动端 |
| resnet18 | 11.7M | 69.8% | ~28ms | 训练最稳定 |
| resnet34 | 21.8M | 73.3% | ~45ms | 精度略高 |

> **默认模型**: `mobilenet_v3_large` — 在P100上约15-20分钟完成50 epoch。
> 在29类兵种上测试准确率84.93%，验证准确率90.81%。

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `mobilenet_v3_large` | torchvision 模型名 |
| `--data_dir` | `training/data` | 数据目录（含 images/ 和 annotations/） |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小（GPU: 64, CPU: 4-16） |
| `--lr` | 3e-4 | 初始学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 (L2正则) |
| `--label_smoothing` | 0.1 | 标签平滑 (减少过拟合) |
| `--clip_grad` | 1.0 | 梯度裁剪阈值 |
| `--input_size` | 224 | 模型输入尺寸 |
| `--device` | auto | auto/cpu/cuda |
| `--num_workers` | 2 | DataLoader 工作进程数 |
| `--early_stop` | 10 | Early Stopping 耐心值 (0=关闭) |
| `--warmup_epochs` | 3 | 学习率 warmup 轮数 |
| `--no_pretrain` | False | 不使用预训练权重 |
| `--no_amp` | False | 禁用混合精度训练 |

学习率调度: 线性Warmup + Cosine退火。

---

## 锚框检测系统

### 概述

分类模型训练完成后，通过锚框检测系统实现端到端的战场兵种检测。检测流程:

```
战场截图 → 锚框生成 → 裁剪锚框区域 → 分类模型推理 → NMS后处理 → 检测结果
```

### 锚框方案 V2

基于战场几何结构（15×11六边形网格，1062×664分辨率）生成候选锚框。

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 列间距 | 44px | 六边形列间距 |
| 行间距 | 42px | 六边形行间距 |
| 偶数行col0中心X | 233 | even-r offset 右偏 |
| 奇数行col0中心X | 211 | even-r offset 左对齐 |
| Row 0 中心Y | 125 | 首行基准Y |

**单格兵锚框**:

| 参数 | 值 |
|------|-----|
| 基础宽度 | 44px |
| 基础高度 | 44px |
| 高度扩展 | +44px (向上1格，总高=88px) |
| 宽度扩展 | +22px (±0.5格，wide变体) |
| 垂直偏移 | -5px |

**双格兵锚框**:

| 参数 | 值 |
|------|-----|
| 基础宽度 | 88px (2列间距) |
| 基础高度 | 44px |
| 中心位置 | 同行相邻hex中心的中点 |

**2种水平变体**:

| 变体 | 单格宽度 | 双格宽度 | 用途 |
|------|---------|---------|------|
| `center` | 44px | 88px | 精确对中，基础宽度 |
| `wide` | 66px | 110px | 左右对称扩大，覆盖精灵偏移/数字框 |

**锚框统计**:

| 类型 | 数量 | 计算 |
|------|------|------|
| 单格 | 330 | 165 hex × 2 variants |
| 双格 | 308 | 154 hex-pairs × 2 variants |
| **总计** | **638** | |

### NMS 后处理（V3 集群策略）

检测结果经过两阶段NMS处理:

**阶段1: Hex级冲突解决**
- 同一六边形位置只保留置信度最高的检测
- 单格兵获得置信度偏置（`single_tile_bias=1.10`），避免被双格锚框误判
- 双格锚框占用2个hex，先到先得

**阶段2: 同类邻近压制（V3 集群策略）**
- 按Chebyshev距离≤d（默认d=1）构建同类检测的连通分量（Union-Find）
- 根据集群内高置信度（≥0.85）检测数量决定策略:
  - **≥2个高置信度**: 全部保留（可能是不同的同类单位），低置信度仅在不靠近高置信度时保留
  - **1个高置信度**: 保留它，压制邻近低置信度（视为同一单位的重复检测）
  - **0个高置信度**: 保留置信度最高的1个（至少保留1个，避免漏检）

### 检测器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `confidence_threshold` | 0.70 | 最低置信度阈值 |
| `same_class_distance` | 1 | 同类邻近压制距离（Chebyshev，0=关闭） |
| `same_class_anchor_threshold` | 0.85 | 高置信度判定阈值 |
| `single_tile_bias` | 1.10 | 单格兵置信度偏置系数 |

### 检测器使用方法

```python
from backend.app.battlefield.battlefield_detector import BattlefieldDetector

# 初始化检测器
detector = BattlefieldDetector(
    model_path="training/outputs/best_mobilenet_v3_large.pth",
    confidence_threshold=0.70,
    same_class_distance=1,
)

# 检测战场截图
result = detector.detect("screenshot.png")

# 查看结果
for det in result.detections:
    print(f"({det.anchor.col},{det.anchor.row}): "
          f"{det.class_name} ({det.confidence:.3f}) "
          f"[{det.anchor.anchor_type}]")

# 获取hex坐标映射
bf_map = detector.detect_to_battlefield_map("screenshot.png")

# 生成叠加图（调试用）
from backend.app.battlefield.battlefield_detector import draw_detection_overlay
draw_detection_overlay("screenshot.png", result, "overlay.png")
```

### CPU 延迟估算（638锚框）

| 模型 | 批处理推理 | 预处理 | 总计 |
|------|-----------|--------|------|
| mobilenet_v3_large | ~2.3s | ~1.9s | **~4.2s** |
| mobilenet_v3_small | ~1.3s | ~1.9s | **~3.2s** |
| shufflenet_v2_x1_0 | ~1.0s | ~1.9s | **~2.9s** |

---

## 常用命令

```bash
# 默认训练 (MobileNetV3-Large, 29类, 动态5:1分割)
python scripts/train.py --data_dir data --model mobilenet_v3_large --epochs 50 --batch_size 64

# CPU 快速验证 (5 epoch, 固定随机种子可复现)
python scripts/train.py --data_dir data --model mobilenet_v3_large --epochs 5 --batch_size 4 --device cpu --no_amp --seed 42

# 轻量模型 (CPU推理优先)
python scripts/train.py --data_dir data --model mobilenet_v3_small --epochs 50 --batch_size 128

# 生成完整标注文件
python scripts/generate_annotations.py

# 规范命名图片
python scripts/rename_images.py

# 评估模型 (使用动态随机验证集)
python scripts/train.py --evaluate outputs/best_mobilenet_v3_large.pth --data_dir data

# 导出 ONNX 模型
python scripts/train.py --export_onnx outputs/best_mobilenet_v3_large.pth

# 锚框生成器测试
python backend/app/battlefield/test_anchor_detector.py

# 检测器测试（无模型，验证框架）
python backend/app/battlefield/battlefield_detector.py

# 检测器测试（加载模型）
python backend/app/battlefield/battlefield_detector.py \
    --image screenshot.png \
    --model training/outputs/best_mobilenet_v3_large.pth \
    --output result_overlay.png

# 锚框导出为JSON
python backend/app/battlefield/battlefield_detector.py \
    --export_anchors anchors.json

# 打印延迟估算
python backend/app/battlefield/battlefield_detector.py --latency
```

---

## 项目结构

```
training/
├── README.md                        # 本文件
├── .gitignore                       # Git 忽略规则
├── data/
│   ├── images/                      # 训练图片（按兵种分文件夹，29类）
│   │   ├── 0/  (79张)
│   │   ├── 1/  (64张)
│   │   ├── ...
│   │   └── 28/ (63张)
│   └── annotations/                 # 标注文件
│       ├── full.csv                 # 完整标注 (动态分割的输入)
│       ├── train.csv                # 最近一次训练的train split
│       ├── val.csv                  # 最近一次训练的val split
│       ├── summary.json             # 数据集摘要
│       └── creature_index.xlsx      # 全兵种索引 (191类)
├── scripts/
│   ├── dataset.py                   # PyTorch Dataset (CSV/内存双模式)
│   ├── data_processor.py            # 数据发现 + 动态随机分割 (5:1)
│   ├── model_factory.py             # 模型构建 (build_model)
│   ├── train_loop.py                # 训练/验证循环 (train_epoch, validate_epoch)
│   ├── train.py                     # 训练主脚本 (CLI+编排+评估+导出)
│   ├── generate_annotations.py      # 完整标注生成
│   └── rename_images.py             # 图片重命名 (规范格式)
└── outputs/                         # 训练输出 (不上传git)
    ├── best_mobilenet_v3_large.pth   # 最佳模型 (~51MB)
    ├── training_curves_*.png         # 训练曲线图
    ├── history_*.json                # 训练历史
    └── eval_*.json                   # 评估结果

backend/app/battlefield/             # 战场检测模块 (与training联动)
├── anchor_generator.py              # 锚框生成器 (638个锚框)
├── battlefield_detector.py          # 战场检测器 (分类+NMS)
└── test_anchor_detector.py          # 检测系统测试
```

---

## FAQ

### Q1: CUDA out of memory
减小 `--batch_size`，例如从 64 降到 32 或 16。P100 (16GB) 上 mobilenet_v3_large 可安全使用 batch_size=64（约6GB显存）。

### Q2: 数据不够，每类只有几十张图？
- 训练时 `--label_smoothing 0.1` 减少过拟合
- 使用较小模型 (mobilenet_v3_large 或 mobilenet_v3_small)
- 当前29类兵种每类62-79张，总计1845张，已验证可达到84.93%测试准确率

### Q3: 如何继续中断的训练？
目前版本暂不支持断点续训。建议在 Kaggle 上开启 "Save Output" 保存最佳模型。

### Q4: CPU 推理如何优化？
1. 训练时选轻量模型: `--model mobilenet_v3_small`
2. 导出 ONNX 后用 ONNX Runtime 推理
3. 导出: `python scripts/train.py --export_onnx outputs/best_model.pth`
4. 检测时使用较小输入尺寸或减少锚框变体

### Q5: 新增兵种类别后如何操作？
1. 将新兵种图片按编号放入 `data/images/{新编号}/`
2. 运行: `python scripts/generate_annotations.py`
3. 训练时模型自动适配 creature_index.xlsx 中的类别数

### Q6: 验证集准确率不提升？
- 检查图片是否按兵种正确分文件夹
- 尝试减小学习率: `--lr 1e-4`
- 增加 warmup: `--warmup_epochs 5`
- 使用离线数据增强扩充训练集

### Q7: 检测器输出太多/太少检测框？
- **太多**: 提高 `confidence_threshold`（如 0.80），开启 `same_class_distance=1`
- **太少**: 降低 `confidence_threshold`（如 0.50），调整 `single_tile_bias`
- **双格兵被识别为单格**: 降低 `single_tile_bias`（如 1.05）
- **单格兵被双格误判**: 提高 `single_tile_bias`（如 1.15）

### Q8: 锚框检测和分类模型的关系？
分类模型（如 MobileNetV3-Large）是检测系统的核心推理引擎。检测器生成锚框 → 裁剪每个锚框区域 → 送入分类模型识别 → NMS后处理去重。分类模型的准确率直接决定检测质量。

### Q9: 训练/验证集是怎么分割的？
每次训练时动态随机分割，不使用预分配CSV：
- 比例: 训练:验证 = 5:1（约83%训练，17%验证）
- 每类保证至少1张进验证集（仅1张图时全进训练集）
- 如需固定分割复现结果，使用 `--seed 42` 参数
- 分割结果保存在 `data/annotations/train.csv` 和 `val.csv`（每次训练覆盖）
