# H3HOTA Creature Classifier — 训练框架

英雄无敌3:深渊号角（HotA）兵种图像识别模型训练框架。基于 PyTorch + torchvision，支持 CPU/GPU/混合精度训练。

## 目录

- [当前测试数据](#当前测试数据)
- [快速开始 (首次训练测试)](#快速开始-首次训练测试)
- [Kaggle 训练指南](#kaggle-训练指南)
- [本地 CPU 测试](#本地-cpu-测试)
- [数据准备](#数据准备)
- [模型选型](#模型选型)
- [训练输出说明](#训练输出说明)
- [添加更多兵种](#添加更多兵种)
- [常见问题](#常见问题)

---

## 当前测试数据

**首次训练测试：仅 Castle 阵营前 10 种兵（label 0-9）**

| Label | 兵种英文名 | 中文名 | 等级 | 图片数 |
|-------|-----------|--------|------|--------|
| 0 | Pikeman | 枪兵 | 1 | 20 |
| 1 | Halberdier | 戟兵 | 1(upg) | 10 |
| 2 | Archer | 弩箭手 | 2 | 10 |
| 3 | Marksman | 重弩手 | 2(upg) | 10 |
| 4 | Griffin | 狮鹫 | 3 | 10 |
| 5 | Royal Griffin | 皇家狮鹫 | 3(upg) | 10 |
| 6 | Swordsman | 剑士 | 4 | 10 |
| 7 | Crusader | 十字军 | 4(upg) | 10 |
| 8 | Monk | 僧侣 | 5 | 10 |
| 9 | Zealot | 狂信徒 | 5(upg) | 10 |

- **总图片数**: 110 张 (含数据增强变体)
- **训练集**: 77 张 | **验证集**: 12 张 | **测试集**: 21 张
- **数据目录**: `data/test_run_0_9/`

---

## 快速开始 (首次训练测试)

### 1. 安装依赖

```bash
pip install torch torchvision pillow matplotlib scikit-learn
```

### 2. 运行训练

```bash
# 使用 EfficientNet-B0 (推荐), 50 epochs
python scripts/train.py --data_dir data/test_run_0_9 --model efficientnet_b0 --epochs 50 --batch_size 16

# 使用 MobileNetV3-Large (更轻量)
python scripts/train.py --data_dir data/test_run_0_9 --model mobilenet_v3_large --epochs 50 --batch_size 32

# CPU 快速测试 (仅验证流程, 5 epochs)
python scripts/train.py --data_dir data/test_run_0_9 --model efficientnet_b0 --epochs 5 --batch_size 8 --device cpu
```

### 3. 查看结果

训练完成后在 `outputs/` 目录查看:
- `training_curves_{model}.png` — loss/accuracy 曲线图
- `history_{model}.json` — 训练历史数据
- `best_{model}.pth` — 最佳模型权重
- `confusion_matrix_{model}.png` — 混淆矩阵
- `eval_{model}.json` — 测试集评估结果

---

## Kaggle 训练指南

### 环境说明

| 项目 | 值 |
|------|-----|
| GPU | P100 (16GB 显存可用) |
| 框架 | PyTorch + torchvision |
| 推荐模型 | EfficientNet-B0 (5.3M 参数) |
| 推荐 Batch Size | 64 (P100 约4GB显存) |

### Step 1: 上传数据到 Kaggle

1. 将整个 `training/` 文件夹打包为 ZIP
2. 在 Kaggle Notebook 中上传为 Dataset 或直接上传文件
3. 解压到 `/kaggle/working/`

### Step 2: 安装依赖

在 Kaggle Notebook 的第一个 cell:

```python
!pip install torch torchvision pillow matplotlib scikit-learn -q
```

### Step 3: 运行训练

```python
import sys
sys.path.insert(0, '/kaggle/working/training')

# 方式一: 命令行方式
!python /kaggle/working/training/scripts/train.py \
    --data_dir /kaggle/working/training/data/test_run_0_9 \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-4

# 方式二: Python 调用
from training.scripts.train import train
import argparse

args = argparse.Namespace(
    model='efficientnet_b0',
    data_dir='/kaggle/working/training/data/test_run_0_9',
    epochs=50,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.1,
    clip_grad=1.0,
    input_size=224,
    device='auto',
    num_workers=2,
    early_stop=10,
    warmup_epochs=3,
    no_pretrain=False,
    no_amp=False,
)
history = train(args)
print(f"Best val accuracy: {history['best_val_acc']:.2f}%")
```

### Step 4: 下载模型

```python
# 查看生成的输出文件
import os
for f in sorted(os.listdir('/kaggle/working/training/outputs/')):
    print(f)

# Kaggle 会自动保存 outputs/ 目录下的文件
# 训练完成后可在 Notebook 右侧 "Output" 标签页下载
```

### Kaggle 显存建议

| 模型 | 推荐 Batch | 预计显存 |
|------|-----------|---------|
| EfficientNet-B0 | 64-128 | 4-7 GB |
| MobileNetV3-Large | 64-128 | 3-6 GB |
| MobileNetV3-Small | 128-256 | 2-4 GB |
| ResNet-18 | 32-64 | 5-9 GB |

P100 有 16GB 显存，以上配置均足够。

---

## 本地 CPU 测试

用于验证代码流程是否正常（不需要 GPU）:

```bash
# 仅 5 epochs, 小 batch, 快速验证
python scripts/train.py \
    --data_dir data/test_run_0_9 \
    --model efficientnet_b0 \
    --epochs 5 \
    --batch_size 4 \
    --device cpu \
    --no_amp

# 预期: 5 分钟内完成, 确认无报错即可
```

---

## 数据准备

### 当前训练数据

首次测试使用 `data/test_run_0_9/`:
```
data/test_run_0_9/
├── images/              # 110 张图片 (10类 × 10-20张)
│   ├── 0_0.png          # Pikeman 原始图
│   ├── 0_0_aug01.png    # Pikeman 增强变体
│   └── ...
└── annotations/         # 标注文件
    ├── train.csv        # 训练集
    ├── val.csv          # 验证集
    ├── test.csv         # 测试集
    ├── full.csv         # 完整标注
    └── summary.json     # 数据集摘要
```

### 图片命名规范

`{label_index}_{序号}.png`

- `label_index`: 兵种标签索引 (0=Pikeman, 1=Halberdier, ...)
- `序号`: 该兵种的第几张图片 (0, 1, 2, ...)
- 增强变体自动添加 `_aug01`, `_aug02` 等后缀

### 添加更多图片

```bash
# 1. 将新图片放入对应目录 (按命名规范)
# 2. 运行数据增强扩充数据集
python scripts/augment_images.py \
    --input data/test_run_0_9/images \
    --output data/test_run_0_9/images \
    --variations 9

# 3. 重新生成标注
# (手动运行 generate_annotations.py 或重新准备数据)
```

### 从截图准备新兵种

参见 `截图要求.txt` (项目根目录) 了解截图规范。

截图预处理脚本:
```bash
python scripts/preprocess_screenshots.py --help
```

---

## 模型选型

详见 [MODEL_RECOMMENDATION.md](MODEL_RECOMMENDATION.md)，7 个候选模型的完整对比。

| 模型 | 参数量 | ImageNet Top-1 | CPU推理 | 推荐场景 |
|------|--------|---------------|---------|---------|
| **EfficientNet-B0** | 5.3M | 77.1% | ~22ms | **首选**: 精度/效率最佳平衡 |
| MobileNetV3-Large | 5.5M | 75.2% | ~18ms | CPU推理优先 |
| MobileNetV3-Small | 2.5M | 67.4% | ~10ms | 极致轻量 |
| ResNet-18 | 11.7M | 69.8% | ~28ms | 训练最稳定 |
| ShuffleNetV2 x1.0 | 2.3M | 69.4% | ~8ms | ARM/移动端 |
| MobileViT-XXS | 1.3M | 69.0% | ~15ms | CNN+Transformer混合 |
| DeiT-Tiny | 5.7M | 72.2% | ~35ms | 纯Transformer |

### 首次测试推荐

对于 10 类兵种首次测试，推荐:

1. **EfficientNet-B0** — 5.3M 参数，预期 85%+ 准确率
2. **MobileNetV3-Large** — 如果 CPU 推理速度优先

---

## 训练输出说明

训练完成后 `outputs/` 目录包含:

| 文件 | 说明 |
|------|------|
| `best_{model}.pth` | 最佳模型 checkpoint (含权重+优化器状态+训练历史) |
| `history_{model}.json` | 训练历史 (每 epoch 的 loss/acc/lr) |
| `training_curves_{model}.png` | loss-epoch + accuracy-epoch 双图 |
| `confusion_matrix_{model}.png` | 测试集混淆矩阵 |
| `eval_{model}.json` | 测试集评估指标 |

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | efficientnet_b0 | torchvision 模型名 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 (GPU 64, CPU 8) |
| `--lr` | 3e-4 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--label_smoothing` | 0.1 | 标签平滑 |
| `--clip_grad` | 1.0 | 梯度裁剪 |
| `--input_size` | 224 | 输入图片尺寸 |
| `--device` | auto | auto/cpu/cuda |
| `--data_dir` | data | 数据目录路径 |
| `--early_stop` | 10 | Early stopping patience (0=关闭) |
| `--warmup_epochs` | 3 | 学习率 warmup 轮数 |
| `--no_pretrain` | False | 不使用 ImageNet 预训练权重 |
| `--no_amp` | False | 禁用混合精度训练 |

---

## 添加更多兵种

当需要扩展训练到更多兵种时:

### 1. 添加图片

将新兵种图片放入 `data/test_run_0_9/images/` (或新数据目录)，命名遵循 `{label_index}_{序号}.png`。

### 2. 重新生成标注

使用 Python 脚本重新生成标注文件:

```python
import json, csv, random
from pathlib import Path
from collections import defaultdict

# 配置
IMG_DIR = Path('data/test_run_0_9/images')
ANN_DIR = Path('data/test_run_0_9/annotations')
LABEL_FILE = Path('gamedata/creature_labels.json')
TARGET_LABELS = range(10)  # 修改为需要的 label 范围

# ... 运行标注生成逻辑
```

### 3. 数据增强

```bash
python scripts/augment_images.py --input data/test_run_0_9/images --variations 9
```

### 4. 重新训练

```bash
python scripts/train.py --data_dir data/test_run_0_9 --model efficientnet_b0 --epochs 50
```

---

## 常见问题

### Q: ImportError: No module named 'torch'
安装 PyTorch: `pip install torch torchvision`

### Q: CUDA out of memory
减小 batch_size: `--batch_size 16` 或 `--batch_size 8`

### Q: 训练集太小，精度很低
每个兵种建议至少 10 张图片。使用数据增强扩充:
```bash
python scripts/augment_images.py --variations 19
```

### Q: 如何从已有模型继续训练？
在 train.py 中加载 checkpoint:
```python
checkpoint = torch.load('outputs/best_efficientnet_b0.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# 然后正常调用 train()
```

### Q: 如何导出 ONNX 用于推理部署？
```bash
python scripts/train.py --export_onnx outputs/best_efficientnet_b0.pth
```

---

## 目录结构

```
training/
├── README.md                          # 本文件
├── BATTLEFIELD_RECOGNITION.md         # 战场三模型流水线方案
├── MODEL_RECOMMENDATION.md            # 7模型选型报告
├── QUANTIZATION_PLAN.md               # 量化方案预备
├── IMAGE_ADDITION_GUIDE.md            # 图片添加指南
├── BULWARK_IMAGE_SOURCES.md           # Bulwark 图片来源
├── data/
│   └── test_run_0_9/                  # 首次测试数据 (10类)
│       ├── images/                    # 110张训练图片
│       └── annotations/              # 标注CSV文件
├── scripts/
│   ├── dataset.py                     # PyTorch Dataset (CPU/GPU兼容)
│   ├── train.py                       # 训练主脚本 (AMP/Warmup/图表/ONNX)
│   ├── augment_images.py              # 数据增强
│   ├── generate_annotations.py        # 全量标注生成 (189类)
│   ├── preprocess_screenshots.py      # 截图预处理 (~22k张管理)
│   ├── create_xlsx_mapping.py         # XLSX映射表
│   ├── rename_images.py               # 批量重命名
│   ├── remap_labels.py                # 标签重映射
│   └── download_bulwark_images.py     # Bulwark图片下载
└── outputs/                           # 训练输出 (模型/图表/评估)
```
