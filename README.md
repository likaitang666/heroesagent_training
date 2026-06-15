# H3HOTA Creature Classifier — 训练框架

英雄无敌3:深渊号角（HotA）兵种图像识别模型训练框架。基于 PyTorch + torchvision，支持 CPU/GPU/混合精度训练，专为 Kaggle Notebook 环境优化。

## 目录

- [快速开始](#快速开始)
- [Kaggle Notebook 训练指南](#kaggle-notebook-训练指南)
- [数据集说明](#数据集说明)
- [数据增强](#数据增强)
- [模型选型](#模型选型)
- [训练参数说明](#训练参数说明)
- [常用命令](#常用命令)
- [FAQ](#faq)

---

## 快速开始

### 本地 CPU 快速验证（5 epoch）

```bash
python scripts/train.py --data_dir data --model efficientnet_b0 --epochs 5 --batch_size 4 --device cpu --no_amp
```

### GPU 完整训练

```bash
python scripts/train.py --data_dir data --model efficientnet_b0 --epochs 50 --batch_size 64
```

### 仅评估已有模型

```bash
python scripts/train.py --evaluate outputs/best_efficientnet_b0.pth --data_dir data
```

---

## Kaggle Notebook 训练指南

### 环境说明

| 项目 | 值 |
|------|-----|
| GPU | P100 (16GB 显存) |
| 框架 | PyTorch + torchvision |
| 推荐模型 | EfficientNet-B0 (5.3M 参数) |
| 推荐 Batch Size | 64 (约 8GB 显存) |

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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ========== Cell 2: 安装依赖 ==========
!pip install matplotlib -q

# ========== Cell 3: 开始训练 ==========
# 将 data_dir 改为你的数据集路径
# Dataset 内路径格式: /kaggle/input/your-dataset/data/
!python scripts/train.py \
    --data_dir /kaggle/input/herosagent-training/data \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-4 \
    --warmup_epochs 3 \
    --early_stop 10
```

也可以使用 Python 调用方式:

```python
import sys
sys.path.insert(0, '/kaggle/working/training')
from training.scripts.train import train
import argparse

args = argparse.Namespace(
    model='efficientnet_b0',
    data_dir='/kaggle/input/herosagent-training/data',
    epochs=50, batch_size=64, lr=3e-4,
    weight_decay=1e-4, label_smoothing=0.1,
    clip_grad=1.0, input_size=224,
    device='auto', num_workers=2,
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
| efficientnet_b0 | 64 | ~8GB |
| efficientnet_b1 | 48 | ~12GB |
| mobilenet_v3_large | 96 | ~6GB |
| resnet18 | 64 | ~6GB |
| resnet34 | 48 | ~10GB |
| resnet50 | 32 | ~14GB |
| shufflenet_v2_x1_0 | 128 | ~4GB |

---

## 数据集说明

### 当前数据 (191类：189兵种 + 障碍物 + 空地)

图片按兵种编号分文件夹组织: `data/images/{编号}/{编号}_{序号}.png`

- **训练数据**: 10类兵种有真实截图 (Factory阵营兵种，编号157-166)，约649张
- **合成数据**: 障碍物(189)和空地(190)由 `generate_background_data.py` 生成
- **全部类别**: 模型训练191类，包含全部189兵种 + 障碍物 + 空地
- 分割比例: 70/15/15（分层分割，每类独立采样）
- 图片格式: PNG，统一命名 `{兵种编号}_{序号}.png`

### 添加新数据

将新图片按兵种编号放入对应子文件夹:

```
data/images/
├── 0/   ← 兵种0的图片 (已存在79张)
├── 1/
├── ...
└── 10/  ← 新增兵种10的图片放这里
```

然后重新生成标注:

```bash
python scripts/generate_annotations.py
```

---

## 数据增强

### 数据增强

训练时通过 `dataset.py` 中 `get_default_transforms` 自动应用在线增强:
- `RandomResizedCrop` (scale 0.7-1.0)
- `RandomHorizontalFlip` (50%，不包含垂直翻转)
- `ColorJitter` (亮度/对比度/饱和度)
- `RandomErasing` (随机遮挡, 20%)

如需离线增强，可在 Kaggle 上使用 albumentations 等库进行。

### 在线增强（训练时自动应用）

`dataset.py` 中 `get_default_transforms` 在训练时自动应用:
- `RandomResizedCrop` (scale 0.7-1.0)
- `RandomHorizontalFlip` (50%，不包含垂直翻转)
- `ColorJitter` (亮度/对比度/饱和度)
- `RandomErasing` (随机遮挡, 20%)

---

## 模型选型

所有模型基于 torchvision，使用 ImageNet 预训练权重。

| 模型 | 参数量 | Top-1 | 推理(CPU) | 推荐场景 |
|------|--------|-------|----------|---------|
| **efficientnet_b0** | 5.3M | 77.1% | ~22ms | **首选**: 精度/效率最佳平衡 |
| mobilenet_v3_large | 5.5M | 75.2% | ~18ms | CPU推理优先 |
| mobilenet_v3_small | 2.5M | 67.4% | ~10ms | 极致轻量 |
| resnet18 | 11.7M | 69.8% | ~28ms | 训练最稳定 |
| resnet34 | 21.8M | 73.3% | ~45ms | 精度略高 |
| shufflenet_v2_x1_0 | 2.3M | 69.4% | ~8ms | ARM/移动端 |

> 首选 `efficientnet_b0`，P100 上约 15-20 分钟完成 50 epoch。

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | efficientnet_b0 | torchvision 模型名 |
| `--data_dir` | training/data | 数据目录（含 images/ 和 annotations/） |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小（GPU: 64, CPU: 4-16） |
| `--lr` | 3e-4 | 初始学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 (L2正则) |
| `--label_smoothing` | 0.1 | 标签平滑 |
| `--clip_grad` | 1.0 | 梯度裁剪阈值 |
| `--input_size` | 224 | 模型输入尺寸 |
| `--device` | auto | auto/cpu/cuda |
| `--num_workers` | 2 | DataLoader 工作进程数 |
| `--early_stop` | 10 | Early Stopping 耐心值 (0=关闭) |
| `--warmup_epochs` | 3 | 学习率 warmup 轮数 |
| `--no_pretrain` | False | 不使用预训练权重 |
| `--no_amp` | False | 禁用混合精度训练 |

---

## 常用命令

```bash
# 首次训练 (191类，含障碍物和空地)
python scripts/train.py --data_dir data --model efficientnet_b0 --epochs 5 --batch_size 4 --device cpu --no_amp

# CPU 快速验证 (5 epoch)
python scripts/train.py --data_dir data --model efficientnet_b0 --epochs 5 --batch_size 4 --device cpu --no_amp

# 轻量模型
python scripts/train.py --data_dir data --model mobilenet_v3_large --epochs 50 --batch_size 96

# 生成标注文件
python scripts/generate_annotations.py

# 生成障碍物/空地背景数据
python scripts/generate_background_data.py

# 评估模型
python scripts/train.py --evaluate outputs/best_efficientnet_b0.pth --data_dir data

# 导出 ONNX 模型
python scripts/train.py --export_onnx outputs/best_efficientnet_b0.pth
```

---

## 项目结构

```
training/
├── README.md                        # 本文件
├── .gitignore                       # Git 忽略规则
├── data/
│   ├── images/                      # 训练图片（按兵种分文件夹）
│   │   ├── 157/ (Kobold)
│   │   ├── 158/ (Kobold Foreman)
│   │   ├── ...
│   │   ├── 189/ (obstacle 合成)
│   │   └── 190/ (empty 合成)
│   └── annotations/                 # 标注文件
│       ├── train.csv                # 训练集
│       ├── val.csv                  # 验证集
│       ├── test.csv                 # 测试集
│       ├── full.csv                 # 全部标注
│       └── summary.json             # 数据集摘要
├── scripts/
│   ├── dataset.py                   # PyTorch Dataset (CPU/GPU)
│   ├── train.py                     # 训练主脚本
│   ├── generate_annotations.py      # 标注生成
│   └── generate_background_data.py  # 障碍物/空地合成数据
└── outputs/                         # 训练输出 (不上传git)
```

---

## FAQ

### Q1: CUDA out of memory
减小 `--batch_size`，例如从 64 降到 32 或 16。P100 (16GB) 上 efficientnet_b0 可安全使用 batch_size=64。

### Q2: 数据不够，每类只有几十张图？
- 训练时 `--label_smoothing 0.1` 减少过拟合
- 使用较小模型 (efficientnet_b0 或 mobilenet_v3_large)
- 在Kaggle上可使用 albumentations 等库做离线增强

### Q3: 如何继续中断的训练？
目前版本暂不支持断点续训。建议在 Kaggle 上开启 "Save Output" 保存最佳模型。

### Q4: CPU 推理如何优化？
1. 训练时选轻量模型: `--model mobilenet_v3_large`
2. 导出 ONNX 后用 ONNX Runtime 推理
3. 导出: `python scripts/train.py --export_onnx outputs/best_model.pth`

### Q5: 新增兵种类别后如何操作？
1. 将新兵种图片按编号放入 `data/images/{新编号}/`
2. 运行: `python scripts/generate_annotations.py`
3. 训练时模型自动适配新的类别数

### Q6: 验证集准确率不提升？
- 检查图片是否按兵种正确分文件夹
- 尝试减小学习率: `--lr 1e-4`
- 增加 warmup: `--warmup_epochs 5`
- 使用离线数据增强扩充训练集
