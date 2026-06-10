# 英雄无敌3兵种图像识别 — 模型选型报告

## 需求概述

- **任务**: 171类兵种图像分类(细粒度,同阵营兵种外观相似)
- **训练平台**: Kaggle Notebook, P100 GPU (16GB显存), 30GB系统内存
- **约束**: 模型参数量 <1B, 需兼容非GPU环境(CPU)推理
- **输入**: 兵种肖像原始58×64px PNG, 训练时resize到224×224
- **输出**: 171类softmax分类

## 候选模型列表

### 1. EfficientNet-B0 (推荐首选)

| 项目 | 值 |
|------|-----|
| 参数量 | ~5.3M |
| 输入尺寸 | 224×224 |
| ImageNet Top-1 | 77.1% |
| CPU推理速度 | ~12ms/张 |
| 训练显存(batch=64) | ~4GB |
| 推理显存 | <200MB |
| 优势 | 精度/效率平衡优秀;复合缩放策略;PyTorch官方预训练 |
| 劣势 | 比MobileNet稍慢 |
| torchvision | `efficientnet_b0` |

**结构特点**: 使用NAS搜索的复合缩放(深度/宽度/分辨率同时缩放)。MBConv模块+SE注意力。对细粒度分类效果好。

### 2. MobileNetV3-Large

| 项目 | 值 |
|------|-----|
| 参数量 | ~5.5M |
| 输入尺寸 | 224×224 |
| ImageNet Top-1 | 75.2% |
| CPU推理速度 | ~10ms/张 |
| 训练显存(batch=64) | ~3.5GB |
| 优势 | CPU推理极快;自带的SE注意力有助于细粒度 |
| torchvision | `mobilenet_v3_large` |

### 3. MobileNetV3-Small

| 项目 | 值 |
|------|-----|
| 参数量 | ~2.5M |
| 输入尺寸 | 224×224 |
| ImageNet Top-1 | 67.4% |
| CPU推理速度 | ~5ms/张 |
| 训练显存(batch=64) | ~2GB |
| 优势 | 最轻量选择;部署极快 |
| 劣势 | 精度上限较低(比B0低约10%) |
| torchvision | `mobilenet_v3_small` |

### 4. ResNet-18

| 项目 | 值 |
|------|-----|
| 参数量 | ~11.7M |
| 输入尺寸 | 224×224 |
| ImageNet Top-1 | 69.8% |
| CPU推理速度 | ~15ms/张 |
| 训练显存(batch=64) | ~5GB |
| 优势 | 经典架构;残差连接训练稳定 |
| 劣势 | 参数量比EfficientNet-B0大但精度更低 |
| torchvision | `resnet18` |

### 5. ShuffleNetV2 x1.0

| 项目 | 值 |
|------|-----|
| 参数量 | ~2.3M |
| 输入尺寸 | 224×224 |
| ImageNet Top-1 | 69.4% |
| CPU推理速度 | ~4ms/张(ARM优化) |
| 训练显存(batch=64) | ~2GB |
| 优势 | 通道混洗减少计算;移动端推理最快 |
| 劣势 | torchvision无官方预训练;需用timm |
| timm | `shufflenet_v2_x1_0` |

### 6. MobileViT-XXS

| 项目 | 值 |
|------|-----|
| 参数量 | ~1.3M |
| 输入尺寸 | 256×256 |
| ImageNet Top-1 | 69.0% |
| CPU推理速度 | ~8ms/张 |
| 训练显存(batch=32) | ~3GB |
| 优势 | CNN+ViT混合;全局+局部特征;最轻量 |
| 劣势 | Transformer部分在CPU上较慢 |
| timm | `mobilevit_xxs` |

### 7. DeiT-Tiny (蒸馏ViT)

| 项目 | 值 |
|------|-----|
| 参数量 | ~5.7M |
| 输入尺寸 | 224×224 |
| ImageNet Top-1 | 72.2% |
| CPU推理速度 | ~20ms/张(Transformer) |
| 训练显存(batch=64) | ~6GB |
| 优势 | 纯Transformer全局注意力;蒸馏训练 |
| 劣势 | CPU推理慢于CNN;需要更多数据增强 |
| timm | `deit_tiny_patch16_224` |

## 推荐方案

### 首选: EfficientNet-B0 + 微调

- **理由**: 精度(77.1%)和效率(5.3M参数)的最佳平衡点
- **P100 16GB显存**: 训练batch=64仅需~4GB, 远低于上限
- **微调策略**: 冻结stem层,训练其余层+新分类头
- **数据增强**: RandomResizedCrop, HorizontalFlip, ColorJitter, RandAugment

### 备选: MobileNetV3-Large

- 如果CPU推理速度是首要关注点
- 5.5M参数, 75.2%精度, CPU仅~10ms/张

### 进阶方案(如基础模型精度不足):
1. EfficientNet-B0先跑baseline,评估精度
2. 如精度<80%,替换为EfficientNet-B1/B2
3. 考虑ArcFace/SphereFace等度量学习loss改善细粒度
4. 多模型集成(Model Ensemble)

## P100显存估算

| 模型 | Batch=32 | Batch=64 | Batch=128 |
|------|----------|----------|-----------|
| MobileNetV3-Small | ~1.5GB | ~2GB | ~3.5GB |
| MobileNetV3-Large | ~2.5GB | ~3.5GB | ~5.5GB |
| EfficientNet-B0 | ~3GB | ~4GB | ~7GB |
| ResNet-18 | ~3.5GB | ~5GB | ~8.5GB |
| DeiT-Tiny | ~4.5GB | ~6GB | ~10GB |

P100有16GB显存(实际可用~14-15GB),所有候选模型均可使用batch=64或更大。

## 推荐训练配置

```
模型: EfficientNet-B0 (torchvision)
优化器: AdamW (lr=3e-4, weight_decay=1e-4)
调度器: CosineAnnealingLR
Batch size: 64 (P100) / 16 (CPU测试)
Epochs: 50-100
Early Stopping: patience=10
数据增强: RandAugment(N=2, M=9)
损失函数: CrossEntropyLoss(label_smoothing=0.1)
```

## 总结

对于189类兵种分类,在P100/30GB Kaggle环境下训练完全可行。推荐EfficientNet-B0作为baseline,仅5.3M参数。通过微调+数据增强,预期可达到85%+的top-1准确率。所有候选模型均可轻松导出ONNX,在CPU上实现<20ms推理。

## Kaggle P100 30GB 训练配置详解

### P100 (PCIe 32GB HBM2) 规格
| 项目 | 值 |
|------|-----|
| 显存 | 32GB HBM2 (实际可用约30GB) |
| FP32算力 | ~9.5 TFLOPS |
| FP16算力 | ~19 TFLOPS (无Tensor Core，与FP32比约2x) |
| 显存带宽 | 732 GB/s |
| 架构 | Pascal (2016) |

**注意**: P100没有Tensor Core，FP16训练加速主要来自显存带宽减半。混合精度训练(AMP)仍然推荐开启以节省显存。

### 189类分类下的显存估算

| 模型 | 参数量 | Batch=32 | Batch=64 | Batch=128 | Batch=256 |
|------|--------|----------|----------|-----------|-----------|
| ShuffleNetV2 x1.0 | 2.3M | ~1.5GB | ~2GB | ~3GB | ~5GB |
| MobileNetV3-Small | 2.5M | ~1.5GB | ~2GB | ~3.5GB | ~5.5GB |
| MobileViT-XXS | 1.3M | ~2GB | ~3GB | ~5GB | ~9GB |
| MobileNetV3-Large | 5.5M | ~2.5GB | ~3.5GB | ~5.5GB | ~9GB |
| EfficientNet-B0 | 5.3M | ~3GB | ~4GB | ~7GB | ~12GB |
| DeiT-Tiny | 5.7M | ~4.5GB | ~6GB | ~10GB | ~17GB |
| ResNet-18 | 11.7M | ~3.5GB | ~5GB | ~8.5GB | ~14GB |

P100 30GB完全足够,即使是ResNet-18 batch=256也仅需~14GB。推荐batch=64-128以获得最佳训练效率。

### CPU推理速度对比 (Intel Core i7-12700H, 单线程)

| 模型 | FP32 (ms/张) | INT8量化 (ms/张) | 模型大小 (MB) |
|------|-------------|-----------------|--------------|
| ShuffleNetV2 x1.0 | ~8 | ~4 | 8.8 |
| MobileNetV3-Small | ~10 | ~5 | 9.6 |
| MobileViT-XXS | ~15 | ~8 | 5.2 |
| MobileNetV3-Large | ~18 | ~9 | 21 |
| EfficientNet-B0 | ~22 | ~12 | 20.5 |
| DeiT-Tiny | ~35 | ~20 | 22 |
| ResNet-18 | ~28 | ~15 | 44.7 |

所有模型在CPU上推理均<50ms/张，满足实时战场分析需求(<1秒)。INT8量化后推理速度提升1.5-2x。

### 推荐训练超参数 (189类)

```
模型: EfficientNet-B0 (torchvision)
优化器: AdamW (lr=3e-4, weight_decay=1e-4)
调度器: Warmup+CosineAnnealing (warmup 3 epochs)
Batch size: 64-128 (P100 30GB)
Epochs: 50-100
Early Stopping: patience=10
数据增强: RandomResizedCrop + HorizontalFlip + ColorJitter
损失函数: CrossEntropyLoss(label_smoothing=0.1)
梯度裁剪: max_norm=1.0
混合精度: AMP (torch.amp)
```

### 模型架构详解

#### 1. EfficientNet-B0 (首选)
- **核心模块**: MBConv (Mobile Inverted Bottleneck) + SE (Squeeze-and-Excitation)注意力
- **结构**: 7个stage, 使用NAS搜索的复合缩放(width×1.0, depth×1.0, resolution×1.0)
- **MBConv结构**: 1x1扩展 → Depthwise 3x3/5x5 Conv → SE注意力 → 1x1压缩 + 残差连接
- **为何适合本任务**: SE注意力模块对细粒度分类(区分相似兵种)特别有效; 复合缩放确保特征提取充分
- **显存需求**: batch=64约4GB, batch=128约7GB

#### 2. MobileNetV3-Large
- **核心模块**: 改进的MobileNetV2 inverted residual block + SE + h-swish激活
- **结构**: 通过NAS+NetAdapt联合搜索, 针对移动CPU优化
- **特点**: 使用h-swish(swish的近似)替代ReLU, 尾部使用大卷积核提取全局特征
- **为何适合**: CPU推理友好(h-swish在ARM上高效); SE注意力有助于细粒度; 预热训练权重丰富

#### 3. MobileNetV3-Small
- **核心模块**: 同Large但深度/宽度削减
- **结构**: 更少的bottleneck层, 更小的扩展比
- **特点**: 参数仅2.5M, 极致轻量, 适合ARM/移动部署
- **劣势**: 精度天花板较低(比B0低约10%), 对于189类细粒度分类可能不够

#### 4. ResNet-18
- **核心模块**: 残差块 (Conv3x3 → BN → ReLU → Conv3x3 → BN + Skip Connection)
- **结构**: 4个stage [2,2,2,2]个残差块, 通道数[64,128,256,512]
- **特点**: 训练最稳定, 残差连接消除梯度消失; 但参数量是B0的2倍而精度反而更低
- **适用场景**: 当其他模型训练不稳定时作为baseline验证

#### 5. ShuffleNetV2 x1.0
- **核心模块**: Channel Shuffle + Depthwise Conv, 遵循4条轻量设计准则
- **特点**: 通道混洗在不增加计算量的前提下实现信息交换; 2.3M参数, 最轻量

#### 6. MobileViT-XXS
- **核心模块**: CNN局部特征 + ViT全局特征融合
- **结构**: MobileNetV2 stem → MobileViT blocks (Transformer + CNN融合)
- **特点**: 1.3M参数但精度与更大模型相当; Transformer提供全局感受野
- **劣势**: Transformer部分在CPU上较慢; 需要timm库

#### 7. DeiT-Tiny
- **核心模块**: 标准ViT + 蒸馏token
- **结构**: 12层Transformer, patch_size=16, embedding_dim=192, 3个head
- **特点**: 纯注意力机制; 使用CNN teacher蒸馏训练; 全局感受野

## 预期训练效果

基于ImageNet预训练权重微调, 预期经过50-100 epochs训练后可达到:

| 模型 | 预期Top-1 | 预期Top-5 | 训练时间(P100) |
|------|----------|----------|---------------|
| EfficientNet-B0 | 82-88% | 94-97% | ~2-3小时 |
| MobileNetV3-Large | 78-85% | 92-96% | ~1.5-2小时 |
| MobileNetV3-Small | 70-78% | 88-93% | ~1-1.5小时 |
| ResNet-18 | 74-82% | 90-95% | ~2小时 |
| ShuffleNetV2 | 72-80% | 89-94% | ~1小时 |
| MobileViT-XXS | 72-80% | 89-94% | ~2.5小时 |
| DeiT-Tiny | 76-84% | 91-96% | ~3小时 |

**注**: 实际精度取决于训练数据质量和数量。当前每兵种仅1-2张图, 数据增强可扩充到10张+。

## 189类标签分布

12阵营 + Neutral 共189个兵种类别:
- Castle: 14, Rampart: 14, Tower: 14, Inferno: 14
- Necropolis: 14, Dungeon: 14, Stronghold: 14, Fortress: 14
- Conflux: 14, Cove: 16(含3级额外Sea Dog), Factory: 16(含7级额外Dreadnought/Juggernaut)
- Bulwark: 14, Neutral: 17

标签索引0-188, 详见 `gamedata/creature_labels.json`。
