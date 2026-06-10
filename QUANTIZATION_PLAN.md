# 模型量化预备方案

## 背景

如果训练后的模型在CPU上推理时间过长(>100ms/张),需要进行量化处理以减小模型体积和加速推理。

## 量化方案对比

### 1. PyTorch 动态量化 (Dynamic Quantization)

**原理**: 仅量化权重(INT8),激活值在推理时动态量化。最简单,精度损失最小。

```python
import torch
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
torch.save(model.state_dict(), "model_quantized.pth")
```

| 项目 | 值 |
|------|-----|
| 模型压缩率 | ~4x (FP32→INT8) |
| 推理加速 | 1.5-2x (CPU) |
| 精度损失 | <1% |
| 实现难度 | 低 (3行代码) |
| GPU支持 | 否(仅CPU加速) |

### 2. PyTorch 静态量化 (Static Quantization)

**原理**: 权重+激活值均量化为INT8。需要校准数据集来确定激活值范围。

| 项目 | 值 |
|------|-----|
| 模型压缩率 | ~4x |
| 推理加速 | 2-4x (CPU) |
| 精度损失 | 1-3% |
| 实现难度 | 中(需要校准数据和模型改造) |

### 3. 半精度 FP16

**原理**: 将模型从FP32转为FP16。GPU上显著加速,CPU上支持有限。

```python
model = model.half()  # 转换为FP16
```

| 项目 | 值 |
|------|-----|
| 模型压缩率 | ~2x |
| 推理加速 | 1.5-2x (GPU), CPU效果有限 |
| 精度损失 | <0.5% |
| 实现难度 | 低 |

**重要**: P100和大多数CPU对FP16支持有限。P100的FP16性能与FP32接近(无Tensor Core)。仅推荐在有Tensor Core的GPU上使用。

### 4. ONNX Runtime + INT8量化

**原理**: 导出ONNX格式,使用ONNX Runtime进行INT8量化推理。

```python
import onnx
import onnxruntime as ort
# 导出ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
# 使用ONNX Runtime INT8推理
session = ort.InferenceSession("model.onnx",
    providers=['CPUExecutionProvider'])
```

| 项目 | 值 |
|------|-----|
| 模型压缩率 | ~4x |
| 推理加速 | 2-5x (CPU,相比PyTorch) |
| 精度损失 | 1-2% |
| 实现难度 | 中 |
| 优势 | 跨平台,ARM/x86均可,无需PyTorch环境 |

### 5. OpenVINO (Intel CPU优化)

**原理**: Intel官方推理优化框架,针对Intel CPU深度优化。

| 项目 | 值 |
|------|-----|
| 推理加速 | 3-8x (Intel CPU) |
| 精度损失 | <1% |
| 实现难度 | 中(需安装OpenVINO) |
| 限制 | 仅Intel CPU有效,AMD CPU效果有限 |

### 6. TensorRT (NVIDIA GPU)

**原理**: NVIDIA官方推理优化框架,FP16/INT8量化+内核融合。

| 项目 | 值 |
|------|-----|
| 推理加速 | 3-10x (NVIDIA GPU) |
| 精度损失 | <1% |
| 实现难度 | 高(需构建引擎) |
| 限制 | 仅NVIDIA GPU |

## 推荐实施路径

### 步骤1: 基准测试 (必须先做)

```python
import time
import torch
from training.scripts.dataset import CreatureDataset, get_default_transforms

# 测量当前模型推理时间
model.eval()
with torch.no_grad():
    for img in test_images:
        start = time.time()
        output = model(img)
        elapsed = time.time() - start

avg_time = total_time / num_images
print(f"平均推理时间: {avg_time*1000:.1f}ms/张")
```

### 步骤2: 根据基准选择方案

- **CPU推理 >100ms/张**: 优先尝试ONNX Runtime + INT8量化
- **CPU推理 20-100ms/张**: 使用PyTorch动态量化(最简单)
- **CPU推理 <20ms/张**: 无需量化,当前速度已达标
- **GPU推理**: 使用FP16(如有Tensor Core)或保持FP32

### 步骤3: ONNX导出脚本

```python
# export_onnx.py
import torch
from training.scripts.train import build_model

model = build_model("efficientnet_b0", num_classes=171)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "creature_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)
print("ONNX模型已导出: creature_classifier.onnx")
```

### 步骤4: 精度验证

量化后必须在测试集上验证精度,确保精度损失在可接受范围内(<3%)。

## 预期效果 (EfficientNet-B0, 189类)

| 方案 | 模型大小 | CPU推理(ms/张) | 预期精度 |
|------|---------|---------------|---------|
| 原始FP32 | ~21MB | ~22ms | baseline |
| 动态量化INT8 | ~6MB | ~12ms | ~baseline-0.5% |
| ONNX+INT8 | ~6MB | ~10ms | ~baseline-1% |
| 静态量化INT8 | ~6MB | ~9ms | ~baseline-2% |

## 其他模型量化预期

| 模型 | FP32大小 | INT8大小 | CPU FP32 | CPU INT8 |
|------|---------|---------|----------|----------|
| ShuffleNetV2 x1.0 | 8.8MB | 2.5MB | ~8ms | ~4ms |
| MobileNetV3-Small | 9.6MB | 2.8MB | ~10ms | ~5ms |
| MobileNetV3-Large | 21MB | 5.8MB | ~18ms | ~9ms |
| EfficientNet-B0 | 20.5MB | 5.7MB | ~22ms | ~12ms |
| ResNet-18 | 44.7MB | 11.5MB | ~28ms | ~15ms |

## 量化感知训练 (QAT) 预备方案

如果动态/静态量化后精度损失超过3%, 可考虑量化感知训练:

```python
import torch.quantization as quant

# 1. 在模型定义中加入QuantStub/DeQuantStub
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model = quant.prepare_qat(model, inplace=True)

# 2. 用较小的学习率继续训练几个epoch
for epoch in range(5):
    train_one_epoch(model, train_loader, optimizer, device)

# 3. 转换为量化模型
model = quant.convert(model.eval(), inplace=True)
```

**QAT优势**: 精度损失通常<0.5%, 推理速度达到静态量化水平。
**QAT劣势**: 需要额外训练时间(通常5-10个epoch); 需要校准数据集。

## 部署策略

### 场景1: GPU可用 (如用户有NVIDIA显卡)
- 使用原始FP32模型
- 推理速度: <5ms/张 (GPU batch推理)
- 无需量化

### 场景2: 仅CPU可用 (通用场景)
- 使用ONNX Runtime + INT8量化
- 推理速度: ~10ms/张
- 安装: `pip install onnxruntime`

### 场景3: 低端硬件/嵌入式
- 使用MobileNetV3-Small + PyTorch动态量化
- 推理速度: ~5ms/张
- 模型大小: <3MB

### 场景4: 极致优化
- 导出ONNX → 使用OpenVINO (Intel CPU) 或 TensorRT (NVIDIA GPU)
- 推理速度: <3ms/张

## 量化实施检查清单

- [ ] 完成EfficientNet-B0训练, 获得baseline准确率
- [ ] 测量baseline CPU推理速度
- [ ] 如速度>100ms: 必须量化; >20ms: 建议量化; <20ms: 可选
- [ ] 尝试动态量化 (3行代码, 最简单)
- [ ] 验证量化后精度损失 (<3%为可接受)
- [ ] 如精度损失>3%: 尝试QAT
- [ ] 导出ONNX用于跨平台部署
- [ ] 集成到Agent项目的`backend/app/rag/`模块中

## 总结

由于我们选的模型(如EfficientNet-B0)本身只有~5.3M参数,推理已经很快(~22ms/张CPU)。量化主要是"锦上添花"而非必需。对189类兵种分类:
1. **优先完成训练**, 测量实际baseline
2. 如CPU推理<50ms, 无需量化
3. 如需量化, 优先使用ONNX Runtime INT8 (平衡速度和精度)
4. 如精度要求极高, 使用QAT
