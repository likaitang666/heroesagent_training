# 训练图片添加指南（新标准）

## 命名规范

所有兵种图片必须使用统一命名格式：

```
{label_index}_{序号}.png
```

- `label_index`: 兵种标签ID（0~170），从 `gamedata/creature_labels.json` 查找
- `序号`: 同一兵种的图片编号（0, 1, 2, ...），从0开始递增

**示例**:

| 文件名 | 含义 |
|--------|------|
| `0_0.png` | 枪兵(Pikeman, label=0)第1张 |
| `0_1.png` | 枪兵第2张 |
| `70_0.png` | 黑龙(Black Dragon, label=70)第1张 |
| `70_1.png` | 黑龙第2张 |

**禁止**:
- 不要使用中文文件名
- 不要使用 `creature_` 前缀的旧命名（不再兼容）
- 不要使用空格或特殊字符

## 快速查找 label_index

### 方法1: 查看 XLSX 映射表

```bash
cd F:/桌面/test3
python training/scripts/create_xlsx_mapping.py
```
打开 `training/data/creature_image_mapping.xlsx`，在"标签ID"列查找。

### 方法2: 命令行查询

```bash
cd F:/桌面/test3
"E:/anaconda/python.exe" -c "
import json
with open('gamedata/creature_labels.json', encoding='utf-8') as f:
    data = json.load(f)
for lbl in data['labels']:
    print(f\"[{lbl['label']:3d}] {lbl['name_en']:<30} {lbl['name_zh']}\")
"
```

### 方法3: 按兵种名搜索

```bash
cd F:/桌面/test3
"E:/anaconda/python.exe" -c "
from gamedata.creature_sparse_vectors import get_label_by_name
# 查英文名
print(get_label_by_name('Pikeman'))        # -> 0
# 查中文名
print(get_label_by_name('枪兵', lang='zh'))  # -> 0
"
```

## 添加图片步骤

### 步骤1: 准备图片

**图片来源**:
- HotA Wiki: https://heroes.thelazy.net/index.php/List_of_creatures_(HotA) — 点击兵种进入详情页，右键保存肖像
- 游戏截图: 打开兵种信息界面截图，裁剪出58×64px的兵种肖像区域
- 数据增强: 从已有图片自动生成变体（见步骤3）

**图片要求**:
- 格式: PNG（必须）
- 最小尺寸: 58×64px（原始兵种肖像尺寸），训练时自动resize到224×224
- 背景: 建议统一使用游戏内纯色背景

### 步骤2: 放入图片

将图片文件按命名规范命名后放入：

```
training/data/images/
```

> 也接受放入 `images/creatures/`，运行标注脚本时会自动复制到 `training/data/images/`。

**命名范例**（以添加第2张枪兵图片为例）:

```
training/data/images/14_1.png
```

如果该兵种已有 `14_0.png`，新图片序号应为 `1`（递增）。

### 步骤3: 数据增强（推荐，快速扩充）

如果手动收集的图片不足（建议≥10张/兵种），使用数据增强生成变体：

```bash
cd F:/桌面/test3
python training/scripts/augment_images.py --variations 9
```

这会为每张原图生成9个变体（旋转、翻转、亮度/对比度/色彩调整、锐化等）。

变体文件命名: `{label_index}_{序号}_aug{N}.png`（如 `14_0_aug0.png`）

### 步骤4: 重新生成标注

```bash
cd F:/桌面/test3
python training/scripts/generate_annotations.py
```

脚本会自动:
1. 扫描所有图片，匹配到对应兵种标签
2. 按分层策略划分 train/val/test 集
3. 输出数据覆盖报告

### 步骤5: 检查覆盖情况

```bash
# 查看数据不足的兵种
cat training/data/annotations/low_data.txt

# 查看完全缺失图片的兵种
cat training/data/annotations/missing.txt

# 查看统计摘要
cat training/data/annotations/summary.json
```

## 数据集划分策略

| 每类图片数 | 训练集 | 验证集 | 测试集 |
|-----------|--------|--------|--------|
| ≥10张 | 80% | 10% | 10% |
| 2-9张 | ≥1张 | 剩余约67% | 剩余约33% |
| =1张 | 1张 | 0 | 0 |

**推荐目标**: 每个兵种 ≥10张图片（含增强变体）

## 测试集说明

按照 DESIGN.md 要求，当前已收集的图片（`images/creatures/` 中以 `{label}_0.png` 命名的文件）作为**测试集基准**：
- 这些图片代表官方标准肖像，用于评估模型的baseline准确率
- 后续手动收集的新图片放入训练集/验证集
- 数据增强生成的变体也归入训练集

当运行 `generate_annotations.py` 时，分层分割会自动处理以上逻辑。

## 当前数据状态

| 指标 | 数值 |
|------|------|
| 总兵种数 | 171 |
| 已有图片覆盖 | 157 个兵种 |
| 完全缺失 | 14 个兵种（全部为 Bulwark/棱堡阵营，HotA 1.8 新增） |
| 多张图片(≥2) | 仅 2 个兵种 |

**急需**: 14 个 Bulwark 兵种的图片 + 其余兵种的额外图片（目标 ≥10张/兵种）

## 批量添加工作流

```bash
# 1. 查找目标兵种的 label_index
cd F:/桌面/test3
"E:/anaconda/python.exe" -c "
from gamedata.creature_sparse_vectors import get_label_by_name
print(get_label_by_name('Pikeman'))  # 替换为目标兵种名
"

# 2. 将图片命名为 {label_index}_{序号}.png，放入 training/data/images/

# 3. （可选）数据增强
python training/scripts/augment_images.py --variations 9

# 4. 重新生成标注
python training/scripts/generate_annotations.py

# 5. 检查结果
cat training/data/annotations/low_data.txt
```

## 常见问题

**Q: 如何知道某个兵种的 label_index？**
A: 运行 `python training/scripts/create_xlsx_mapping.py` 打开XLSX查看，或使用 `get_label_by_name()` 函数查询。

**Q: 图片放错位置了怎么办？**
A: 直接移动文件到正确位置后重新运行 `generate_annotations.py` 即可。

**Q: 同兵种多张图片序号冲突怎么办？**
A: 查看 `training/data/images/` 中该兵种已有的最大序号，新图片从最大序号+1开始。

**Q: 为什么不能用旧的中文/英文文件名？**
A: 新标准使用 label_index 作为唯一标识，避免中英文命名歧义和编码问题。`generate_annotations.py` 不再支持旧格式匹配。

**Q: 数量够了但训练准确率还是低？**
A: 检查图片质量（是否统一背景、清晰度），考虑增加图片多样性（不同角度/光照/背景），或使用更强的数据增强策略。
