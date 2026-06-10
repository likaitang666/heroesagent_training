# Bulwark (棱堡) 兵种图片获取指南

## 概述

Bulwark是HotA 1.8新增的冰雪阵营，共14个兵种。经网络搜索，在 heroes.thelazy.net 和 heroes.v.thelazy.net 上找到了其中11个兵种的肖像图片。3个兵种暂无独立wiki页面。

## 图片命名规则

下载后将图片重命名为 `{label_index}_0.png` 格式放入 `training/data/images/`:

| label_index | 英文名 | 中文名 | 文件名 |
|-------------|--------|--------|--------|
| 157 | Kobold | 狗头人 | 157_0.png |
| 158 | Kobold Foreman | 狗头人工长 | 158_0.png |
| 159 | Billy Goat | 山羚 | 159_0.png |
| 160 | Ram | 雪羚 | 160_0.png |
| 161 | Snow Elf | 雪精灵 | 161_0.png |
| 162 | Steel Elf | 铁甲雪精灵 | 162_0.png |
| 163 | Yeti | 雪怪 | 163_0.png |
| 164 | Yeti Runemaster | 控符雪怪 | 164_0.png |
| 165 | Shaman | 萨满 | 165_0.png |
| 166 | Great Shaman | 萨满长老 | 166_0.png |
| 167 | Mammoth | 猛犸 | 167_0.png |
| 168 | War Mammoth | 战斗猛犸 | 168_0.png |
| 169 | Jotunn | 冰霜巨人 | 169_0.png |
| 170 | Jotunn Warlord | 冰霜领主 | 170_0.png |

## 已找到的图片URL (11/14)

### Kobold (label=157)
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Kobold.png
- 直接下载: 访问上述URL, 右键保存图片, 重命名为 157_0.png

### Kobold Foreman (label=158)
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Kobold_Foreman.png
- 重命名为: 158_0.png

### Snow Elf (label=161)
- 58x64 PNG 肖像: https://heroes.thelazy.net/index.php/File:Snow_Elf_portrait.png
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Snow_Elf_(HotA).png
- 29x32 GIF: https://heroes.thelazy.net/index.php/File:Creature_portrait_Snow_Elf_small.gif
- 重命名为: 161_0.png

### Yeti (label=163)
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Yeti.png
- 重命名为: 163_0.png

### Yeti Runemaster (label=164)
- 58x64 PNG 肖像: http://heroes.v.thelazy.net/index.php/File:Yeti_Runemaster_portrait.png
- 重命名为: 164_0.png

### Shaman (label=165)
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Shaman.png
- 58x64 PNG 肖像: http://heroes.v.thelazy.net/index.php/File:Shaman_portrait.png
- 重命名为: 165_0.png

### Great Shaman (label=166)
- 58x64 PNG 肖像: http://heroes.v.thelazy.net/index.php/File:Great_Shaman_portrait.png
- 重命名为: 166_0.png

### Mammoth (label=167)
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Mammoth.png
- 58x64 PNG 肖像: http://heroes.v.thelazy.net/index.php/File:Mammoth_(HotA)_portrait.png
- 重命名为: 167_0.png

### War Mammoth (label=168)
- 58x64 PNG 肖像: http://heroes.v.thelazy.net/index.php/File:War_Mammoth_portrait.png
- 重命名为: 168_0.png

### Jotunn (label=169)
- 58x64 PNG 肖像: https://heroes.thelazy.net/index.php/File:Jotunn_portrait.png
- 100x130 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Jotunn.png
- 重命名为: 169_0.png

### Jotunn Warlord (label=170)
- 58x64 PNG 肖像: http://heroes.v.thelazy.net/index.php/File:Jotunn_Warlord_portrait.png
- 100x133 PNG: http://heroes.v.thelazy.net/index.php/File:Creature_Jotunn_Warlord.png
- 重命名为: 170_0.png

## 缺失图片 (3/14)

以下兵种暂无独立wiki页面, 图片需要通过其他方式获取:

| label_index | 英文名 | 中文名 | 建议获取方式 |
|-------------|--------|--------|-------------|
| 159 | Billy Goat | 山羚 | 从HotA游戏客户端截图 |
| 160 | Ram | 雪羚 | 从HotA游戏客户端截图 |
| 162 | Steel Elf | 铁甲雪精灵 | 从HotA游戏客户端截图 |

## 下载步骤

1. 打开上述URL, 在页面中找到图片
2. 右键 -> 另存为PNG格式
3. 将文件按上表重命名为 `{label_index}_0.png`
4. 放入 `F:\桌面\test3\training\data\images\`
5. 运行 `python training/scripts/generate_annotations.py` 更新标注

## 注意事项

- Wiki图片尺寸通常为58x64px(肖像)或100x130px(大图), 训练时自动resize到224x224
- 建议下载58x64px的portrait版本作为标准输入 (与游戏中显示一致)
- 文件扩展名统一使用 .png
