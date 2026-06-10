"""图像标注文件生成脚本 — 从子文件夹结构生成训练标注。

图片组织格式: data/images/{label_index}/{label_index}_{idx}.png

用法:
    cd F:/桌面/test3 && python training/scripts/generate_annotations.py

生成:
    - training/data/annotations/train.csv      训练集标注
    - training/data/annotations/val.csv        验证集标注
    - training/data/annotations/test.csv       测试集标注
    - training/data/annotations/full.csv       完整标注
    - training/data/annotations/summary.json   标注摘要

分割策略:
    - >=10张/类: 70/15/15 分层分割 (优先分配训练集)
    - 2-9张/类: train至少1张，剩余按67/33分给val/test
    - =1张/类: 全部放入train
"""

import json
import csv
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = Path(__file__).parent.parent  # 训练模块根目录 (文件夹名无关)
IMAGES_DIR = TRAINING_ROOT / "data" / "images"
ANNOTATIONS_DIR = TRAINING_ROOT / "data" / "annotations"

sys.path.insert(0, str(TRAINING_ROOT / "scripts"))


def stratified_split(
    per_class: dict[int, list[dict]],
    random_seed: int = 42,
) -> dict[str, list[dict]]:
    """按兵种分层分割数据集 (70/15/15 优先)。

    策略:
    - 每类>=10张: 70/15/15
    - 每类2-9张: train至少1张，剩余按67/33分给val/test
    - 每类=1张: 全部放入train
    """
    random.seed(random_seed)
    train_anns: list[dict] = []
    val_anns: list[dict] = []
    test_anns: list[dict] = []

    for label_idx, items in per_class.items():
        items_shuffled = items.copy()
        random.shuffle(items_shuffled)
        n = len(items_shuffled)

        if n >= 10:
            train_end = max(1, int(n * 0.7))
            val_end = max(train_end + 1, int(n * 0.85))
            train_anns.extend(items_shuffled[:train_end])
            val_anns.extend(items_shuffled[train_end:val_end])
            test_anns.extend(items_shuffled[val_end:])
        elif n >= 2:
            train_anns.append(items_shuffled[0])
            remaining = items_shuffled[1:]
            if len(remaining) >= 2:
                mid = max(1, int(len(remaining) * 0.67))
                val_anns.extend(remaining[:mid])
                test_anns.extend(remaining[mid:])
            elif len(remaining) == 1:
                val_anns.extend(remaining)
        else:
            train_anns.extend(items_shuffled)

    random.shuffle(train_anns)
    random.shuffle(val_anns)
    random.shuffle(test_anns)

    return {"train": train_anns, "val": val_anns, "test": test_anns}


def generate_annotations() -> None:
    """从子文件夹结构生成标注CSV文件。"""
    if not IMAGES_DIR.exists():
        print(f"[ERROR] 图片目录不存在: {IMAGES_DIR}")
        sys.exit(1)

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有数字命名子文件夹
    subdirs = sorted(
        [d for d in IMAGES_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if not subdirs:
        print("[ERROR] 未找到兵种子文件夹 (如 0/, 1/, ...)")
        print(f"请确保图片放在: {IMAGES_DIR}/0/, {IMAGES_DIR}/1/ 等子文件夹中")
        sys.exit(1)

    print(f"发现 {len(subdirs)} 个兵种文件夹")

    # 按兵种收集图片
    per_class: dict[int, list[dict]] = defaultdict(list)
    total_images = 0

    for subdir in subdirs:
        label_idx = int(subdir.name)
        png_files = sorted(subdir.glob("*.png"))
        jpg_files = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.jpeg"))
        all_files = png_files + jpg_files

        for img_file in all_files:
            ann = {
                "image": f"{label_idx}/{img_file.name}",
                "label_index": label_idx,
                "name_en": "",
                "name_zh": "",
                "faction": "",
                "level": 0,
                "is_upgraded": False,
            }
            per_class[label_idx].append(ann)
            total_images += 1

        print(f"  [{label_idx}/] {len(all_files)} 张图片")

    print(f"\n总计: {total_images} 张图片, {len(per_class)} 个兵种")

    # 分层分割
    splits = stratified_split(per_class)

    # 写入CSV文件
    fieldnames = ["image", "label_index", "name_en", "name_zh",
                   "faction", "level", "is_upgraded"]
    for split_name, split_data in splits.items():
        csv_path = ANNOTATIONS_DIR / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_data)
        classes = set(a["label_index"] for a in split_data)
        print(f"  {split_name}: {len(split_data)} 张, {len(classes)} 类")

    # 完整标注
    all_annotations = [a for items in per_class.values() for a in items]
    full_csv = ANNOTATIONS_DIR / "full.csv"
    with open(full_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_annotations)
    print(f"  full: {len(all_annotations)} 张 -> {full_csv}")

    # 摘要
    summary = {
        "total_images": total_images,
        "num_classes": len(per_class),
        "classes": sorted(per_class.keys()),
        "splits": {k: {"count": len(v), "classes": len(set(a["label_index"] for a in v))}
                   for k, v in splits.items()},
        "images_dir": str(IMAGES_DIR),
    }
    summary_path = ANNOTATIONS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    print(f"\n摘要已保存: {summary_path}")

    print(f"\n{'='*60}")
    print(f"数据集分割完成:")
    print(f"  总图片: {total_images}")
    print(f"  训练集: {len(splits['train'])} 张")
    print(f"  验证集: {len(splits['val'])} 张")
    print(f"  测试集: {len(splits['test'])} 张")


if __name__ == "__main__":
    generate_annotations()
