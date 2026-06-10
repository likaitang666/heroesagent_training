"""图像标注文件生成脚本 — 将兵种图片映射到标签并创建训练标注。

图片命名格式: {label_index}_{index}.png (如 14_0.png 表示 Pikeman 的第0张图)

用法:
    cd F:/桌面/test3 && python training/scripts/generate_annotations.py

生成:
    - training/data/annotations/train.csv      训练集标注
    - training/data/annotations/val.csv        验证集标注
    - training/data/annotations/test.csv       测试集标注
    - training/data/annotations/full.csv       完整标注
    - training/data/annotations/summary.json   标注摘要
    - training/data/annotations/missing.txt    缺失图片清单
    - training/data/annotations/low_data.txt   数据不足兵种清单

分割策略:
    - >=10张/类: 80/10/10 分层分割
    - 2-9张/类: train至少1张，剩余按67/33分给val/test
    - =1张/类: 全部放入train，标注"需扩充数据"
"""

import json
import csv
import random
import shutil
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
IMAGES_SRC = PROJECT_ROOT / "images" / "creatures"
IMAGES_DST = TRAINING_ROOT / "data" / "images"
ANNOTATIONS_DIR = TRAINING_ROOT / "data" / "annotations"
LABELS_FILE = PROJECT_ROOT / "gamedata" / "creature_labels.json"

sys.path.insert(0, str(PROJECT_ROOT))


def load_labels() -> dict:
    if not LABELS_FILE.exists():
        print(f"[ERROR] 标签文件不存在: {LABELS_FILE}")
        print("请先运行: python training/scripts/create_xlsx_mapping.py")
        sys.exit(1)
    with open(LABELS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _normalize(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def match_image_to_label(image_stem: str, labels: list[dict]) -> Optional[dict]:
    """匹配图片文件名到标签。

    支持两种命名格式:
    1. 新格式: {label_index}_{index} (如 14_0.png)
    2. 旧格式: creature_{name_en} 或包含 name_en 的任意名
    """
    # 尝试新格式: 纯数字_label
    m = re.match(r"^(\d+)_\d+$", image_stem)
    if m:
        label_idx = int(m.group(1))
        for label in labels:
            if label["label"] == label_idx:
                return label
        return None

    # 回退到旧格式匹配
    norm_stem = _normalize(image_stem)
    best_label: Optional[dict] = None
    best_len = 0

    for label in labels:
        name_en = _normalize(label["name_en"])
        expected = f"creature_{name_en}"
        if norm_stem == expected:
            return label
        if name_en in norm_stem:
            if len(name_en) > best_len:
                best_label = label
                best_len = len(name_en)

    return best_label


def stratified_split(
    per_class: dict[int, list[dict]],
    random_seed: int = 42,
) -> dict[str, list[dict]]:
    """按兵种分层分割数据集。

    策略:
    - 每类>=10张: 80/10/10 分割
    - 每类2-9张: 确保train至少1张，其余80/20分给train/val
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
            train_end = max(1, int(n * 0.8))
            val_end = max(train_end + 1, int(n * 0.9))
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
    labels_data = load_labels()
    labels = labels_data["labels"]
    print(f"加载 {len(labels)} 个标签")

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DST.mkdir(parents=True, exist_ok=True)

    # 按兵种分组收集标注
    per_class: dict[int, list[dict]] = defaultdict(list)
    unmatched_images: list[str] = []
    matched_labels: set[str] = set()

    image_files = sorted(IMAGES_SRC.glob("*.png"))
    print(f"发现 {len(image_files)} 张图片")

    for img_path in image_files:
        label = match_image_to_label(img_path.stem, labels)
        if label:
            label_idx = label.get("label") if label.get("label") is not None else label.get("index", 0)
            ann = {
                "image": img_path.name,
                "label_index": label_idx,
                "name_en": label["name_en"],
                "name_zh": label["name_zh"],
                "faction": label["faction"],
                "level": label["level"],
                "is_upgraded": label["is_upgraded"],
            }
            per_class[label_idx].append(ann)
            matched_labels.add(label["name_en"])

            dst = IMAGES_DST / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
        else:
            unmatched_images.append(img_path.name)

    total_annotations = sum(len(v) for v in per_class.values())
    print(f"匹配成功: {total_annotations} 张 ({len(per_class)} 个兵种)")

    # 检查缺失和不足的兵种
    missing_labels: list[dict] = []
    low_data_labels: list[dict] = []
    for label in labels:
        if label["name_en"] not in matched_labels:
            missing_labels.append(label)
        else:
            label_idx = label.get("label") if label.get("label") is not None else label.get("index", 0)
            count = len(per_class.get(label_idx, []))
            if count < 2:
                low_data_labels.append((label, count))

    print(f"缺失图片(标签无对应图): {len(missing_labels)} 个")
    print(f"数据不足(<2张/类): {len(low_data_labels)} 个")

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
        # 统计该split覆盖的兵种数
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

    # 缺失清单
    if missing_labels:
        _write_missing_report(missing_labels)
    if low_data_labels:
        _write_low_data_report(low_data_labels)
    if unmatched_images:
        _write_unmatched_report(unmatched_images)

    # 摘要
    summary = {
        "total_labels": len(labels),
        "total_images": len(image_files),
        "matched": total_annotations,
        "classes_with_images": len(per_class),
        "classes_missing": len(missing_labels),
        "classes_low_data": len(low_data_labels),
        "unmatched_images": len(unmatched_images),
        "splits": {k: {"count": len(v), "classes": len(set(a["label_index"] for a in v))}
                   for k, v in splits.items()},
        "images_dir": str(IMAGES_DST),
        "note": "数据不足的兵种(<2张图)全部分配到训练集。请添加更多图片后重新生成。"
        if low_data_labels else "",
    }
    summary_path = ANNOTATIONS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    print(f"\n摘要已保存: {summary_path}")

    # 打印关键统计
    print(f"\n{'='*60}")
    print(f"数据集分割完成:")
    print(f"  总图片: {total_annotations}")
    print(f"  训练集: {len(splits['train'])} 张")
    print(f"  验证集: {len(splits['val'])} 张")
    print(f"  测试集: {len(splits['test'])} 张")
    if low_data_labels:
        print(f"\n  [警告] {len(low_data_labels)} 个兵种数据不足(<2张图)")
        print(f"  建议每个兵种至少10张图片后再开始训练")
        print(f"  详见: {ANNOTATIONS_DIR / 'low_data.txt'}")
    if missing_labels:
        print(f"\n  [警告] {len(missing_labels)} 个兵种完全缺失图片")
        print(f"  详见: {ANNOTATIONS_DIR / 'missing.txt'}")


def _write_missing_report(missing_labels: list[dict]) -> None:
    path = ANNOTATIONS_DIR / "missing.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"缺失图片清单 (共 {len(missing_labels)} 个兵种):\n\n")
        f.write(f"{'索引':<6} {'英文名':<30} {'中文名':<16} {'阵营':<14} {'等级':<6}\n")
        f.write("-" * 80 + "\n")
        for label in missing_labels:
            idx = label.get("label") if label.get("label") is not None else label.get("index", 0)
            f.write(f"[{idx:<4}] {label['name_en']:<30} {label['name_zh']:<16} "
                    f"{label.get('faction', '?'):<14} Lv.{label.get('level', '?')}\n")
    print(f"  缺失清单: {path}")


def _write_low_data_report(low_data_labels: list[tuple]) -> None:
    path = ANNOTATIONS_DIR / "low_data.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"数据不足兵种清单 (共 {len(low_data_labels)} 个兵种, 每类<2张图):\n\n")
        f.write("这些兵种的全部图片均被分配到训练集，无法创建验证/测试集。\n")
        f.write("请为以下每个兵种添加更多图片(建议>=10张)后重新运行本脚本。\n\n")
        f.write(f"{'索引':<6} {'英文名':<30} {'中文名':<16} {'当前数量':<10}\n")
        f.write("-" * 70 + "\n")
        for label, count in low_data_labels:
            idx = label.get("label") if label.get("label") is not None else label.get("index", 0)
            f.write(f"[{idx:<4}] {label['name_en']:<30} {label.get('name_zh', ''):<16} {count}张\n")
    print(f"  数据不足清单: {path}")


def _write_unmatched_report(unmatched: list[str]) -> None:
    path = ANNOTATIONS_DIR / "unmatched.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"未匹配图片清单 (共 {len(unmatched)} 张):\n\n")
        for name in unmatched:
            f.write(f"  {name}\n")


if __name__ == "__main__":
    generate_annotations()
