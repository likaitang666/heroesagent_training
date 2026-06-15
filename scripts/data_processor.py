"""数据处理模块 — 图片发现、随机分割、标注生成。

每次训练时动态随机分割训练/验证集 (5:1比例), 不使用预分配。
保证每类至少有1个验证样本 (除非该类仅有1张图)。

用法:
    from training.scripts.data_processor import (
        discover_images, random_train_val_split, save_annotations,
    )
    samples = discover_images("training/data/images")
    train, val = random_train_val_split(samples, train_ratio=5/6)
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional


def discover_images(images_dir: str | Path) -> list[dict]:
    """扫描图片目录, 发现所有兵种图片。

    Args:
        images_dir: data/images/ 目录, 内含数字命名的子文件夹

    Returns:
        标注列表 [{"image": "0/0_0.png", "label_index": 0, ...}, ...]
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_path}")

    subdirs = sorted(
        [d for d in images_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not subdirs:
        raise ValueError(f"未找到兵种子文件夹 (如 0/, 1/, ...) 在: {images_path}")

    samples: list[dict] = []
    for subdir in subdirs:
        label_idx = int(subdir.name)
        png_files = sorted(subdir.glob("*.png"))
        jpg_files = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.jpeg"))
        for img_file in png_files + jpg_files:
            samples.append({
                "image": f"{label_idx}/{img_file.name}",
                "label_index": label_idx,
                "name_en": "",
                "name_zh": "",
                "faction": "",
                "level": 0,
                "is_upgraded": False,
            })
    return samples


def random_train_val_split(
    samples: list[dict],
    train_ratio: float = 5 / 6,
    random_seed: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """按兵种分层随机分割训练/验证集 (5:1)。

    每类至少1张进验证集 (除非该类仅1张图, 则全进训练集)。
    不使用预分配CSV, 每次调用重新随机分割。

    Args:
        samples: discover_images() 返回的标注列表
        train_ratio: 训练集比例 (默认 5/6 ≈ 83.3%)
        random_seed: 随机种子 (None=不固定)

    Returns:
        (train_annotations, val_annotations)
    """
    if random_seed is not None:
        random.seed(random_seed)

    per_class: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        per_class[s["label_index"]].append(s)

    train_anns: list[dict] = []
    val_anns: list[dict] = []

    for label_idx, items in per_class.items():
        n = len(items)
        items_shuffled = items.copy()
        random.shuffle(items_shuffled)

        if n <= 1:
            train_anns.extend(items_shuffled)
        else:
            val_count = max(1, n - int(n * train_ratio))
            train_count = n - val_count
            train_anns.extend(items_shuffled[:train_count])
            val_anns.extend(items_shuffled[train_count:])

    random.shuffle(train_anns)
    random.shuffle(val_anns)

    return train_anns, val_anns


def save_annotations(
    samples: list[dict],
    output_dir: str | Path,
    prefix: str = "full",
) -> Path:
    """保存标注列表为CSV文件。

    Args:
        samples: 标注列表
        output_dir: 输出目录 (annotations/)
        prefix: 文件名前缀

    Returns:
        保存的CSV文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{prefix}.csv"

    fieldnames = ["image", "label_index", "name_en", "name_zh",
                   "faction", "level", "is_upgraded"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    return csv_path


def save_summary(
    samples: list[dict],
    train: list[dict],
    val: list[dict],
    output_dir: str | Path,
) -> Path:
    """保存数据集摘要JSON。

    Args:
        samples: 全部标注
        train: 训练集标注
        val: 验证集标注
        output_dir: 输出目录

    Returns:
        JSON文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"

    train_classes = set(a["label_index"] for a in train)
    val_classes = set(a["label_index"] for a in val)

    summary = {
        "total_images": len(samples),
        "num_classes": len(set(a["label_index"] for a in samples)),
        "train_count": len(train),
        "val_count": len(val),
        "train_classes": len(train_classes),
        "val_classes": len(val_classes),
        "split_ratio": f"{len(train)}:{len(val)}",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    return summary_path


def print_split_summary(
    samples: list[dict],
    train: list[dict],
    val: list[dict],
) -> None:
    """打印分割摘要到终端。"""
    per_class = defaultdict(list)
    for s in samples:
        per_class[s["label_index"]].append(s)

    train_per_class = defaultdict(int)
    for a in train:
        train_per_class[a["label_index"]] += 1
    val_per_class = defaultdict(int)
    for a in val:
        val_per_class[a["label_index"]] += 1

    print(f"数据集分割 (5:1 随机, 每类保证验证≥1):")
    print(f"  总图片: {len(samples)}, {len(per_class)} 类")
    print(f"  训练集: {len(train)} 张 ({len(train_per_class)} 类)")
    print(f"  验证集: {len(val)} 张 ({len(val_per_class)} 类)")
    print(f"  无测试集 (全部数据用于训练/验证)")
