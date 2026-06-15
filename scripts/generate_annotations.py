"""图像标注生成脚本 — 从子文件夹结构生成完整标注。

图片组织格式: data/images/{label_index}/{label_index}_{idx}.png

训练时不再使用预分配CSV, 由 data_processor.random_train_val_split() 动态分割。

用法:
    cd F:/桌面/test3 && python training/scripts/generate_annotations.py

生成:
    - training/data/annotations/full.csv       完整标注
    - training/data/annotations/summary.json   数据集摘要
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"

sys.path.insert(0, str(PROJECT_ROOT))

from training.scripts.data_processor import (
    discover_images, save_annotations, save_summary,
)


def generate_annotations() -> None:
    """扫描图片目录, 生成完整标注CSV和摘要。"""
    images_dir = TRAINING_ROOT / "data" / "images"
    annotations_dir = TRAINING_ROOT / "data" / "annotations"

    if not images_dir.exists():
        print(f"[ERROR] 图片目录不存在: {images_dir}")
        sys.exit(1)

    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 发现所有图片
    samples = discover_images(str(images_dir))
    if not samples:
        print("[ERROR] 未找到任何图片")
        sys.exit(1)

    num_classes = len(set(s["label_index"] for s in samples))
    print(f"发现 {len(samples)} 张图片, {num_classes} 个类别")

    # 保存完整标注
    full_csv = save_annotations(samples, annotations_dir, "full")
    print(f"  完整标注: {full_csv}")

    # 保存摘要
    summary_path = annotations_dir / "summary.json"
    summary = {
        "total_images": len(samples),
        "num_classes": num_classes,
        "images_dir": str(images_dir),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    print(f"  摘要: {summary_path}")


if __name__ == "__main__":
    generate_annotations()
