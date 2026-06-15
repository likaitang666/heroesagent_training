"""图像标注生成脚本 — 从子文件夹结构生成完整标注。

图片组织格式: data/images/{label_index}/{label_index}_{idx}.png

训练时不再使用预分配CSV, 由 data_processor.random_train_val_split() 动态分割。

用法:
    cd heroesagent-training && python scripts/generate_annotations.py

生成:
    - data/annotations/full.csv       完整标注
    - data/annotations/summary.json   数据集摘要
"""

import json
import sys
from pathlib import Path

# 将scripts目录加入sys.path, 支持直接导入同级模块 (不依赖父目录名)
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_processor import (
    discover_images, save_annotations,
)

# 项目根目录 (heroesagent-training/)
PROJECT_ROOT = SCRIPT_DIR.parent


def generate_annotations() -> None:
    """扫描图片目录, 生成完整标注CSV和摘要。"""
    images_dir = PROJECT_ROOT / "data" / "images"
    annotations_dir = PROJECT_ROOT / "data" / "annotations"

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
