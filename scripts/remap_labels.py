"""将兵种图片从旧标签索引重命名为新标签索引。

creature_labels.json 已按照 creature_index.xlsx 重新排序后，
旧标签索引需要映射到新标签索引。本脚本处理所有图片文件的重命名。

采用两轮重命名策略避免冲突:
  1. 将所有文件重命名为临时前缀 TEMP_old{N}_
  2. 再将临时文件重命名为新标签索引

用法:
    cd F:/桌面/test3 && python training/scripts/remap_labels.py [--dry-run]
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = Path(__file__).parent.parent  # 训练模块根目录 (文件夹名无关)
IMAGES_DIRS = [
    PROJECT_ROOT / "images" / "creatures",
    TRAINING_ROOT / "data" / "images",
]
REMAP_FILE = PROJECT_ROOT / "gamedata" / "label_remapping.json"


def load_remapping() -> dict[int, int]:
    """加载旧->新标签映射。"""
    with open(REMAP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data["old_to_new"].items()}


def remap_images(dry_run: bool = False) -> dict:
    """两轮重命名所有图片。"""
    old_to_new = load_remapping()
    print(f"加载 {len(old_to_new)} 条标签映射")

    stats = {"renamed": 0, "skipped": 0, "errors": 0}

    for images_dir in IMAGES_DIRS:
        if not images_dir.exists():
            print(f"[SKIP] 目录不存在: {images_dir}")
            continue

        png_files = sorted(images_dir.glob("*.png"))
        print(f"\n处理目录: {images_dir} ({len(png_files)} 张图片)")

        # === 第一轮: 重命名为临时名称 ===
        temp_plan: list[tuple[Path, Path]] = []
        for img_path in png_files:
            stem = img_path.stem
            m = re.match(r"^(\d+)_(\d+)(_aug\d+)?$", stem)
            if not m:
                print(f"  [SKIP] 无法解析文件名: {img_path.name}")
                stats["skipped"] += 1
                continue

            old_label = int(m.group(1))
            new_label = old_to_new.get(old_label)

            if new_label is None or new_label == old_label:
                stats["skipped"] += 1
                continue

            rest = f"_{m.group(2)}{m.group(3) or ''}"
            temp_name = f"__REMAP_{old_label}_{new_label}{rest}{img_path.suffix}"
            temp_path = img_path.parent / temp_name
            temp_plan.append((img_path, temp_path))

        print(f"  第一轮(TEMP): {len(temp_plan)} 张")

        if temp_plan:
            for old_path, temp_path in temp_plan:
                if dry_run:
                    print(f"  [DRY1] {old_path.name} -> {temp_path.name}")
                else:
                    old_path.rename(temp_path)

        # === 第二轮: 从临时名称重命名为目标名称 ===
        round2 = 0
        for temp_name in sorted(images_dir.glob("__REMAP_*.png")):
            m = re.match(r"^__REMAP_(\d+)_(\d+)_(\d+)(_aug\d+)?\.png$", temp_name.name)
            if not m:
                continue
            new_label = int(m.group(2))
            seq = m.group(3)
            aug = m.group(4) or ""
            final_name = f"{new_label}_{seq}{aug}.png"
            final_path = temp_name.parent / final_name

            if final_path.exists() and final_path != temp_name:
                print(f"  [WARN] 目标已存在: {final_name}")
                stats["errors"] += 1
                continue

            if dry_run:
                print(f"  [DRY2] {temp_name.name} -> {final_name}")
            else:
                temp_name.rename(final_path)
            round2 += 1

        stats["renamed"] += round2
        print(f"  第二轮(FINAL): {round2} 张")

    print(f"\n{'='*60}")
    print(f"{'DRY RUN - ' if dry_run else ''}重命名统计:")
    print(f"  重命名: {stats['renamed']} 张")
    print(f"  跳过: {stats['skipped']} 张")
    print(f"  错误: {stats['errors']} 张")
    return stats


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("=== DRY RUN 模式 (不实际修改文件) ===\n")
    remap_images(dry_run=dry)
    if not dry:
        print("\n提示: 重命名完成后请重新运行 generate_annotations.py 更新标注文件")
