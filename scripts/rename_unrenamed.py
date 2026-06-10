"""
将未改名的QQ截图重命名为统一格式: {label_index}_{idx}.png

用法:
    cd F:/桌面/test3 && python training/scripts/rename_unrenamed.py [--dry-run]
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = Path(__file__).parent.parent  # 训练模块根目录 (文件夹名无关)
IMAGES_DIR = TRAINING_ROOT / "data" / "images"


def rename_folder(folder: Path, label_idx: int, dry_run: bool = False) -> int:
    """重命名文件夹中的所有图片为 {label_idx}_{idx}.png 格式。"""
    existing = sorted(folder.glob("*.png"))
    if not existing:
        return 0

    renamed_count = 0
    # 找出已经是标准格式的文件的最大索引
    max_existing_idx = -1
    for f in existing:
        m = re.match(rf"^{label_idx}_(\d+)\.png$", f.name)
        if m:
            max_existing_idx = max(max_existing_idx, int(m.group(1)))

    next_idx = max_existing_idx + 1
    renamed: list[tuple[str, str]] = []

    for f in existing:
        # 检查是否已经是标准格式
        if re.match(rf"^{label_idx}_\d+\.png$", f.name):
            continue  # 已经命名好,跳过
        # 需要重命名
        new_name = f"{label_idx}_{next_idx}.png"
        new_path = folder / new_name
        renamed.append((f.name, new_name))
        next_idx += 1

        if not dry_run:
            f.rename(new_path)
        renamed_count += 1

    if renamed:
        print(f"  文件夹 {label_idx}/: 重命名 {len(renamed)} 个文件")
        if dry_run:
            for old, new in renamed[:5]:
                print(f"    {old} -> {new}")
            if len(renamed) > 5:
                print(f"    ... 共 {len(renamed)} 个")
    else:
        print(f"  文件夹 {label_idx}/: 全部已命名,无需修改")

    return renamed_count


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY RUN 模式 (不实际修改文件) ===\n")

    if not IMAGES_DIR.exists():
        print(f"[ERROR] 图片目录不存在: {IMAGES_DIR}")
        sys.exit(1)

    # 获取所有数字命名子文件夹
    subdirs = sorted(
        [d for d in IMAGES_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    print(f"找到 {len(subdirs)} 个兵种文件夹: {[d.name for d in subdirs]}")

    total_renamed = 0
    for subdir in subdirs:
        label_idx = int(subdir.name)
        count = rename_folder(subdir, label_idx, dry_run)
        total_renamed += count

    print(f"\n总计重命名: {total_renamed} 个文件")

    if not dry_run and total_renamed > 0:
        print("重命名完成! 请重新运行 generate_annotations.py 更新标注文件。")


if __name__ == "__main__":
    main()
