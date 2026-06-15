"""重命名图片为规范格式 {兵种编号}_{序号}.png

将 data/images/{id}/ 下的 QQ截图...png 等非规范命名的文件
重命名为 {id}_{序号}.png

用法:
    python training/scripts/rename_images.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "training" / "data" / "images"


def rename_images() -> None:
    """遍历所有兵种文件夹，将非规范命名的png文件重命名。"""
    if not IMAGES_DIR.exists():
        print(f"[ERROR] 图片目录不存在: {IMAGES_DIR}")
        sys.exit(1)

    subdirs = sorted(
        [d for d in IMAGES_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    total_renamed = 0

    for subdir in subdirs:
        creature_id = int(subdir.name)
        # 获取该文件夹下所有png文件，按原名排序以保证序号稳定
        png_files = sorted(subdir.glob("*.png"))

        # 分离已规范命名和待重命名的文件
        already_ok = set()
        to_rename = []

        for f in png_files:
            # 规范格式: {id}_{idx}.png
            parts = f.stem.split("_", 1)
            if len(parts) == 2 and parts[0] == subdir.name and parts[1].isdigit():
                already_ok.add(int(parts[1]))
            else:
                to_rename.append(f)

        if not to_rename:
            continue

        # 从已有序号之后开始编号
        next_idx = max(already_ok) + 1 if already_ok else 0

        for old_file in to_rename:
            new_name = f"{subdir.name}_{next_idx}.png"
            new_path = subdir / new_name
            # 防止覆盖已存在的文件
            while new_path.exists():
                next_idx += 1
                new_name = f"{subdir.name}_{next_idx}.png"
                new_path = subdir / new_name
            old_file.rename(new_path)
            next_idx += 1
            total_renamed += 1

        print(f"  [{subdir.name}/] 重命名 {len(to_rename)} 个文件")

    print(f"\n总计重命名: {total_renamed} 个文件")


if __name__ == "__main__":
    rename_images()
