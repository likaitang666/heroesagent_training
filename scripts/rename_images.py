"""将兵种图片从 creature_{name}.png 重命名为 {label_index}_{index}.png 格式。

用法:
    cd F:/桌面/test3 && python training/scripts/rename_images.py [--dry-run]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGES_SRC = PROJECT_ROOT / "images" / "creatures"
IMAGES_DST = PROJECT_ROOT / "training" / "data" / "images"
LABELS_FILE = PROJECT_ROOT / "gamedata" / "creature_labels.json"


def _normalize(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def match_image_to_label(
    image_stem: str, labels: list[dict]
) -> dict | None:
    """匹配图片文件名到标签,使用最长匹配策略。"""
    norm_stem = _normalize(image_stem)
    best_label: dict | None = None
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


def rename_images(dry_run: bool = False) -> dict:
    """重命名所有兵种图片为 {label_index}_{index}.png 格式。"""
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data["labels"]

    image_files = sorted(IMAGES_SRC.glob("*.png"))
    per_class: dict[int, list[Path]] = defaultdict(list)

    unmatched: list[str] = []
    for img_path in image_files:
        label = match_image_to_label(img_path.stem, labels)
        if label:
            per_class[label["label"]].append(img_path)
        else:
            unmatched.append(img_path.name)

    renamed: list[tuple[str, str]] = []
    for label_idx in sorted(per_class.keys()):
        for i, img_path in enumerate(per_class[label_idx]):
            ext = img_path.suffix
            new_name = f"{label_idx}_{i}{ext}"
            new_path = img_path.parent / new_name
            renamed.append((img_path.name, new_name))
            if not dry_run:
                img_path.rename(new_path)

    if unmatched:
        print(f"[WARN] {len(unmatched)} 张图片未匹配: {unmatched}")

    print(f"[OK] 重命名 {len(renamed)} 张图片 ({'DRY RUN' if dry_run else 'DONE'})")
    if dry_run:
        for old, new in renamed[:10]:
            print(f"  {old} -> {new}")
        if len(renamed) > 10:
            print(f"  ... 共 {len(renamed)} 张")

    # 复制到训练目录
    if not dry_run:
        IMAGES_DST.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img_path in sorted(IMAGES_SRC.glob("*.png")):
            dst = IMAGES_DST / img_path.name
            if not dst.exists():
                import shutil
                shutil.copy2(img_path, dst)
                copied += 1
        print(f"[OK] 复制 {copied} 张图片到: {IMAGES_DST}")

    return {
        "renamed": len(renamed),
        "unmatched": len(unmatched),
        "classes": len(per_class),
    }


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("=== DRY RUN 模式(不实际修改文件) ===\n")
    result = rename_images(dry_run=dry)
    print(f"\n统计: {result['classes']} 个兵种, "
          f"{result['renamed']} 张图片, "
          f"{result['unmatched']} 张未匹配")
