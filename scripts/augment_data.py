"""数据增强脚本 — 对小样本类别进行扩充。

对图片数量不足的类别进行数据增强, 使每个类别达到目标图片数。
增强方法 (遵循DESIGN.md): 高斯模糊, 亮度调整, 左右翻转 (不做上下翻转)。

用法:
    python scripts/augment_data.py
    python scripts/augment_data.py --min_count 30 --target_dir data/images
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images"


def augment_image(img: Image.Image, variant: int) -> Image.Image:
    """对单张图片应用一种随机增强变换。

    增强类型: 水平翻转, 高斯模糊, 亮度调整, 组合增强
    不做垂直翻转和旋转 (兵种不会倒立/倾斜)。

    Args:
        img: PIL Image
        variant: 变换变体编号 (0-7)

    Returns:
        增强后的图片
    """
    img = img.copy()

    if variant == 0:
        # 水平翻转
        img = ImageOps.mirror(img)
    elif variant == 1:
        # 轻度高斯模糊 (radius=0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif variant == 2:
        # 中度高斯模糊 (radius=1.3)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.3))
    elif variant == 3:
        # 亮度提升
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.25)
    elif variant == 4:
        # 亮度降低
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.75)
    elif variant == 5:
        # 翻转 + 轻度模糊
        img = ImageOps.mirror(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    elif variant == 6:
        # 翻转 + 亮度提升
        img = ImageOps.mirror(img)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
    elif variant == 7:
        # 翻转 + 亮度降低
        img = ImageOps.mirror(img)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)

    return img


def augment_class_directory(
    class_dir: Path,
    class_id: int,
    min_count: int = 20,
) -> int:
    """对单个类别的图片目录进行增强扩充。

    Args:
        class_dir: 类别图片目录
        class_id: 类别编号
        min_count: 目标最少图片数

    Returns:
        新增图片数量
    """
    existing = sorted(class_dir.glob("*.png"))
    current_count = len(existing)

    if current_count >= min_count:
        return 0

    # 找到当前最大序号
    max_idx = 0
    for f in existing:
        parts = f.stem.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            max_idx = max(max_idx, int(parts[1]))

    needed = min_count - current_count
    variants = list(range(8))  # 8种增强变体

    generated = 0
    next_idx = max_idx + 1
    random.seed(42)  # 固定种子保证可复现

    while generated < needed:
        for src_img_path in existing:
            if generated >= needed:
                break
            # 随机选择原图和增强变体
            variant = variants[generated % len(variants)]
            try:
                src_img = Image.open(src_img_path).convert("RGB")
                aug_img = augment_image(src_img, variant)
                out_name = f"{class_id}_{next_idx}.png"
                out_path = class_dir / out_name
                aug_img.save(out_path, "PNG")
                next_idx += 1
                generated += 1
            except Exception as e:
                print(f"  [WARN] 增强失败 {src_img_path.name}: {e}")
                continue

    return generated


def augment_all(
    images_dir: str | Path,
    min_count: int = 20,
    dry_run: bool = False,
) -> dict[int, int]:
    """扫描所有类别并增强不足的类别。

    Args:
        images_dir: 图片根目录
        min_count: 目标最少图片数
        dry_run: 仅统计不执行

    Returns:
        {class_id: 新增图片数}
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_path}")

    subdirs = sorted(
        [d for d in images_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    results: dict[int, int] = {}
    total_before = 0
    total_after = 0

    for subdir in subdirs:
        class_id = int(subdir.name)
        current = len(list(subdir.glob("*.png")))
        total_before += current

        if current < min_count:
            needed = min_count - current
            if dry_run:
                print(f"  [{class_id}/] {current}张 → 需要+{needed}张")
                results[class_id] = needed
                total_after += min_count
            else:
                added = augment_class_directory(subdir, class_id, min_count)
                results[class_id] = added
                total_after += current + added
                if added > 0:
                    print(f"  [{class_id}/] {current}张 → {current + added}张 (+{added})")
        else:
            total_after += current

    if dry_run:
        print(f"\n总计: {total_before}张 → {total_after}张 "
              f"({len(results)}个类别需增强)")
    else:
        print(f"\n总计: {total_before}张 → {total_after}张 "
              f"({len(results)}个类别已增强)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="兵种图片数据增强")
    parser.add_argument("--min_count", type=int, default=20,
                        help="目标最少图片数 (默认: 20)")
    parser.add_argument("--target_dir", type=str, default=str(DEFAULT_IMAGES_DIR),
                        help="图片目录路径")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅统计不执行")
    args = parser.parse_args()

    if args.dry_run:
        print("[数据增强] 预览模式\n")
    else:
        print("[数据增强] 开始执行\n")
        sys.stdout.reconfigure(encoding='utf-8')

    augment_all(args.target_dir, args.min_count, args.dry_run)
