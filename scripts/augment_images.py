"""兵种图片数据增强脚本 — 为训练集图片生成增强变体。

增强策略 (仅训练集):
    - 水平翻转 (左右镜像, 50%概率)
    - 随机旋转 (±20度)
    - 随机裁剪+拉伸回原尺寸
    - 随机遮挡 (RandomErasing区域)
    - 亮度/对比度/饱和度调整
    - 轻微拉伸变形 (perspective-like scale jitter)
    注意: 不做上下翻转 (兵种不会倒立)

用法:
    # 基础用法 — 为每张训练图生成9个变体
    python training/scripts/augment_images.py --variations 9

    # 仅预览
    python training/scripts/augment_images.py --variations 3 --dry_run

    # 指定输入输出目录
    python training/scripts/augment_images.py --input training/data/images --output training/data/augmented --variations 9
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = Path(__file__).parent.parent  # 训练模块根目录 (文件夹名无关)
DEFAULT_INPUT = TRAINING_ROOT / "data" / "images"
DEFAULT_OUTPUT = TRAINING_ROOT / "data" / "augmented"

sys.path.insert(0, str(TRAINING_ROOT / "scripts"))


def add_random_occlusion(img: Image.Image) -> Image.Image:
    """在图片上添加随机遮挡块 (模拟战斗中部分遮挡)。"""
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    num_blocks = random.randint(1, 3)
    for _ in range(num_blocks):
        # 遮挡块尺寸: 图片的10%-25%
        bw = random.randint(int(w * 0.08), int(w * 0.25))
        bh = random.randint(int(h * 0.08), int(h * 0.25))
        x = random.randint(0, w - bw)
        y = random.randint(0, h - bh)
        # 随机灰色块模拟遮挡
        color = random.randint(20, 80)
        draw.rectangle([x, y, x + bw, y + bh], fill=(color, color, color))

    return img_copy


def random_stretch(img: Image.Image) -> Image.Image:
    """随机拉伸变形 — 对宽或高做不均匀缩放。"""
    w, h = img.size
    # 对宽高分别做0.85-1.15的缩放,模拟不同截图视角
    scale_w = random.uniform(0.85, 1.15)
    scale_h = random.uniform(0.85, 1.15)
    new_w = max(16, int(w * scale_w))
    new_h = max(16, int(h * scale_h))
    stretched = img.resize((new_w, new_h), Image.BILINEAR)
    # 裁剪或填充回原尺寸
    if new_w > w:
        left = (new_w - w) // 2
        stretched = stretched.crop((left, 0, left + w, new_h))
    elif new_w < w:
        padded = Image.new("RGB", (w, new_h), (0, 0, 0))
        left = (w - new_w) // 2
        padded.paste(stretched, (left, 0))
        stretched = padded
    if new_h > h:
        top = (new_h - h) // 2
        stretched = stretched.crop((0, top, w, top + h))
    elif new_h < h:
        padded = Image.new("RGB", (w, h), (0, 0, 0))
        top = (h - new_h) // 2
        padded.paste(stretched, (0, top))
        stretched = padded

    return stretched.resize((w, h), Image.BILINEAR)


def random_crop_and_resize(img: Image.Image) -> Image.Image:
    """随机裁剪一部分后拉伸回原尺寸 (模拟不同截图范围)。"""
    w, h = img.size
    crop_scale = random.uniform(0.75, 0.95)
    crop_w = int(w * crop_scale)
    crop_h = int(h * crop_scale)
    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.BILINEAR)


def augment_image(img: Image.Image, variation: int) -> Image.Image:
    """对单张图片应用一种随机增强组合。

    Args:
        img: PIL Image (RGB)
        variation: 变体编号(决定增强随机种子)

    Returns:
        增强后的图片
    """
    seed = variation * 42 + hash(variation) % 10000
    random.seed(seed)

    # 1. 水平翻转 (50%概率) — 不做垂直翻转
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. 旋转 (-20 ~ +20度)
    angle = random.uniform(-20, 20)
    if abs(angle) > 1:
        img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    # 3. 亮度调整 (0.75 ~ 1.35)
    brightness = random.uniform(0.75, 1.35)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 4. 对比度调整 (0.75 ~ 1.25)
    contrast = random.uniform(0.75, 1.25)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 5. 饱和度调整 (0.7 ~ 1.3)
    saturation = random.uniform(0.7, 1.3)
    img = ImageEnhance.Color(img).enhance(saturation)

    # 6. 随机裁剪+拉伸 (40%概率)
    if random.random() < 0.4:
        img = random_crop_and_resize(img)

    # 7. 随机遮挡 (30%概率)
    if random.random() < 0.3:
        img = add_random_occlusion(img)

    # 8. 拉伸变形 (25%概率)
    if random.random() < 0.25:
        img = random_stretch(img)

    # 9. 轻微锐化 (20%概率)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.SHARPEN)

    return img


def augment_subfolder(
    input_subdir: Path,
    output_subdir: Path,
    num_variations: int,
    dry_run: bool = False,
) -> dict:
    """对单个兵种子文件夹中的所有图片生成增强变体。

    Args:
        input_subdir: 输入图片子文件夹 (如 data/images/0/)
        output_subdir: 输出子文件夹
        num_variations: 每张图片生成的变体数
        dry_run: 预览模式

    Returns:
        该子文件夹的统计信息
    """
    png_files = sorted(input_subdir.glob("*.png"))
    jpg_files = (sorted(input_subdir.glob("*.jpg")) +
                 sorted(input_subdir.glob("*.jpeg")))
    all_files = png_files + jpg_files

    if not all_files:
        return {"original": 0, "generated": 0, "skipped": 0, "errors": 0}

    label_name = input_subdir.name
    generated = 0
    skipped = 0
    errors = 0

    for img_path in all_files:
        stem = img_path.stem
        # 跳过已经是增强变体的文件
        if "_aug" in stem or stem.endswith("_var"):
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] 无法读取 {label_name}/{img_path.name}: {e}")
            errors += 1
            continue

        for v in range(num_variations):
            try:
                augmented = augment_image(img, v)
                out_name = f"{stem}_aug{v+1:02d}.png"
                out_path = output_subdir / out_name

                if not dry_run:
                    augmented.save(out_path, "PNG")
                generated += 1
            except Exception as e:
                print(f"  [ERROR] 增强失败 {label_name}/{img_path.name} var{v}: {e}")
                errors += 1

    return {
        "original": len(all_files),
        "generated": generated,
        "skipped": skipped,
        "errors": errors,
    }


def augment_directory(
    input_dir: Path,
    output_dir: Path,
    num_variations: int,
    dry_run: bool = False,
) -> dict:
    """对目录中所有兵种子文件夹生成增强变体。

    Args:
        input_dir: 输入根目录 (含 0/, 1/, 2/... 子文件夹)
        output_dir: 输出根目录
        num_variations: 每张原始图片生成的变体数量
        dry_run: 不实际保存文件

    Returns:
        统计信息字典
    """
    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return {"total_original": 0, "total_generated": 0}

    # 获取所有数字命名子文件夹
    subdirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if not subdirs:
        print("[ERROR] 未找到兵种子文件夹 (如 0/, 1/, ...)")
        return {"total_original": 0, "total_generated": 0}

    total_original = 0
    total_generated = 0
    total_skipped = 0
    total_errors = 0

    for subdir in subdirs:
        out_subdir = output_dir / subdir.name
        if not dry_run:
            out_subdir.mkdir(parents=True, exist_ok=True)

        result = augment_subfolder(subdir, out_subdir, num_variations, dry_run)
        total_original += result["original"]
        total_generated += result["generated"]
        total_skipped += result["skipped"]
        total_errors += result["errors"]

        if result["original"] > 0:
            status = f"+{result['generated']}张" if result["generated"] > 0 else "无变化"
            print(f"  [{subdir.name}/] {result['original']}张原始 -> {status}"
                  + (f" ({result['errors']}错误)" if result["errors"] else ""))

    result = {
        "total_original": total_original,
        "total_generated": total_generated,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
    }

    print(f"\n完成:")
    print(f"  原始图片: {total_original} 张")
    print(f"  跳过: {total_skipped} 张")
    print(f"  新生成变体: {total_generated} 张")
    if total_errors:
        print(f"  错误: {total_errors} 张")
    if not dry_run:
        print(f"  输出目录: {output_dir}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="兵种图片数据增强")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help=f"输入图片根目录 (默认: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"输出图片根目录 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--variations", type=int, default=9,
                        help="每张图片生成的变体数量 (默认: 9)")
    parser.add_argument("--dry_run", action="store_true",
                        help="预览模式,不实际保存文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print("=" * 60)
    print("兵种图片数据增强")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"  每张变体数: {args.variations}")
    print(f"  模式: {'预览(不保存)' if args.dry_run else '正式'}")
    print(f"  增强策略: 水平翻转 | 旋转±20° | 裁剪拉伸 | 遮挡 | 伸缩变形")
    print("=" * 60)

    augment_directory(input_dir, output_dir, args.variations, args.dry_run)

    if not dry_run:
        print(f"\n提示: 增强完成后请重新运行 generate_annotations.py 生成标注文件")


if __name__ == "__main__":
    main()
