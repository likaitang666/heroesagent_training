"""兵种图片数据增强脚本 — 从现有图片生成变体以扩充训练集。

用法:
    # 为每张图片生成5个变体
    python training/scripts/augment_images.py --variations 5

    # 指定输入输出目录
    python training/scripts/augment_images.py --input training/data/images/ --output training/data/images/ --variations 10

    # 预览模式(不实际保存)
    python training/scripts/augment_images.py --variations 3 --dry_run

增强策略:
    - 随机旋转 (±15度)
    - 水平翻转
    - 亮度/对比度调整
    - 轻微缩放
    - 高斯噪声
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
DEFAULT_INPUT = TRAINING_ROOT / "data" / "images"
DEFAULT_OUTPUT = TRAINING_ROOT / "data" / "images"

sys.path.insert(0, str(PROJECT_ROOT))


def augment_image(img: Image.Image, variation: int) -> Image.Image:
    """对单张图片应用一种随机增强。

    Args:
        img: PIL Image (RGB)
        variation: 变体编号(决定增强类型)

    Returns:
        增强后的图片
    """
    seed = variation * 42
    random.seed(seed)

    # 水平翻转 (50%概率)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 旋转 (-15 ~ +15度)
    angle = random.uniform(-15, 15)
    if abs(angle) > 1:
        img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    # 亮度调整 (0.8 ~ 1.3)
    brightness = random.uniform(0.8, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 对比度调整 (0.8 ~ 1.2)
    contrast = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 饱和度调整 (0.8 ~ 1.2)
    saturation = random.uniform(0.8, 1.2)
    img = ImageEnhance.Color(img).enhance(saturation)

    # 轻微缩放 (0.9 ~ 1.0, 即从90%区域裁剪)
    if random.random() < 0.3:
        w, h = img.size
        scale = random.uniform(0.85, 0.95)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.BILINEAR)

    return img


def augment_directory(
    input_dir: Path,
    output_dir: Path,
    num_variations: int,
    dry_run: bool = False,
) -> dict:
    """对目录中所有图片生成增强变体。

    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录(可与input相同)
        num_variations: 每张图片生成的变体数量
        dry_run: 不实际保存文件

    Returns:
        统计信息字典
    """
    png_files = sorted(input_dir.glob("*.png"))
    jpg_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    all_files = png_files + jpg_files

    if not all_files:
        print("[ERROR] 未找到图片文件")
        return {"total_original": 0, "total_generated": 0}

    print(f"找到 {len(all_files)} 张原始图片")
    generated = 0
    skipped = 0
    errors = 0

    for img_path in all_files:
        stem = img_path.stem
        # 跳过已经是增强变体的文件
        if "_aug" in stem or "_var" in stem:
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] 无法读取 {img_path.name}: {e}")
            errors += 1
            continue

        for v in range(num_variations):
            try:
                augmented = augment_image(img, v)
                out_name = f"{stem}_aug{v+1:02d}.png"
                out_path = output_dir / out_name

                if not dry_run:
                    augmented.save(out_path, "PNG")
                generated += 1
            except Exception as e:
                print(f"  [ERROR] 增强失败 {img_path.name} var{v}: {e}")
                errors += 1

        if generated % 50 == 0 and generated > 0:
            print(f"  已生成 {generated} 张...")

    result = {
        "total_original": len(all_files),
        "total_generated": generated,
        "skipped": skipped,
        "errors": errors,
    }

    print(f"\n完成:")
    print(f"  原始图片: {result['total_original']} 张")
    print(f"  跳过(已是变体): {skipped} 张")
    print(f"  新生成变体: {generated} 张")
    print(f"  错误: {errors} 张")
    if not dry_run:
        print(f"  输出目录: {output_dir}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="兵种图片数据增强")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help=f"输入图片目录 (默认: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"输出图片目录 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--variations", type=int, default=5,
                        help="每张图片生成的变体数量 (默认: 5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="预览模式,不实际保存文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("兵种图片数据增强")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"  每张变体数: {args.variations}")
    print(f"  模式: {'预览(不保存)' if args.dry_run else '正式'}")
    print("=" * 60)

    augment_directory(input_dir, output_dir, args.variations, args.dry_run)

    if not args.dry_run:
        print("\n提示: 增强完成后请重新运行 generate_annotations.py 更新标注文件")


if __name__ == "__main__":
    main()
