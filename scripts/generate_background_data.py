"""为障碍物(189)和空地(190)类别生成合成训练图片。

从战场样式的空地区域裁剪样本，或生成合成数据。
每类生成约60-70张图片用于训练。
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
IMAGES_DIR = TRAINING_ROOT / "data" / "images"

# 战场颜色方案
EMPTY_COLORS = [
    (76, 115, 60), (82, 120, 65), (70, 110, 55), (88, 125, 70),
    (79, 118, 63), (85, 122, 68), (73, 112, 58), (90, 128, 72),
    (68, 108, 52), (94, 130, 75), (77, 116, 61), (84, 121, 66),
    (95, 132, 77), (72, 111, 56), (80, 119, 64), (87, 124, 69),
    (92, 129, 73), (75, 114, 59), (83, 121, 67), (89, 126, 71),
    (78, 117, 62), (86, 123, 68), (91, 127, 74), (71, 109, 54),
]

OBSTACLE_COLORS = [
    (85, 75, 60), (90, 80, 65), (80, 70, 55), (95, 85, 70),
    (100, 90, 75), (78, 68, 53), (88, 78, 63), (92, 82, 67),
    (82, 72, 57), (97, 87, 72), (75, 65, 50), (84, 74, 59),
    (93, 83, 68), (79, 69, 54), (87, 77, 62), (94, 84, 69),
    (81, 71, 56), (89, 79, 64), (96, 86, 71), (77, 67, 52),
]


def create_empty_image(size: int = 224, seed: int = 0) -> Image.Image:
    """生成空地类图片: 绿色基底+细微噪声,模拟空六边形格。"""
    rng = np.random.RandomState(seed)
    base_color = np.array(EMPTY_COLORS[rng.randint(0, len(EMPTY_COLORS))])

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        noise = rng.randint(-15, 15, (size, size)).astype(np.int16)
        channel = base_color[c] + noise
        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # 添加细微网格线
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    line_color = tuple(max(0, c - 20) for c in base_color)

    # 随机画一些淡线模拟六边形边界
    for _ in range(rng.randint(0, 3)):
        if rng.random() > 0.5:
            x = rng.randint(10, size - 10)
            draw.line([(x, 0), (x, size)], fill=line_color, width=1)
        else:
            y = rng.randint(10, size - 10)
            draw.line([(0, y), (size, y)], fill=line_color, width=1)

    return img_pil.filter(ImageFilter.GaussianBlur(radius=0.3))


def create_obstacle_image(size: int = 224, seed: int = 0) -> Image.Image:
    """生成障碍物类图片: 暗色基底+粗糙纹理,模拟岩石/树木等障碍物。"""
    rng = np.random.RandomState(seed)
    base_color = np.array(OBSTACLE_COLORS[rng.randint(0, len(OBSTACLE_COLORS))])

    img = np.zeros((size, size, 3), dtype=np.uint8)

    # 较粗糙的纹理
    noise_scale = rng.randint(2, 5)
    small_noise = rng.randint(-25, 25, (size // noise_scale, size // noise_scale))
    img_small = np.zeros((size // noise_scale, size // noise_scale, 3), dtype=np.uint8)
    for c in range(3):
        img_small[:, :, c] = np.clip(base_color[c] + small_noise, 0, 255)

    img_pil = Image.fromarray(img_small).resize((size, size), Image.NEAREST)

    # 添加不规则形状模拟障碍物
    draw = ImageDraw.Draw(img_pil)
    for _ in range(rng.randint(2, 6)):
        cx = rng.randint(size // 4, 3 * size // 4)
        cy = rng.randint(size // 4, 3 * size // 4)
        rx = rng.randint(15, size // 3)
        ry = rng.randint(10, size // 4)
        shape_color = tuple(max(0, min(255, c + rng.randint(-30, 10))) for c in base_color)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=shape_color)

    # 添加一些不规则边缘
    img_arr = np.array(img_pil)
    edge_noise = rng.randint(-10, 10, (size, size, 3)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + edge_noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr).filter(ImageFilter.GaussianBlur(radius=0.5))


def generate_data(num_samples: int = 65):
    """为两类各生成num_samples张训练图片。"""

    for class_id, class_name, create_func in [
        (189, "obstacle", create_obstacle_image),
        (190, "empty", create_empty_image),
    ]:
        out_dir = IMAGES_DIR / str(class_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            img = create_func(seed=class_id * 1000 + i)
            img_path = out_dir / f"{class_id}_{i}.png"
            img.save(img_path)

        print(f"[{class_name}] 生成 {num_samples} 张图片 -> {out_dir}")


if __name__ == "__main__":
    generate_data()
    print("障碍物/空地训练图片生成完成!")
