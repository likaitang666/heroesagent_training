"""PyTorch Dataset — 加载兵种图片及标签。

支持从CSV标注文件读取,兼容train/val/test split。
图片自动resize到模型输入尺寸,支持数据增强。
"""

import csv
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image


class CreatureDataset(Dataset):
    """兵种图像分类数据集。

    Args:
        csv_path: 标注CSV文件路径 (含 image, label_index 列)
        images_dir: 图片所在目录
        transform: torchvision transforms (训练时含增强,验证/测试时仅标准化)
        target_size: 模型输入尺寸, 默认224

    CSV格式:
        image,label_index,name_en,name_zh,faction,level,is_upgraded
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform: Optional[Callable] = None,
        target_size: int = 224,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size

        self.samples: list[dict] = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image": row["image"],
                    "label_index": int(row["label_index"]),
                    "name_en": row.get("name_en", ""),
                    "name_zh": row.get("name_zh", ""),
                    "faction": row.get("faction", ""),
                })

        if not self.samples:
            raise ValueError(f"CSV文件无有效数据: {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img_path = self.images_dir / sample["image"]

        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        label = sample["label_index"]
        return image, label

    def get_class_distribution(self) -> dict[int, int]:
        """统计每类样本数。"""
        from collections import Counter
        return Counter(s["label_index"] for s in self.samples)

    @property
    def num_classes(self) -> int:
        return len(set(s["label_index"] for s in self.samples))


def get_default_transforms(train: bool = True, input_size: int = 224):
    """获取默认数据增强pipeline。

    Args:
        train: True返回训练增强, False返回验证标准化
        input_size: 输入尺寸
    """
    import torchvision.transforms as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
