"""PyTorch Dataset — 加载兵种图片及标签。

支持从CSV标注文件读取, 兼容train/val/test split。
图片自动resize到模型输入尺寸, 支持数据增强。
图片路径为相对于images_dir的相对路径 (如 "0/0_0.png")。
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
        images_dir: 图片所在根目录 (内含 0/, 1/, ... 子文件夹)
        transform: torchvision transforms (训练时含增强,验证/测试时仅标准化)
        target_size: 模型输入尺寸, 默认224

    CSV格式:
        image,label_index,name_en,name_zh,faction,level,is_upgraded
        image 列存储相对路径如 "0/0_0.png"
    """

    # 总类别数 (189兵种 + 障碍物 + 空地)
    TOTAL_CLASSES = 191

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform: Optional[Callable] = None,
        target_size: int = 224,
        num_classes: int = 191,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size
        self._num_classes = num_classes

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

    @classmethod
    def from_annotations(
        cls,
        annotations: list[dict],
        images_dir: str,
        transform: Optional[Callable] = None,
        target_size: int = 224,
    ) -> "CreatureDataset":
        """从标注列表直接创建数据集 (无需CSV文件)。

        Args:
            annotations: [{"image": "0/0_0.png", "label_index": 0}, ...]
            images_dir: 图片所在根目录
            transform: torchvision transforms
            target_size: 模型输入尺寸
        """
        dataset = cls.__new__(cls)
        dataset.images_dir = Path(images_dir)
        dataset.transform = transform
        dataset.target_size = target_size
        dataset.samples = [
            {
                "image": a["image"],
                "label_index": int(a["label_index"]),
                "name_en": a.get("name_en", ""),
                "name_zh": a.get("name_zh", ""),
                "faction": a.get("faction", ""),
            }
            for a in annotations
        ]
        if not dataset.samples:
            raise ValueError("标注列表为空")
        return dataset

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
        """总类别数 (191 = 189兵种 + 障碍物 + 空地)。"""
        return self._num_classes

    @staticmethod
    def get_total_classes(xlsx_path: Optional[str] = None) -> int:
        """从creature_index.xlsx获取总类别数 (默认191)。"""
        if xlsx_path is None:
            xlsx_path = str(
                Path(__file__).parent.parent / "data" / "annotations" / "creature_index.xlsx"
            )
        try:
            import openpyxl
            wb = openpyxl.load_workbook(xlsx_path, read_only=True)
            ws = wb.active
            count = sum(1 for _ in ws.iter_rows(min_row=2, values_only=True))
            wb.close()
            return max(count, 1)
        except Exception:
            return len(set(s["label_index"] for s in self.samples))


def get_default_transforms(train: bool = True, input_size: int = 224):
    """获取默认数据增强pipeline。

    训练增强: 随机裁剪+缩放, 水平翻转, 颜色抖动, 随机擦除(遮挡)
    验证: 仅resize+标准化

    注意: 不做垂直翻转 (兵种不会倒立)

    Args:
        train: True返回训练增强, False返回验证标准化
        input_size: 输入尺寸
    """
    import torchvision.transforms as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
