"""兵种图像分类 — 模型微调训练脚本。

支持:
- CPU / GPU 自动检测, 混合精度训练(AMP)
- 多种模型: mobilenet_v3_large, efficientnet_b0, resnet18 等
- checkpoint保存, Early Stopping, 学习率调度(Warmup+Cosine)
- loss-epoch曲线, 准确率曲线, 混淆矩阵自动生成
- 小数据集友好 (Gradient Clipping, Label Smoothing, 数据增强)
- ONNX导出 (用于量化部署)

数据分割: 每次训练动态随机分割 (5:1 训练:验证), 不使用预分配CSV。

用法:
    cd heroesagent-training
    python scripts/train.py
    python scripts/train.py --model mobilenet_v3_large --epochs 50 --batch_size 64
    python scripts/train.py --device cpu --batch_size 16 --epochs 5 --no_amp
    python scripts/train.py --evaluate outputs/best_mobilenet_v3_large.pth
    python scripts/train.py --export_onnx outputs/best_mobilenet_v3_large.pth
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 将scripts目录加入sys.path, 支持直接导入同级模块 (不依赖父目录名)
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import CreatureDataset, get_default_transforms
from data_processor import (
    discover_images, random_train_val_split,
    save_annotations, save_summary, print_split_summary,
)
from model_factory import build_model
from train_loop import train_epoch, validate_epoch

# 项目根目录 (heroesagent-training/)
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DEFAULT_DATA_DIR / "annotations"
IMAGES_DIR = DEFAULT_DATA_DIR / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _setup_log(log_file: Optional[str]) -> object:
    """设置日志双写 (stdout + 文件), 返回文件句柄供最后关闭。"""
    if not log_file:
        return None
    f = open(log_file, "w", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class _Tee:
        def __init__(self, orig, file_):
            self.orig = orig
            self.file = file_
        def write(self, s):
            self.orig.write(s)
            self.orig.flush()
            self.file.write(s)
            self.file.flush()
        def flush(self):
            self.orig.flush()
            self.file.flush()

    sys.stdout = _Tee(orig_stdout, f)
    sys.stderr = _Tee(orig_stderr, f)
    return f


# ============================================================
# 主训练函数
# ============================================================

def train(args: argparse.Namespace) -> dict:
    """执行完整训练流程, 返回训练历史记录。

    数据流程:
      1. 扫描 images/ 目录发现所有图片
      2. 按5:1随机分割训练/验证集 (每类保证验证≥1)
      3. 构建模型并训练
    """
    # --- 日志 ---
    log_handle = _setup_log(getattr(args, "log_file", None))

    # --- 设备 ---
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[训练] 设备: {device}")
    use_amp = False
    if device.type == "cuda":
        print(f"[训练] GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[训练] 显存: {mem_gb:.1f}GB")
        use_amp = not args.no_amp

    # --- 数据发现 & 动态分割 ---
    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    print(f"\n[数据] 扫描图片目录: {images_dir}")
    all_samples = discover_images(str(images_dir))
    print(f"[数据] 发现 {len(all_samples)} 张图片, "
          f"{len(set(s['label_index'] for s in all_samples))} 个类别")

    train_anns, val_anns = random_train_val_split(
        all_samples, train_ratio=5 / 6, random_seed=args.seed,
    )
    print_split_summary(all_samples, train_anns, val_anns)

    # 保存本次分割结果 (可复现)
    # Kaggle input目录只读时自动回退到 output_dir/annotations/
    output_dir = Path(args.output_dir)
    save_dir = annotations_dir
    try:
        annotations_dir.mkdir(parents=True, exist_ok=True)
        save_annotations(train_anns, annotations_dir, "train")
        save_annotations(val_anns, annotations_dir, "val")
        save_annotations(all_samples, annotations_dir, "full")
        save_summary(all_samples, train_anns, val_anns, annotations_dir)
    except OSError:
        save_dir = output_dir / "annotations"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[数据] [WARN] annotations目录只读, 回退保存到: {save_dir}")
        save_annotations(train_anns, save_dir, "train")
        save_annotations(val_anns, save_dir, "val")
        save_annotations(all_samples, save_dir, "full")
        save_summary(all_samples, train_anns, val_anns, save_dir)

    # --- 总类别数 ---
    num_classes = CreatureDataset.get_total_classes(
        str(annotations_dir / "creature_index.xlsx")
    )
    print(f"[训练] 模型类别数: {num_classes} (来自creature_index.xlsx)")

    # --- 数据集 ---
    train_dataset = CreatureDataset.from_annotations(
        train_anns, str(images_dir),
        transform=get_default_transforms(train=True, input_size=args.input_size),
        target_size=args.input_size,
    )
    val_dataset = CreatureDataset.from_annotations(
        val_anns, str(images_dir),
        transform=get_default_transforms(train=False, input_size=args.input_size),
        target_size=args.input_size,
    )

    if len(train_dataset) < 50:
        print(f"[训练] [WARN] 训练集很小 ({len(train_dataset)}张), "
              f"建议使用数据增强扩充")

    train_loader = DataLoader(
        train_dataset, batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    ) if len(val_dataset) > 0 else None

    # --- 模型 ---
    model = build_model(args.model, num_classes, pretrained=not args.no_pretrain)
    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[训练] 模型: {args.model}, "
          f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")

    # --- 优化器 & 调度器 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 混合精度
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- 训练记录 ---
    history: dict = {
        "model": args.model,
        "device": str(device),
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "timestamp": datetime.now().isoformat(),
    }

    best_val_acc = 0.0
    best_model_path = output_dir / f"best_{args.model}.pth"
    output_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    print(f"\n[训练] 开始训练 ({args.epochs} epochs)...")
    print("-" * 65)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.clip_grad,
        )

        val_loss, val_acc = float("nan"), float("nan")
        if val_loader:
            val_loss, val_acc, _, _ = validate_epoch(
                model, val_loader, criterion, device,
            )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        t_elapsed = time.time() - t_start

        val_str = (f"V Loss: {val_loss:.4f} | V Acc: {val_acc:.2f}%"
                   if val_loader else "V: N/A")
        print(f"Epoch {epoch:3d}/{args.epochs} | LR: {lr_now:.1e} | "
              f"T Loss: {train_loss:.4f} | T Acc: {train_acc:.2f}% | "
              f"{val_str} | {t_elapsed:.1f}s")

        # checkpoint
        current_score = val_acc if val_loader else train_acc
        if current_score > best_val_acc:
            best_val_acc = current_score
            history["best_val_acc"] = current_score
            history["best_epoch"] = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "num_classes": num_classes,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc if val_loader else train_acc,
                "history": history,
            }, best_model_path)
            print(f"  -> 保存最佳模型 (Score: {current_score:.2f}%)")
        else:
            patience_counter += 1

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n[训练] Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"[训练] 完成! 总时间: {total_time / 60:.1f}min")
    print(f"[训练] 最佳分数: {best_val_acc:.2f}% (Epoch {history['best_epoch']})")
    print(f"[训练] 模型: {best_model_path}")

    # 保存训练历史
    history_path = output_dir / f"history_{args.model}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, ensure_ascii=False, indent=2, fp=f)

    # 生成图表
    plot_training_curves(history, args.model, output_dir)

    if log_handle:
        log_handle.close()

    return history


# ============================================================
# 图表生成
# ============================================================

def plot_training_curves(history: dict, model_name: str,
                        output_dir: Optional[Path] = None) -> None:
    """生成 loss-epoch 和 accuracy-epoch 图。"""
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss曲线
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
        if any(not (isinstance(v, float) and v != v) for v in history["val_loss"]):
            valid_val = [(i, v) for i, v in enumerate(history["val_loss"], 1)
                         if not (isinstance(v, float) and v != v)]
            if valid_val:
                ax1.plot([e for e, _ in valid_val], [v for _, v in valid_val],
                         "r-", label="Val Loss", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss Curve - {model_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy曲线
        ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
        if any(not (isinstance(v, float) and v != v) for v in history["val_acc"]):
            valid_val = [(i, v) for i, v in enumerate(history["val_acc"], 1)
                         if not (isinstance(v, float) and v != v)]
            if valid_val:
                ax2.plot([e for e, _ in valid_val], [v for _, v in valid_val],
                         "r-", label="Val Acc", linewidth=1.5)
        ax2.axhline(y=history.get("best_val_acc", 0), color="g", linestyle="--",
                    alpha=0.5, label=f"Best: {history.get('best_val_acc', 0):.1f}%")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Accuracy Curve - {model_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"training_curves_{model_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[训练] 图表已保存: {output_path}")

    except ImportError:
        print("[训练] matplotlib未安装,跳过图表生成")


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate_model(
    model_path: str, model_name: str, device: torch.device,
    input_size: int = 224, data_dir: str = "",
    output_dir: Optional[Path] = None,
) -> dict:
    """在验证集上评估已训练模型, 生成混淆矩阵。"""
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    ddir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    images_dir = ddir / "images"

    all_samples = discover_images(str(images_dir))
    _, val_anns = random_train_val_split(all_samples, train_ratio=5 / 6, random_seed=42)

    val_dataset = CreatureDataset.from_annotations(
        val_anns, str(images_dir),
        transform=get_default_transforms(train=False, input_size=input_size),
        target_size=input_size,
        num_classes=191,
    )
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = CreatureDataset.get_total_classes(
        str(ddir / "annotations" / "creature_index.xlsx")
    )
    model = build_model(model_name, num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = 100.0 * correct / total
    result = {
        "val_accuracy": acc,
        "val_samples": total,
        "val_correct": correct,
    }
    print(f"[评估] 验证准确率: {acc:.2f}% ({correct}/{total})")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"eval_{model_name}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, ensure_ascii=False, indent=2, fp=f)

    if total >= 10 and len(set(all_labels)) >= 3:
        try:
            plot_confusion_matrix(all_labels, all_preds, model_name, output_dir)
        except Exception as e:
            print(f"[评估] 混淆矩阵生成失败: {e}")

    return result


def plot_confusion_matrix(labels: list[int], preds: list[int], model_name: str,
                         output_dir: Optional[Path] = None) -> None:
    """生成混淆矩阵图。"""
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds)
        classes = sorted(set(labels))

        fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.3),
                                        max(10, len(classes) * 0.3)))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=6, color="white" if cm[i, j] > cm.max() / 2
                            else "black")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = output_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[评估] 混淆矩阵: {cm_path}")

    except ImportError:
        print("[评估] sklearn未安装,跳过混淆矩阵")


# ============================================================
# 导出ONNX (用于量化部署)
# ============================================================

def export_onnx(model_path: str, model_name: str,
                output_path: Optional[str] = None,
                output_dir: Optional[Path] = None) -> str:
    """导出模型为ONNX格式。

    Args:
        model_path: .pth checkpoint路径
        model_name: torchvision模型名
        output_path: 输出ONNX路径 (默认: output_dir/{model_name}.onnx)
        output_dir: 输出目录 (默认: OUTPUTS_DIR)
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    if output_path is None:
        output_path = str(output_dir / f"{model_name}.onnx")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint.get("history", {}).get("num_classes", 191)
    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    print(f"[导出] ONNX模型: {output_path}")
    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="英雄无敌3兵种图像分类 — 模型微调训练")
    parser.add_argument("--model", type=str, default="mobilenet_v3_large",
                        help="torchvision模型名 (默认: mobilenet_v3_large)")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"数据目录路径 (默认: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR),
                        help=f"输出目录 (模型/标注/图表). "
                             f"Kaggle上应设为 /kaggle/working/ (默认: {OUTPUTS_DIR})")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (None=每次不同)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="仅评估: 指定模型.pth路径")
    parser.add_argument("--export_onnx", type=str, default=None,
                        help="导出ONNX: 指定模型.pth路径")
    parser.add_argument("--log_file", type=str, default=None,
                        help="训练日志输出路径 (同时输出到stdout和文件)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.export_onnx:
        export_onnx(args.export_onnx, args.model, output_dir=output_dir)
    elif args.evaluate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluate_model(args.evaluate, args.model, device,
                       args.input_size, args.data_dir, output_dir=output_dir)
    else:
        train(args)
