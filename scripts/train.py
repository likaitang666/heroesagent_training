"""兵种图像分类 — 模型微调训练脚本。

支持:
- CPU / GPU 自动检测, 混合精度训练(AMP)
- 多种模型: efficientnet_b0, mobilenet_v3_large, resnet18 等
- checkpoint保存, Early Stopping, 学习率调度
- loss-epoch曲线, 准确率曲线, 混淆矩阵自动生成
- 小数据集友好 (Gradient Clipping, Label Smoothing, 数据增强)
- ONNX导出 (用于量化部署)

用法:
    # 基础训练
    python training/scripts/train.py

    # 指定模型和参数
    python training/scripts/train.py --model efficientnet_b0 --epochs 50 --batch_size 64

    # CPU训练 (仅测试流程)
    python training/scripts/train.py --device cpu --batch_size 16 --epochs 5

    # 仅评估已有模型
    python training/scripts/train.py --evaluate training/outputs/best_efficientnet_b0.pth

    # 导出ONNX
    python training/scripts/train.py --export_onnx training/outputs/best_efficientnet_b0.pth
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = Path(__file__).parent.parent  # 训练模块根目录 (文件夹名无关)
DEFAULT_DATA_DIR = TRAINING_ROOT / "data"
OUTPUTS_DIR = TRAINING_ROOT / "outputs"

sys.path.insert(0, str(TRAINING_ROOT / "scripts"))

from dataset import CreatureDataset, get_default_transforms


# ============================================================
# 模型工厂
# ============================================================

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """构建分类模型,自动替换分类头。

    Args:
        model_name: torchvision模型名
        num_classes: 分类数
        pretrained: 是否使用ImageNet预训练权重

    Returns:
        替换了分类头的模型
    """
    import torchvision.models as models

    model_func = getattr(models, model_name, None)
    if model_func is None:
        available = [m for m in dir(models)
                     if m[0].islower() and not m.startswith("_")]
        raise ValueError(f"未知模型: {model_name}\n可用: {', '.join(available)}")

    weights = "IMAGENET1K_V1" if pretrained else None
    model = model_func(weights=weights)

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"无法找到模型分类头: {model_name}")

    return model


# ============================================================
# 训练/验证循环
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional["torch.amp.GradScaler"] = None,
    clip_grad: float = 1.0,
) -> tuple[float, float]:
    """训练一个epoch, 返回(loss, accuracy)。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """验证一个epoch, 返回(loss, accuracy, predictions, labels)。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


# ============================================================
# 主训练函数
# ============================================================

def train(args: argparse.Namespace) -> dict:
    """执行完整训练流程,返回训练历史记录。"""
    # --- 设备 ---
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[训练] 设备: {device}")
    use_amp = False
    if device.type == "cuda":
        print(f"[训练] GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[训练] 显存: {mem_gb:.1f}GB")
        use_amp = not args.no_amp

    # --- 数据集 ---
    data_dir = Path(args.data_dir)
    train_csv = data_dir / "annotations" / "train.csv"
    val_csv = data_dir / "annotations" / "val.csv"
    images_dir = data_dir / "images"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"训练标注不存在: {train_csv}\n"
            f"请先运行: python training/scripts/generate_annotations.py"
        )

    train_dataset = CreatureDataset(
        str(train_csv), str(images_dir),
        transform=get_default_transforms(train=True, input_size=args.input_size),
        target_size=args.input_size,
    )

    val_exists = val_csv.exists()
    if val_exists:
        val_dataset = CreatureDataset(
            str(val_csv), str(images_dir),
            transform=get_default_transforms(train=False, input_size=args.input_size),
            target_size=args.input_size,
        )
    else:
        val_dataset = None

    num_classes = train_dataset.num_classes
    print(f"[训练] 训练集: {len(train_dataset)} 张, "
          f"验证集: {len(val_dataset) if val_dataset else 0} 张")
    print(f"[训练] 类别数: {num_classes}")

    if len(train_dataset) < 50:
        print(f"[训练] [WARN] 训练集很小 ({len(train_dataset)}张), "
              f"建议使用数据增强扩充")

    train_loader = DataLoader(
        train_dataset, batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

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

    # 使用线性warmup + cosine退火
    from torch.optim.lr_scheduler import LambdaLR
    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

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
        "val_samples": len(val_dataset) if val_dataset else 0,
        "timestamp": datetime.now().isoformat(),
    }

    best_val_acc = 0.0
    best_model_path = OUTPUTS_DIR / f"best_{args.model}.pth"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
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

        val_str = f"V Loss: {val_loss:.4f} | V Acc: {val_acc:.2f}%" if val_loader else "V: N/A"
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
    history_path = OUTPUTS_DIR / f"history_{args.model}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, ensure_ascii=False, indent=2, fp=f)

    # 生成图表
    plot_training_curves(history, args.model)

    # 测试集评估 (如果存在)
    test_csv = data_dir / "annotations" / "test.csv"
    if test_csv.exists():
        print(f"\n[评估] 在测试集上评估最佳模型...")
        evaluate_model(str(best_model_path), args.model, device, args.input_size, args.data_dir)

    return history


# ============================================================
# 图表生成
# ============================================================

def plot_training_curves(history: dict, model_name: str) -> None:
    """生成 loss-epoch 和 accuracy-epoch 图。"""
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
        output_path = OUTPUTS_DIR / f"training_curves_{model_name}.png"
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
) -> dict:
    """在测试集上评估已训练模型,生成混淆矩阵。"""
    ddir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    test_csv = ddir / "annotations" / "test.csv"
    images_dir = ddir / "images"

    if not test_csv.exists():
        print("[评估] 测试集不存在,跳过评估")
        return {}

    test_dataset = CreatureDataset(
        str(test_csv), str(images_dir),
        transform=get_default_transforms(train=False, input_size=input_size),
        target_size=input_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = test_dataset.num_classes
    model = build_model(model_name, num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = 100.0 * correct / total
    result = {
        "test_accuracy": acc,
        "test_samples": total,
        "test_correct": correct,
    }
    print(f"[评估] 测试准确率: {acc:.2f}% ({correct}/{total})")

    # 保存评估结果
    result_path = OUTPUTS_DIR / f"eval_{model_name}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, ensure_ascii=False, indent=2, fp=f)

    # 生成混淆矩阵图 (只在样本足够时)
    if total >= 10 and len(set(all_labels)) >= 3:
        try:
            plot_confusion_matrix(all_labels, all_preds, model_name)
        except Exception as e:
            print(f"[评估] 混淆矩阵生成失败: {e}")

    return result


def plot_confusion_matrix(labels: list[int], preds: list[int], model_name: str) -> None:
    """生成混淆矩阵图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import numpy as np

        cm = confusion_matrix(labels, preds)
        classes = sorted(set(labels))

        fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.3),
                                        max(10, len(classes) * 0.3)))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        # 只标注非零
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = OUTPUTS_DIR / f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[评估] 混淆矩阵: {cm_path}")

    except ImportError:
        print("[评估] sklearn未安装,跳过混淆矩阵")


# ============================================================
# 导出ONNX (用于量化部署)
# ============================================================

def export_onnx(model_path: str, model_name: str,
                output_path: Optional[str] = None) -> str:
    """导出模型为ONNX格式, 用于ONNX Runtime推理和量化。

    Args:
        model_path: .pth checkpoint路径
        model_name: torchvision模型名
        output_path: 输出ONNX路径 (默认: outputs/{model_name}.onnx)

    Returns:
        输出的ONNX文件路径
    """
    if output_path is None:
        output_path = str(OUTPUTS_DIR / f"{model_name}.onnx")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint.get("history", {}).get("num_classes", 189)
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
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        help="torchvision模型名 (默认: efficientnet_b0)")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"数据目录路径 (默认: {DEFAULT_DATA_DIR})")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数 (默认: 50)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批次大小 (默认: 64, CPU建议16)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率 (默认: 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="梯度裁剪阈值 (默认: 1.0)")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping patience (0=关闭)")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="学习率warmup轮数 (默认: 3)")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="不使用预训练权重")
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用混合精度训练")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="仅评估: 指定模型.pth路径")
    parser.add_argument("--export_onnx", type=str, default=None,
                        help="导出ONNX: 指定模型.pth路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.export_onnx:
        export_onnx(args.export_onnx, args.model)
    elif args.evaluate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluate_model(args.evaluate, args.model, device, args.input_size, args.data_dir)
    else:
        train(args)
