"""训练启动包装 — 解决PowerShell管道阻塞问题，直接写日志文件。"""
import sys
import time
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# 重定向stdout到日志文件
LOG_PATH = SCRIPT_DIR.parent / "training_output.log"
log_file = open(str(LOG_PATH), "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

import torch
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", default=True)
    parser.add_argument("--model", type=str, default="mobilenet_v3_large")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Device: {args.device}")
    print("=" * 60)

    try:
        history = train(args)
        print(f"\nTraining completed! Best val_acc: {history['best_val_acc']:.2f}%")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        log_file.close()
