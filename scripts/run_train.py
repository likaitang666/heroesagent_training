"""训练启动器 — 直接写日志文件，避免管道缓冲问题。"""
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

LOG_PATH = SCRIPT_DIR.parent / "training_output.log"

class TeeWriter:
    """同时写文件和stdout的writer。"""
    def __init__(self, filepath: Path):
        self.file = open(str(filepath), "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

sys.stdout = TeeWriter(LOG_PATH)  # type: ignore
sys.stderr = sys.stdout

print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log: {LOG_PATH}")

import torch
from train import train

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default="mobilenet_v3_large")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true", default=True)
    args = parser.parse_args()

    try:
        history = train(args)
        print(f"\nDone! Best val_acc: {history['best_val_acc']:.2f}% at epoch {history.get('best_epoch', '?')}")
    except Exception:
        print(f"\nCRASHED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        traceback.print_exc()
        sys.exit(1)
