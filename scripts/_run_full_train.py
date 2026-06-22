"""完整训练启动器 — 重定向输出到日志文件，避免PowerShell管道问题。"""
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.stdout.reconfigure(encoding='utf-8')

LOG = open(str(SCRIPT_DIR.parent / "training_output.log"), "w", encoding="utf-8", buffering=1)

def tee(msg: str) -> None:
    LOG.write(msg + "\n")
    LOG.flush()

class Tee:
    def write(self, s):
        LOG.write(s)
        LOG.flush()
    def flush(self):
        LOG.flush()

sys.stdout = Tee()
sys.stderr = Tee()

tee(f"=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
tee(f"Log file: {SCRIPT_DIR.parent / 'training_output.log'}")

try:
    import torch
    tee(f"PyTorch: {torch.__version__}")
    from train import train, parse_args

    # Build args manually (avoid argparse on PowerShell)
    import argparse
    args = argparse.Namespace(
        model="mobilenet_v3_large",
        data_dir=str(SCRIPT_DIR.parent / "data"),
        output_dir=str(SCRIPT_DIR.parent / "outputs"),
        epochs=10,
        batch_size=16,
        lr=3e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
        clip_grad=1.0,
        input_size=224,
        seed=None,
        device="cpu",
        num_workers=0,
        early_stop=10,
        warmup_epochs=3,
        no_pretrain=False,
        no_amp=True,
        evaluate=None,
        export_onnx=None,
    )

    tee(f"Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    tee(f"Data: {args.data_dir}")
    tee(f"Output: {args.output_dir}")

    history = train(args)
    tee(f"\n=== Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    tee(f"Best val_acc: {history['best_val_acc']:.2f}% (epoch {history.get('best_epoch', '?')})")
    tee(f"Total epochs: {len(history['train_loss'])}")

except Exception:
    tee(f"\n=== CRASHED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    tee(traceback.format_exc())
    sys.exit(1)
finally:
    LOG.close()
