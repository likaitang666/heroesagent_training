"""最小化epoch测试 — 直接测试train_epoch是否崩溃。"""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.stdout.reconfigure(encoding='utf-8')

import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CreatureDataset, get_default_transforms
from data_processor import discover_images, random_train_val_split
from model_factory import build_model
from train_loop import train_epoch

LOG_PATH = SCRIPT_DIR.parent / "training_output.log"
log = open(str(LOG_PATH), "w", encoding="utf-8")
sys.stdout = log
sys.stderr = log

try:
    print("Step 1: discover_images...", flush=True)
    samples = discover_images("data/images")
    print(f"  Found {len(samples)}", flush=True)

    print("Step 2: split...", flush=True)
    train_a, val_a = random_train_val_split(samples, train_ratio=5/6, random_seed=42)
    print(f"  Train={len(train_a)}, Val={len(val_a)}", flush=True)

    print("Step 3: dataset...", flush=True)
    train_ds = CreatureDataset.from_annotations(train_a, "data/images",
        transform=get_default_transforms(train=True), target_size=224)
    print(f"  DS={len(train_ds)}", flush=True)

    print("Step 4: loader...", flush=True)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    print(f"  Batches={len(loader)}", flush=True)

    print("Step 5: model...", flush=True)
    model = build_model("mobilenet_v3_large", 191, pretrained=True)
    print(f"  Built", flush=True)

    print("Step 6: train_epoch (first 10 batches test)...", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 只测试前10个batch
    model.train()
    for i, (images, labels) in enumerate(loader):
        if i >= 10:
            break
        print(f"  Batch {i}: images={images.shape}", flush=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Step 6: OK — 10 batches completed successfully!", flush=True)

    print("\nNow running full epoch...", flush=True)
    train_loss, train_acc = train_epoch(model, loader, criterion, optimizer,
                                        torch.device("cpu"), None, 1.0)
    print(f"Full epoch done: loss={train_loss:.4f}, acc={train_acc:.2f}%", flush=True)

except Exception as e:
    print(f"\nCRASH: {e}", flush=True)
    traceback.print_exc(file=log)
    sys.exit(1)
finally:
    log.close()
