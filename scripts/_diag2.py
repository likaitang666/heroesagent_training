"""诊断2 — 捕获所有异常，直接写文件。"""
import sys, time, traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.stdout.reconfigure(encoding='utf-8')

LOG = open(str(SCRIPT_DIR.parent / "diag2_output.log"), "w", encoding="utf-8", buffering=1)

def log(msg: str) -> None:
    LOG.write(msg + "\n")
    LOG.flush()

try:
    log(f"Start: {time.strftime('%H:%M:%S')}")
    import torch
    from torch.utils.data import DataLoader, Subset
    import torch.nn as nn
    from dataset import CreatureDataset, get_default_transforms
    from data_processor import discover_images, random_train_val_split
    from model_factory import build_model

    log("Step 1: discover...")
    samples = discover_images("data/images")
    log(f"  Total: {len(samples)}")

    log("Step 2: split...")
    train_a, val_a = random_train_val_split(samples, train_ratio=5/6, random_seed=42)
    log(f"  Train: {len(train_a)}, Val: {len(val_a)}")

    # Use only 500 images for quick test
    log("Step 3: subset 500 images...")
    train_a = train_a[:500]
    log(f"  Subset: {len(train_a)}")

    log("Step 4: dataset...")
    train_ds = CreatureDataset.from_annotations(train_a, "data/images",
        transform=get_default_transforms(train=True), target_size=224)
    log(f"  DS len: {len(train_ds)}")

    log("Step 5: loader...")
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    log(f"  Batches: {len(loader)}")

    log("Step 6: model...")
    model = build_model("mobilenet_v3_large", 191, pretrained=True)
    log("  Built")

    log("Step 7: test 5 batches...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    device = torch.device("cpu")

    model.train()
    for i, (images, labels) in enumerate(loader):
        if i >= 5:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        log(f"  Batch {i}: loss={loss.item():.4f}")

    log("SUCCESS — 5 batches completed!")

    # Now test one full epoch
    log("Step 8: full epoch (500 images, batch=16)...")
    t0 = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if i % 5 == 0:
            log(f"  Batch {i}/{len(loader)}: loss={loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    t1 = time.time()
    log(f"Full epoch done: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%, time={t1-t0:.1f}s")

except Exception:
    log(f"CRASH at {time.strftime('%H:%M:%S')}")
    log(traceback.format_exc())
finally:
    LOG.close()
    print(f"Done — see diag2_output.log")
