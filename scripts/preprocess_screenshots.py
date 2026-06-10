"""截图数据预处理 — 将~20,000张游戏截图组织为训练数据。

基于 截图要求.txt:
- 189个兵种，每个兵种120张截图
- 24种地形/背景 × 5种状态 = 120张/兵种
- 总共约22,680张截图
- 命名规则: {creature_id}_{序号}.png (序号0-119)

24种地形分类:
  1-11: 标准地形 (草地/沙地/雪地/沼泽/粗糙/地下/熔岩/水域/高地/荒地/泥土)
  12:    船上
  13-24: 特殊地形 (圣地/诅咒之地/魔法平原/岩石地/火焰之地/清澈池塘/
                 疾风之地/石化之地/力量之地/知识之地/禁魔之地/幸运之地)

5种状态 (每个地形):
  0: 朝右(进攻方)
  1: 朝左(防守方)
  2: 攻击
  3: 被攻击
  4: 防御/被攻击(防御姿态)

序号→(地形,状态) 映射:
  序号 = terrain_index * 5 + state_index
  例如: 序号0  = 草地+朝右, 序号1 = 草地+朝左, ..., 序号5 = 沙地+朝右

用法:
    cd F:/桌面/test3 && python training/scripts/preprocess_screenshots.py

子命令:
    --show-mapping    显示序号→(地形,状态)映射表
    --validate        验证截图目录和数量
    --organize        将原始截图整理到分兵种子目录
    --generate-annotations  生成训练标注文件
    --all             执行全部步骤
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
SCREENSHOTS_DIR = TRAINING_ROOT / "data" / "screenshots"
RAW_DIR = SCREENSHOTS_DIR / "raw"
ORGANIZED_DIR = SCREENSHOTS_DIR / "organized"
ANNOTATIONS_DIR = SCREENSHOTS_DIR / "annotations"
LABELS_FILE = PROJECT_ROOT / "gamedata" / "creature_labels.json"

# ============================================================
# 地形和状态定义
# ============================================================

# 11种标准地形
NORMAL_TERRAINS: list[str] = [
    "grass",        # 草地
    "sand",         # 沙地
    "snow",         # 雪地
    "swamp",        # 沼泽
    "rough",        # 粗糙地
    "subterranean", # 地下
    "lava",         # 熔岩
    "water",        # 水域
    "highland",     # 高地
    "wasteland",    # 荒地
    "dirt",         # 泥土
]

# 船上 (特殊背景)
BOAT_TERRAIN: str = "boat"

# 12种特殊地形 (战场魔法地形/特殊效果地形)
SPECIAL_TERRAINS: list[str] = [
    "holy_ground",       # 圣地
    "cursed_ground",     # 诅咒之地
    "magic_plains",      # 魔法平原
    "rockland",          # 岩石地
    "fiery_fields",      # 火焰之地
    "lucid_pools",       # 清澈池塘
    "gale_grounds",      # 疾风之地
    "petrified_land",    # 石化之地
    "power_grounds",     # 力量之地
    "knowledge_grounds", # 知识之地
    "anti_magic_grounds",# 禁魔之地
    "lucky_grounds",     # 幸运之地
]

# 合成完整24种地形列表
ALL_TERRAINS: list[str] = NORMAL_TERRAINS + [BOAT_TERRAIN] + SPECIAL_TERRAINS

# 5种状态
STATES: list[str] = [
    "facing_right",    # 朝右(进攻方)
    "facing_left",     # 朝左(防守方)
    "attacking",       # 攻击
    "being_attacked",  # 被攻击
    "defending",       # 防御/被攻击(防御姿态)
]

TERRAIN_NAMES_ZH: dict[str, str] = {
    "grass": "草地", "sand": "沙地", "snow": "雪地", "swamp": "沼泽",
    "rough": "粗糙地", "subterranean": "地下", "lava": "熔岩", "water": "水域",
    "highland": "高地", "wasteland": "荒地", "dirt": "泥土",
    "boat": "船上",
    "holy_ground": "圣地", "cursed_ground": "诅咒之地",
    "magic_plains": "魔法平原", "rockland": "岩石地",
    "fiery_fields": "火焰之地", "lucid_pools": "清澈池塘",
    "gale_grounds": "疾风之地", "petrified_land": "石化之地",
    "power_grounds": "力量之地", "knowledge_grounds": "知识之地",
    "anti_magic_grounds": "禁魔之地", "lucky_grounds": "幸运之地",
}

STATE_NAMES_ZH: dict[str, str] = {
    "facing_right": "朝右(进攻方)",
    "facing_left": "朝左(防守方)",
    "attacking": "攻击",
    "being_attacked": "被攻击",
    "defending": "防御(被攻击)",
}

NUM_TERRAINS = len(ALL_TERRAINS)   # 24
NUM_STATES = len(STATES)           # 5
IMAGES_PER_CREATURE = NUM_TERRAINS * NUM_STATES  # 120


def seq_to_terrain_state(seq: int) -> tuple[int, int]:
    """将序号(0-119)映射到(terrain_index, state_index)。

    seq = terrain_index * NUM_STATES + state_index
    """
    if not 0 <= seq < IMAGES_PER_CREATURE:
        raise ValueError(f"序号{seq}超出范围[0, {IMAGES_PER_CREATURE})")
    terrain_idx = seq // NUM_STATES
    state_idx = seq % NUM_STATES
    return terrain_idx, state_idx


def terrain_state_to_seq(terrain_idx: int, state_idx: int) -> int:
    """将(terrain_index, state_index)映射回序号(0-119)。"""
    if not 0 <= terrain_idx < NUM_TERRAINS:
        raise ValueError(f"地形索引{terrain_idx}超出范围[0, {NUM_TERRAINS})")
    if not 0 <= state_idx < NUM_STATES:
        raise ValueError(f"状态索引{state_idx}超出范围[0, {NUM_STATES})")
    return terrain_idx * NUM_STATES + state_idx


def build_expected_files() -> dict[int, list[str]]:
    """构建每个兵种的预期文件列表。

    Returns:
        {creature_id: ["{id}_{seq}.png", ...]} 共189个兵种, 每个120个文件
    """
    expected: dict[int, list[str]] = {}
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels_data = json.load(f)

    for label in labels_data["labels"]:
        cid = label["label"]
        files = [f"{cid}_{seq}.png" for seq in range(IMAGES_PER_CREATURE)]
        expected[cid] = files
    return expected


# ============================================================
# 映射表导出
# ============================================================

def show_mapping() -> None:
    """打印完整序号→(地形,状态)映射表。"""
    print("=" * 70)
    print(f"截图序号 → (地形, 状态) 映射表")
    print(f"地形数: {NUM_TERRAINS} | 状态数: {NUM_STATES} | "
          f"每兵种图片数: {IMAGES_PER_CREATURE}")
    print("=" * 70)

    print(f"\n{'序号范围':<14} {'地形':<25} {'状态':<20}")
    print("-" * 70)
    for terrain_idx, terrain_name in enumerate(ALL_TERRAINS):
        start_seq = terrain_idx * NUM_STATES
        end_seq = start_seq + NUM_STATES - 1
        terrain_zh = TERRAIN_NAMES_ZH.get(terrain_name, terrain_name)
        print(f"[{start_seq:3d}-{end_seq:3d}]    {terrain_name:<25} "
              f"{terrain_zh:<16}")
        for state_idx, state_name in enumerate(STATES):
            seq = start_seq + state_idx
            state_zh = STATE_NAMES_ZH.get(state_name, state_name)
            print(f"  {seq:3d}            {state_name:<30} {state_zh}")

    # 保存为CSV方便查阅
    csv_path = SCREENSHOTS_DIR / "mapping_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "地形(en)", "地形(zh)", "状态(en)", "状态(zh)"])
        for seq in range(IMAGES_PER_CREATURE):
            ti, si = seq_to_terrain_state(seq)
            writer.writerow([
                seq, ALL_TERRAINS[ti], TERRAIN_NAMES_ZH.get(ALL_TERRAINS[ti], ""),
                STATES[si], STATE_NAMES_ZH.get(STATES[si], ""),
            ])
    print(f"\n映射表已保存: {csv_path}")


# ============================================================
# 验证截图
# ============================================================

def validate_screenshots() -> dict:
    """验证截图目录, 检查缺失和格式错误。

    Returns:
        验证报告字典
    """
    if not RAW_DIR.exists():
        print(f"[ERROR] 截图目录不存在: {RAW_DIR}")
        print("请将截图放入该目录后重试。")
        return {"status": "error", "missing_dir": str(RAW_DIR)}

    png_files = sorted(RAW_DIR.glob("*.png"))
    print(f"发现 {len(png_files)} 张截图")

    expected = build_expected_files()
    total_expected = sum(len(v) for v in expected.values())
    print(f"预期总数: {total_expected} 张 ({len(expected)} 兵种 × {IMAGES_PER_CREATURE} 张)")

    # 按兵种统计
    per_creature: dict[int, set[int]] = defaultdict(set)
    invalid_names: list[str] = []

    import re
    for f in png_files:
        m = re.match(r"^(\d+)_(\d+)\.png$", f.name)
        if m:
            cid = int(m.group(1))
            seq = int(m.group(2))
            if 0 <= seq < IMAGES_PER_CREATURE:
                per_creature[cid].add(seq)
            else:
                invalid_names.append(f.name)
        else:
            invalid_names.append(f.name)

    if invalid_names:
        print(f"\n[WARN] {len(invalid_names)} 个文件命名格式异常:")
        for n in invalid_names[:20]:
            print(f"  {n}")
        if len(invalid_names) > 20:
            print(f"  ... 共 {len(invalid_names)} 个")

    # 检查缺失
    missing: dict[int, list[int]] = {}
    for cid, expected_seqs in expected.items():
        actual_seqs = per_creature.get(cid, set())
        expected_set = set(range(IMAGES_PER_CREATURE))
        miss = sorted(expected_set - actual_seqs)
        if miss:
            missing[cid] = miss

    complete_count = len(expected) - len(missing)
    print(f"\n完整兵种(120张全齐): {complete_count}/{len(expected)}")
    print(f"缺失兵种: {len(missing)}")
    if missing:
        with open(LABELS_FILE, encoding="utf-8") as f:
            labels = json.load(f)["labels"]
        idx_to_name = {l["label"]: l["name_en"] for l in labels}

        for cid in sorted(missing.keys())[:10]:
            miss_count = len(missing[cid])
            name = idx_to_name.get(cid, f"Unknown({cid})")
            print(f"  [{cid:3d}] {name:<25} 缺失 {miss_count:3d} 张")

    # 保存详细缺失报告
    report_path = SCREENSHOTS_DIR / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_files": len(png_files),
        "expected_total": total_expected,
        "complete_creatures": complete_count,
        "missing_creatures": len(missing),
        "invalid_names": len(invalid_names),
        "missing_details": {
            str(cid): [seq_to_terrain_state(s) for s in seqs]
            for cid, seqs in missing.items()
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, ensure_ascii=False, indent=2, fp=f)
    print(f"\n详细报告: {report_path}")

    return {"status": "ok", "report": report}


# ============================================================
# 整理截图
# ============================================================

def organize_screenshots() -> dict:
    """将raw目录的截图按兵种分目录整理。"""
    if not RAW_DIR.exists():
        print(f"[ERROR] 截图目录不存在: {RAW_DIR}")
        return {"status": "error"}

    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = {l["label"]: l for l in json.load(f)["labels"]}

    import re
    png_files = sorted(RAW_DIR.glob("*.png"))
    organized_count = 0

    for f in png_files:
        m = re.match(r"^(\d+)_(\d+)\.png$", f.name)
        if not m:
            continue

        cid = int(m.group(1))
        seq = int(m.group(2))
        creature = labels.get(cid)
        if not creature:
            continue

        # 创建兵种子目录
        faction = creature.get("faction", "Unknown")
        name_en = creature["name_en"]
        creature_dir = ORGANIZED_DIR / f"{faction}" / f"{cid:03d}_{name_en}"
        creature_dir.mkdir(parents=True, exist_ok=True)

        dst = creature_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            organized_count += 1

    print(f"整理完成: {organized_count} 张截图 -> {ORGANIZED_DIR}")
    return {"status": "ok", "organized": organized_count}


# ============================================================
# 生成截图标注文件
# ============================================================

def generate_screenshot_annotations(
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    random_seed: int = 42,
) -> dict:
    """为截图生成训练/验证/测试标注文件。

    分层策略: 每个兵种的每种(地形,状态)组合至少1张放入训练集,
    剩余按比例分配。

    Args:
        train_ratio: 训练集比例
        val_ratio: 验证集比例 (测试集=1-train_ratio-val_ratio)
        random_seed: 随机种子
    """
    import random
    random.seed(random_seed)

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 扫描所有截图
    raw_files = list(RAW_DIR.glob("*.png")) if RAW_DIR.exists() else []
    organized_files = list(ORGANIZED_DIR.rglob("*.png")) if ORGANIZED_DIR.exists() else []

    import re
    samples: list[dict] = []
    seen: set[str] = set()

    for f in raw_files + organized_files:
        if f.name in seen:
            continue
        seen.add(f.name)

        m = re.match(r"^(\d+)_(\d+)\.png$", f.name)
        if not m:
            continue
        cid = int(m.group(1))
        seq = int(m.group(2))
        if not 0 <= seq < IMAGES_PER_CREATURE:
            continue
        terrain_idx, state_idx = seq_to_terrain_state(seq)

        samples.append({
            "file": f.name,
            "creature_id": cid,
            "seq": seq,
            "terrain_idx": terrain_idx,
            "terrain": ALL_TERRAINS[terrain_idx],
            "state_idx": state_idx,
            "state": STATES[state_idx],
        })

    if not samples:
        print("[WARN] 未找到有效截图。请先将截图放入 training/data/screenshots/raw/")
        return {"status": "empty"}

    print(f"有效截图: {len(samples)} 张")

    # 按(兵种,地形,状态)分组
    from itertools import groupby
    samples.sort(key=lambda s: (s["creature_id"], s["terrain_idx"], s["state_idx"]))

    train_set: list[dict] = []
    val_set: list[dict] = []
    test_set: list[dict] = []

    for (cid, tid, sid), group in groupby(
        samples, key=lambda s: (s["creature_id"], s["terrain_idx"], s["state_idx"])
    ):
        items = list(group)
        random.shuffle(items)
        n = len(items)

        if n >= 3:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            train_set.extend(items[:n_train])
            val_set.extend(items[n_train:n_train + n_val])
            test_set.extend(items[n_train + n_val:])
        elif n == 2:
            train_set.append(items[0])
            val_set.append(items[1])
        else:
            train_set.append(items[0])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # 写入CSV
    fieldnames = [
        "file", "creature_id", "seq", "terrain_idx", "terrain",
        "state_idx", "state",
    ]
    splits = {"train": train_set, "val": val_set, "test": test_set}

    for split_name, split_data in splits.items():
        csv_path = ANNOTATIONS_DIR / f"screenshot_{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_data)
        print(f"  {split_name}: {len(split_data)} 张 -> {csv_path}")

    # 摘要
    summary = {
        "total_samples": len(samples),
        "creatures_covered": len(set(s["creature_id"] for s in samples)),
        "terrains_covered": len(set(s["terrain_idx"] for s in samples)),
        "states_covered": len(set(s["state_idx"] for s in samples)),
        "splits": {k: len(v) for k, v in splits.items()},
    }
    summary_path = ANNOTATIONS_DIR / "screenshot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    print(f"摘要: {summary_path}")
    return {"status": "ok", "summary": summary}


# ============================================================
# 命令行入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="截图数据预处理 — 管理~20,000张兵种截图"
    )
    parser.add_argument("--show-mapping", action="store_true",
                        help="显示序号→(地形,状态)映射表")
    parser.add_argument("--validate", action="store_true",
                        help="验证截图完整性和命名")
    parser.add_argument("--organize", action="store_true",
                        help="将截图按兵种分目录整理")
    parser.add_argument("--generate-annotations", action="store_true",
                        help="生成训练/验证/测试标注文件")
    parser.add_argument("--all", action="store_true",
                        help="执行全部步骤")
    parser.add_argument("--train-ratio", type=float, default=0.80,
                        help="训练集比例 (默认: 0.80)")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="验证集比例 (默认: 0.10)")

    args = parser.parse_args()

    if not any([args.show_mapping, args.validate, args.organize,
                args.generate_annotations, args.all]):
        parser.print_help()
        return

    if args.show_mapping or args.all:
        show_mapping()

    if args.validate or args.all:
        print("\n" + "=" * 60)
        validate_screenshots()

    if args.organize or args.all:
        print("\n" + "=" * 60)
        organize_screenshots()

    if args.generate_annotations or args.all:
        print("\n" + "=" * 60)
        generate_screenshot_annotations(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )


if __name__ == "__main__":
    main()
