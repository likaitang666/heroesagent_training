"""下载Bulwark(棱堡)阵营兵种图片。

从 heroes.thelazy.net / heroes.v.thelazy.net 下载肖像图片到训练目录。
11个兵种有可用URL, 3个(Billy Goat/Ram/Steel Elf)暂无Wiki页面。

用法:
    cd F:/桌面/test3 && python training/scripts/download_bulwark_images.py
"""

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = Path(__file__).parent.parent  # 训练模块根目录 (文件夹名无关)
IMAGES_DST = TRAINING_ROOT / "data" / "images"
IMAGES_SRC = PROJECT_ROOT / "images" / "creatures"
LABELS_FILE = PROJECT_ROOT / "gamedata" / "creature_labels.json"

# Bulwark兵种图片URL (来自 training/BULWARK_IMAGE_SOURCES.md)
# 使用新标签索引
BULWARK_URLS: dict[str, list[str]] = {
    "Kobold": [
        "http://heroes.v.thelazy.net/index.php/File:Creature_Kobold.png",
    ],
    "Kobold Foreman": [
        "http://heroes.v.thelazy.net/index.php/File:Creature_Kobold_Foreman.png",
    ],
    "Snow Elf": [
        "https://heroes.thelazy.net/index.php/File:Snow_Elf_portrait.png",
        "http://heroes.v.thelazy.net/index.php/File:Creature_Snow_Elf_(HotA).png",
    ],
    "Yeti": [
        "http://heroes.v.thelazy.net/index.php/File:Creature_Yeti.png",
    ],
    "Yeti Runemaster": [
        "http://heroes.v.thelazy.net/index.php/File:Yeti_Runemaster_portrait.png",
    ],
    "Shaman": [
        "http://heroes.v.thelazy.net/index.php/File:Creature_Shaman.png",
        "http://heroes.v.thelazy.net/index.php/File:Shaman_portrait.png",
    ],
    "Great Shaman": [
        "http://heroes.v.thelazy.net/index.php/File:Great_Shaman_portrait.png",
    ],
    "Mammoth": [
        "http://heroes.v.thelazy.net/index.php/File:Creature_Mammoth.png",
        "https://heroes.thelazy.net/index.php/File:Mammoth_(HotA)_portrait.png",
    ],
    "War Mammoth": [
        "http://heroes.v.thelazy.net/index.php/File:War_Mammoth_portrait.png",
    ],
    "Jotunn": [
        "https://heroes.thelazy.net/index.php/File:Jotunn_portrait.png",
        "http://heroes.v.thelazy.net/index.php/File:Creature_Jotunn.png",
    ],
    "Jotunn Warlord": [
        "http://heroes.v.thelazy.net/index.php/File:Jotunn_Warlord_portrait.png",
        "http://heroes.v.thelazy.net/index.php/File:Creature_Jotunn_Warlord.png",
    ],
}


def get_label_by_name(name_en: str) -> int | None:
    """根据英文名获取标签索引。"""
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    for lbl in labels:
        if lbl["name_en"] == name_en:
            return lbl["label"]
    return None


def try_direct_image_url(page_url: str) -> str | None:
    """尝试从Wiki文件页提取直接图片URL。

    MediaWiki文件页通常可通过在文件名前加特殊路径直接访问。
    也尝试常见图片CDN前缀。
    """
    # heroes.thelazy.net 的直接图片路径模式
    # 文件页: /index.php/File:Creature_Kobold.png
    # 直接图片通常需要特殊路径
    return None


def download_image(url: str, save_path: Path) -> bool:
    """下载单张图片。"""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 100:
                print(f"  [WARN] 下载内容过小 ({len(data)} bytes): {url}")
                return False
            save_path.write_bytes(data)
            print(f"  [OK] {save_path.name} ({len(data)} bytes)")
            return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
        return False


def main() -> None:
    IMAGES_DST.mkdir(parents=True, exist_ok=True)
    IMAGES_SRC.mkdir(parents=True, exist_ok=True)

    # 检查已有图片
    existing: set[str] = set()
    for d in [IMAGES_DST, IMAGES_SRC]:
        if d.exists():
            existing.update(f.stem for f in d.glob("*.png"))

    print("=" * 60)
    print("下载 Bulwark (棱堡) 兵种图片")
    print(f"目标目录: {IMAGES_DST}")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for name_en, urls in BULWARK_URLS.items():
        label = get_label_by_name(name_en)
        if label is None:
            print(f"\n[SKIP] {name_en}: 标签未找到")
            fail_count += 1
            continue

        print(f"\n{name_en} (label={label}):")
        for i, url in enumerate(urls):
            # 命名: {label}_{序号}.png
            seq = i  # 第一张URL用序号0, 第二张用序号1
            filename = f"{label}_{seq}.png"
            stem = f"{label}_{seq}"

            if stem in existing:
                print(f"  [SKIP] 已存在: {filename}")
                skip_count += 1
                continue

            # 尝试下载
            if download_image(url, IMAGES_DST / filename):
                # 同时复制到images/creatures
                src_copy = IMAGES_SRC / filename
                if not src_copy.exists():
                    try:
                        import shutil
                        shutil.copy2(IMAGES_DST / filename, src_copy)
                    except Exception:
                        pass
                success_count += 1
            else:
                fail_count += 1

    # 报告缺失兵种
    print(f"\n{'='*60}")
    print(f"下载完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skip_count}")
    print(f"\n以下3个兵种暂无Wiki图片URL, 需从游戏客户端截图:")
    print(f"  [159] Billy Goat (山羚)")
    print(f"  [160] Ram (雪羚)")
    print(f"  [162] Steel Elf (铁甲雪精灵)")

    # 重新生成标注
    if success_count > 0:
        print(f"\n提示: 图片下载后请运行 generate_annotations.py 更新标注文件")


if __name__ == "__main__":
    main()
