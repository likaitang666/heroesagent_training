"""自动战场截图采集脚本 — 用于训练数据收集。

自动控制游戏进入战场并截图，截图保存为训练数据集格式。
利用已有的game_control模块实现自动化操作。

用法:
    python training/scripts/auto_screenshot.py
    python training/scripts/auto_screenshot.py --count 100 --delay 2.0
    python training/scripts/auto_screenshot.py --mode battle --output data/battle_screenshots/

模式:
    - battle: 在战场中自动截图(需手动先进入战场)
    - adventure: 在冒险地图中自动截图
    - menu: 扫描各个菜单界面
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8')


class AutoScreenshotCollector:
    """自动截图采集器。

    使用示例:
        collector = AutoScreenshotCollector(output_dir="training/data/battle_screenshots")
        collector.connect_to_game()
        collector.collect_battle_screenshots(count=50, interval=1.5)
    """

    def __init__(
        self,
        output_dir: str = "training/data/screenshots",
        resolution: tuple[int, int] = (1062, 664),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self._screenshot_count = 0
        self._log: list[dict[str, Any]] = []

    def connect_to_game(self) -> bool:
        """连接游戏窗口。"""
        from backend.app.game_control.window_manager import WindowManager

        wm = WindowManager()
        window = wm.find_game_window()
        if not window:
            print("[错误] 未找到游戏窗口，请先启动游戏(HoMM3 HD)")
            return False

        wm.activate_window(window.hwnd)
        print(f"[连接] 已连接游戏窗口: {window.title} ({window.width}x{window.height})")
        return True

    def take_screenshot(self, label: str = "") -> Optional[str]:
        """截取当前游戏画面并保存。

        Args:
            label: 截图标签(用于文件名和分类)

        Returns:
            保存的文件路径，失败返回None
        """
        from backend.app.game_control.game_control_tools import take_screenshot as _take

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{ts}.png" if label else f"screen_{ts}.png"
        filepath = self.output_dir / filename

        result = _take(str(filepath))
        if result.get("success") and filepath.exists():
            self._screenshot_count += 1
            self._log.append({
                "index": self._screenshot_count,
                "label": label,
                "filepath": str(filepath),
                "timestamp": ts,
            })
            print(f"[截图 {self._screenshot_count}] {filename}")
            return str(filepath)
        else:
            print(f"[错误] 截图失败: {filepath}")
            return None

    def wait(self, seconds: float = 1.0):
        """等待指定秒数。"""
        time.sleep(seconds)

    def press_key(self, key: str):
        """按下键盘按键。"""
        from backend.app.game_control.game_control_tools import press_key as _press
        _press(key)

    def click_at(self, x: int, y: int):
        """点击指定坐标。"""
        from backend.app.game_control.mouse_controller import MouseController
        from backend.app.game_control.window_manager import WindowManager
        wm = WindowManager()
        window = wm.find_game_window()
        hwnd = window.hwnd if window else 0
        rect = window.rect if window else (0, 0, 1062, 664)
        mouse = MouseController(hwnd, rect)
        mouse.move_to(x, y)
        mouse.click(x, y)

    def collect_battle_screenshots(
        self,
        count: int = 50,
        interval: float = 1.5,
        random_delay: bool = True,
    ) -> int:
        """在战场中自动截图(需已在战场界面)。

        在战场中周期性地截图，捕获不同动画帧和状态。

        Args:
            count: 截图数量
            interval: 截图间隔(秒)
            random_delay: 是否添加随机延迟(模拟真实操作)

        Returns:
            实际截图数量
        """
        import random

        print(f"\n=== 战场自动截图 ===")
        print(f"目标: {count}张 | 间隔: {interval}s | 输出: {self.output_dir}")
        print("按Ctrl+C停止\n")

        try:
            for i in range(count):
                # 随机微小延迟(避免完全规律的截图)
                if random_delay:
                    actual_interval = interval + random.uniform(-0.3, 0.3)
                    time.sleep(max(0.1, actual_interval))
                else:
                    time.sleep(interval)

                label = f"battle_{i+1:04d}"
                self.take_screenshot(label)

        except KeyboardInterrupt:
            print(f"\n[停止] 用户中断，已截图{self._screenshot_count}张")

        self._save_log()
        return self._screenshot_count

    def collect_field_of_view_screenshots(
        self,
        rows: int = 11,
        cols: int = 15,
    ) -> int:
        """扫描战场每个格位并截图(用于锚框验证和训练数据)。

        点击每个战场格位，截图记录格位高亮状态。
        注意: 此方法需要游戏在战场中且至少有一个部队可行动。

        Args:
            rows: 战场行数
            cols: 战场列数

        Returns:
            截图数量
        """
        from backend.app.game_control.coordinate_mapper import CoordinateMapper

        mapper = CoordinateMapper()
        print(f"\n=== 战场格位扫描截图 ===")
        print(f"扫描范围: {cols}×{rows} = {cols*rows}个格位")

        for row in range(rows):
            for col in range(cols):
                pixel = mapper.hex_to_pixel(col, row)
                if pixel:
                    px, py = pixel
                    # 点击格位
                    self.click_at(px, py)
                    self.wait(0.2)
                    # 截图
                    label = f"hex_{col:02d}_{row:02d}"
                    self.take_screenshot(label)
                    # ESC取消选择
                    self.press_key("escape")
                    self.wait(0.1)

        self._save_log()
        return self._screenshot_count

    def collect_menu_screenshots(self) -> int:
        """扫描各菜单界面截图(用于UI识别训练)。

        需要在游戏主菜单开始，自动探索各子菜单。
        """
        print(f"\n=== 菜单界面截图 ===")

        # 已知的菜单按钮坐标 (来自UI/UI.xlsx)
        menu_positions = {
            "main_menu": [
                ("new_game", 400, 300),
                ("load_game", 400, 360),
                ("high_scores", 400, 420),
                ("credits", 400, 480),
                ("quit", 400, 540),
            ],
        }

        # 截图主菜单
        self.take_screenshot("menu_main")

        # 点击各个按钮并截图
        for menu_name, positions in menu_positions.items():
            for btn_name, px, py in positions:
                self.click_at(px, py)
                self.wait(0.5)
                self.take_screenshot(f"menu_{btn_name}")
                self.press_key("escape")
                self.wait(0.3)

        self._save_log()
        return self._screenshot_count

    def get_collection_stats(self) -> dict[str, Any]:
        """获取采集统计信息。"""
        return {
            "total_screenshots": self._screenshot_count,
            "output_dir": str(self.output_dir),
            "log_entries": len(self._log),
        }

    def _save_log(self):
        """保存截图日志。"""
        log_path = self.output_dir / "screenshot_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total": self._screenshot_count,
                "entries": self._log,
            }, f, ensure_ascii=False, indent=2)
        print(f"[日志] 已保存到 {log_path}")


# ============================================================
# Tool函数 (供LLM调用)
# ============================================================

def auto_screenshot_collection(
    mode: str = "battle",
    count: int = 30,
    output_dir: str = "training/data/screenshots",
) -> dict[str, Any]:
    """LLM可调用的自动截图采集工具。

    Args:
        mode: 采集模式 (battle/adventure/menu/hex_scan)
        count: 截图数量
        output_dir: 输出目录

    Returns:
        采集结果统计
    """
    collector = AutoScreenshotCollector(output_dir=output_dir)

    if not collector.connect_to_game():
        return {"success": False, "error": "无法连接游戏窗口"}

    if mode == "battle":
        collected = collector.collect_battle_screenshots(count=count)
    elif mode == "hex_scan":
        collected = collector.collect_field_of_view_screenshots()
    elif mode == "menu":
        collected = collector.collect_menu_screenshots()
    else:
        return {"success": False, "error": f"未知模式: {mode}"}

    return {
        "success": True,
        "mode": mode,
        "screenshots_collected": collected,
        "output_dir": str(collector.output_dir),
        "stats": collector.get_collection_stats(),
    }


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="自动战场截图采集脚本")
    parser.add_argument("--mode", choices=["battle", "hex_scan", "menu"],
                        default="battle", help="采集模式")
    parser.add_argument("--count", type=int, default=50, help="截图数量")
    parser.add_argument("--delay", type=float, default=1.5, help="截图间隔(秒)")
    parser.add_argument("--output", type=str, default="training/data/screenshots",
                        help="输出目录")
    parser.add_argument("--no-random", action="store_true", help="禁用随机延迟")

    args = parser.parse_args()

    collector = AutoScreenshotCollector(output_dir=args.output)

    if not collector.connect_to_game():
        print("\n[提示] 请先启动游戏(HoMM3 HD)，确保窗口可见。")
        sys.exit(1)

    if args.mode == "battle":
        collector.collect_battle_screenshots(
            count=args.count, interval=args.delay,
            random_delay=not args.no_random,
        )
    elif args.mode == "hex_scan":
        collector.collect_field_of_view_screenshots()
    elif args.mode == "menu":
        collector.collect_menu_screenshots()

    stats = collector.get_collection_stats()
    print(f"\n=== 采集完成 ===")
    print(f"总截图: {stats['total_screenshots']}张")
    print(f"输出目录: {stats['output_dir']}")


if __name__ == "__main__":
    main()
