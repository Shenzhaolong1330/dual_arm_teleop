#!/usr/bin/env python3
"""
用 record_cfg.yaml 里的 oculus_config 连接 Quest，实时打印 delta，用于核对 channel_signs / 平移映射。

用法（在 dual_arm_teleop 根目录）:
  python -m scripts.tools.verify_oculus_mapping
  python -m scripts.tools.verify_oculus_mapping --config scripts/config/record_cfg.yaml

验证步骤（每次只动一只手 + 按住对应 Grip）:
  1) 手仅水平左右平移 → 应主要看到 left/right_delta_ee_pose.y 变化（机器人 y=左）
  2) 手仅朝身体方向前后拉推（深度）→ 应主要看到 .x 变化（机器人 x=前）
  3) 手仅上下 → 应主要看到 .z 变化
  4) 再各绕一根轴慢慢转手柄，看 .rx .ry .rz 哪个在动、符号是否符合直觉

若某轴方向反了，只改该轴在 channel_signs 里对应那一项的符号。
Ctrl+C 退出。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(root / "scripts" / "config" / "record_cfg.yaml"),
        help="record_cfg.yaml 路径",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["record"]

    from scripts.core.run_record import RecordConfig

    rc = RecordConfig(cfg)
    tcfg = rc.create_teleop_config()
    from teleoperators import OculusTeleop

    teleop = OculusTeleop(tcfg)
    print(__doc__)
    print(f"Oculus IP: {tcfg.ip}")
    print(f"left_pose_scaler:  {tcfg.left_pose_scaler}")
    print(f"left_channel_signs: {tcfg.left_channel_signs}")
    print(f"right_pose_scaler:  {tcfg.right_pose_scaler}")
    print(f"right_channel_signs:{tcfg.right_channel_signs}")
    print("连接中...\n")
    teleop.connect()

    try:
        while True:
            a = teleop.get_action()
            lz = f"{a.get('left_delta_ee_pose.z', 0):+.4f}"
            ly = f"{a.get('left_delta_ee_pose.y', 0):+.4f}"
            lx = f"{a.get('left_delta_ee_pose.x', 0):+.4f}"
            rz = f"{a.get('right_delta_ee_pose.z', 0):+.4f}"
            ry = f"{a.get('right_delta_ee_pose.y', 0):+.4f}"
            rx = f"{a.get('right_delta_ee_pose.x', 0):+.4f}"
            print(
                f"\rL xyz [{lx},{ly},{lz}]  R xyz [{rx},{ry},{rz}]  "
                f"L rxyz [{a.get('left_delta_ee_pose.rx',0):+.3f},{a.get('left_delta_ee_pose.ry',0):+.3f},{a.get('left_delta_ee_pose.rz',0):+.3f}]  "
                f"R rxyz [{a.get('right_delta_ee_pose.rx',0):+.3f},{a.get('right_delta_ee_pose.ry',0):+.3f},{a.get('right_delta_ee_pose.rz',0):+.3f}]    ",
                end="",
                flush=True,
            )
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
