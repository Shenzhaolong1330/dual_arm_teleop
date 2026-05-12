#!/usr/bin/env python3
"""Hardware smoke test for FrankaDualArm.

This script can connect to the real dual-Franka RPC server and RealSense
cameras using scripts/config/record_cfg.yaml.

Default behavior is conservative: connect, read one observation, read cameras,
then disconnect. Use --move to send a tiny Cartesian motion. Use
--cycle-grippers to command grippers.

Examples:
    python robots/dual_franka/test_franka_dual_arm_hardware.py
    python robots/dual_franka/test_franka_dual_arm_hardware.py --move --arm left
    python robots/dual_franka/test_franka_dual_arm_hardware.py --move --cycle-grippers
    python robots/dual_franka/test_franka_dual_arm_hardware.py --skip-cameras --move-distance 0.003 --move
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig

from robots.dual_franka.config_franka import FrankaDualArmConfig
from robots.dual_franka.franka_dual_arm import FrankaDualArm


AXES = ["x", "y", "z", "rx", "ry", "rz"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_record_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "record" not in data:
        raise ValueError(f"Invalid record config: {path}")
    return data["record"]


def _make_camera_configs(record_cfg: dict[str, Any]) -> dict[str, RealSenseCameraConfig]:
    cameras = record_cfg["cameras"]
    fps = int(record_cfg.get("fps", 30))
    width = int(cameras.get("width", 424))
    height = int(cameras.get("height", 240))

    def camera(serial_key: str) -> RealSenseCameraConfig:
        return RealSenseCameraConfig(
            serial_number_or_name=str(cameras[serial_key]),
            fps=fps,
            width=width,
            height=height,
            color_mode=ColorMode.RGB,
            use_depth=False,
            rotation=Cv2Rotation.NO_ROTATION,
        )

    return {
        "left_wrist_image": camera("left_wrist_cam_serial"),
        "right_wrist_image": camera("right_wrist_cam_serial"),
        "head_image": camera("head_cam_serial"),
    }


def _make_robot_config(record_cfg: dict[str, Any], args: argparse.Namespace) -> FrankaDualArmConfig:
    robot_cfg = record_cfg["robot"]
    if record_cfg.get("robot_type") != "franka_dual_arm":
        raise ValueError(
            f"record.robot_type must be franka_dual_arm, got {record_cfg.get('robot_type')!r}"
        )

    cameras = {} if args.skip_cameras else _make_camera_configs(record_cfg)
    return FrankaDualArmConfig(
        robot_ip=str(robot_cfg.get("robot_ip", "127.0.0.1")),
        robot_port=int(robot_cfg.get("robot_port", 4242)),
        rpc_timeout_sec=float(robot_cfg.get("rpc_timeout_sec", args.rpc_timeout)),
        cameras=cameras,
        debug=not args.move,
        use_gripper=bool(robot_cfg.get("use_gripper", True)) and not args.no_gripper,
        gripper_max_open=float(robot_cfg.get("gripper_max_open", 0.085)),
        gripper_force=float(robot_cfg.get("gripper_force", 10.0)),
        gripper_speed=float(robot_cfg.get("gripper_speed", 0.1)),
        close_threshold=float(robot_cfg.get("close_threshold", 0.5)),
        gripper_reverse=bool(robot_cfg.get("gripper_reverse", False)),
        open_grippers_on_connect=bool(args.open_grippers_on_connect),
        reset_opens_grippers=bool(args.reset_opens_grippers),
        reset_go_home=bool(args.go_home),
        go_home_duration_sec=float(robot_cfg.get("go_home_duration_sec", 5.0)),
        go_home_rate_hz=float(robot_cfg.get("go_home_rate_hz", 50.0)),
        control_mode=str(record_cfg.get("control_mode", "oculus")),
        max_cartesian_delta=float(args.max_cartesian_delta),
        max_rotation_delta=float(args.max_rotation_delta),
    )


def _zero_action() -> dict[str, float]:
    action: dict[str, float] = {}
    for side in ("left", "right"):
        for axis in AXES:
            action[f"{side}_delta_ee_pose.{axis}"] = 0.0
    return action


def _motion_action(arm: str, axis: str, amount: float) -> dict[str, float]:
    action = _zero_action()
    sides = ("left", "right") if arm == "both" else (arm,)
    for side in sides:
        action[f"{side}_delta_ee_pose.{axis}"] = float(amount)
    return action


def _axis_value(obs: dict[str, Any], side: str, axis: str) -> float | None:
    value = obs.get(f"{side}_ee_pose.{axis}")
    return None if value is None else float(value)


def _print_axis_delta(before: dict[str, Any], after: dict[str, Any], arm: str, axis: str) -> None:
    sides = ("left", "right") if arm == "both" else (arm,)
    print("[MOVE] observed displacement:")
    for side in sides:
        start = _axis_value(before, side, axis)
        end = _axis_value(after, side, axis)
        if start is None or end is None:
            print(f"  {side}: unavailable")
            continue
        print(f"  {side}: {end - start:+.5f}m ({axis}: {start:.5f} -> {end:.5f})")


def _distance_chunks(total_distance: float, max_step: float) -> list[float]:
    if abs(total_distance) < 1e-12:
        return []
    safe_step = max(abs(max_step), 1e-6)
    steps = max(1, int(math.ceil(abs(total_distance) / safe_step)))
    chunk = total_distance / steps
    return [chunk] * steps


def _execute_incremental_move(
    robot: FrankaDualArm,
    *,
    arm: str,
    axis: str,
    distance: float,
    max_step: float,
    step_period_sec: float,
) -> None:
    chunks = _distance_chunks(distance, max_step)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[MOVE] step {idx}/{len(chunks)}: {axis} {chunk:+.5f}m")
        robot.send_action(_motion_action(arm, axis, chunk))
        time.sleep(step_period_sec)


def _print_observation_summary(obs: dict[str, Any]) -> None:
    print("[OBS] selected robot state:")
    for key in (
        "left_ee_pose.x",
        "left_ee_pose.y",
        "left_ee_pose.z",
        "right_ee_pose.x",
        "right_ee_pose.y",
        "right_ee_pose.z",
        "left_gripper_state_norm",
        "right_gripper_state_norm",
    ):
        if key in obs:
            print(f"  {key}: {obs[key]}")

    image_keys = [key for key, value in obs.items() if hasattr(value, "shape")]
    if image_keys:
        print("[CAM] frames:")
        for key in image_keys:
            print(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")


def _countdown(seconds: int = 3) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"[WARN] Robot motion starts in {remaining}s. Press Ctrl-C to abort.")
        time.sleep(1.0)


def run(args: argparse.Namespace) -> int:
    record_cfg = _load_record_cfg(args.config)
    robot_config = _make_robot_config(record_cfg, args)
    robot = FrankaDualArm(robot_config)

    print("[INFO] Connecting FrankaDualArm")
    print(f"  RPC: {robot_config.robot_ip}:{robot_config.robot_port}")
    print(f"  cameras: {list(robot_config.cameras.keys()) or 'disabled'}")
    print(f"  debug: {robot_config.debug}")
    robot.connect()

    try:
        print("[INFO] Reading observation")
        obs = robot.get_observation()
        _print_observation_summary(obs)

        if args.go_home:
            print("[INFO] Calling robot.reset() with reset_go_home=true")
            robot.reset()

        if args.move:
            _countdown(args.countdown_sec)
            before_move = robot.get_observation()
            print(
                f"[MOVE] {args.arm} arm(s): total {args.move_distance:+.5f}m on {args.axis}; "
                f"max step {args.max_cartesian_delta:.5f}m"
            )
            _execute_incremental_move(
                robot,
                arm=args.arm,
                axis=args.axis,
                distance=args.move_distance,
                max_step=args.max_cartesian_delta,
                step_period_sec=args.step_period_sec,
            )
            print(f"[MOVE] holding for {args.hold_sec:.2f}s")
            time.sleep(args.hold_sec)
            after_move = robot.get_observation()
            _print_axis_delta(before_move, after_move, args.arm, args.axis)

            if args.return_after_move:
                print("[MOVE] returning by the opposite delta")
                _execute_incremental_move(
                    robot,
                    arm=args.arm,
                    axis=args.axis,
                    distance=-args.move_distance,
                    max_step=args.max_cartesian_delta,
                    step_period_sec=args.step_period_sec,
                )
                time.sleep(args.hold_sec)
                after_return = robot.get_observation()
                _print_axis_delta(before_move, after_return, args.arm, args.axis)

        if args.cycle_grippers and robot_config.use_gripper:
            print("[GRIPPER] half-close both grippers, then reopen")
            robot.send_action({"left_gripper_cmd_bin": 0.5, "right_gripper_cmd_bin": 0.5})
            time.sleep(args.settle_sec)
            robot.send_action({"left_gripper_cmd_bin": 1.0, "right_gripper_cmd_bin": 1.0})
            time.sleep(args.settle_sec)

        print("[INFO] Final observation")
        _print_observation_summary(robot.get_observation())
        print("[PASS] Hardware smoke test completed")
        return 0
    finally:
        if not args.keep_connected:
            print("[INFO] Disconnecting")
            robot.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "scripts" / "config" / "record_cfg.yaml",
    )
    parser.add_argument("--rpc-timeout", type=float, default=30.0)
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--open-grippers-on-connect", action="store_true")
    parser.add_argument("--reset-opens-grippers", action="store_true")
    parser.add_argument("--keep-connected", action="store_true")

    parser.add_argument("--move", action="store_true", help="send a tiny Cartesian motion")
    parser.add_argument("--arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--move-distance", type=float, default=0.02)
    parser.add_argument("--max-cartesian-delta", type=float, default=0.01)
    parser.add_argument("--max-rotation-delta", type=float, default=0.05)
    parser.add_argument("--step-period-sec", type=float, default=0.2)
    parser.add_argument("--hold-sec", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--countdown-sec", type=int, default=3)
    parser.add_argument("--return-after-move", action="store_true")

    parser.add_argument("--go-home", action="store_true", help="call server go_home through robot.reset()")
    parser.add_argument("--cycle-grippers", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user")
        return 130
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
