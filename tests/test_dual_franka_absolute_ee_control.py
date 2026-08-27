#!/usr/bin/env python3
"""Command-line smoke script for dual Franka absolute EE pose control."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/deepcybo/Le-nero/dual_arm_teleop")

import numpy as np

sys.path.insert(0, str(REPO_ROOT))

_RPC_CLIENT_PATH = REPO_ROOT / "robots" / "dual_franka" / "dual_franka_robotiq_rpc_client.py"
_RPC_CLIENT_SPEC = importlib.util.spec_from_file_location("dual_franka_rpc_client_cli", _RPC_CLIENT_PATH)
if _RPC_CLIENT_SPEC is None or _RPC_CLIENT_SPEC.loader is None:
    raise RuntimeError(f"Failed to load RPC client module from {_RPC_CLIENT_PATH}")
franka_rpc = importlib.util.module_from_spec(_RPC_CLIENT_SPEC)
_RPC_CLIENT_SPEC.loader.exec_module(franka_rpc)

DualFrankaRobotiqRpcClient = franka_rpc.DualFrankaRobotiqRpcClient
_absolute_target_to_delta = franka_rpc._absolute_target_to_delta
_pose_from_side_observation = franka_rpc._pose_from_side_observation

np.set_printoptions(precision=6, suppress=True)


def parse_pose(value: str) -> np.ndarray:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"pose must be JSON list of 6 numbers: {exc}") from exc
    try:
        pose = np.asarray(parsed, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("pose must be a list of 6 finite numbers") from exc
    if pose.size != 6 or not np.all(np.isfinite(pose)):
        raise argparse.ArgumentTypeError(f"pose must contain exactly 6 finite numbers, got {parsed!r}")
    return pose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move dual Franka end-effectors to absolute torso-frame target poses.",
    )
    parser.add_argument(
        "--server-host",
        default=os.environ.get("FRANKA_RPC_HOST", "172.16.0.1"),
        help="ZeroRPC server host. Default: env FRANKA_RPC_HOST or 172.16.0.1.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.environ.get("FRANKA_RPC_PORT", "4242")),
        help="ZeroRPC server port. Default: env FRANKA_RPC_PORT or 4242.",
    )
    parser.add_argument(
        "--rpc-timeout-sec",
        type=float,
        default=float(os.environ.get("FRANKA_RPC_TIMEOUT_SEC", "30")),
    )
    parser.add_argument(
        "--left_target",
        "--left-target",
        dest="left_target",
        type=parse_pose,
        required=True,
        help='Left absolute target pose, e.g. "[0.5, 0.5, 0.2, 1.57, -1.57, 0.7]".',
    )
    parser.add_argument(
        "--right_target",
        "--right-target",
        dest="right_target",
        type=parse_pose,
        required=True,
        help='Right absolute target pose, e.g. "[0.5, -0.5, 0.2, -1.57, -1.57, -0.7]".',
    )
    parser.add_argument("--rate_hz", "--rate-hz", dest="rate_hz", type=float, default=50.0)
    parser.add_argument(
        "--max_translation_speed",
        "--max-translation-speed",
        dest="max_translation_speed",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_rotation_speed",
        "--max-rotation-speed",
        dest="max_rotation_speed",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--max_translation_step",
        "--max-translation-step",
        dest="max_translation_step",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--max_rotation_step",
        "--max-rotation-step",
        dest="max_rotation_step",
        type=float,
        default=0.015,
    )
    parser.add_argument("--settle_time_sec", "--settle-time-sec", dest="settle_time_sec", type=float, default=0.8)
    parser.add_argument(
        "--position_tolerance_m",
        "--position-tolerance-m",
        dest="position_tolerance_m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--rotation_tolerance_rad",
        "--rotation-tolerance-rad",
        dest="rotation_tolerance_rad",
        type=float,
        default=0.008,
    )
    parser.add_argument(
        "--max_correction_iters",
        "--max-correction-iters",
        dest="max_correction_iters",
        type=int,
        default=2,
    )
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=3000)
    parser.add_argument("--compact", action="store_true", help="Print compact JSON for result payloads.")
    return parser


def read_ee_poses(client: DualFrankaRobotiqRpcClient) -> tuple[np.ndarray, np.ndarray]:
    obs = client.get_observation()
    left = np.array(_pose_from_side_observation(obs, "left_arm"), dtype=float)
    right = np.array(_pose_from_side_observation(obs, "right_arm"), dtype=float)
    return left, right


def print_pose(name: str, pose: np.ndarray) -> None:
    print(f"{name}: {np.array2string(np.asarray(pose), precision=6, suppress_small=True)}")


def pose_error(current: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = np.array(_absolute_target_to_delta(current.tolist(), target.tolist()), dtype=float)
    return {
        "raw_current_minus_target": (current - target).tolist(),
        "residual_command_to_target": residual.tolist(),
        "translation_norm_m": float(np.linalg.norm(residual[:3])),
        "rotation_norm_rad": float(np.linalg.norm(residual[3:])),
    }


def print_error(name: str, error: dict[str, Any]) -> None:
    print_pose(f"{name} raw current-target", np.asarray(error["raw_current_minus_target"], dtype=float))
    print_pose(f"{name} residual command", np.asarray(error["residual_command_to_target"], dtype=float))
    print(f"{name} residual norm m/rad:", error["translation_norm_m"], error["rotation_norm_rad"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = DualFrankaRobotiqRpcClient(
        ip=args.server_host,
        port=args.server_port,
        timeout=args.rpc_timeout_sec,
    )
    try:
        print("server:", f"{args.server_host}:{args.server_port}")
        print("ping:", client.ping())

        left0, right0 = read_ee_poses(client)
        print_pose("left0", left0)
        print_pose("right0", right0)
        print_pose("left_target", args.left_target)
        print_pose("right_target", args.right_target)

        result = client.dual_robot_move_to_ee_pose(
            args.left_target.tolist(),
            args.right_target.tolist(),
            delta=False,
            wait=False,
            smooth=True,
            rate_hz=args.rate_hz,
            max_translation_speed=args.max_translation_speed,
            max_rotation_speed=args.max_rotation_speed,
            max_translation_step=args.max_translation_step,
            max_rotation_step=args.max_rotation_step,
            settle_time_sec=args.settle_time_sec,
            position_tolerance_m=args.position_tolerance_m,
            rotation_tolerance_rad=args.rotation_tolerance_rad,
            max_correction_iters=args.max_correction_iters,
            max_steps=args.max_steps,
        )

        left1, right1 = read_ee_poses(client)
        left_error = pose_error(left1, args.left_target)
        right_error = pose_error(right1, args.right_target)

        print("client ok:", result.get("ok"))
        print("trajectory:", json.dumps(result.get("trajectory"), indent=None if args.compact else 2, ensure_ascii=False, default=str))
        print("final_error:", json.dumps(result.get("final_error"), indent=None if args.compact else 2, ensure_ascii=False, default=str))
        print_pose("left1", left1)
        print_pose("right1", right1)
        print_pose("left actual delta", left1 - left0)
        print_pose("right actual delta", right1 - right0)
        print_error("left", left_error)
        print_error("right", right_error)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
