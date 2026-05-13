#!/usr/bin/env python3
"""Open-loop RPC motion probe for dual Franka.

This script bypasses the teleop loop and camera reads. It talks directly to the
dual_franka_robotiq_rpc_server over ZeroRPC and sends a fixed stream of small
Cartesian delta commands at a fixed rate. The goal is to isolate server/client
and low-level controller responsiveness from the rest of the data-collection
stack.

Examples:
    python robots/dual_franka/test_franka_rpc_open_loop.py
    python robots/dual_franka/test_franka_rpc_open_loop.py --arm right --axis y
    python robots/dual_franka/test_franka_rpc_open_loop.py --delta 0.0005 --rate-hz 60
    python robots/dual_franka/test_franka_rpc_open_loop.py --go-home-before --print-each-step
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from robots.dual_franka.dual_franka_robotiq_rpc_client import DualFrankaRobotiqRpcClient


AXES = ("x", "y", "z", "rx", "ry", "rz")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_record_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "record" not in data:
        raise ValueError(f"Invalid record config: {path}")
    return data["record"]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _side_robot_state(obs: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    side_obs = _as_mapping(obs.get(side))
    robot_state = side_obs.get("robot_state")
    return _as_mapping(robot_state) if robot_state is not None else side_obs


def _extract_xyz(obs: Mapping[str, Any], side: str) -> tuple[float, float, float] | None:
    state = _side_robot_state(obs, side)
    eef_pose = _as_mapping(state.get("eef_pose"))
    position = eef_pose.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        return (float(position[0]), float(position[1]), float(position[2]))

    end_pose = state.get("end_pose")
    if isinstance(end_pose, (list, tuple)) and len(end_pose) >= 3:
        return (float(end_pose[0]), float(end_pose[1]), float(end_pose[2]))
    return None


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def _print_stats(name: str, values_ms: list[float]) -> None:
    if not values_ms:
        print(f"[STATS] {name}: no samples")
        return
    sorted_values = sorted(values_ms)
    print(
        f"[STATS] {name}: "
        f"mean={mean(sorted_values):7.2f} ms  "
        f"min={sorted_values[0]:7.2f} ms  "
        f"p95={_percentile(sorted_values, 0.95):7.2f} ms  "
        f"max={sorted_values[-1]:7.2f} ms"
    )


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def _clip_delta(
    axis: str,
    requested: float,
    max_cartesian_delta: float | None,
    max_rotation_delta: float | None,
) -> float:
    limit = max_rotation_delta if axis.startswith("r") else max_cartesian_delta
    if limit is None or limit <= 0.0:
        return requested
    if abs(requested) <= limit:
        return requested
    clipped = math.copysign(limit, requested)
    print(
        f"[WARN] Requested delta {requested:+.6f} exceeds per-step limit {limit:.6f} "
        f"for axis {axis}; clipped to {clipped:+.6f}"
    )
    return clipped


def _build_motion_action(arm: str, axis: str, amount: float) -> dict[str, Any]:
    vector = [0.0] * 6
    vector[AXES.index(axis)] = float(amount)
    translation = vector[:3]
    rotation_rotvec = vector[3:]

    side_keys = ("left_arm", "right_arm") if arm == "both" else (f"{arm}_arm",)
    action: dict[str, Any] = {}
    for side_key in side_keys:
        action[side_key] = {
            "motion": {
                "translation": list(translation),
                "rotation_rotvec": list(rotation_rotvec),
            }
        }
    return action


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"[WARN] Open-loop motion starts in {remaining}s. Press Ctrl-C to abort.")
        time.sleep(1.0)


def _print_pose_delta(before: Mapping[str, Any], after: Mapping[str, Any], arm: str) -> None:
    sides = ("left_arm", "right_arm") if arm == "both" else (f"{arm}_arm",)
    print("[OBS] end-effector position change:")
    for side in sides:
        start = _extract_xyz(before, side)
        end = _extract_xyz(after, side)
        if start is None or end is None:
            print(f"  {side}: unavailable")
            continue
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        print(
            f"  {side}: "
            f"dx={dx:+.5f}  dy={dy:+.5f}  dz={dz:+.5f}  "
            f"start=({start[0]:+.5f}, {start[1]:+.5f}, {start[2]:+.5f})  "
            f"end=({end[0]:+.5f}, {end[1]:+.5f}, {end[2]:+.5f})"
        )


def run(args: argparse.Namespace) -> int:
    record_cfg = _load_record_cfg(args.config)
    robot_cfg = record_cfg["robot"]

    if record_cfg.get("robot_type") != "franka_dual_arm":
        raise ValueError(
            f"record.robot_type must be franka_dual_arm, got {record_cfg.get('robot_type')!r}"
        )

    max_cartesian_delta = _optional_float(robot_cfg.get("max_cartesian_delta"))
    max_rotation_delta = _optional_float(robot_cfg.get("max_rotation_delta"))
    safe_delta = _clip_delta(args.axis, args.delta, max_cartesian_delta, max_rotation_delta)
    go_home_duration_sec = float(robot_cfg.get("go_home_duration_sec", 5.0))
    effective_rpc_timeout = float(args.rpc_timeout)
    if args.go_home_before:
        effective_rpc_timeout = max(effective_rpc_timeout, go_home_duration_sec + 10.0)

    client = DualFrankaRobotiqRpcClient(
        ip=str(robot_cfg.get("robot_ip", "127.0.0.1")),
        port=int(robot_cfg.get("robot_port", 4242)),
        timeout=effective_rpc_timeout,
    )
    print(f"[INFO] Connected to RPC server tcp://{robot_cfg.get('robot_ip', '127.0.0.1')}:{robot_cfg.get('robot_port', 4242)}")
    print(f"[INFO] ping={client.ping()}")
    print(
        f"[INFO] Open-loop test: arm={args.arm} axis={args.axis} delta={safe_delta:+.6f} "
        f"rate={args.rate_hz:.1f}Hz steps/half={args.steps_per_half_cycle} cycles={args.cycles}"
    )
    print(f"[INFO] rpc_timeout={effective_rpc_timeout:.1f}s")

    try:
        if args.go_home_before:
            print("[INFO] Going home before open-loop test")
            client.go_home(
                "both",
                go_home_duration_sec,
                float(robot_cfg.get("go_home_rate_hz", 50.0)),
            )
            time.sleep(args.settle_sec)

        before_obs = client.get_observation()
        if args.countdown_sec > 0:
            _countdown(args.countdown_sec)

        period_sec = 1.0 / max(args.rate_hz, 1e-6)
        send_ms: list[float] = []
        loop_ms: list[float] = []
        step_count_samples: list[int] = []

        total_half_cycles = args.cycles * 2
        total_commands = total_half_cycles * args.steps_per_half_cycle
        next_deadline = time.perf_counter()
        command_index = 0

        for half_cycle_idx in range(total_half_cycles):
            # sign = 1.0 if half_cycle_idx % 2 == 0 else -1.0
            sign = 1.0
            signed_delta = sign * safe_delta
            for _ in range(args.steps_per_half_cycle):
                command_index += 1
                action = _build_motion_action(args.arm, args.axis, signed_delta)

                loop_start = time.perf_counter()
                rpc_start = time.perf_counter()
                result = client.step(action)
                rpc_elapsed_ms = (time.perf_counter() - rpc_start) * 1e3
                send_ms.append(rpc_elapsed_ms)

                if isinstance(result, Mapping):
                    step_count = result.get("step_count")
                    if step_count is not None:
                        try:
                            step_count_samples.append(int(step_count))
                        except (TypeError, ValueError):
                            pass

                next_deadline += period_sec
                remaining = next_deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                loop_ms.append((time.perf_counter() - loop_start) * 1e3)

                if args.print_each_step:
                    print(
                        f"[STEP] {command_index:04d}/{total_commands} "
                        f"delta={signed_delta:+.6f} "
                        f"rpc={rpc_elapsed_ms:7.2f} ms"
                    )

        after_obs = client.get_observation()
        print("[INFO] Open-loop test completed")
        _print_stats("rpc_step", send_ms)
        _print_stats("loop_period", loop_ms)
        if step_count_samples:
            print(
                f"[INFO] server step_count: start={step_count_samples[0]} "
                f"end={step_count_samples[-1]}"
            )
        _print_pose_delta(before_obs if isinstance(before_obs, Mapping) else {}, after_obs if isinstance(after_obs, Mapping) else {}, args.arm)
        print(
            f"[INFO] Suggested next check: compare target loop period {period_sec * 1000:.2f} ms "
            f"with measured loop_period stats"
        )
        return 0
    finally:
        client.close()
        print("[INFO] RPC client closed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "scripts" / "config" / "record_cfg.yaml",
        help="Path to record_cfg.yaml",
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="left")
    parser.add_argument("--axis", choices=AXES, default="x")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.001,
        help="Per-step Cartesian delta. Meters for x/y/z, radians for rx/ry/rz.",
    )
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--steps-per-half-cycle", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--rpc-timeout", type=float, default=30.0)
    parser.add_argument("--countdown-sec", type=int, default=3)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--go-home-before", action="store_true")
    parser.add_argument("--print-each-step", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
