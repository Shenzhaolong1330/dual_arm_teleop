"""Stop the selected robot without running its normal connect/enable sequence."""

import argparse
import json
import math
from pathlib import Path
import time

import yaml

from scripts.core.robot_selection import add_selection_arguments, default_record_cfg_path, load_selected_robot


def _stop_nero(config: dict, sides: tuple[str, ...], timeout: float) -> dict:
    import zerorpc

    client = zerorpc.Client(timeout=timeout, heartbeat=20)
    results = {}
    try:
        client.connect(f"tcp://{config['robot_ip']}:{config['robot_port']}")
        # Always attempt the other arm even if the first command/status fails.
        for side in sides:
            try:
                if client.robot_stop(f"{side}_robot") is not True:
                    raise RuntimeError("Server rejected the stop request")
                status_fn = getattr(client, f"{side}_robot_get_arm_status")
                deadline = time.monotonic() + timeout
                while True:
                    status = status_fn()
                    if status.get("arm_status") == 1:  # SDK EMERGENCY_STOP
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Damped stop was not confirmed: {status}")
                    time.sleep(0.05)
                results[side] = {"ok": True, "action": "damped_emergency_stop", "state": status}
            except Exception as exc:
                results[side] = {"ok": False, "error": str(exc)}
    finally:
        client.close()  # Transport only; never enables, resets or moves an arm.
    return results


def _stop_flexiv(config: dict, sides: tuple[str, ...], timeout: float) -> dict:
    import flexivrdk

    results = {}
    for side in sides:
        robot = None
        try:
            # Do not use FlexivDualArm.connect(): it can enable/home/change tools.
            robot = flexivrdk.Robot(
                config[f"{side}_robot_sn"], config.get("network_interface_whitelist", [])
            )
            robot.Stop()
            deadline = time.monotonic() + timeout
            while not (robot.stopped() and robot.mode() == flexivrdk.Mode.IDLE):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Stop/IDLE was not confirmed")
                time.sleep(0.05)
            results[side] = {"ok": True, "action": "stop_idle", "brakes_engaged": "unverified"}
        except Exception as exc:
            results[side] = {"ok": False, "error": str(exc)}
        finally:
            robot = None  # Release this one-shot RDK connection.
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stop the robot selected by record.robot_type. Does not home or enable it.",
        epilog="Exit teleoperation/recording before use. Nero may sink under gravity; "
               "Flexiv Rizon4s Stop enters IDLE and does not guarantee brake engagement.",
    )
    parser.add_argument("--config", "--config-path", type=Path, default=default_record_cfg_path())
    add_selection_arguments(parser)
    parser.add_argument("--arm", choices=("both", "left", "right"), default="both")
    parser.add_argument("--dry-run", action="store_true", help="Show target without connecting hardware.")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="RPC timeout and per-arm status wait in seconds; excludes RDK blocking calls.")
    args = parser.parse_args(argv)
    path = args.config.expanduser().resolve()
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be positive and finite.")
        robot_type, config = load_selected_robot(path, robot_type=args.robot_type,
                                               station_path=args.station_config,
                                               use_station=not args.no_station_config)
        if robot_type not in ("nero_dual_arm", "flexiv_dual_arm"):
            raise ValueError(f"robot-stop does not support {robot_type!r}; no hardware commands sent.")
        sides = ("left", "right") if args.arm == "both" else (args.arm,)
        summary = {"config": str(path), "robot_type": robot_type, "arms": sides}
        if robot_type == "nero_dual_arm":
            if not isinstance(config.get("robot_ip"), str) or not config["robot_ip"].strip():
                raise ValueError("Nero robot_ip must be explicitly configured.")
            config["robot_port"] = int(config.get("robot_port", 4242))
            if not 1 <= config["robot_port"] <= 65535:
                raise ValueError("Nero robot_port must be between 1 and 65535.")
            summary.update(robot_ip=config["robot_ip"], robot_port=config["robot_port"],
                           action="damped_emergency_stop",
                           note="Nero 阻尼急停后可能因重力缓慢下沉；这不是切断硬件电源。")
            stop = _stop_nero
        else:
            for side in sides:
                if not isinstance(config.get(f"{side}_robot_sn"), str) or not config[f"{side}_robot_sn"].strip():
                    raise ValueError(f"Flexiv {side}_robot_sn must be configured.")
            if len(sides) == 2 and config["left_robot_sn"] == config["right_robot_sn"]:
                raise ValueError("Left and right Flexiv serial numbers must differ.")
            summary.update({f"{side}_robot_sn": config[f"{side}_robot_sn"] for side in sides})
            summary.update(action="stop_idle", note="Rizon4s 仅停止并进入 IDLE；不保证抱闸/失能。")
            stop = _stop_flexiv
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY_RUN_OK: no hardware connected; no commands sent.")
        return
    print("请确保遥操作/数采已退出，其他控制程序仍可能重新控制机械臂。", flush=True)
    try:
        results = stop(config, sides, args.timeout)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1) from exc
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False), flush=True)
    if not all(result["ok"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
