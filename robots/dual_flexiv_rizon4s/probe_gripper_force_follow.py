"""Empty-gripper experiment: bounded aperture feedback through Grasp, no arm control.

This is an experimental diagnostic, not the recording controller. Recording must
be stopped and both grippers empty and clear. No robot Enable/mode calls or Init.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("force", "move", "production"), default="force")
    parser.add_argument("--integral-gain", type=float, default=0.0)
    args = parser.parse_args()
    import flexivrdk
    robots, grippers = {}, {}
    adapter = None
    data = {"experimental": True, "arm_motion_commands_sent": False,
            "mode": args.mode, "kp": 300.0, "kd": 10.0,
            "ki": args.integral_gain, "force_cap_N": 5.0, "samples": []}
    with ThreadPoolExecutor(max_workers=2) as pool:
        def both(method, values):
            futures = [pool.submit(getattr(grippers[s], method), *v) for s, v in values.items()]
            for future in futures:
                future.result()
        try:
            for side, sn in [("left", "Rizon4s-063173"), ("right", "Rizon4s-063278")]:
                robot = flexivrdk.Robot(sn, ["192.168.2.200"])
                robots[side] = robot
                if robot.fault() or not robot.operational():
                    raise RuntimeError(f"{side} not operational")
                gripper = flexivrdk.Gripper(robot)
                grippers[side] = gripper
                gripper.Enable("Flexiv-GN01")
                p = gripper.params()
                if not (p.min_force <= -5 < 5 <= p.max_force):
                    raise ValueError("Hardware force limits incompatible")
            if args.mode == "production":
                import sys
                from dataclasses import fields
                import yaml
                root = Path(__file__).resolve().parents[2]
                sys.path.insert(0, str(root))
                from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
                from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm
                config = yaml.safe_load((root / "scripts/config/robots/flexiv_config.yaml").read_text())["robot"]
                names = {f.name for f in fields(FlexivDualArmConfig)}
                config = {k: v for k, v in config.items() if k in names}
                config.update(debug=False, cameras={})
                adapter = FlexivDualArm(FlexivDualArmConfig(**config))
                adapter._left_gripper, adapter._right_gripper = grippers["left"], grippers["right"]
                adapter.is_connected = True
                data["production_mode"] = config["gripper_control_mode"]
            both("Move", {s: (0.05, 0.05, 20.0) for s in grippers})
            deadline = time.monotonic() + 4
            while True:
                if all(abs(g.states().width - 0.05) < 0.002 for g in grippers.values()):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Setup width not reached")
                time.sleep(0.02)
            previous = {s: g.states().width for s, g in grippers.items()}
            velocities = {s: 0.0 for s in grippers}
            integrals = {s: 0.0 for s in grippers}
            previous_errors = {s: 0.0 for s in grippers}
            start = last = time.monotonic()
            last_move_time = -float("inf")
            for frame in range(240 if args.mode == "move" else 510):
                now = time.monotonic()
                dt = max(now - last, 1 / 120)
                target = 0.05 + 0.02 * math.sin(2 * math.pi * 0.3 * (now - start))
                if frame >= 240:
                    target = (0.05, 0.06, 0.04)[min(2, (frame - 240) // 90)]
                row = {"t": now - start, "target": target}
                commands = {}
                for side, g in grippers.items():
                    state = g.states()
                    width = state.width
                    if not (math.isfinite(width) and 0.015 < width < 0.085):
                        raise RuntimeError(f"{side} outside experiment width boundary")
                    velocities[side] = 0.75 * velocities[side] + 0.25 * (width - previous[side]) / dt
                    # Positive Grasp force closes, the opposite of width velocity.
                    error = width - target
                    if error * previous_errors[side] < 0:
                        integrals[side] = 0.0
                    integrals[side] = max(-1.5, min(1.5, integrals[side] + args.integral_gain * error * dt))
                    force = max(-5.0, min(5.0, 300 * error + 10 * velocities[side] + integrals[side]))
                    previous_errors[side] = error
                    commands[side] = (force,)
                    row[side] = {"width": width, "command_force": force,
                                 "measured_force": state.force, "moving": state.is_moving}
                    previous[side] = width
                if args.mode == "production":
                    for side in grippers:
                        row[side].pop("command_force")
                    action = {f"{s}_gripper_width": target for s in grippers}
                    returned = adapter.send_action(action)
                    if returned != action:
                        raise AssertionError("Accepted aperture labels differ from requested goals")
                elif args.mode == "force":
                    both("Grasp", commands)
                elif now - last_move_time >= 0.2:
                    both("Move", {s: (target, 0.05, 20.0) for s in grippers})
                    last_move_time = now
                row["delivery_sec"] = time.monotonic() - now
                data["samples"].append(row)
                last = now
                time.sleep(max(0, start + (frame + 1) / 30 - time.monotonic()))
            if adapter is not None:
                for endpoint in (0.0, 0.1):
                    start_endpoint = time.monotonic()
                    for frame in range(60):
                        adapter.send_action({f"{s}_gripper_width": endpoint for s in grippers})
                        time.sleep(max(0, start_endpoint + (frame + 1) / 30 - time.monotonic()))
                    widths = {s: g.states().width for s, g in grippers.items()}
                    data.setdefault("endpoints", []).append({"target": endpoint, "widths": widths})
                    if any(abs(w - endpoint) > 0.002 for w in widths.values()):
                        raise AssertionError("Endpoint did not reach requested width")
                adapter.send_action({f"{s}_gripper_width": 0.03 for s in grippers})
                watchdog_samples = []
                for _ in range(30):
                    time.sleep(0.02)
                    watchdog_samples.append({s: g.states().width for s, g in grippers.items()})
                data["watchdog_samples"] = watchdog_samples
                data["watchdog_stopped"] = all(c._deadline is None for c in adapter._gripper_feedback.values())
                if not data["watchdog_stopped"]:
                    raise AssertionError("Watchdog did not stop after input loss")
                for side in grippers:
                    tail = [r[side] for r in watchdog_samples[-10:]]
                    if max(tail) - min(tail) > 0.001:
                        raise AssertionError("Gripper width did not hold after watchdog stop")
                adapter._close_gripper_feedback()
            both("Stop", {s: () for s in grippers})
            both("Move", {s: (0.07, 0.05, 20.0) for s in grippers})
            time.sleep(1.5)
            for side in grippers:
                widths = [r[side]["width"] for r in data["samples"]]
                errors = [abs(r[side]["width"] - r["target"]) for r in data["samples"]]
                print(side, "range_mm", round((max(widths)-min(widths))*1000, 2),
                      "mae_mm", round(sum(errors)/len(errors)*1000, 2), flush=True)
        except BaseException as exc:
            data["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if adapter is not None:
                adapter._close_gripper_feedback()
            for side, g in grippers.items():
                try:
                    g.Stop()
                except Exception as exc:
                    data.setdefault("stop_errors", {})[side] = str(exc)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
