"""Compare single and repeated GN01 Move requests, without arm motion commands.

Run only with recording stopped and both grippers empty and clear. This does not
enable robots, switch arm modes, initialize grippers, or change station settings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-ip", default="192.168.2.200")
    args = parser.parse_args()
    import flexivrdk

    results = {"sdk_version": flexivrdk.__version__, "profiles": [], "error": None}
    robots, grippers = {}, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        def move_both(target):
            calls = [pool.submit(g.Move, target, 0.05, 20.0) for g in grippers.values()]
            for call in calls:
                call.result()

        def settle(target):
            move_both(target)
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                states = [g.states() for g in grippers.values()]
                if all(abs(s.width - target) < 0.002 and not s.is_moving for s in states):
                    return
                time.sleep(0.02)
            raise TimeoutError(f"Setup width {target} was not reached")

        try:
            for side, sn in [("left", "Rizon4s-063173"), ("right", "Rizon4s-063278")]:
                robot = flexivrdk.Robot(sn, [args.host_ip])
                robots[side] = robot
                if robot.fault() or not robot.operational():
                    raise RuntimeError(f"{side} robot not ready; no automatic recovery attempted")
                gripper = flexivrdk.Gripper(robot)
                grippers[side] = gripper
                gripper.Enable("Flexiv-GN01")
                params = gripper.params()
                if not (params.min_width <= 0.029 < 0.07 <= params.max_width
                        and params.min_vel <= 0.05 <= params.max_vel
                        and params.min_force <= 20 <= params.max_force):
                    raise ValueError(f"{side} hardware limits do not permit the bounded test")

            for mode in ("single", "repeat_identical_30hz", "repeat_offset_30hz"):
                settle(0.07)
                profile = {"mode": mode, "samples": [], "commands": []}
                results["profiles"].append(profile)
                start = time.monotonic()
                for frame in range(90):
                    if frame == 0 or mode != "single":
                        target = 0.03
                        if mode == "repeat_offset_30hz":
                            target += 0.001 if frame % 2 else -0.001
                        sent = time.monotonic()
                        move_both(target)
                        profile["commands"].append({
                            "t": sent - start, "target": target,
                            "delivery_sec": time.monotonic() - sent,
                        })
                    sample = {"t": time.monotonic() - start}
                    for side, gripper in grippers.items():
                        state = gripper.states()
                        sample[side] = {"width": state.width, "force": state.force,
                                        "moving": state.is_moving}
                    profile["samples"].append(sample)
                    time.sleep(max(0.0, start + (frame + 1) / 30 - time.monotonic()))
                for side in grippers:
                    widths = [s[side]["width"] for s in profile["samples"]]
                    print(mode, side, "range_mm", round((max(widths)-min(widths))*1000, 2),
                          "final_mm", round(widths[-1]*1000, 2), flush=True)
                save()
            settle(0.07)
        except BaseException as exc:
            results["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for side, gripper in grippers.items():
                try:
                    gripper.Stop()
                except Exception as exc:
                    results.setdefault("stop_errors", {})[side] = str(exc)
            save()


if __name__ == "__main__":
    main()
