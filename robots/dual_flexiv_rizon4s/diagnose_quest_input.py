"""Quest-only input diagnostic; waits for real controller input before sampling.

No robot/controller connections or motion commands. Keep the headset worn and
Reader foreground throughout. Move each controller with its side Grip held.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core.run_record import RecordConfig
from teleoperators.oculus_teleoperator import OculusTeleop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=float, default=60)
    parser.add_argument("--sample-seconds", type=float, default=25)
    args = parser.parse_args()
    config = RecordConfig(yaml.safe_load((ROOT / "scripts/config/record_cfg.yaml").read_text())["record"])
    teleop = OculusTeleop(config.create_teleop_config())
    teleop.connect()
    reader = teleop.oculus_robot._oculus_reader
    rows, previous = [], {}
    started = time.monotonic()
    active_start = None
    next_log = 0
    try:
        while True:
            now = time.monotonic()
            transforms, buttons = reader.get_transformations_and_buttons()
            action = teleop.get_action()
            distinct = (all(h in transforms for h in ("l", "r"))
                        and np.linalg.norm(transforms["l"][:3, 3] - transforms["r"][:3, 3]) > 0.02)
            if active_start is None and distinct and (buttons.get("LG") or buttons.get("RG")):
                active_start = now
                previous = {}
                print("VALID INPUT: sampling both hands for 25 seconds", flush=True)
            row = {"t": now - started, "active_t": now - active_start if active_start is not None else None,
                   "poses": list(transforms), "distinct_hands": bool(distinct)}
            for hand, side, grip in (("l", "left", "LG"), ("r", "right", "RG")):
                raw = float(np.linalg.norm(transforms[hand][:3, 3] - previous[hand][:3, 3])) if hand in transforms and hand in previous else 0
                output = float(np.linalg.norm([action[f"{side}_delta_ee_pose.{axis}"] for axis in ("x", "y", "z")]))
                row[side] = {"grip": bool(buttons.get(grip, False)), "raw_mm": 1000 * raw, "output_mm": 1000 * output}
            rows.append(row)
            previous = {k: v.copy() for k, v in transforms.items()}
            if now - started >= next_log:
                print(json.dumps(row), flush=True)
                next_log += 2
            if active_start is None and now - started >= args.wait_seconds:
                break
            if active_start is not None and now - active_start >= args.sample_seconds:
                break
            time.sleep(1 / 30)
    finally:
        result = {"arm_motion_commands_sent": False, "valid_input_seen": active_start is not None, "samples": rows}
        for side in ("left", "right"):
            selected = [r[side] for r in rows if r["active_t"] is not None and r[side]["grip"]]
            result[side] = {"grip_frames": len(selected), "raw_path_mm": sum(r["raw_mm"] for r in selected),
                            "output_path_mm": sum(r["output_mm"] for r in selected)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT", json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
        reader.running = False
        reader.thread.join(timeout=0.5)
        teleop.disconnect()


if __name__ == "__main__":
    main()
