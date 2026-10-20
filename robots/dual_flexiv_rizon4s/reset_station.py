"""Run a one-shot station reset using the existing robot-reset implementation.

The GN01/48V Grav requires Init before Move. Recording can keep initialization
disabled on connect, but an explicit station reset enables it for that gripper.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

import yaml


def build_reset_config(source: Path) -> dict:
    from scripts.core import reset_robot as shared_reset

    config = yaml.safe_load(source.read_text())
    if "record" not in config:
        config = {"record": {"robot_type": "flexiv_dual_arm", "robot": config["robot"]}}
    config = copy.deepcopy(config)
    record = config["record"]
    robot_type = record["robot_type"]
    if robot_type != "flexiv_dual_arm":
        return config

    detail_path = record.get("das_config_path") or record.get("robot_config_path")
    if detail_path:
        from scripts.core.run_record import _resolve_das_config_path

        root = Path(shared_reset.__file__).resolve().parents[2]
        detail_path = _resolve_das_config_path(robot_type, root / "scripts", root, detail_path)
        detail = yaml.safe_load(detail_path.read_text())
        robot = dict(detail.get("record", detail)["robot"])
        robot.update(record.get("robot") or {})
    else:
        robot = dict(record.get("robot") or shared_reset._load_robot_cfg_from_das(robot_type))

    needs_gn01_init = bool(robot.get("use_gripper")) and any(
        robot.get(f"{side}_gripper_name") == "Flexiv-GN01" for side in ("left", "right")
    )
    robot.update(
        reset_go_home=True,
        go_home_on_connect=False,
        initialize_gripper_on_connect=needs_gn01_init or bool(robot.get("initialize_gripper_on_connect")),
        use_cartesian_servo_thread=False,
        switch_cartesian_mode_on_connect=False,
        zero_ft_sensor_on_connect=False,
        stop_grippers_on_disconnect=False,
    )
    record["robot"] = robot
    return config


def main(argv: list[str] | None = None):
    from robots import create_robot_config
    from scripts.core import reset_robot as shared_reset

    parser = argparse.ArgumentParser(description="Home the station's Flexiv arms and open the grippers.")
    parser.add_argument(
        "--config", "--config-path", type=Path,
        default=Path(__file__).resolve().with_name("station_deepcybo_5090.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show config without connecting hardware.")
    args = parser.parse_args(argv)
    source = args.config.expanduser().resolve()
    config = build_reset_config(source)
    record = config["record"]
    robot_type = record["robot_type"]
    robot = record.get("robot") or shared_reset._load_robot_cfg_from_das(robot_type)
    validated = create_robot_config(robot_type, **{**robot, "debug": False})
    summary = {"config": str(source), "robot_type": robot_type}
    for key in (
        "left_robot_sn", "right_robot_sn", "left_home_joints", "right_home_joints",
        "reset_go_home", "home_joint_max_vel", "home_joint_max_acc", "home_joint_timeout_sec",
        "initialize_gripper_on_connect", "gripper_max_open", "gripper_init_timeout_sec",
        "use_cartesian_servo_thread", "switch_cartesian_mode_on_connect",
    ):
        if hasattr(validated, key):
            summary[key] = getattr(validated, key)
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_OK: no hardware connected; no motion commands sent.")
        return
    with tempfile.TemporaryDirectory(prefix="flexiv-reset-") as folder:
        path = Path(folder) / "reset.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        shared_reset.main(["--config", str(path)])


if __name__ == "__main__":
    main()
