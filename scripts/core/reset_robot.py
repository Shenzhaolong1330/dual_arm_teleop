import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from robots import (
    SUPPORTED_ROBOTS,
    create_robot_config,
    create_robot,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_record_cfg_path() -> Path:
    return _default_scripts_dir() / "config" / "record_cfg.yaml"


ROBOT_DETAIL_CONFIG_FILES = {
    "franka": "franka_config.yaml",
    "franka_dual_arm": "franka_config.yaml",
    "nero_dual_arm": "nero_cofig.yaml",
}


def _load_robot_cfg_from_das(robot_type: str) -> Dict[str, Any]:
    config_name = ROBOT_DETAIL_CONFIG_FILES.get(robot_type)
    if config_name is None:
        raise ValueError(
            "No DAS_config mapping is defined for robot_type="
            f"{robot_type!r}. Add record.robot or extend ROBOT_DETAIL_CONFIG_FILES."
        )
    das_config_path = _default_scripts_dir() / "DAS_config" / config_name
    with open(das_config_path, "r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"DAS config must be a mapping: {das_config_path}")
    detail_cfg = loaded.get("record", loaded)
    if not isinstance(detail_cfg, dict) or "robot" not in detail_cfg:
        raise ValueError(f"DAS config must contain a `robot` mapping: {das_config_path}")
    return dict(detail_cfg["robot"])


def _load_record_cfg_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "record" not in cfg:
        raise ValueError(f"Reset config must contain a top-level `record` mapping: {cfg_path}")
    return cfg


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Reset a configured robot to its home position.")
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=_default_record_cfg_path(),
        help="Path to record_cfg.yaml.",
    )
    args = parser.parse_args(argv)
    cfg = _load_record_cfg_yaml(args.config_path)

    record_cfg = cfg["record"]
    robot_type = record_cfg.get("robot_type", "dobot_dual_arm")
    robot_cfg = dict(record_cfg.get("robot") or _load_robot_cfg_from_das(robot_type))
    robot_cfg["debug"] = False
    
    # 创建机器人配置
    robot_config = create_robot_config(
        robot_type=robot_type,
        **robot_cfg,
    )
    
    # 创建机器人实例并连接
    robot = create_robot(robot_type, robot_config)
    print("----------",robot.name)
    robot.connect()
    
    # 重置机器人到初始位置
    logging.info("Resetting robot to home position...")
    robot.reset()
    
    # 断开连接
    # robot.disconnect()
    logging.info("Robot reset completed successfully.")

if __name__ == "__main__":
    main()
