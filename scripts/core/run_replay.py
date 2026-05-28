import argparse
import copy
import time
import yaml
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from pathlib import Path
from typing import Dict, Any
from robots import SUPPORTED_ROBOTS, create_robot_config, create_robot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_record_cfg_path() -> Path:
    return _default_scripts_dir() / "config" / "record_cfg.yaml"


ROBOT_DETAIL_CONFIG_FILES = {
    "franka": "franka_config.yaml",
    "franka_dual_arm": "franka_config.yaml",
    "nero_dual_arm": "nero_cofig.yaml",
}


def _load_das_config(robot_type: str) -> Dict[str, Any]:
    config_name = ROBOT_DETAIL_CONFIG_FILES.get(robot_type)
    if config_name is None:
        raise ValueError(
            "No DAS_config mapping is defined for robot_type="
            f"{robot_type!r}. Add replay.robot or extend ROBOT_DETAIL_CONFIG_FILES."
        )
    das_config_path = _default_scripts_dir() / "DAS_config" / config_name
    with open(das_config_path, "r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"DAS config must be a mapping: {das_config_path}")
    detail_cfg = loaded.get("record", loaded)
    if not isinstance(detail_cfg, dict):
        raise ValueError(f"DAS config `record` section must be a mapping: {das_config_path}")
    return detail_cfg


def _hydrate_replay_robot_details(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hydrated = copy.deepcopy(cfg)
    robot_type = hydrated.get("robot_type", "dobot_dual_arm")
    if "robot" in hydrated and hydrated.get("control_mode") is not None:
        return hydrated

    detail_cfg = _load_das_config(robot_type)
    hydrated.setdefault("robot", copy.deepcopy(detail_cfg["robot"]))
    teleop_cfg = detail_cfg.get("teleop", {})
    hydrated.setdefault("control_mode", teleop_cfg.get("control_mode", "oculus"))
    return hydrated


def _load_record_cfg_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "replay" not in cfg:
        raise ValueError(f"Replay config must contain a top-level `replay` mapping: {cfg_path}")
    return cfg


class ReplayConfig:
    def __init__(self, cfg: Dict[str, Any]):
        robot = cfg["robot"]

        # global config
        self.dataset_name: str = cfg["dataset_name"]
        self.episode_idx: int = int(cfg.get("episode_idx", 0))

        # robot config
        # Support both `robot_ip` and legacy `ip` in YAML replay config.
        self.robot_ip: str = robot.get("robot_ip", robot.get("ip", "localhost"))
        self.robot_port: int = robot.get("robot_port", 4242)
        self.control_mode: str = cfg.get("control_mode", "oculus")
        # Finish behavior: mirror run_record defaults.
        self.reset_on_finish: bool = cfg.get("reset_on_finish", True)
        self.disconnect_on_finish: bool = cfg.get("disconnect_on_finish", False)
        
        # Robot type selection (default to dobot_dual_arm for backward compatibility)
        self.robot_type: str = cfg.get("robot_type", "dobot_dual_arm")
        if self.robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot type: {self.robot_type}. "
                f"Supported types: {SUPPORTED_ROBOTS}"
            )

def run_replay(replay_cfg: ReplayConfig):
    episode_idx = replay_cfg.episode_idx

    robot_config = create_robot_config(
        replay_cfg.robot_type,
        robot_ip=replay_cfg.robot_ip,
        robot_port=replay_cfg.robot_port,
        debug=False,
        control_mode=replay_cfg.control_mode
    )
    
    robot = create_robot(replay_cfg.robot_type, robot_config)
    robot.connect()
    dataset = LeRobotDataset(replay_cfg.dataset_name)
    episode_indices = dataset.hf_dataset["episode_index"]
    selected_frame_indices = [
        i for i, ep in enumerate(episode_indices) if int(ep) == episode_idx
    ]

    if not selected_frame_indices:
        available_eps = sorted({int(ep) for ep in episode_indices})
        raise ValueError(
            f"Episode index {episode_idx} not found in dataset {replay_cfg.dataset_name}. "
            f"Available episodes: {available_eps[:20]}"
            + ("..." if len(available_eps) > 20 else "")
        )

    action_names = dataset.features["action"]["names"]
    log_say(
        f"Replaying episode {episode_idx} with {len(selected_frame_indices)} frames"
    )
    for frame_idx in selected_frame_indices:
        t0 = time.perf_counter()
        action_vec = dataset.hf_dataset[frame_idx]["action"]
        action = {
            name: float(action_vec[i]) for i, name in enumerate(action_names)
        }
        # print(f"action: {action}")
        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    # Match run_record finish behavior: reset to home first, then optional disconnect.
    if replay_cfg.reset_on_finish:
        try:
            robot.reset()
        except Exception as reset_err:
            logging.warning(f"[WARNING] reset_on_finish failed: {reset_err}")

    if replay_cfg.disconnect_on_finish:
        robot.disconnect()
    else:
        logging.warning("[INFO] Skip robot.disconnect() to avoid stop/e-stop at session end.")

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Replay a recorded LeRobot episode.")
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

    replay_cfg = ReplayConfig(_hydrate_replay_robot_details(cfg["replay"]))

    run_replay(replay_cfg)


if __name__ == "__main__":
    main()
