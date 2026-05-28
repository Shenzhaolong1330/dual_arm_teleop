import argparse
import copy
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from robots import (
    SUPPORTED_ROBOTS,
    create_robot_config,
    create_robot,
)
from teleoperators import (
    OculusTeleopConfig,
    OculusTeleop,
)
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
# from send2trash import send2trash
import shutil
import termios, sys
import time
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data
from dataclasses import field
from scripts.core.policy_config_utils import (
    build_policy_config,
    load_policy_yaml,
    resolve_policy_config_path,
    self_test_policy_config_loader,
)

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

RUN_MIX_MOVEMENT_EPS = 1e-4
RUN_MIX_CHANGE_EPS = 5e-3
RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS = 0.1
SUCCESS_ANNOTATION_TRUE = {
    "y",
    "yes",
    "1",
    "true",
    "success",
    "succeeded",
    "done",
    "complete",
    "completed",
}
SUCCESS_ANNOTATION_FALSE = {
    "n",
    "no",
    "0",
    "false",
    "fail",
    "failed",
    "failure",
    "aborted",
    "incomplete",
}
SUCCESS_POLICY_NONE = "none"
SUCCESS_POLICY_EXPLICIT = "explicit"
SUCCESS_POLICY_RECORDED_IS_SUCCESS = "recorded_is_success"
SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE = "allow_missing_for_smoke"
VALID_RECORD_SUCCESS_POLICIES = {
    SUCCESS_POLICY_NONE,
    SUCCESS_POLICY_EXPLICIT,
    SUCCESS_POLICY_RECORDED_IS_SUCCESS,
    SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE,
}
RUN_MODE_RECORD = "run_record"
RUN_MODE_POLICY = "run_policy"
RUN_MODE_MIX = "run_mix"
POLICY_RUN_MODES = {RUN_MODE_POLICY, RUN_MODE_MIX}
VALID_RUN_MODES = {RUN_MODE_RECORD, RUN_MODE_POLICY, RUN_MODE_MIX}
GRIPPER_COMMAND_KEY_CANDIDATES = {
    "left": ("left_gripper_cmd", "left_gripper_cmd_bin"),
    "right": ("right_gripper_cmd", "right_gripper_cmd_bin"),
}
FRANKA_EXTRA_ROBOT_CONFIG_KEYS = (
    "schema_mode",
    "rpc_timeout_sec",
    "open_grippers_on_connect",
    "reset_opens_grippers",
    "reset_go_home",
    "go_home_duration_sec",
    "go_home_rate_hz",
    "max_cartesian_delta",
    "max_rotation_delta",
)

def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_project_root() -> Path:
    return _default_scripts_dir().parent


def _default_record_cfg_path() -> Path:
    return _default_scripts_dir() / "config" / "record_cfg.yaml"


ROBOT_DETAIL_CONFIG_FILES = {
    "franka": "franka_config.yaml",
    "franka_dual_arm": "franka_config.yaml",
    "nero_dual_arm": "nero_cofig.yaml",
}

ROBOT_DETAIL_CONFIG_KEYS = ("teleop", "robot", "cameras")


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_das_config_path(
    robot_type: str,
    scripts_dir: Path,
    project_root: Path,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_absolute():
            return path
        candidates = (
            project_root / path,
            scripts_dir / path,
            scripts_dir / "DAS_config" / path,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    config_name = ROBOT_DETAIL_CONFIG_FILES.get(robot_type)
    if config_name is None:
        raise ValueError(
            "No DAS_config mapping is defined for robot_type="
            f"{robot_type!r}. Add record.das_config_path or extend "
            "ROBOT_DETAIL_CONFIG_FILES."
        )
    return scripts_dir / "DAS_config" / config_name


def _load_robot_detail_cfg(
    robot_type: str,
    scripts_dir: Path,
    project_root: Path,
    explicit_path: str | Path | None = None,
) -> Dict[str, Any]:
    das_config_path = _resolve_das_config_path(
        robot_type,
        scripts_dir=scripts_dir,
        project_root=project_root,
        explicit_path=explicit_path,
    )
    with open(das_config_path, "r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"DAS config must be a mapping: {das_config_path}")
    detail_cfg = loaded.get("record", loaded)
    if not isinstance(detail_cfg, dict):
        raise ValueError(f"DAS config `record` section must be a mapping: {das_config_path}")

    missing = [key for key in ROBOT_DETAIL_CONFIG_KEYS if key not in detail_cfg]
    if missing:
        raise ValueError(
            f"DAS config is missing required section(s) {missing}: {das_config_path}"
        )
    return {key: copy.deepcopy(detail_cfg[key]) for key in ROBOT_DETAIL_CONFIG_KEYS}


def _hydrate_record_robot_details(
    cfg: Dict[str, Any],
    scripts_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    hydrated = copy.deepcopy(cfg)
    robot_type = hydrated.get("robot_type", "dobot_dual_arm")
    explicit_path = hydrated.get("das_config_path") or hydrated.get("robot_config_path")
    needs_robot_details = any(key not in hydrated for key in ROBOT_DETAIL_CONFIG_KEYS)
    if not needs_robot_details and explicit_path is None and robot_type not in ROBOT_DETAIL_CONFIG_FILES:
        return hydrated

    detail_cfg = _load_robot_detail_cfg(
        robot_type,
        scripts_dir=scripts_dir,
        project_root=project_root,
        explicit_path=explicit_path,
    )
    for key in ROBOT_DETAIL_CONFIG_KEYS:
        if isinstance(hydrated.get(key), dict):
            hydrated[key] = _deep_merge_dicts(detail_cfg[key], hydrated[key])
        else:
            hydrated[key] = detail_cfg[key]
    return hydrated


def _validate_local_pretrained_path(pretrained_path: str | Path | None) -> None:
    """Fail early when an absolute/local checkpoint path is misspelled."""
    if not pretrained_path:
        return

    raw_path = str(pretrained_path)
    path = Path(raw_path).expanduser()
    is_local_reference = path.is_absolute() or raw_path.startswith(("~", ".")) or path.exists()
    if not is_local_reference:
        return

    if not path.is_dir():
        raise FileNotFoundError(
            "Local pretrained_path does not exist or is not a directory: "
            f"{path}\n"
            "Expected a checkpoint directory containing config.json and model.safetensors. "
            "For example: .../checkpoints/010000/pretrained_model"
        )

    missing = [name for name in ("config.json", "model.safetensors") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Local pretrained_path is missing required file(s): {missing}\n"
            f"Path: {path}"
        )


def _normalize_record_success_policy(task_cfg: Dict[str, Any]) -> str:
    success_policy = task_cfg.get("success_policy")
    if success_policy is None:
        return (
            SUCCESS_POLICY_EXPLICIT
            if bool(task_cfg.get("annotate_success", False))
            else SUCCESS_POLICY_NONE
        )

    success_policy = str(success_policy).strip().lower()
    if success_policy not in VALID_RECORD_SUCCESS_POLICIES:
        raise ValueError(
            "`record.task.success_policy` must be one of "
            f"{sorted(VALID_RECORD_SUCCESS_POLICIES)}. Got: {success_policy!r}"
        )
    return success_policy


class RecordConfig:
    """Configuration class for recording sessions."""
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        scripts_dir: Path | None = None,
        project_root: Path | None = None,
        config_source_name: str = "record_cfg.yaml",
        force_policy_config: bool = False,
    ):
        self.scripts_dir = Path(scripts_dir) if scripts_dir is not None else _default_scripts_dir()
        self.project_root = (
            Path(project_root) if project_root is not None else self.scripts_dir.parent
        )
        self.config_source_name = config_source_name
        cfg = _hydrate_record_robot_details(
            cfg,
            scripts_dir=self.scripts_dir,
            project_root=self.project_root,
        )
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        policy = cfg.get("policy")
        teleop = cfg["teleop"]
        debug_cfg = cfg.get("debug", {})
        debug_options = debug_cfg if isinstance(debug_cfg, dict) else {}
        
        # Global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = bool(
            debug_options.get("robot_debug", cfg.get("robot_debug", False))
            if isinstance(debug_cfg, dict)
            else cfg.get("debug", True)
        )
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: Path = Path(cfg.get("dataset_path", HF_LEROBOT_HOME / self.repo_id))
        self.dataset_name: str | None = cfg.get("dataset_name")
        self.dataset_root: Path | None = (
            Path(cfg["dataset_root"]) if cfg.get("dataset_root") is not None else None
        )
        self.user_info: str = cfg.get("user_notes", None)
        self.run_mode: str = str(cfg.get("run_mode", RUN_MODE_RECORD)).strip().lower()
        if self.run_mode not in VALID_RUN_MODES:
            raise ValueError(
                f"Unsupported `record.run_mode`: {self.run_mode!r}. "
                f"Supported: {sorted(VALID_RUN_MODES)}"
            )
        self.rename_map: dict[str, str] = field(default_factory=dict)
        # Finish behavior: by default reset to home and keep connection to avoid server stop on close.
        self.reset_on_finish: bool = cfg.get("reset_on_finish", True)
        self.disconnect_on_finish: bool = cfg.get("disconnect_on_finish", False)
        # Finish behavior: by default reset to home and keep connection to avoid server stop on close.
        self.reset_on_finish: bool = cfg.get("reset_on_finish", True)
        self.disconnect_on_finish: bool = cfg.get("disconnect_on_finish", False)
        
        # Robot type selection
        self.robot_type: str = cfg.get("robot_type", "dobot_dual_arm")
        if self.robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot type: {self.robot_type}. "
                f"Supported types: {SUPPORTED_ROBOTS}"
            )
        
        # Teleop config - parse based on control mode
        self.control_mode = teleop.get("control_mode", "oculus")
        self.dual_arm = teleop.get("dual_arm", True)
        self._parse_teleop_config(teleop)
        
        # Policy config is only required when the run mode actually executes a policy.
        self.policy_type: str | None = None
        self.policy_config_path: Path | None = None
        self.policy = None
        needs_policy = force_policy_config or self.run_mode in POLICY_RUN_MODES
        if needs_policy:
            if policy is None:
                raise ValueError(
                    "`record.policy` must be a mapping with at least `type` and "
                    "`config_path` when `record.run_mode` is run_policy/run_mix "
                    "or when policy dry-run is requested. "
                    f"Current run_mode: {self.run_mode!r}"
                )
            self._parse_policy_config(policy)
        elif policy is None:
            logging.info(
                "`record.policy` is empty; skipping policy config for run_mode=%s",
                self.run_mode,
            )
        
        # Robot config
        self.robot_ip: str = robot.get("robot_ip", "localhost")
        self.robot_port: int = robot.get("robot_port", 4242)
        self.use_gripper: bool = robot["use_gripper"]
        self.close_threshold = robot.get("close_threshold", 0.5)
        self.gripper_reverse: bool = robot.get("gripper_reverse", False)
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.085)
        self.gripper_force: float = robot.get("gripper_force", 10.0)
        self.gripper_speed: float = robot.get("gripper_speed", 0.1)
        self.robot_extra_config: dict[str, Any] = {
            key: robot[key]
            for key in FRANKA_EXTRA_ROBOT_CONFIG_KEYS
            if key in robot and robot[key] is not None
        }
        
        # Task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", False)
        self.resume_dataset: str = task.get("resume_dataset", "")
        self.success_policy: str = _normalize_record_success_policy(task)
        self.annotate_success: bool = bool(task.get("annotate_success", False))
        self.default_success: bool = bool(task.get("default_success", False))
        self.success_prompt: str = task.get(
            "success_prompt",
            "====== [ANNOTATE] Was this episode successful and suitable "
            "for full-episode training?",
        )
        
        # Time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        # save metadata period (number of episodes between metadata writes)
        # YAML uses `save_meta_period` — use the same name here.
        self.save_meta_period: int = time.get("save_meta_period", 1)
        
        # Cameras config (3 RealSense cameras: left wrist, right wrist, head)
        self.left_wrist_cam_serial: str = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial: str = cam["right_wrist_cam_serial"]
        self.head_cam_serial: str = cam["head_cam_serial"]
        self.cam_width: int = cam["width"]
        self.cam_height: int = cam["height"]
        
        # Storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)
    
    def _parse_teleop_config(self, teleop: Dict[str, Any]) -> None:
        """Parse teleoperation configuration based on control mode."""
        if self.control_mode == "oculus":
            oculus_cfg = teleop.get("oculus_config", {})
            self.use_gripper = oculus_cfg.get("use_gripper", True)
            self.oculus_ip = oculus_cfg.get("ip", "192.168.110.62")
            self.pose_scaler = oculus_cfg.get("pose_scaler", [1.0, 1.0])
            self.channel_signs = oculus_cfg.get("channel_signs", [1, 1, 1, 1, 1, 1])
            self.visualize_placo = oculus_cfg.get("visualize_placo", False)
            self.action_smoothing_alpha = oculus_cfg.get("action_smoothing_alpha", 0.35)
            self.mirror_teleop = oculus_cfg.get("mirror_teleop", False)
            if self.dual_arm:
                self.left_pose_scaler = oculus_cfg.get("left_pose_scaler", self.pose_scaler)
                self.right_pose_scaler = oculus_cfg.get("right_pose_scaler", self.pose_scaler)
                self.left_channel_signs = oculus_cfg.get("left_channel_signs", self.channel_signs)
                self.right_channel_signs = oculus_cfg.get("right_channel_signs", self.channel_signs)
        
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")
    
    def _parse_policy_config(self, policy: Dict[str, Any]) -> None:
        """Parse policy configuration."""
        if not isinstance(policy, dict):
            raise ValueError(
                "`record.policy` must be a mapping with at least `type` and "
                f"`config_path`. Got: {type(policy).__name__}"
            )
        if not policy.get("type"):
            raise ValueError("`record.policy.type` is required when policy config is enabled.")
        self.policy_type = str(policy["type"]).strip().lower()
        self.policy_config_path = resolve_policy_config_path(
            policy,
            scripts_dir=self.scripts_dir,
            project_root=self.project_root,
            mode="reason",
        )
        policy_yaml = load_policy_yaml(self.policy_config_path)
        self.policy = build_policy_config(
            self.policy_type,
            policy_yaml,
            legacy_policy_dict=policy,
            legacy_source_name=self.config_source_name,
            config_path=self.policy_config_path,
            mode="reason",
        )
        _validate_local_pretrained_path(self.policy.pretrained_path)
    
    def create_teleop_config(self):
        """Create teleoperation configuration object."""
        if self.control_mode == "oculus":
            if self.dual_arm:
                return OculusTeleopConfig(
                    use_gripper=self.use_gripper,
                    ip=self.oculus_ip,
                    left_pose_scaler=self.left_pose_scaler,
                    right_pose_scaler=self.right_pose_scaler,
                    left_channel_signs=self.left_channel_signs,
                    right_channel_signs=self.right_channel_signs,
                    action_smoothing_alpha=self.action_smoothing_alpha,
                    mirror_teleop=self.mirror_teleop,
                    visualize_placo=self.visualize_placo,
                )
            return OculusTeleopConfig(
                use_gripper=self.use_gripper,
                ip=self.oculus_ip,
                pose_scaler=self.pose_scaler,
                channel_signs=self.channel_signs,
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")


def handle_incomplete_dataset(dataset_path):
    if dataset_path.exists():
        print(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to delete it? (y/n): ").strip().lower()
        if ans == "y":
            print(f"====== [DELETE] Removing folder: {dataset_path} ======")
            # Send to trash
            # send2trash(dataset_path)
            shutil.rmtree(dataset_path)
            print("====== [DONE] Incomplete dataset folder deleted successfully. ======")
        else:
            print("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")


def _resolve_record_dataset_root(
    dataset_name: str,
    run_mode: str,
    dataset_root: Path | None = None,
) -> Path:
    if dataset_root is not None:
        return dataset_root
    base = Path(HF_LEROBOT_HOME) / dataset_name
    return base


def _clone_action_feature(action_feature: dict[str, Any]) -> dict[str, Any]:
    names = action_feature.get("names")
    return {
        "dtype": action_feature["dtype"],
        "shape": tuple(action_feature["shape"]),
        "names": list(names) if isinstance(names, list) else names,
    }


def _action_names_from_dataset(dataset: LeRobotDataset | None) -> list[str]:
    if dataset is None:
        return []
    action_feature = dataset.features.get(ACTION)
    if action_feature is None:
        return []
    names = action_feature.get("names")
    return list(names) if names is not None else []


def _complete_action_dict(
    action: dict[str, Any],
    action_names: list[str],
    fallback_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_action = fallback_action or {}
    completed = dict(action)
    for name in action_names:
        if name not in completed:
            completed[name] = fallback_action.get(name, 0.0)
    return completed


def _missing_or_invalid_action_names(
    action: dict[str, Any],
    action_names: list[str],
) -> list[str]:
    missing: list[str] = []
    for name in action_names:
        if name not in action or action[name] is None:
            missing.append(name)
            continue
        try:
            value = float(action[name])
        except (TypeError, ValueError):
            missing.append(name)
            continue
        if not np.isfinite(value):
            missing.append(name)
    return missing


def _is_arm_override_active(
    teleop_raw_action: dict[str, Any],
    movement_eps: float = RUN_MIX_MOVEMENT_EPS,
) -> tuple[bool, str]:
    """Detect if teleop currently provides an arm/body override signal."""

    if bool(teleop_raw_action.get("reset_requested", False)):
        return True, "reset_requested"

    if bool(teleop_raw_action.get("left_grip_pressed", False)):
        return True, "left_grip_pressed"
    if bool(teleop_raw_action.get("right_grip_pressed", False)):
        return True, "right_grip_pressed"
    if bool(teleop_raw_action.get("is_expert_override", False)):
        return True, "is_expert_override"

    for key, value in teleop_raw_action.items():
        if key == "reset_requested":
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue

        lower_key = key.lower()
        if "delta" in lower_key or "pose" in lower_key:
            if abs(num) > movement_eps:
                return True, f"{key} motion={num:.4f}"

    return False, "no_arm_override_signal"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_gripper_cmd(value: float) -> float:
    return min(1.0, max(0.0, value))


def _flatten_feature_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    if isinstance(names, dict):
        flattened: list[str] = []
        for value in names.values():
            flattened.extend(_flatten_feature_names(value))
        return flattened
    if isinstance(names, (list, tuple)):
        flattened = []
        for value in names:
            flattened.extend(_flatten_feature_names(value))
        return flattened
    return [str(names)]


def _action_names_from_features_or_names(features_or_names: Any) -> list[str]:
    if features_or_names is None:
        return []
    if isinstance(features_or_names, (list, tuple)):
        return [str(name) for name in features_or_names]
    if isinstance(features_or_names, dict):
        action_feature = features_or_names.get(ACTION)
        if isinstance(action_feature, dict):
            return _flatten_feature_names(action_feature.get("names"))
        if "names" in features_or_names:
            return _flatten_feature_names(features_or_names.get("names"))
        return [str(name) for name in features_or_names.keys()]
    return []


def resolve_gripper_command_keys(features_or_names: Any) -> dict[str, str]:
    """Resolve public gripper command keys from a robot/dataset action schema."""

    action_names = _action_names_from_features_or_names(features_or_names)
    action_name_set = set(action_names)
    resolved: dict[str, str] = {}
    for arm, candidates in GRIPPER_COMMAND_KEY_CANDIDATES.items():
        for key in candidates:
            if key in action_name_set:
                resolved[arm] = key
                break
    return resolved


def get_gripper_action_keys(robot) -> dict[str, str]:
    return resolve_gripper_command_keys(getattr(robot, "action_features", {}))


def _candidate_gripper_keys(arm: str, gripper_keys: dict[str, str]) -> list[str]:
    preferred = gripper_keys.get(arm)
    keys = [preferred] if preferred else []
    keys.extend(GRIPPER_COMMAND_KEY_CANDIDATES.get(arm, ()))
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _gripper_command_value(
    arm: str,
    source: dict[str, Any] | None,
    gripper_keys: dict[str, str],
) -> float | None:
    if source is None:
        return None
    for key in _candidate_gripper_keys(arm, gripper_keys):
        if key not in source:
            continue
        value = _float_or_none(source.get(key))
        if value is not None:
            return _clip_gripper_cmd(value)
    return None


def normalize_gripper_command_keys(
    action: dict[str, Any],
    gripper_keys: dict[str, str],
) -> dict[str, Any]:
    """Copy known gripper aliases into the schema-selected action key."""

    normalized = dict(action)
    for arm, expected_key in gripper_keys.items():
        if expected_key in normalized:
            continue
        for candidate in _candidate_gripper_keys(arm, gripper_keys):
            if candidate in normalized:
                normalized[expected_key] = normalized[candidate]
                break
    return normalized


def _reset_gripper_soft_takeover(state: dict[str, dict[str, Any]]) -> None:
    for arm_state in state.values():
        arm_state["active"] = False
        arm_state["hold"] = None
        arm_state["manual"] = False
        arm_state["ignore_until_released"] = False
        arm_state["released_to_policy"] = False
        arm_state["waiting_logged"] = False


def _current_gripper_cmd(
    arm: str,
    raw_obs: dict[str, Any],
    last_exec_action: dict[str, Any] | None,
    fallback_action: dict[str, Any],
    gripper_keys: dict[str, str],
) -> float | None:
    for source in (last_exec_action, raw_obs, fallback_action):
        value = _gripper_command_value(arm, source, gripper_keys)
        if value is not None:
            return value
    return None


def _gripper_request_reason(
    arm: str,
    teleop_raw_action: dict[str, Any],
    last_teleop_raw_action: dict[str, Any] | None,
    state: dict[str, dict[str, Any]],
    gripper_keys: dict[str, str],
    change_eps: float = RUN_MIX_CHANGE_EPS,
) -> str | None:
    arm_state = state[arm]
    if bool(teleop_raw_action.get(f"{arm}_gripper_release_requested", False)):
        should_log_release = (
            arm_state["active"]
            or arm_state["manual"]
            or not arm_state.get("released_to_policy", False)
            or not arm_state.get("ignore_until_released", False)
        )
        arm_state["active"] = False
        arm_state["hold"] = None
        arm_state["manual"] = False
        arm_state["ignore_until_released"] = True
        arm_state["released_to_policy"] = True
        arm_state["waiting_logged"] = False
        if should_log_release:
            logging.info(
                "[run_mix] %s gripper released to policy. waiting for trigger release before reacquire.",
                arm,
            )
        return None

    if arm_state.get("ignore_until_released", False):
        if bool(teleop_raw_action.get(f"{arm}_trigger_pressed", False)):
            return None
        arm_state["ignore_until_released"] = False
        return None

    if bool(teleop_raw_action.get(f"{arm}_trigger_pressed", False)):
        return f"{arm}_trigger_pressed"

    if last_teleop_raw_action is None:
        return None

    current = _gripper_command_value(arm, teleop_raw_action, gripper_keys)
    previous = _gripper_command_value(arm, last_teleop_raw_action, gripper_keys)
    if current is None or previous is None:
        return None

    delta = current - previous
    if abs(delta) > change_eps:
        return f"{arm}_gripper_trigger_changed"
    return None


def _copy_arm_channels(target_action: dict[str, Any], expert_action: dict[str, Any]) -> set[str]:
    copied: set[str] = set()
    for arm in ("left", "right"):
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            key = f"{arm}_delta_ee_pose.{axis}"
            if key in expert_action:
                target_action[key] = expert_action[key]
                copied.add(key)

    if expert_action.get("reset_requested", False):
        target_action["reset_requested"] = True

    return copied


def _clip_gripper_channels(action: dict[str, Any], gripper_keys: dict[str, str]) -> None:
    for arm in ("left", "right"):
        key = gripper_keys.get(arm)
        if key is None:
            continue
        value = _float_or_none(action.get(key))
        if value is not None:
            action[key] = _clip_gripper_cmd(value)


def _apply_gripper_channel_control(
    arm: str,
    target_action: dict[str, Any],
    expert_action: dict[str, Any],
    raw_obs: dict[str, Any],
    last_exec_action: dict[str, Any] | None,
    last_teleop_raw_action: dict[str, Any] | None,
    fallback_action: dict[str, Any],
    state: dict[str, dict[str, Any]],
    gripper_request_reason: str | None,
    hold_without_manual: bool,
    gripper_keys: dict[str, str],
) -> tuple[bool, str | None]:
    """Apply per-gripper teleop control without letting policy fight that gripper."""
    key = gripper_keys.get(arm)
    if key is None:
        return False, None
    if key not in target_action or key not in expert_action:
        return False, None

    arm_state = state[arm]
    if gripper_request_reason is not None:
        arm_state["manual"] = True
        arm_state["released_to_policy"] = False

    if not arm_state["manual"]:
        if arm_state.get("released_to_policy", False):
            # The user explicitly handed this gripper back to the policy (B for
            # right, Y for left). Do not let the arm-override hold logic below
            # freeze the gripper while the human keeps moving the arm.
            arm_state["hold"] = None
            arm_state["active"] = False
            arm_state["waiting_logged"] = False
            return False, None

        if hold_without_manual:
            hold = _current_gripper_cmd(
                arm,
                raw_obs,
                last_exec_action,
                fallback_action,
                gripper_keys,
            )
            if hold is not None:
                target_action[key] = hold
            return False, None

        arm_state["hold"] = None
        arm_state["active"] = False
        arm_state["waiting_logged"] = False
        return False, None

    teleop_cmd = _float_or_none(expert_action[key])
    if teleop_cmd is None:
        return False, None
    teleop_cmd = _clip_gripper_cmd(teleop_cmd)

    if arm_state["hold"] is None:
        hold = _current_gripper_cmd(
            arm,
            raw_obs,
            last_exec_action,
            fallback_action,
            gripper_keys,
        )
        arm_state["hold"] = teleop_cmd if hold is None else hold

    hold = arm_state["hold"]
    if not arm_state["active"]:
        previous_teleop_cmd = None
        if last_teleop_raw_action is not None:
            previous_teleop_cmd = _gripper_command_value(arm, last_teleop_raw_action, gripper_keys)

        takeover_matched = abs(teleop_cmd - hold) <= RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS
        if previous_teleop_cmd is not None:
            takeover_matched = takeover_matched or (
                abs(previous_teleop_cmd - hold) <= RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS
            )

        if takeover_matched:
            arm_state["active"] = True
            if arm_state["waiting_logged"]:
                logging.info(
                    "[run_mix] %s gripper soft takeover acquired. hold=%.3f teleop=%.3f",
                    arm,
                    hold,
                    teleop_cmd,
                )
        else:
            target_action[key] = hold
            if not arm_state["waiting_logged"]:
                logging.info(
                    "[run_mix] %s gripper soft takeover waiting. hold=%.3f teleop=%.3f",
                    arm,
                    hold,
                    teleop_cmd,
                )
                arm_state["waiting_logged"] = True
            return True, f"{arm}_gripper_waiting"

    target_action[key] = teleop_cmd
    return True, gripper_request_reason or f"{arm}_gripper_active"


def run_mix_record_loop(
    robot,
    teleop,
    policy,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset | None,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    events: dict,
    fps: int,
    control_time_s: int | float,
    single_task: str,
    display_data: bool,
    success_policy: str = SUCCESS_POLICY_NONE,
) -> dict[str, Any]:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_episode_t = time.perf_counter()
    timestamp_s = 0.0
    last_teleop_raw_action: dict[str, Any] | None = None

    expert_exec_steps = 0
    policy_exec_steps = 0
    saved_steps = 0
    intervention_count = 0
    complete_expert_label_steps = 0

    last_action_source = "policy"
    last_exec_action: dict[str, Any] | None = None
    action_names = _action_names_from_dataset(dataset)
    gripper_action_keys = resolve_gripper_command_keys(action_names)
    if gripper_action_keys:
        logging.info("[run_mix] gripper action keys: %s", gripper_action_keys)
    missing_gripper_warning_keys: set[str] = set()
    intervention_segment_id = -1
    intervention_active = False
    gripper_soft_takeover = {
        "left": {
            "active": False,
            "hold": None,
            "manual": False,
            "ignore_until_released": False,
            "released_to_policy": False,
            "waiting_logged": False,
        },
        "right": {
            "active": False,
            "hold": None,
            "manual": False,
            "ignore_until_released": False,
            "released_to_policy": False,
            "waiting_logged": False,
        },
    }
    while timestamp_s < control_time_s:
        loop_start_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        raw_obs = robot.get_observation()
        obs_processed = robot_observation_processor(raw_obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # (1) Policy inference.
        policy_action = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        policy_action_processed = make_robot_action(policy_action, dataset.features)
        policy_action_processed = _complete_action_dict(policy_action_processed, action_names)
        policy_action_processed = normalize_gripper_command_keys(policy_action_processed, gripper_action_keys)

        # (2) Default execute policy action. Teleop may override selected channels below.
        exec_action = dict(policy_action_processed)
        action_source = "policy"
        is_expert = False
        frame_role = "policy"
        frame_segment_id = -1

        # (3) Expert override is split by channel: arm/body and grippers are independent.
        teleop_raw_action = teleop.get_action()
        expert_action_raw = normalize_gripper_command_keys(
            dict(teleop_action_processor((teleop_raw_action, raw_obs))),
            gripper_action_keys,
        )
        for arm, key in gripper_action_keys.items():
            if key in expert_action_raw or key in missing_gripper_warning_keys:
                continue
            logging.warning(
                "[run_mix] expert action is missing expected %s gripper key `%s`; "
                "known aliases are %s. This frame will not be a complete expert label.",
                arm,
                key,
                list(GRIPPER_COMMAND_KEY_CANDIDATES[arm]),
            )
            missing_gripper_warning_keys.add(key)
        expert_action_missing_names = _missing_or_invalid_action_names(expert_action_raw, action_names)
        expert_action = dict(expert_action_raw)
        is_arm_override, arm_override_reason = _is_arm_override_active(teleop_raw_action)
        gripper_request_reasons = {
            arm: _gripper_request_reason(
                arm,
                teleop_raw_action,
                last_teleop_raw_action,
                gripper_soft_takeover,
                gripper_action_keys,
            )
            for arm in ("left", "right")
        }

        override_reasons: list[str] = []
        overridden_action_names: set[str] = set()

        if is_arm_override:
            overridden_action_names.update(_copy_arm_channels(exec_action, expert_action))
            override_reasons.append(arm_override_reason)
            # Flush policy temporal state only for arm/body interventions. Gripper-only
            # control should not disturb policy arm motion.
            policy.reset()

        for arm, gripper_reason in gripper_request_reasons.items():
            gripper_overridden, gripper_override_reason = _apply_gripper_channel_control(
                arm=arm,
                target_action=exec_action,
                expert_action=expert_action,
                raw_obs=raw_obs,
                last_exec_action=last_exec_action,
                last_teleop_raw_action=last_teleop_raw_action,
                fallback_action=policy_action_processed,
                state=gripper_soft_takeover,
                gripper_request_reason=gripper_reason,
                hold_without_manual=is_arm_override,
                gripper_keys=gripper_action_keys,
            )
            if gripper_overridden and gripper_override_reason is not None:
                gripper_key = gripper_action_keys.get(arm)
                if gripper_key is not None:
                    overridden_action_names.add(gripper_key)
                override_reasons.append(gripper_override_reason)

        if override_reasons:
            action_source = (
                "expert"
                if action_names and overridden_action_names.issuperset(set(action_names))
                else "mixed"
            )
            is_expert = True
            if last_action_source == "policy":
                logging.info(
                    "[run_mix] source->%s (teleop override). reason=%s",
                    action_source,
                    ", ".join(override_reasons),
                )
        elif last_action_source in {"expert", "mixed"}:
            logging.info(
                "[run_mix] source->policy (no expert override detected). last_reason=%s",
                "no_channel_override_signal",
            )
            _reset_gripper_soft_takeover(gripper_soft_takeover)
        else:
            pass

        _clip_gripper_channels(exec_action, gripper_action_keys)
        # Keep the expert label independent from policy/sent action. Missing
        # dimensions are zero-filled only so the raw LeRobot schema can be
        # written; export drops these rows via expert_label_complete=False.
        released_to_policy_action_names = [
            name
            for arm in ("left", "right")
            if gripper_soft_takeover[arm].get("released_to_policy", False)
            for name in [gripper_action_keys.get(arm)]
            if name is not None
        ]
        expert_action_missing_names = sorted(
            set(expert_action_missing_names) | set(released_to_policy_action_names)
        )
        expert_action = _complete_action_dict(expert_action_raw, action_names, fallback_action={})
        last_action_source = action_source

        reset_requested = bool(teleop_raw_action.get("reset_requested", False))
        waiting_only = bool(override_reasons) and all("waiting" in reason for reason in override_reasons)
        expert_label_complete = (
            bool(is_expert)
            and not waiting_only
            and not reset_requested
            and len(expert_action_missing_names) == 0
        )
        if is_expert and not waiting_only:
            if not intervention_active:
                intervention_segment_id += 1
                intervention_count += 1
                frame_role = "takeover_start"
            else:
                frame_role = "recovery"
            if reset_requested:
                frame_role = "reset"
            frame_segment_id = intervention_segment_id
            intervention_active = True
        elif waiting_only:
            frame_role = "ignore"
            frame_segment_id = -1
            intervention_active = False
        else:
            frame_role = "policy"
            frame_segment_id = -1
            intervention_active = False

        logging.debug(
            "[run_mix] action_source=%s frame_role=%s segment=%s reason=%s"
            " left_grip=%s right_grip=%s reset=%s",
            action_source,
            frame_role,
            frame_segment_id,
            ",".join(override_reasons) if override_reasons else "no_channel_override_signal",
            teleop_raw_action.get("left_grip_pressed", False),
            teleop_raw_action.get("right_grip_pressed", False),
            teleop_raw_action.get("reset_requested", False),
        )

        last_teleop_raw_action = teleop_raw_action

        # (4) Execute action.
        robot_action_to_send = robot_action_processor((exec_action, raw_obs))
        sent_action = normalize_gripper_command_keys(
            _complete_action_dict(dict(robot_action_to_send), action_names, fallback_action=exec_action),
            gripper_action_keys,
        )
        sent_action_raw = robot.send_action(sent_action)
        sent_action = normalize_gripper_command_keys(
            _complete_action_dict(
                dict(sent_action_raw or sent_action),
                action_names,
                fallback_action=exec_action,
            ),
            gripper_action_keys,
        )
        last_exec_action = dict(sent_action)

        if action_source in {"expert", "mixed"}:
            expert_exec_steps += 1
        else:
            policy_exec_steps += 1
        if expert_label_complete:
            complete_expert_label_steps += 1

        # Raw run_mix logs intentionally keep the full timeline. `action` remains
        # the actual mixed command sent to the robot; export later rewrites
        # `action = expert_action` only when expert_label_complete=True.
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            policy_action_frame = build_dataset_frame(
                dataset.features,
                policy_action_processed,
                prefix="policy_action",
            )
            expert_action_frame = build_dataset_frame(
                dataset.features,
                expert_action,
                prefix="expert_action",
            )
            sent_action_frame = build_dataset_frame(dataset.features, sent_action, prefix="sent_action")
            frame = {
                **observation_frame,
                **action_frame,
                **policy_action_frame,
                **expert_action_frame,
                **sent_action_frame,
                "action_source": action_source,
                "is_expert": np.array([is_expert], dtype=np.bool_),
                "intervention_segment_id": np.array([frame_segment_id], dtype=np.int64),
                "frame_role": frame_role,
                "expert_label_complete": np.array([expert_label_complete], dtype=np.bool_),
                "expert_action_missing": ",".join(expert_action_missing_names),
                "task": single_task,
            }
            if "success" in dataset.features:
                frame["success"] = np.array([False], dtype=np.bool_)
            if "success_policy" in dataset.features:
                frame["success_policy"] = success_policy
            if "success_inferred_from_recorded_episode" in dataset.features:
                frame["success_inferred_from_recorded_episode"] = np.array([False], dtype=np.bool_)
            dataset.add_frame(frame)
            saved_steps += 1

        if display_data:
            log_rerun_data(observation=obs_processed, action=sent_action)

        dt_s = time.perf_counter() - loop_start_t
        busy_wait(1 / fps - dt_s)
        timestamp_s = time.perf_counter() - start_episode_t

    return {
        "expert_exec_steps": expert_exec_steps,
        "policy_exec_steps": policy_exec_steps,
        "saved_steps": saved_steps,
        "expert_frame_ratio": expert_exec_steps / max(1, expert_exec_steps + policy_exec_steps),
        "intervention_count": intervention_count,
        "complete_expert_label_steps": complete_expert_label_steps,
    }


def _prompt_episode_success(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    prompt_text = prompt if "[y/n]" in prompt.lower() else f"{prompt} [{suffix}]: "
    while True:
        try:
            answer = input(prompt_text).strip().lower()
        except EOFError:
            logging.warning(
                "[run_mix] no stdin available for success annotation; using default=%s",
                default,
            )
            return bool(default)
        if not answer:
            return bool(default)
        if answer in SUCCESS_ANNOTATION_TRUE:
            return True
        if answer in SUCCESS_ANNOTATION_FALSE:
            return False
        logging.info("====== [WARNING] Please answer y/yes or n/no. ======")


def _set_episode_success_annotation(
    dataset: LeRobotDataset,
    *,
    success: bool | None,
    success_policy: str,
    inferred_from_recorded_episode: bool,
) -> None:
    if dataset.episode_buffer is None:
        return
    size = int(dataset.episode_buffer.get("size", 0) or 0)
    if "success" in dataset.episode_buffer and success is not None:
        dataset.episode_buffer["success"] = [
            np.array([success], dtype=np.bool_)
            for _ in range(size)
        ]
    if "success_policy" in dataset.episode_buffer:
        dataset.episode_buffer["success_policy"] = [success_policy for _ in range(size)]
    if "success_inferred_from_recorded_episode" in dataset.episode_buffer:
        dataset.episode_buffer["success_inferred_from_recorded_episode"] = [
            np.array([inferred_from_recorded_episode], dtype=np.bool_)
            for _ in range(size)
        ]


def run_record(record_cfg: RecordConfig):
    print("====== [START] Starting recording ======")
    dataset_name = None
    dataset_root = None
    try:
        if record_cfg.dataset_name is not None:
            dataset_name = record_cfg.dataset_name
            data_version = "manual"
        else:
            dataset_name, data_version = generate_dataset_name(record_cfg)
        dataset_root = _resolve_record_dataset_root(
            dataset_name,
            record_cfg.run_mode,
            dataset_root=record_cfg.dataset_root,
        )

        # Check joint offsets
        # if not record_cfg.debug:
        #     check_joint_offsets(record_cfg)        
        
        # Create RealSenseCamera configurations (3 cameras: left wrist, right wrist, head)
        left_wrist_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.left_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        right_wrist_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.right_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        head_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.head_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        # Create the robot and teleoperator configurations
        camera_config = {
            "left_wrist_image": left_wrist_image_cfg,
            "right_wrist_image": right_wrist_image_cfg,
            "head_image": head_image_cfg,
        }
        
        # Create teleop config using the new method
        teleop_config = record_cfg.create_teleop_config()
        
        # Create robot configuration dynamically based on robot_type
        robot_config_kwargs = {
            "robot_ip": record_cfg.robot_ip,
            "robot_port": record_cfg.robot_port,
            "cameras": camera_config,
            "debug": record_cfg.debug,
            "use_gripper": record_cfg.use_gripper,
            "gripper_max_open": record_cfg.gripper_max_open,
            "gripper_force": record_cfg.gripper_force,
            "gripper_speed": record_cfg.gripper_speed,
            "close_threshold": record_cfg.close_threshold,
            "gripper_reverse": record_cfg.gripper_reverse,
            "control_mode": record_cfg.control_mode,
            **record_cfg.robot_extra_config,
        }
        robot_config = create_robot_config(
            record_cfg.robot_type,
            **robot_config_kwargs,
        )
        
        # Initialize the robot dynamically based on robot_type
        robot = create_robot(record_cfg.robot_type, robot_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}
        if record_cfg.run_mode == RUN_MODE_MIX:
            # Extend dataset schema for DAgger mixed collection metadata.
            action_feature = dataset_features[ACTION]
            dataset_features["policy_action"] = _clone_action_feature(action_feature)
            dataset_features["expert_action"] = _clone_action_feature(action_feature)
            dataset_features["sent_action"] = _clone_action_feature(action_feature)
            dataset_features["action_source"] = {"dtype": "string", "shape": (1,), "names": None}
            dataset_features["is_expert"] = {"dtype": "bool", "shape": (1,), "names": None}
            dataset_features["intervention_segment_id"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }
            dataset_features["frame_role"] = {"dtype": "string", "shape": (1,), "names": None}
            dataset_features["expert_label_complete"] = {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            }
            dataset_features["expert_action_missing"] = {
                "dtype": "string",
                "shape": (1,),
                "names": None,
            }
            if (
                record_cfg.annotate_success
                or record_cfg.success_policy == SUCCESS_POLICY_RECORDED_IS_SUCCESS
            ):
                dataset_features["success"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }
            if record_cfg.success_policy != SUCCESS_POLICY_NONE:
                dataset_features["success_policy"] = {"dtype": "string", "shape": (1,), "names": None}
                dataset_features["success_inferred_from_recorded_episode"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }

        if record_cfg.run_mode == RUN_MODE_MIX:
            logging.info("====== [RUN_MIX] Mix mode config ======")
            logging.info(
                "[run_mix] robot_type=%s control_mode=%s fps=%s episode_time=%s reset_time=%s",
                record_cfg.robot_type,
                record_cfg.control_mode,
                record_cfg.fps,
                record_cfg.episode_time_sec,
                record_cfg.reset_time_sec,
            )
            logging.info(
                "[run_mix] policy=%s policy_device=%s pretrained_path=%s",
                type(record_cfg.policy).__name__,
                record_cfg.policy.device,
                record_cfg.policy.pretrained_path,
            )
            logging.info(
                "[run_mix] teleop dual_arm=%s oculus_ip=%s left_scaler=%s left_signs=%s right_scaler=%s right_signs=%s",
                record_cfg.dual_arm,
                getattr(record_cfg, "oculus_ip", "n/a"),
                getattr(record_cfg, "left_pose_scaler", None),
                getattr(record_cfg, "left_channel_signs", None),
                getattr(record_cfg, "right_pose_scaler", None),
                getattr(record_cfg, "right_channel_signs", None),
            )
            logging.info(
                "[run_mix] override detection: movement_eps=%.4f change_eps=%.4f gripper_soft_takeover_eps=%.4f",
                RUN_MIX_MOVEMENT_EPS,
                RUN_MIX_CHANGE_EPS,
                RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS,
            )
            logging.info(
                "[run_mix] success_policy=%s annotate_success=%s",
                record_cfg.success_policy,
                record_cfg.annotate_success,
            )

        if record_cfg.resume:
            dataset = LeRobotDataset(
                dataset_name,
                root=dataset_root,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer()
            sanity_check_dataset_robot_compatibility(dataset, robot, record_cfg.fps, dataset_features)
        else:
            # # Create the dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=record_cfg.fps,
                features=dataset_features,
                robot_type=robot.name,
                root=dataset_root,
                use_videos=True,
                image_writer_threads=4,
            )
        # Set the episode metadata buffer size to 1, so that each episode is saved immediately
        dataset.meta.metadata_buffer_size = record_cfg.save_meta_period

        # Initialize keyboard listener.
        # Rerun visualization can introduce periodic stalls when transport is unstable,
        # so only initialize it when display is explicitly enabled.
        # Initialize keyboard listener.
        # Rerun visualization can introduce periodic stalls when transport is unstable,
        # so only initialize it when display is explicitly enabled.
        _, events = init_keyboard_listener()
        if record_cfg.display:
            init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        preprocessor = None
        postprocessor = None

        # configure the teleop and policy
        if record_cfg.run_mode == RUN_MODE_RECORD:
            logging.info("====== [INFO] Running in teleoperation mode ======")
            teleop = OculusTeleop(teleop_config)
            policy = None
        elif record_cfg.run_mode == RUN_MODE_POLICY:
            logging.info("====== [INFO] Running in policy mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = None
        elif record_cfg.run_mode == RUN_MODE_MIX:
            logging.info("====== [INFO] Running in mixed mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = OculusTeleop(teleop_config)
        else:
            raise ValueError(
                f"Unsupported run_mode: {record_cfg.run_mode}. "
                f"Supported: {RUN_MODE_RECORD} | {RUN_MODE_POLICY} | {RUN_MODE_MIX}"
            )
        
        if policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=record_cfg.policy,
                pretrained_path=record_cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, {}),  # 使用空字典作为rename_map
                preprocessor_overrides={
                    "device_processor": {"device": record_cfg.policy.device},
                    "rename_observations_processor": {"rename_map": {}},  # 使用空字典作为rename_map
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        episode_idx = 0
        run_mix_episode_stats: list[dict[str, Any]] = []

        while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"====== [RECORD] Recording episode {episode_idx + 1} of {record_cfg.num_episodes} ======")
            if record_cfg.run_mode == RUN_MODE_MIX:
                mix_stats = run_mix_record_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    events=events,
                    fps=record_cfg.fps,
                    control_time_s=record_cfg.episode_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                    success_policy=record_cfg.success_policy,
                )
                mix_stats["episode_index"] = episode_idx
                run_mix_episode_stats.append(mix_stats)
                logging.info(
                    "[run_mix] policy_exec_steps=%d expert_exec_steps=%d saved_steps=%d "
                    "expert_frame_ratio=%.4f interventions=%d complete_expert_labels=%d",
                    mix_stats["policy_exec_steps"],
                    mix_stats["expert_exec_steps"],
                    mix_stats["saved_steps"],
                    mix_stats["expert_frame_ratio"],
                    mix_stats["intervention_count"],
                    mix_stats["complete_expert_label_steps"],
                )
            else:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    control_time_s=record_cfg.episode_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            rerecord_requested = bool(events["rerecord_episode"])
            if rerecord_requested:
                logging.info("Re-recording episode requested: discard current episode and enter reset state.")
                events["rerecord_episode"] = False
                # Left arrow also sets exit_early=True. Clear it here so reset phase does not exit immediately.
                events["exit_early"] = False
                if dataset.episode_buffer is not None:
                    dataset.clear_episode_buffer()
            elif record_cfg.run_mode == RUN_MODE_MIX:
                has_recorded_frames = (
                    dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0
                )
                if has_recorded_frames:
                    if (
                        record_cfg.success_policy == SUCCESS_POLICY_EXPLICIT
                        and record_cfg.annotate_success
                    ):
                        success = _prompt_episode_success(
                            record_cfg.success_prompt,
                            default=record_cfg.default_success,
                        )
                        _set_episode_success_annotation(
                            dataset,
                            success=success,
                            success_policy=record_cfg.success_policy,
                            inferred_from_recorded_episode=False,
                        )
                        mix_stats["success"] = success
                        mix_stats["success_policy"] = record_cfg.success_policy
                        mix_stats["success_inferred_from_recorded_episode"] = False
                        logging.info("[run_mix] episode %d success=%s", episode_idx + 1, success)
                    elif record_cfg.success_policy == SUCCESS_POLICY_RECORDED_IS_SUCCESS:
                        _set_episode_success_annotation(
                            dataset,
                            success=True,
                            success_policy=record_cfg.success_policy,
                            inferred_from_recorded_episode=True,
                        )
                        mix_stats["success"] = True
                        mix_stats["success_policy"] = record_cfg.success_policy
                        mix_stats["success_inferred_from_recorded_episode"] = True
                        logging.info(
                            "[run_mix] episode %d success=True inferred from saved recorded episode",
                            episode_idx + 1,
                        )
                    elif record_cfg.success_policy == SUCCESS_POLICY_ALLOW_MISSING_FOR_SMOKE:
                        _set_episode_success_annotation(
                            dataset,
                            success=None,
                            success_policy=record_cfg.success_policy,
                            inferred_from_recorded_episode=False,
                        )
                        mix_stats["success_policy"] = record_cfg.success_policy
                    dataset.save_episode()
                else:
                    logging.warning(
                        "[run_mix] episode %d has no recorded frames; skip saving this episode.",
                        episode_idx + 1,
                    )
            else:
                dataset.save_episode()

            # Reset the environment between episodes, and also before a re-record attempt.
            if not events["stop_recording"] and (episode_idx < record_cfg.num_episodes - 1 or rerecord_requested):
                while True:
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    user_input = input("====== [WAIT] Press Enter to reset the environment ======")
                    if user_input == "":
                        break  
                    else:
                        logging.info("====== [WARNING] Please press only Enter to continue ======")

                logging.info("====== [RESET] Resetting the environment ======")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=record_cfg.reset_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            if rerecord_requested:
                continue

            episode_idx += 1

        # Clean up
        logging.info("Stop recording")

        # Reset robot to home position at the end (same intent as pressing A in teleop).
        if record_cfg.reset_on_finish:
            try:
                robot.reset()
            except Exception as reset_err:
                logging.warning(f"[WARNING] reset_on_finish failed: {reset_err}")

        # Optional disconnect. For Nero, disconnect triggers client.close() -> robot_stop on server.
        if record_cfg.disconnect_on_finish:
            robot.disconnect()
        else:
            logging.info("[INFO] Skip robot.disconnect() to avoid stop/e-stop at session end.")

        if teleop is not None:
            teleop.disconnect()
        dataset.finalize()

        update_dataset_info(record_cfg, dataset_name, data_version)
        if record_cfg.push_to_hub:
            dataset.push_to_hub()

        result = {
            "dataset_name": dataset_name,
            "dataset_root": str(dataset_root),
            "data_version": data_version,
        }
        if record_cfg.run_mode == RUN_MODE_MIX:
            total_expert = sum(s["expert_exec_steps"] for s in run_mix_episode_stats)
            total_policy = sum(s["policy_exec_steps"] for s in run_mix_episode_stats)
            result["run_mix_stats"] = {
                "episodes": run_mix_episode_stats,
                "expert_frame_ratio": total_expert / max(1, total_expert + total_policy),
                "expert_episode_count": sum(
                    1 for s in run_mix_episode_stats if s["intervention_count"] > 0
                ),
                "intervention_count": sum(s["intervention_count"] for s in run_mix_episode_stats),
                "saved_steps": sum(s["saved_steps"] for s in run_mix_episode_stats),
                "complete_expert_label_steps": sum(
                    s["complete_expert_label_steps"] for s in run_mix_episode_stats
                ),
            }
        return result

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        dataset_path = dataset_root if dataset_root is not None else Path(HF_LEROBOT_HOME) / str(dataset_name)
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        dataset_path = dataset_root if dataset_root is not None else Path(HF_LEROBOT_HOME) / str(dataset_name)
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def _load_record_cfg_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "record" not in cfg:
        raise ValueError(f"Record config must contain a top-level `record` mapping: {cfg_path}")
    return cfg


def dry_run_policy_config(cfg_path: Path) -> RecordConfig:
    scripts_dir = _default_scripts_dir()
    project_root = _default_project_root()
    cfg = _load_record_cfg_yaml(cfg_path)
    record_cfg = RecordConfig(
        cfg["record"],
        scripts_dir=scripts_dir,
        project_root=project_root,
        config_source_name=str(cfg_path),
        force_policy_config=True,
    )
    logging.info("====== [POLICY CONFIG DRY-RUN] OK ======")
    logging.info("policy.type: %s", record_cfg.policy_type)
    logging.info("policy.config_path: %s", record_cfg.policy_config_path)
    logging.info("policy.config_class: %s", type(record_cfg.policy).__name__)
    logging.info("policy.device: %s", record_cfg.policy.device)
    logging.info("policy.pretrained_path: %s", record_cfg.policy.pretrained_path)
    return record_cfg


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Record LeRobot dual-arm data.")
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=_default_record_cfg_path(),
        help="Path to record_cfg.yaml.",
    )
    parser.add_argument(
        "--dry-run-policy-config",
        action="store_true",
        help="Load record_cfg.yaml and the referenced policy yaml, build the policy config, then exit.",
    )
    parser.add_argument(
        "--self-test-policy-config",
        action="store_true",
        help="Run minimal in-process checks for the policy config loader, then exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test_policy_config:
        self_test_policy_config_loader()
        return

    if args.dry_run_policy_config:
        dry_run_policy_config(args.config_path)
        return

    scripts_dir = _default_scripts_dir()
    project_root = _default_project_root()
    cfg = _load_record_cfg_yaml(args.config_path)
    record_cfg = RecordConfig(
        cfg["record"],
        scripts_dir=scripts_dir,
        project_root=project_root,
        config_source_name=str(args.config_path),
    )
    run_record(record_cfg)

if __name__ == "__main__":
    main()
