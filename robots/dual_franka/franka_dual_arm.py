"""LeRobot Robot implementation for ROS2 dual-Franka + Robotiq."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_franka import FrankaDualArmConfig
from .dual_franka_robotiq_rpc_client import FrankaDualArmClient

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _as_np(values: Any, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float).reshape(-1)
    n = min(length, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _robot_state_from_side(side_state: Mapping[str, Any]) -> Mapping[str, Any]:
    robot_state = side_state.get("robot_state")
    return _as_mapping(robot_state) if robot_state is not None else side_state


def _ee_pose_from_side(side_state: Mapping[str, Any]) -> np.ndarray:
    robot_state = _robot_state_from_side(side_state)
    if "end_pose" in side_state:
        return _as_np(side_state.get("end_pose"), 6)
    if "end_pose" in robot_state:
        return _as_np(robot_state.get("end_pose"), 6)

    eef_pose = _as_mapping(robot_state.get("eef_pose"))
    if not eef_pose:
        return np.zeros(6, dtype=float)
    position = _as_np(eef_pose.get("position"), 3)
    quat = _as_np(eef_pose.get("orientation_xyzw"), 4)
    rotvec = np.zeros(3, dtype=float)
    if np.linalg.norm(quat) > 1e-12:
        try:
            rotvec = R.from_quat(quat).as_rotvec()
        except ValueError:
            rotvec = np.zeros(3, dtype=float)
    return np.concatenate([position, rotvec])


def _gripper_open_fraction(gripper_state: Any, fallback: float) -> float:
    if isinstance(gripper_state, Mapping):
        if "open_fraction" in gripper_state:
            return _clamp(_safe_float(gripper_state.get("open_fraction"), fallback), 0.0, 1.0)
        if "position" in gripper_state:
            position = _safe_float(gripper_state.get("position"), fallback)
            open_position = _safe_float(gripper_state.get("open_position"), 0.0)
            closed_position = _safe_float(gripper_state.get("closed_position"), 1.0)
            span = closed_position - open_position
            if abs(span) > 1e-9:
                closed_fraction = (position - open_position) / span
                return _clamp(1.0 - closed_fraction, 0.0, 1.0)
            return _clamp(position, 0.0, 1.0)
        return fallback
    return _clamp(_safe_float(gripper_state, fallback), 0.0, 1.0)


def _gripper_open_fraction_from_side(side_state: Mapping[str, Any], fallback: float) -> float:
    if "gripper" in side_state:
        return _gripper_open_fraction(side_state.get("gripper"), fallback)
    sensors = _as_mapping(side_state.get("sensors"))
    if "robotiq" in sensors:
        return _gripper_open_fraction(sensors.get("robotiq"), fallback)
    return fallback


class FrankaDualArm(Robot):
    """Dual Franka robot controlled through a ROS2 ZeroRPC bridge."""

    config_class = FrankaDualArmConfig
    name = "franka_dual_arm"

    def __init__(self, config: FrankaDualArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        self._robot: Optional[FrankaDualArmClient] = None
        self._prev_observation: Optional[dict[str, Any]] = None
        self._num_joints_per_arm = int(config.num_joints_per_arm)

        self._last_left_gripper_open: Optional[float] = None
        self._last_right_gripper_open: Optional[float] = None
        self._left_gripper_state = 1.0
        self._right_gripper_state = 1.0
        self._warned_joint_control = False
        self._delta_clip_warn_count = 0
        self._nonfinite_action_warn_count = 0

    # ==================== Connection ====================

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        logger.info("[FRANKA] Connecting to dual-arm server %s:%s", self.config.robot_ip, self.config.robot_port)
        self._robot = FrankaDualArmClient(
            ip=self.config.robot_ip,
            port=self.config.robot_port,
            timeout=self.config.rpc_timeout_sec,
        )
        logger.info("[FRANKA] Server ping: %s", self._robot.ping())

        if self.config.use_gripper:
            try:
                self._robot.gripper_initialize()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FRANKA] Gripper initialize failed: %s", exc)
            if self.config.open_grippers_on_connect:
                self._open_both_grippers(blocking=True)

        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info("[CAM] %s connected", cam_name)

        self._is_connected = True
        logger.info("[FRANKA] %s connected", self.name)

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        for cam in self.cameras.values():
            cam.disconnect()
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._is_connected = False
        logger.info("[FRANKA] %s disconnected", self.name)

    # ==================== Reset ====================

    def reset(self) -> None:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        if self.config.reset_go_home:
            logger.info("[FRANKA] Moving both arms to server home pose")
            self._robot.go_home(
                "both",
                self.config.go_home_duration_sec,
                self.config.go_home_rate_hz,
            )
        else:
            logger.info("[FRANKA] Resetting target poses to current poses")
            self._robot.reset()
        if self.config.use_gripper and self.config.reset_opens_grippers:
            self._open_both_grippers(blocking=True)

    def _open_both_grippers(self, blocking: bool = True) -> None:
        if self._robot is None:
            return
        self._robot.left_gripper_goto(
            width=self.config.gripper_max_open,
            speed=self.config.gripper_speed,
            force=self.config.gripper_force,
            blocking=blocking,
        )
        self._robot.right_gripper_goto(
            width=self.config.gripper_max_open,
            speed=self.config.gripper_speed,
            force=self.config.gripper_force,
            blocking=blocking,
        )
        self._last_left_gripper_open = 1.0
        self._last_right_gripper_open = 1.0

    # ==================== Actions ====================

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        if action.get("reset_requested", False):
            self.reset()
            return action

        server_action: dict[str, Any] = {}
        gripper_updates: list[tuple[str, float]] = []

        if not self.config.debug:
            if "left_delta_ee_pose.x" in action or "right_delta_ee_pose.x" in action:
                self._add_cartesian_step_action(server_action, action)
            elif all(f"left_joint_{i + 1}.pos" in action for i in range(self._num_joints_per_arm)):
                self._send_action_joint(action)

        if self.config.use_gripper:
            if "left_gripper_cmd_bin" in action:
                self._add_gripper_step_action(
                    server_action,
                    "left",
                    float(action["left_gripper_cmd_bin"]),
                    gripper_updates,
                )
            if "right_gripper_cmd_bin" in action:
                self._add_gripper_step_action(
                    server_action,
                    "right",
                    float(action["right_gripper_cmd_bin"]),
                    gripper_updates,
                )

        if server_action:
            self._robot.step(server_action)
            for side, open_fraction in gripper_updates:
                if side == "left":
                    self._last_left_gripper_open = open_fraction
                else:
                    self._last_right_gripper_open = open_fraction

        return action

    def _add_cartesian_step_action(self, server_action: dict[str, Any], action: dict[str, Any]) -> None:
        left_delta, right_delta = self._cartesian_deltas_from_action(action)
        if np.linalg.norm(left_delta) >= 1e-9:
            server_action.setdefault("left_arm", {})["motion"] = {
                "translation": left_delta[:3].tolist(),
                "rotation_rotvec": left_delta[3:].tolist(),
            }
        if np.linalg.norm(right_delta) >= 1e-9:
            server_action.setdefault("right_arm", {})["motion"] = {
                "translation": right_delta[:3].tolist(),
                "rotation_rotvec": right_delta[3:].tolist(),
            }

    def _cartesian_deltas_from_action(self, action: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in axes], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in axes], dtype=float)

        if not np.all(np.isfinite(left_delta)) or not np.all(np.isfinite(right_delta)):
            self._nonfinite_action_warn_count += 1
            if self._nonfinite_action_warn_count <= 5 or self._nonfinite_action_warn_count % 100 == 0:
                logger.warning(
                    "[FRANKA] Non-finite cartesian action received; replacing NaN/Inf with 0 "
                    "(left=%s right=%s)",
                    left_delta.tolist(),
                    right_delta.tolist(),
                )
            left_delta = np.nan_to_num(left_delta, nan=0.0, posinf=0.0, neginf=0.0)
            right_delta = np.nan_to_num(right_delta, nan=0.0, posinf=0.0, neginf=0.0)

        raw_left = left_delta.copy()
        raw_right = right_delta.copy()
        left_delta[:3] = np.clip(left_delta[:3], -self.config.max_cartesian_delta, self.config.max_cartesian_delta)
        right_delta[:3] = np.clip(right_delta[:3], -self.config.max_cartesian_delta, self.config.max_cartesian_delta)
        left_delta[3:] = np.clip(left_delta[3:], -self.config.max_rotation_delta, self.config.max_rotation_delta)
        right_delta[3:] = np.clip(right_delta[3:], -self.config.max_rotation_delta, self.config.max_rotation_delta)
        if not np.allclose(raw_left, left_delta) or not np.allclose(raw_right, right_delta):
            self._delta_clip_warn_count += 1
            if self._delta_clip_warn_count <= 5 or self._delta_clip_warn_count % 100 == 0:
                logger.warning(
                    "[FRANKA] Cartesian action clipped to per-step limits "
                    "(max_translation=%.4fm max_rotation=%.4frad); raw_left=%s raw_right=%s",
                    self.config.max_cartesian_delta,
                    self.config.max_rotation_delta,
                    raw_left.tolist(),
                    raw_right.tolist(),
                )
        return left_delta, right_delta

    def _send_action_cartesian(self, action: dict[str, Any]) -> None:
        left_delta, right_delta = self._cartesian_deltas_from_action(action)

        if np.linalg.norm(left_delta) < 1e-9 and np.linalg.norm(right_delta) < 1e-9:
            return

        self._robot.dual_robot_move_to_ee_pose(left_delta, right_delta, delta=True, wait=False)

    def _send_action_joint(self, action: dict[str, Any]) -> None:
        if not self._warned_joint_control:
            logger.warning(
                "[FRANKA] Joint actions were provided, but this server controls "
                "cartesian equilibrium_pose topics. Joint command is ignored."
            )
            self._warned_joint_control = True

    def _handle_gripper(self, side: str, value: float) -> None:
        if self._robot is None:
            return
        server_action: dict[str, Any] = {}
        gripper_updates: list[tuple[str, float]] = []
        self._add_gripper_step_action(server_action, side, value, gripper_updates)
        if server_action:
            self._robot.step(server_action)
            for update_side, open_fraction in gripper_updates:
                if update_side == "left":
                    self._last_left_gripper_open = open_fraction
                else:
                    self._last_right_gripper_open = open_fraction

    def _add_gripper_step_action(
        self,
        server_action: dict[str, Any],
        side: str,
        value: float,
        gripper_updates: list[tuple[str, float]],
    ) -> None:
        open_fraction = _clamp(value, 0.0, 1.0)
        if self.config.gripper_reverse:
            open_fraction = 1.0 - open_fraction
        width = open_fraction * self.config.gripper_max_open

        if side == "left":
            if self._last_left_gripper_open is not None and abs(open_fraction - self._last_left_gripper_open) < 1e-4:
                return
            side_key = "left_arm"
        else:
            if self._last_right_gripper_open is not None and abs(open_fraction - self._last_right_gripper_open) < 1e-4:
                return
            side_key = "right_arm"

        server_action.setdefault(side_key, {})["gripper"] = {
            "width": width,
            "max_velocity": self.config.gripper_speed,
            "max_effort": self.config.gripper_force,
        }
        gripper_updates.append((side, open_fraction))

    # ==================== Observations ====================

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        try:
            state = self._robot.get_full_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FRANKA] get_full_state failed: %s", exc)
            if self._prev_observation is not None:
                return self._prev_observation
            raise

        obs: dict[str, Any] = {}
        left_side = _as_mapping(state.get("left_arm", {}))
        right_side = _as_mapping(state.get("right_arm", {}))
        left_robot_state = _robot_state_from_side(left_side)
        right_robot_state = _robot_state_from_side(right_side)

        left_joints = _as_np(left_robot_state.get("joint_positions"), self._num_joints_per_arm)
        right_joints = _as_np(right_robot_state.get("joint_positions"), self._num_joints_per_arm)

        for i in range(self._num_joints_per_arm):
            obs[f"left_joint_{i + 1}.pos"] = float(left_joints[i])
            obs[f"right_joint_{i + 1}.pos"] = float(right_joints[i])

        left_pose = _ee_pose_from_side(left_side)
        right_pose = _ee_pose_from_side(right_side)
        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs[f"left_ee_pose.{axis}"] = float(left_pose[i])
            obs[f"right_ee_pose.{axis}"] = float(right_pose[i])

        if self.config.use_gripper:
            left_grip = _gripper_open_fraction_from_side(left_side, self._left_gripper_state)
            right_grip = _gripper_open_fraction_from_side(right_side, self._right_gripper_state)
            if self.config.gripper_reverse:
                left_grip = 1.0 - left_grip
                right_grip = 1.0 - right_grip
            self._left_gripper_state = _clamp(left_grip, 0.0, 1.0)
            self._right_gripper_state = _clamp(right_grip, 0.0, 1.0)
            obs["left_gripper_state_norm"] = self._left_gripper_state
            obs["right_gripper_state_norm"] = self._right_gripper_state
            obs["left_gripper_cmd_bin"] = (
                self._last_left_gripper_open
                if self._last_left_gripper_open is not None
                else self._left_gripper_state
            )
            obs["right_gripper_cmd_bin"] = (
                self._last_right_gripper_open
                if self._last_right_gripper_open is not None
                else self._right_gripper_state
            )

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.read()

        self._prev_observation = obs
        return obs

    # ==================== Features ====================

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        if self.config.control_mode in {"oculus", "spacemouse"}:
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"left_delta_ee_pose.{axis}"] = float
                features[f"right_delta_ee_pose.{axis}"] = float
        else:
            for i in range(self._num_joints_per_arm):
                features[f"left_joint_{i + 1}.pos"] = float
                features[f"right_joint_{i + 1}.pos"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for i in range(self._num_joints_per_arm):
            features[f"left_joint_{i + 1}.pos"] = float
            features[f"right_joint_{i + 1}.pos"] = float
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_ee_pose.{axis}"] = float
            features[f"right_ee_pose.{axis}"] = float
        if self.config.use_gripper:
            features["left_gripper_state_norm"] = float
            features["right_gripper_state_norm"] = float
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam_name: (cam.height, cam.width, 3)
            for cam_name, cam in self.cameras.items()
        }

    # ==================== Robot Interface ====================

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected

    def configure(self) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = FrankaDualArmConfig(debug=True)
    robot = FrankaDualArm(cfg)
    robot.connect()
    print(robot.get_observation().keys())
    robot.disconnect()
