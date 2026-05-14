"""LeRobot Robot implementation for ROS2 dual-Franka + Robotiq."""

from __future__ import annotations

import logging
import threading
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
        self._cached_rpc_state: Optional[dict[str, Any]] = None
        self._num_joints_per_arm = int(config.num_joints_per_arm)

        self._last_left_gripper_open: Optional[float] = None
        self._last_right_gripper_open: Optional[float] = None
        self._left_gripper_state = 1.0
        self._right_gripper_state = 1.0
        self._warned_joint_control = False
        self._delta_clip_warn_count = 0
        self._nonfinite_action_warn_count = 0
        self._action_debug_enabled = False
        self._action_debug_every_n = 30
        self._action_debug_count = 0
        self._camera_stop_event = threading.Event()
        self._camera_threads: dict[str, threading.Thread] = {}
        self._frame_lock = threading.Lock()
        self._latest_frames: dict[str, Any] = {}

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
            self._start_camera_thread(cam_name, cam)

        self._is_connected = True
        logger.info("[FRANKA] %s connected", self.name)

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._stop_camera_threads()
        for cam in self.cameras.values():
            cam.disconnect()
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._is_connected = False
        self._cached_rpc_state = None
        self._prev_observation = None
        logger.info("[FRANKA] %s disconnected", self.name)

    # ==================== Reset ====================

    def reset(self) -> None:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        if self.config.reset_go_home:
            self.go_home()
        else:
            logger.info("[FRANKA] Resetting target poses to current poses")
            self._robot.reset()
            self._cached_rpc_state = None
        if self.config.use_gripper and self.config.reset_opens_grippers:
            self._open_both_grippers(blocking=True)

    def go_home(self) -> None:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        logger.info("[FRANKA] Moving both arms to server home pose")
        self._robot.go_home(
            "both",
            self.config.go_home_duration_sec,
            self.config.go_home_rate_hz,
        )
        self._cached_rpc_state = None

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
        self._cached_rpc_state = None

    # ==================== Actions ====================

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        sent_action = dict(action)

        if action.get("reset_requested", False):
            self.reset()
            return sent_action

        server_action: dict[str, Any] = {}
        gripper_updates: list[tuple[str, float]] = []

        has_cartesian_action = "left_delta_ee_pose.x" in action or "right_delta_ee_pose.x" in action
        if not self.config.debug:
            if has_cartesian_action:
                self._add_cartesian_step_action(server_action, sent_action, action)
            elif all(f"left_joint_{i + 1}.pos" in action for i in range(self._num_joints_per_arm)):
                self._send_action_joint(action)
        elif has_cartesian_action:
            zero_delta = np.zeros(6, dtype=float)
            self._update_sent_cartesian_action(sent_action, zero_delta, zero_delta, action)

        if self.config.use_gripper:
            if "left_gripper_cmd_bin" in action:
                sent_action["left_gripper_cmd_bin"] = self._add_gripper_step_action(
                    server_action,
                    "left",
                    float(action["left_gripper_cmd_bin"]),
                    gripper_updates,
                )
            if "right_gripper_cmd_bin" in action:
                sent_action["right_gripper_cmd_bin"] = self._add_gripper_step_action(
                    server_action,
                    "right",
                    float(action["right_gripper_cmd_bin"]),
                    gripper_updates,
                )

        if server_action:
            step_result = self._robot.step(server_action)
            self._update_cached_rpc_state_from_step(step_result)
            for side, open_fraction in gripper_updates:
                if side == "left":
                    self._last_left_gripper_open = open_fraction
                else:
                    self._last_right_gripper_open = open_fraction

        self._log_action_debug(sent_action)
        return sent_action

    def set_action_debug(self, enabled: bool, every_n: int = 30) -> None:
        self._action_debug_enabled = bool(enabled)
        self._action_debug_every_n = max(1, int(every_n))
        self._action_debug_count = 0

    def _log_action_debug(self, action: dict[str, Any]) -> None:
        if not self._action_debug_enabled:
            return

        self._action_debug_count += 1
        if self._action_debug_count > 5 and self._action_debug_count % self._action_debug_every_n != 0:
            return

        axes = ["x", "y", "z", "rx", "ry", "rz"]
        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in axes], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in axes], dtype=float)
        logger.info(
            "[FRANKA ACTION] step=%d left_xyz=%.6f right_xyz=%.6f left_rot=%.6f right_rot=%.6f "
            "left_grip=%s right_grip=%s",
            self._action_debug_count,
            float(np.linalg.norm(left_delta[:3])),
            float(np.linalg.norm(right_delta[:3])),
            float(np.linalg.norm(left_delta[3:])),
            float(np.linalg.norm(right_delta[3:])),
            action.get("left_gripper_cmd_bin"),
            action.get("right_gripper_cmd_bin"),
        )

    def _add_cartesian_step_action(
        self,
        server_action: dict[str, Any],
        sent_action: dict[str, Any],
        action: dict[str, Any],
    ) -> None:
        left_delta, right_delta = self._cartesian_deltas_from_action(action)
        self._update_sent_cartesian_action(sent_action, left_delta, right_delta, action)
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

        if self.config.max_cartesian_delta is None and self.config.max_rotation_delta is None:
            return left_delta, right_delta

        raw_left = left_delta.copy()
        raw_right = right_delta.copy()
        if self.config.max_cartesian_delta is not None:
            max_translation = float(self.config.max_cartesian_delta)
            if max_translation > 0.0:
                left_delta[:3] = np.clip(left_delta[:3], -max_translation, max_translation)
                right_delta[:3] = np.clip(right_delta[:3], -max_translation, max_translation)

        if self.config.max_rotation_delta is not None:
            max_rotation = float(self.config.max_rotation_delta)
            if max_rotation > 0.0:
                left_delta[3:] = np.clip(left_delta[3:], -max_rotation, max_rotation)
                right_delta[3:] = np.clip(right_delta[3:], -max_rotation, max_rotation)

        if not np.allclose(raw_left, left_delta) or not np.allclose(raw_right, right_delta):
            self._delta_clip_warn_count += 1
            if self._delta_clip_warn_count <= 5 or self._delta_clip_warn_count % 100 == 0:
                logger.warning(
                    "[FRANKA] Cartesian action clipped to per-step limits "
                    "(max_translation=%s max_rotation=%s); raw_left=%s raw_right=%s",
                    self.config.max_cartesian_delta,
                    self.config.max_rotation_delta,
                    raw_left.tolist(),
                    raw_right.tolist(),
                )
        return left_delta, right_delta

    @staticmethod
    def _update_sent_cartesian_action(
        sent_action: dict[str, Any],
        left_delta: np.ndarray,
        right_delta: np.ndarray,
        source_action: dict[str, Any],
    ) -> None:
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        for index, axis in enumerate(axes):
            left_key = f"left_delta_ee_pose.{axis}"
            if left_key in source_action:
                sent_action[left_key] = float(left_delta[index])

            right_key = f"right_delta_ee_pose.{axis}"
            if right_key in source_action:
                sent_action[right_key] = float(right_delta[index])

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
            step_result = self._robot.step(server_action)
            self._update_cached_rpc_state_from_step(step_result)
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
    ) -> float:
        commanded_open_fraction = _clamp(value, 0.0, 1.0)
        open_fraction = commanded_open_fraction
        if self.config.gripper_reverse:
            open_fraction = 1.0 - open_fraction
        width = open_fraction * self.config.gripper_max_open

        if side == "left":
            if self._last_left_gripper_open is not None and abs(open_fraction - self._last_left_gripper_open) < 1e-4:
                return commanded_open_fraction
            side_key = "left_arm"
        else:
            if self._last_right_gripper_open is not None and abs(open_fraction - self._last_right_gripper_open) < 1e-4:
                return commanded_open_fraction
            side_key = "right_arm"

        server_action.setdefault(side_key, {})["gripper"] = {
            "width": width,
            "max_velocity": self.config.gripper_speed,
            "max_effort": self.config.gripper_force,
        }
        gripper_updates.append((side, open_fraction))
        return commanded_open_fraction

    # ==================== Observations ====================

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        try:
            state = self._get_cached_or_live_state()
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

        latest_frames = self._snapshot_latest_frames()
        for cam_name, cam in self.cameras.items():
            frame = latest_frames.get(cam_name)
            if frame is None:
                frame = cam.read()
                with self._frame_lock:
                    self._latest_frames[cam_name] = frame
            obs[cam_name] = frame

        self._prev_observation = obs
        return obs

    def _update_cached_rpc_state_from_step(self, step_result: Any) -> None:
        if not isinstance(step_result, Mapping):
            return
        observation = step_result.get("observation")
        if isinstance(observation, Mapping):
            self._cached_rpc_state = dict(observation)

    def _get_cached_or_live_state(self) -> dict[str, Any]:
        if self._cached_rpc_state is not None:
            return self._cached_rpc_state
        state = self._robot.get_full_state()
        if not isinstance(state, Mapping):
            raise RuntimeError(f"Unexpected state payload from RPC server: {type(state)!r}")
        self._cached_rpc_state = dict(state)
        return self._cached_rpc_state

    def _start_camera_thread(self, cam_name: str, cam: Any) -> None:
        thread = threading.Thread(
            target=self._camera_read_loop,
            args=(cam_name, cam),
            name=f"{self.name}_{cam_name}_reader",
            daemon=True,
        )
        self._camera_threads[cam_name] = thread
        thread.start()

    def _stop_camera_threads(self) -> None:
        self._camera_stop_event.set()
        for thread in self._camera_threads.values():
            thread.join(timeout=1.0)
        self._camera_threads.clear()
        self._camera_stop_event = threading.Event()
        with self._frame_lock:
            self._latest_frames.clear()

    def _camera_read_loop(self, cam_name: str, cam: Any) -> None:
        while not self._camera_stop_event.is_set():
            try:
                frame = cam.read()
                with self._frame_lock:
                    self._latest_frames[cam_name] = frame
            except Exception as exc:  # noqa: BLE001
                if self._camera_stop_event.is_set():
                    break
                logger.warning("[CAM] %s background read failed: %s", cam_name, exc)
                self._camera_stop_event.wait(timeout=0.1)
                continue

            # RealSense reads block naturally. This tiny wait keeps synthetic test cameras
            # from spinning a tight CPU loop while remaining effectively free in practice.
            self._camera_stop_event.wait(timeout=0.001)

    def _snapshot_latest_frames(self) -> dict[str, Any]:
        with self._frame_lock:
            return dict(self._latest_frames)

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
