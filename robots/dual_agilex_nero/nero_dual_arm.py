"""
Nero dual-arm robot implementation.
Each arm has 7 DOF with agx_gripper as end effector.
Uses Oculus Quest for teleoperation control.
"""

import logging
import time
from typing import Any, Optional
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from robots.dual_arm_schema import (
    AXES,
    action_features as x_embodiment_action_features,
    clip_gripper_width,
    observation_features as x_embodiment_observation_features,
    width_from_normalized_command,
)
from robots.pose_frames import BaseToWorldTransform
from .config_nero import NeroDualArmConfig
from .nero_interface_client import NeroDualArmClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeroDualArm(Robot):
    """
    Dual-arm Nero robot
    Each arm has 7 DOF, total 14 DOF.
    """
    
    config_class = NeroDualArmConfig
    name = "nero_dual_arm"
    
    def __init__(self, config: NeroDualArmConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config
        if config.observation_pose_frame not in {"base", "world"}:
            raise ValueError("observation_pose_frame must be 'base' or 'world'.")
        self._pose_transforms = {}
        if config.observation_pose_frame == "world":
            for side in ("left", "right"):
                name = f"{side}_base_pose_in_world"
                extrinsic = getattr(config, name)
                if extrinsic is None:
                    raise ValueError(f"{name} is required for observation_pose_frame='world'.")
                self._pose_transforms[side] = BaseToWorldTransform(extrinsic, name=name)
        self._is_connected = False
        self._robot: Optional[NeroDualArmClient] = None
        self._prev_observation = None
        self._num_joints_per_arm = 7
        self._cartesian_rejections = {"left": 0, "right": 0}
        
        # Gripper settings
        self._gripper_force = config.gripper_force
        self._left_gripper_target_width = None
        self._right_gripper_target_width = None
        self._left_gripper_width = float(config.gripper_max_open)
        self._right_gripper_width = float(config.gripper_max_open)

        # Action smoothing
        # self._smoothing_alpha = 0.4
        # self._left_smoothed_delta = None
        # self._right_smoothed_delta = None

        # 发送频率控制
        self.action_send_freq = 100.0  # 50Hz
        self.action_send_dt = 1.0 / self.action_send_freq
        self.last_action_send_time = 0.0

    def _should_send_action(self) -> bool:
        """检查是否应该发送action（频率限制）"""
        current_time = time.monotonic()
        if current_time - self.last_action_send_time >= self.action_send_dt:
            self.last_action_send_time = current_time
            return True
        return False

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot.
        
        Args:
            calibrate: Whether to calibrate the robot after connecting.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        
        logger.info("\n" + "=" * 60)
        logger.info("[ROBOT] Connecting to Nero Dual-Arm System")
        logger.info("=" * 60)
        
        # Connect to dual-arm server (single port)
        self._robot = self.check_nero_connection()
        # print("Nero dual-arm connected successfully.")
        
        # Connect to gripper server
        if self.config.use_gripper:
            self.initialize_grippers()
        
        # TODO: Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")
        
        self.is_connected = True
        logger.info(f"[INFO] {self.name} initialization completed successfully.\n")
    
    def check_nero_connection(self) -> NeroDualArmClient:
        """Connect to Nero dual-arm server via zerorpc (single port)."""
        try:
            logger.info("\n===== [ROBOT] Connecting to Nero dual-arm =====")
            
            robot = NeroDualArmClient(
                ip=self.config.robot_ip,
                port=self.config.robot_port
            )
            # print(robot)
            # Get end-effector poses for both arms
            left_ee_pose = robot.left_robot_get_ee_pose()
            right_ee_pose = robot.right_robot_get_ee_pose()
            left_joint_pos = robot.left_robot_get_joint_positions()
            right_joint_pos = robot.right_robot_get_joint_positions()
            # print(left_ee_pose)
            # print(right_ee_pose)
            # print(left_joint_pos)
            # print(right_joint_pos)

            if left_ee_pose is not None and len(left_ee_pose) == 6:
                logger.info(f"[LEFT ARM] End-effector pose: {[round(j, 4) for j in left_ee_pose]}")
            if right_ee_pose is not None and len(right_ee_pose) == 6:
                logger.info(f"[RIGHT ARM] End-effector pose: {[round(j, 4) for j in right_ee_pose]}")
            if left_joint_pos is not None and len(left_joint_pos) == self._num_joints_per_arm:
                logger.info(f"[LEFT ARM] Joint positions: {[round(j, 4) for j in left_joint_pos]}")
            if right_joint_pos is not None and len(right_joint_pos) == self._num_joints_per_arm:
                logger.info(f"[RIGHT ARM] Joint positions: {[round(j, 4) for j in right_joint_pos]}")

            logger.info("===== [ROBOT] Nero dual-arm connected successfully =====\n")
            return robot
            
        except Exception as e:
            logger.error("===== [ERROR] Failed to connect to Nero dual-arm =====")
            logger.error(f"Exception: {e}\n")
            raise
    
    def initialize_grippers(self) -> None:
        """Initialize both grippers."""
        try:
            logger.info("\n===== [GRIPPER] Initializing grippers =====")
            # self._robot.left_gripper_initialize()
            self._robot.left_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            logger.info("[LEFT GRIPPER] Initialized successfully")
            # self._robot.right_gripper_initialize()
            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
                )
            self._left_gripper_target_width = float(self.config.gripper_max_open)
            self._right_gripper_target_width = float(self.config.gripper_max_open)
            self._left_gripper_width = float(self.config.gripper_max_open)
            self._right_gripper_width = float(self.config.gripper_max_open)
            logger.info("[RIGHT GRIPPER] Initialized successfully")
            logger.info("===== [GRIPPER] Grippers initialized successfully =====\n")
        except Exception as e:
            logger.error("===== [ERROR] Failed to initialize grippers =====")
            logger.error(f"Exception: {e}\n")
            raise


    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        logger.info("[ROBOT] Resetting dual-arm system...")
        self._robot.robot_go_home()
        
        if self.config.use_gripper:
            self._robot.left_gripper_goto(
                width=self.config.gripper_min_width,
                force=self._gripper_force,
            )
            self._robot.right_gripper_goto(
                width=self.config.gripper_min_width,
                force=self._gripper_force,
            )
            self._wait_reset_grippers(opening=False)
            self._robot.left_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            widths = self._wait_reset_grippers(opening=True)
            self._left_gripper_target_width = float(self.config.gripper_max_open)
            self._right_gripper_target_width = float(self.config.gripper_max_open)
            self._left_gripper_width = widths["left"]
            self._right_gripper_width = widths["right"]
        
        logger.info("===== [ROBOT] Dual-arm system reset successfully =====\n")

    def _wait_reset_grippers(self, *, opening: bool, timeout: float = 5.0) -> dict[str, float]:
        """Wait for a closed grip, or a settled open grip at its physical stroke limit."""
        started = time.monotonic()
        previous: dict[str, float] = {}
        stable_width: dict[str, float] = {}
        stable_since: dict[str, float] = {}
        while time.monotonic() - started < timeout:
            now = time.monotonic()
            widths = {}
            for side in ("left", "right"):
                state = getattr(self._robot, f"{side}_gripper_get_state")()
                if "width" not in state or not np.isfinite(state["width"]):
                    raise RuntimeError(f"{side} gripper has no valid width feedback during reset")
                width = float(state["width"])
                widths[side] = width
                if side not in stable_width or abs(width - stable_width[side]) > 0.0005:
                    stable_since[side] = now
                    stable_width[side] = width
                previous[side] = width
            if not opening and all(
                # Physical closure and the encoder zero can differ slightly.
                widths[side] <= self.config.gripper_min_width + 0.003
                and now - stable_since[side] >= 0.15
                for side in ("left", "right")
            ):
                logger.info("[GRIPPER] Reset closed widths: %s", widths)
                return widths
            if opening and now - started >= 0.5 and all(
                widths[side] > self.config.gripper_min_width + 0.005
                and now - stable_since[side] >= 0.3
                for side in ("left", "right")
            ):
                logger.info("[GRIPPER] Reset open widths: %s", widths)
                return widths
            time.sleep(0.05)
        phase = "open" if opening else "close"
        raise RuntimeError(f"Gripper reset {phase} timed out; last widths: {previous}")
    
    @property
    def motor_features(self) -> dict[str, type]:
        """Canonical task-space state: EE poses and physical gripper widths."""
        return {
            name: feature
            for name, feature in x_embodiment_observation_features(
                {}, use_gripper=self.config.use_gripper
            ).items()
        }
    
    @property
    def action_features(self) -> dict[str, type]:
        """Canonical action: EE delta and commanded target gripper width in metres."""
        return x_embodiment_action_features(use_gripper=self.config.use_gripper)

    def _clip_gripper_width(self, width: float) -> float:
        return clip_gripper_width(
            width,
            self.config.gripper_min_width,
            self.config.gripper_max_open,
        )

    def _gripper_width_from_action(self, action: dict[str, Any], arm_side: str) -> float | None:
        width_key = f"{arm_side}_gripper_width"
        if width_key in action and action[width_key] is not None:
            return self._clip_gripper_width(float(action[width_key]))

        # Legacy normalized commands remain accepted as input only. They are
        # converted before reaching the hardware, while the public schema and
        # recorded datasets stay in metres.
        for key in (f"{arm_side}_gripper_cmd", f"{arm_side}_gripper_cmd_bin"):
            if key in action and action[key] is not None:
                return width_from_normalized_command(
                    float(action[key]),
                    self.config.gripper_min_width,
                    self.config.gripper_max_open,
                    reverse=self.config.gripper_reverse,
                )
        return None

    def handle_gripper(self, arm_side: str, width: float) -> float:
        if not self.config.use_gripper:
            return float(width)

        width = self._clip_gripper_width(width)
        width_attr = f"_{arm_side}_gripper_width"
        last_width = getattr(self, f"_{arm_side}_gripper_target_width", None)

        # Skip redundant command writes to reduce RPC blocking and gripper bus load.
        if last_width is not None and abs(width - last_width) < self.config.gripper_command_epsilon:
            return float(last_width)
        
        try:
            if arm_side == "left":
                self._robot.left_gripper_goto(
                    width=width,
                    force=self._gripper_force
                )
            else:
                self._robot.right_gripper_goto(
                    width=width,
                    force=self._gripper_force
                )
            setattr(self, width_attr, width)
            setattr(self, f"_{arm_side}_gripper_target_width", width)
        except Exception as e:
            logger.warning(f"[{arm_side.upper()} GRIPPER] zerorpc error: {e}")
            raise
        return width
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent_action = dict(action)

        # Check for reset request
        if action.get("reset_requested", False):
            logger.info("[ROBOT] Reset requested for dual-arm system...")
            self._robot.robot_go_home()
            if self.config.use_gripper:
                self._robot.left_gripper_goto(
                    width=self.config.gripper_max_open,
                    force=self._gripper_force
                )
                self._robot.right_gripper_goto(
                    width=self.config.gripper_max_open,
                    force=self._gripper_force
                )
            self.reset()
            return sent_action

        sent_action.update(self.send_action_cartesian(action))
        
        # Public actions are physical target widths. Return the clipped values
        # so replay and recorded labels exactly match the executable command.
        for side in ("left", "right"):
            width = self._gripper_width_from_action(action, side)
            if width is not None:
                sent_action[f"{side}_gripper_width"] = self.handle_gripper(side, width)

        # t_send_end = time.perf_counter()
        # logger.info(f"[TIMING] send_action total: {(t_send_end-t_send_start)*1000:.2f}ms")

        return sent_action

    def send_action_cartesian(self, action: dict[str, Any]) -> dict[str, float]:
        """Return only increments accepted by the RPC, in the public frame."""
        deltas = {
            side: np.array([action.get(f"{side}_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
            for side in ("left", "right")
        }
        if any(not np.all(np.isfinite(delta)) for delta in deltas.values()):
            raise ValueError("Nero Cartesian actions must be finite")
        effective = {f"{side}_delta_ee_pose.{axis}": 0.0 for side in deltas for axis in AXES}
        if self.config.debug or not self._should_send_action():
            return effective
        for side, public_delta in deltas.items():
            transform = self._pose_transforms.get(side)
            base_delta = transform.delta_to_base(public_delta) if transform is not None else public_delta
            if np.linalg.norm(base_delta) < 0.001:
                continue
            accepted = self._robot.servo_p_OL(f"{side}_robot", base_delta, delta=True)
            if accepted is False:
                self._handle_cartesian_rejection(side)
                # The server did not send this target. Keep its recorded delta
                # at zero, discard it, and still process the other arm.
                continue
            self._cartesian_rejections[side] = 0
            sent_delta = transform.delta_to_world(base_delta) if transform is not None else base_delta
            effective.update({f"{side}_delta_ee_pose.{axis}": float(sent_delta[i]) for i, axis in enumerate(AXES)})
        return effective

    def _handle_cartesian_rejection(self, side: str) -> None:
        """A rejected IK/workspace frame may be skipped only on a healthy arm."""
        # Never suppress RPC failures or infer health from a False response.
        status = getattr(self._robot, f"{side}_robot_get_arm_status")()
        enabled = status.get("joint_enabled") if isinstance(status, dict) else None
        healthy = (
            isinstance(status, dict)
            and status.get("arm_status") == 0
            and status.get("all_joints_enabled") is True
            and isinstance(enabled, (list, tuple))
            and len(enabled) == 7
            and all(enabled)
        )
        if not healthy:
            raise RuntimeError(f"Nero rejected {side} Cartesian action; arm not confirmed healthy: {status}")
        self._cartesian_rejections[side] += 1
        count = self._cartesian_rejections[side]
        if count == 1 or count % 100 == 0:
            logger.warning(
                "[NERO] Skipped %s Cartesian frame: server rejected target "
                "(IK/workspace/cooldown); arm is normal and enabled. "
                "Rejected delta discarded; consecutive rejected frames=%d.",
                side, count,
            )


    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # t_total_start = time.perf_counter()
        
        try:
            # t_query_start = time.perf_counter()
            left_ee_pose = self._robot.left_robot_get_ee_pose()
            # t_query_end = time.perf_counter()
            # logger.info(f"[TIMING] left robot query: {(t_query_end-t_query_start)*1000:.2f}ms")
            
            # t_query_start = time.perf_counter()
            right_ee_pose = self._robot.right_robot_get_ee_pose()
            # t_query_end = time.perf_counter()
            # logger.info(f"[TIMING] right robot query: {(t_query_end-t_query_start)*1000:.2f}ms")
            
        except Exception as e:
            logger.warning(f"[ROBOT] zerorpc error in get_observation: {e}")
            if self._prev_observation is not None:
                return self._prev_observation
            else:
                raise
        
        obs_dict = {}
        
        server_axes = ("x", "y", "z", "rz", "ry", "rx")
        # Preserve the existing RPC wire-axis convention, then transform the
        # canonical [x,y,z,rotvec] pose. Absolute rotations compose on SO(3).
        for side, raw_pose in (("left", left_ee_pose), ("right", right_ee_pose)):
            pose_by_axis = dict(zip(server_axes, raw_pose, strict=True))
            pose = np.array([pose_by_axis[axis] for axis in AXES], dtype=float)
            transform = self._pose_transforms.get(side)
            if transform is not None:
                pose = transform.apply_pose6(pose)
            for i, axis in enumerate(AXES):
                obs_dict[f"{side}_ee_pose.{axis}"] = float(pose[i])
        
        # Gripper width is queried from the controller when available. Use the
        # last requested physical width as a safe fallback during RPC hiccups.
        if self.config.use_gripper:
            for side in ("left", "right"):
                fallback = getattr(self, f"_{side}_gripper_width")
                try:
                    state = getattr(self._robot, f"{side}_gripper_get_state")()
                    width = state.get("width", fallback) if isinstance(state, dict) else fallback
                    width = self._clip_gripper_width(float(width))
                    setattr(self, f"_{side}_gripper_width", width)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s GRIPPER] state read failed: %s", side.upper(), exc)
                    width = fallback
                obs_dict[f"{side}_gripper_width"] = float(width)

        # TODO: Camera images
        # t_cam_total_start = time.perf_counter()
        for cam_key, cam in self.cameras.items():
            # t_cam_start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            # t_cam_end = time.perf_counter()
            # logger.info(f"[TIMING] {cam_key} read: {(t_cam_end-t_cam_start)*1000:.2f}ms")
        # t_cam_total_end = time.perf_counter()
        # logger.info(f"[TIMING] camera total: {(t_cam_total_end-t_cam_total_start)*1000:.2f}ms")
        
        self._prev_observation = obs_dict
        # t_total_end = time.perf_counter()
        # logger.info(f"[TIMING] get_observation total: {(t_total_end-t_total_start)*1000:.2f}ms")
        return obs_dict
    
    def disconnect(self) -> None:
        if not self.is_connected:
            return
        
        # TODO: Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        if self._robot is not None:
            self._robot.close()
        
        self.is_connected = False
        logger.info(f"[INFO] ===== {self.name} disconnected =====")
    
    def calibrate(self) -> None:
        pass
    
    def is_calibrated(self) -> bool:
        return self.is_connected
    
    def configure(self) -> None:
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value
    
    @property
    def cameras_features(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) 
            for cam in self.cameras
        }
    
    @property
    def observation_features(self) -> dict[str, Any]:
        return x_embodiment_observation_features(
            self.cameras,
            use_gripper=self.config.use_gripper,
        )
