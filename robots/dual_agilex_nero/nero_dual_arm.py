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
        self._is_connected = False
        self._robot: Optional[NeroDualArmClient] = None
        self._prev_observation = None
        self._num_joints_per_arm = 7
        
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
        current_time = time.time()
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
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            self._left_gripper_target_width = float(self.config.gripper_max_open)
            self._right_gripper_target_width = float(self.config.gripper_max_open)
            self._left_gripper_width = float(self.config.gripper_max_open)
            self._right_gripper_width = float(self.config.gripper_max_open)
        
        logger.info("===== [ROBOT] Dual-arm system reset successfully =====\n")
    
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

        # Use joint servo control if joint positions are provided
        if not self.config.debug:
            try:
                self.send_action_cartesian(action)
                    
            except Exception as e:
                logger.warning(f"[ROBOT] Action failed: {e}")
                raise
        
        # Public actions are physical target widths. Return the clipped values
        # so replay and recorded labels exactly match the executable command.
        for side in ("left", "right"):
            width = self._gripper_width_from_action(action, side)
            if width is not None:
                sent_action[f"{side}_gripper_width"] = self.handle_gripper(side, width)

        # t_send_end = time.perf_counter()
        # logger.info(f"[TIMING] send_action total: {(t_send_end-t_send_start)*1000:.2f}ms")

        return sent_action

    def send_action_cartesian(self, action: dict[str, Any]) -> None:
        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in AXES])
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in AXES])
        left_norm = float(np.linalg.norm(left_delta))
        right_norm = float(np.linalg.norm(right_delta))

        # 频率限制
        if not self._should_send_action():
            return

        if not self.config.debug:
            try:
                # 左臂：直接传入增量
                if left_norm >= 0.001:
                    # t_servo_start = time.perf_counter()
                    self._robot.servo_p_OL("left_robot", left_delta, delta=True)
                    # t_servo_end = time.perf_counter()
                    # logger.info(f"[TIMING] left servo_p_OL: {(t_servo_end-t_servo_start)*1000:.2f}ms")
                
                # 右臂：直接传入增量
                if right_norm >= 0.001:
                    # t_servo_start = time.perf_counter()
                    self._robot.servo_p_OL("right_robot", right_delta, delta=True)
                    # t_servo_end = time.perf_counter()
                    # logger.info(f"[TIMING] right servo_p_OL: {(t_servo_end-t_servo_start)*1000:.2f}ms")
                    
            except Exception as e:
                logger.warning(f"[DUAL ARM] servo_p_OL failed: {e}")
                raise
        
        # t_cart_end = time.perf_counter()
        # logger.info(f"[TIMING] send_action_cartesian total: {(t_cart_end-t_cart_start)*1000:.2f}ms")


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
        left_pose_by_axis = dict(zip(server_axes, left_ee_pose, strict=True))
        right_pose_by_axis = dict(zip(server_axes, right_ee_pose, strict=True))
        for axis in AXES:
            obs_dict[f"left_ee_pose.{axis}"] = float(left_pose_by_axis[axis])
        for axis in AXES:
            obs_dict[f"right_ee_pose.{axis}"] = float(right_pose_by_axis[axis])
        
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
