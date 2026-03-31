"""
Nero dual-arm robot implementation.
Each arm has 7 DOF with Robotiq 2F-85 gripper as end effector.
Uses Oculus Quest for teleoperation control.
"""

# Unfinished

import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config_nero import NeroDualArmConfig
# from .nero_interface_client import NeroDualArmClient
# from .dual_gripper_client import DualGripperClient

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
        self._gripper_client: Optional[DualGripperClient] = None
        self._prev_observation = None
        self._num_joints_per_arm = 7
        
        # Gripper settings
        self._gripper_force = config.gripper_force
        self._gripper_speed = config.gripper_speed
        self._left_gripper_position = 1.0
        self._right_gripper_position = 1.0
        self._last_left_gripper_position = 1.0
        self._last_right_gripper_position = 1.0
        
        # Action smoothing
        self._smoothing_alpha = 0.4
        self._left_smoothed_delta = None
        self._right_smoothed_delta = None
    
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        
        logger.info("\n" + "=" * 60)
        logger.info("[ROBOT] Connecting to Nero Dual-Arm System")
        logger.info("=" * 60)
        
        # Connect to dual-arm server (single port)
        self._robot = self._check_nero_connection()
        
        # Connect to gripper server
        if self.config.use_gripper:
            self._gripper_client = self._check_gripper_connection()
        
        # Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")
        
        self.is_connected = True
        logger.info(f"[INFO] {self.name} initialization completed successfully.\n")
    
    def _check_nero_connection(self) -> NeroDualArmClient:
        """Connect to Nero dual-arm server via zerorpc (single port)."""
        try:
            logger.info("\n===== [ROBOT] Connecting to Nero dual-arm =====")
            
            robot = NeroDualArmClient(
                ip=self.config.robot_ip,
                port=self.config.robot_port
            )
            
            # Get end-effector poses for both arms
            left_ee_pose = robot.left_robot_get_ee_pose()
            right_ee_pose = robot.right_robot_get_ee_pose()
            
            if left_ee_pose is not None and len(left_ee_pose) == 6:
                logger.info(f"[LEFT ARM] End-effector pose: {[round(j, 4) for j in left_ee_pose]}")
            if right_ee_pose is not None and len(right_ee_pose) == 6:
                logger.info(f"[RIGHT ARM] End-effector pose: {[round(j, 4) for j in right_ee_pose]}")
            
            logger.info("===== [ROBOT] Nero dual-arm connected successfully =====\n")
            return robot
            
        except Exception as e:
            logger.error("===== [ERROR] Failed to connect to Nero dual-arm =====")
            logger.error(f"Exception: {e}\n")
            raise
    
    def _check_gripper_connection(self) -> DualGripperClient:
        """Connect to dual gripper server via zerorpc."""
        try:
            logger.info("\n===== [GRIPPER] Connecting to dual gripper server =====")
            
            gripper_client = DualGripperClient(
                ip=self.config.gripper_ip,
                port=self.config.gripper_port
            )
            
            logger.info("===== [GRIPPER] Dual gripper server connected successfully =====\n")
            gripper_client.left_gripper_initialize()
            gripper_client.left_gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=True
                )
            logger.info("[LEFT GRIPPER] Initialized successfully")

            gripper_client.right_gripper_initialize()
            gripper_client.right_gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=True
                )
            logger.info("[RIGHT GRIPPER] Initialized successfully")

            return gripper_client
            
        except Exception as e:
            logger.error("===== [ERROR] Failed to connect to dual gripper server =====")
            logger.error(f"Exception: {e}\n")
            return DualGripperClient()

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        logger.info("[ROBOT] Resetting dual-arm system...")
        self._robot.robot_go_home()
        
        if self.config.use_gripper:
            self._gripper_client.left_gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=True
            )
            self._gripper_client.right_gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=True
            )
        
        logger.info("===== [ROBOT] Dual-arm system reset successfully =====\n")
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for dual-arm system."""
        features = {}
        
        # Left arm joint positions
        for i in range(self._num_joints_per_arm):
            features[f"left_joint_{i+1}.pos"] = float
        
        # Right arm joint positions
        for i in range(self._num_joints_per_arm):
            features[f"right_joint_{i+1}.pos"] = float
        
        # Left arm end effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_ee_pose.{axis}"] = float
        
        # Right arm end effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_ee_pose.{axis}"] = float
        
        # Gripper states
        if self.config.use_gripper:
            features["left_gripper_state_norm"] = float
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_state_norm"] = float
            features["right_gripper_cmd_bin"] = float
        
        return features
    
    @property
    def action_features(self) -> dict[str, type]:
        features = {}

        # Left arm joint positions
        for i in range(self._num_joints_per_arm):
            features[f"left_joint_{i+1}.pos"] = float
        
        # Right arm joint positions
        for i in range(self._num_joints_per_arm):
            features[f"right_joint_{i+1}.pos"] = float

        # Left arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_delta_ee_pose.{axis}"] = float
        # Right arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_delta_ee_pose.{axis}"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features

    def _handle_gripper(self, arm_side: str, gripper_value: float, is_binary: bool = True) -> None:
        """Handle gripper control for specified arm."""
        if not self.config.use_gripper:
            return
        
        last_pos_attr = f"_last_{arm_side}_gripper_position"
        gripper_pos_attr = f"_{arm_side}_gripper_position"
        
        if is_binary:
            gripper_position = gripper_value
        else:
            gripper_position = 0.0 if gripper_value < self.config.close_threshold else 1.0
        
        if self.config.gripper_reverse:
            gripper_position = 1 - gripper_position
        
        try:
            last_pos = getattr(self, last_pos_attr)
            if gripper_position != last_pos:
                if arm_side == "left":
                    self._gripper_client.left_gripper_goto(
                        width=gripper_position * self.config.gripper_max_open,
                        speed=self._gripper_speed,
                        force=self._gripper_force,
                    )
                else:
                    self._gripper_client.right_gripper_goto(
                        width=gripper_position * self.config.gripper_max_open,
                        speed=self._gripper_speed,
                        force=self._gripper_force,
                    )
                setattr(self, last_pos_attr, gripper_position)
            
            if arm_side == "left":
                gripper_state = self._gripper_client.left_gripper_get_state()
            else:
                gripper_state = self._gripper_client.right_gripper_get_state()
            
            gripper_state_norm = max(0.0, min(1.0, gripper_state["width"] / self.config.gripper_max_open))
            if self.config.gripper_reverse:
                gripper_state_norm = 1 - gripper_state_norm
            setattr(self, gripper_pos_attr, gripper_state_norm)
        except Exception as e:
            logger.warning(f"[{arm_side.upper()} GRIPPER] zerorpc error: {e}")
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check for reset request
        # if action.get("reset_requested", False):
        #     logger.info("[ROBOT] Reset requested for dual-arm system...")
        #     # self._robot.robot_go_home()
        #     # if self.config.use_gripper:
        #     #     self._gripper_client.left_gripper_goto(
        #     #         width=self.config.gripper_max_open,
        #     #         speed=self._gripper_speed,
        #     #         force=self._gripper_force,
        #     #         blocking=True
        #     #     )
        #     #     self._gripper_client.right_gripper_goto(
        #     #         width=self.config.gripper_max_open,
        #     #         speed=self._gripper_speed,
        #     #         force=self._gripper_force,
        #     #         blocking=True
        #     #     )
        #     self.reset()
        #     return action

        # Use joint servo control if joint positions are provided
        if not self.config.debug:
            try:
                # Check if joint positions are in action
                has_joints = all(f"left_joint_{i+1}.pos" in action for i in range(6))
                
                if has_joints:
                    # Use servo_j_delta for joint control
                    self._send_action_joint_servo(action)
                else:
                    # Fallback to Cartesian control
                    self._send_action_cartesian(action)
                    
            except Exception as e:
                logger.warning(f"[ROBOT] Action failed: {e}")
        
        # Handle grippers
        if "left_gripper_cmd_bin" in action:
            self._handle_gripper("left", action["left_gripper_cmd_bin"], is_binary=True)
        if "right_gripper_cmd_bin" in action:
            self._handle_gripper("right", action["right_gripper_cmd_bin"], is_binary=True)

        return action
    
    def _send_action_joint_servo(self, action: dict[str, Any]) -> None:
        """Send action using joint servo control (servo_j_delta)."""
        # Get target joint positions (in radians)
        left_target_joints = np.array([
            action[f"left_joint_{i+1}.pos"] for i in range(self._num_joints_per_arm)
        ])
        right_target_joints = np.array([
            action[f"right_joint_{i+1}.pos"] for i in range(self._num_joints_per_arm)
        ])
        
        # Get current joint positions (in radians)
        # left_current_joints = self._robot.left_robot_get_joint_positions()
        # right_current_joints = self._robot.right_robot_get_joint_positions()
        
        # Compute delta joints (in radians)
        # left_delta_joints = left_target_joints - left_current_joints
        # right_delta_joints = right_target_joints - right_current_joints
        
        # # Safety check: limit max joint delta
        # left_max_delta = np.abs(left_delta_joints).max()
        # right_max_delta = np.abs(right_delta_joints).max()
        
        # if left_max_delta > self.config.max_joint_delta or right_max_delta > self.config.max_joint_delta:
        #     logger.warning(f"[ROBOT] Joint delta too large: left={left_max_delta:.3f}, right={right_max_delta:.3f}")
        #     # Scale down if too large
        #     scale = self.config.max_joint_delta / max(left_max_delta, right_max_delta)
        #     left_delta_joints *= scale
        #     right_delta_joints *= scale
        
        # Send servo commands (client expects radians)
        try:
            # Use servo_j_delta for smooth joint control
            # self._robot.servo_j_delta('left', left_delta_joints, t=0.05, lookahead_time=0.05, gain=300)
            # self._robot.servo_j_delta('right', right_delta_joints, t=0.05, lookahead_time=0.05, gain=300)
            
            # logger.debug(f"[SERVO_J] Left delta (rad): {left_delta_joints}")
            # logger.debug(f"[SERVO_J] Right delta (rad): {right_delta_joints}")
            
            self._robot.servo_j('left', left_target_joints, t=0.05, lookahead_time=0.05, gain=300)
            self._robot.servo_j('right', right_target_joints, t=0.05, lookahead_time=0.05, gain=300)

        except Exception as e:
            logger.warning(f"[SERVO_J] Failed: {e}")
    
    def _send_action_cartesian(self, action: dict[str, Any]) -> None:
        """Send action in oculus mode (delta ee pose) without using loops."""
        # Get delta poses for both arms
        left_delta = np.array([
            action[f"left_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ])
        right_delta = np.array([
            action[f"right_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ])
        # print(f"[ACTION] Left delta pose: {left_delta}")
        # print(f"[ACTION] Right delta pose: {right_delta}") 

        # 使用 dual_robot_move_to_ee_pose 同时控制双臂
        if not self.config.debug:
            import scipy.spatial.transform as st
            
            # 获取当前末端位姿（一次性获取，避免重复调用）
            try:
                left_ee_pose = self._robot.left_robot_get_ee_pose()
                right_ee_pose = self._robot.right_robot_get_ee_pose()
            except Exception as e:
                logger.warning(f"[ROBOT] Failed to get EE poses: {e}")
                return
            
            # 计算左臂目标位姿
            target_left_ee_pose = None
            if np.linalg.norm(left_delta) >= 0.001:
                try:
                    target_left_position = left_ee_pose[:3] + left_delta[:3]
                    current_left_rot = st.Rotation.from_euler("xyz", left_ee_pose[3:])
                    
                    delta_rot_norm = np.linalg.norm(left_delta[3:])
                    if delta_rot_norm < 1e-6:
                        target_left_euler = left_ee_pose[3:]
                        print(f"[LEFT ARM] Rotation delta is very small ({delta_rot_norm:.2e}), keeping current rotation.")
                    else:
                        delta_rot = st.Rotation.from_euler("xyz", left_delta[3:])
                        target_rotation = delta_rot * current_left_rot
                        target_left_euler = target_rotation.as_euler("xyz")
                    
                    target_left_ee_pose = np.concatenate([target_left_position, target_left_euler])
                except Exception as e:
                    logger.warning(f"[LEFT ARM] Failed to compute target pose: {e}")
            
            # 计算右臂目标位姿
            target_right_ee_pose = None
            if np.linalg.norm(right_delta) >= 0.001:
                try:
                    target_right_position = right_ee_pose[:3] + right_delta[:3]
                    current_right_rot = st.Rotation.from_euler("xyz", right_ee_pose[3:])
                    
                    delta_rot_norm = np.linalg.norm(right_delta[3:])
                    if delta_rot_norm < 1e-6:
                        target_right_euler = right_ee_pose[3:]
                        print(f"[RIGHT ARM] Rotation delta is very small ({delta_rot_norm:.2e}), keeping current rotation.")
                    else:
                        delta_rot = st.Rotation.from_euler("xyz", right_delta[3:])
                        target_rotation = delta_rot * current_right_rot
                        target_right_euler = target_rotation.as_euler("xyz")
                    
                    target_right_ee_pose = np.concatenate([target_right_position, target_right_euler])
                except Exception as e:
                    logger.warning(f"[RIGHT ARM] Failed to compute target pose: {e}")
            
            # 使用 dual_robot_move_to_ee_pose 同时控制双臂
            if target_left_ee_pose is not None or target_right_ee_pose is not None:
                try:
                    # 如果某个臂没有目标位姿，使用当前位姿
                    if target_left_ee_pose is None:
                        target_left_ee_pose = left_ee_pose
                    if target_right_ee_pose is None:
                        target_right_ee_pose = right_ee_pose
                    
                    # wait=True 确保动作执行完成后再返回
                    self._robot.dual_robot_move_to_ee_pose(
                        target_left_ee_pose.tolist(),
                        target_right_ee_pose.tolist(),
                        wait=True  # 等待动作完成
                    )
                    print(f"[DUAL ARM] Executed: Left target: {target_left_ee_pose[:3]}, Right target: {target_right_ee_pose[:3]}")
                except Exception as e:
                    logger.warning(f"[DUAL ARM] dual_robot_move_to_ee_pose failed: {e}")
        
        if "left_gripper_cmd_bin" in action:
            self._handle_gripper("left", action["left_gripper_cmd_bin"], is_binary=True)
        if "right_gripper_cmd_bin" in action:
            self._handle_gripper("right", action["right_gripper_cmd_bin"], is_binary=True)


    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            left_joint_pos = self._robot.left_robot_get_joint_positions()
            left_ee_pose = self._robot.left_robot_get_ee_pose()
            
            right_joint_pos = self._robot.right_robot_get_joint_positions()
            right_ee_pose = self._robot.right_robot_get_ee_pose()
            
        except Exception as e:
            logger.warning(f"[ROBOT] zerorpc error in get_observation: {e}")
            if self._prev_observation is not None:
                return self._prev_observation
            else:
                raise
        
        obs_dict = {}
        
        # Left arm observations
        for i in range(len(left_joint_pos)):
            obs_dict[f"left_joint_{i+1}.pos"] = float(left_joint_pos[i])

        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs_dict[f"left_ee_pose.{axis}"] = float(left_ee_pose[i])
        
        # Right arm observations
        for i in range(len(right_joint_pos)):
            obs_dict[f"right_joint_{i+1}.pos"] = float(right_joint_pos[i])

        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs_dict[f"right_ee_pose.{axis}"] = float(right_ee_pose[i])
        
        # Gripper states
        if self.config.use_gripper:
            obs_dict["left_gripper_state_norm"] = self._left_gripper_position
            obs_dict["left_gripper_cmd_bin"] = self._last_left_gripper_position
            obs_dict["right_gripper_state_norm"] = self._right_gripper_position
            obs_dict["right_gripper_cmd_bin"] = self._last_right_gripper_position
        else:
            obs_dict["left_gripper_state_norm"] = None
            obs_dict["left_gripper_cmd_bin"] = None
            obs_dict["right_gripper_state_norm"] = None
            obs_dict["right_gripper_cmd_bin"] = None
        
        # Camera images
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        self._prev_observation = obs_dict
        return obs_dict
    
    def disconnect(self) -> None:
        if not self.is_connected:
            return
        
        for cam in self.cameras.values():
            cam.disconnect()
        
        if self._robot is not None:
            self._robot.close()
        
        if self._gripper_client is not None:
            self._gripper_client.close()
        
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
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) 
            for cam in self.cameras
        }
    
    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}