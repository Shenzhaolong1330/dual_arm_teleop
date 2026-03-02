"""
Dobot Nova5 dual-arm robot implementation.
Each arm has 6 DOF with Robotiq 2F-85 gripper as end effector.
Uses Oculus Quest for teleoperation control.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config_dobot import DobotDualArmConfig, DobotSingleArmConfig
from .dobot_interface_client import DobotArmClient, DobotDualArmClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DobotSingleArm(Robot):
    """
    Single Dobot Nova5 arm with Robotiq 2F-85 gripper.
    """
    
    config_class = DobotSingleArmConfig
    name = "dobot_single_arm"
    
    def __init__(self, config: DobotSingleArmConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self.config = config
        self._is_connected = False
        self._robot: Optional[DobotArmClient] = None
        self._prev_observation = None
        self._num_joints = 6  # Dobot Nova5 has 6 DOF
        
        # Gripper settings (Robotiq 2F-85)
        self._gripper_force = config.gripper_force
        self._gripper_speed = config.gripper_speed
        self._gripper_position = 1.0  # 1.0 = fully open
        self._last_gripper_position = 1.0
        
        # Action smoothing (EMA filter)
        self._smoothing_alpha = 0.4
        self._smoothed_delta = None
        
        # Arm side
        self._arm_side = config.arm_side
    
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self._arm_side.upper()} ARM] Connecting to Dobot Nova5")
        logger.info(f"{'='*60}")
        
        # Connect to robot
        self._robot = self._check_dobot_connection()
        
        # Initialize gripper
        if self.config.use_gripper:
            self._check_gripper_connection()
        
        # Connect cameras
        logger.info(f"\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")
        
        self.is_connected = True
        logger.info(f"[INFO] {self.name} initialization completed successfully.\n")
    
    def _check_dobot_connection(self) -> DobotArmClient:
        """Connect to Dobot arm via zerorpc."""
        try:
            logger.info(f"\n===== [ROBOT] Connecting to Dobot {self._arm_side} arm =====")
            
            robot = DobotArmClient(
                ip='127.0.0.1',
                port=self.config.robot_port,
                arm_side=self._arm_side
            )
            robot.robot_start_joint_impedance_control()
            
            joint_positions = robot.robot_get_joint_positions()
            if joint_positions is not None and len(joint_positions) == self._num_joints:
                formatted_joints = [round(j, 4) for j in joint_positions]
                logger.info(f"[ROBOT] Current joint positions: {formatted_joints}")
                logger.info(f"===== [ROBOT] Dobot {self._arm_side} arm connected successfully =====\n")
            else:
                logger.error("===== [ERROR] Failed to read joint positions =====")
            
            return robot
            
        except Exception as e:
            logger.error(f"===== [ERROR] Failed to connect to Dobot robot =====")
            logger.error(f"Exception: {e}\n")
            raise
    
    def _check_gripper_connection(self):
        """Initialize Robotiq 2F-85 gripper."""
        logger.info(f"\n===== [GRIPPER] Initializing Robotiq 2F-85 gripper...")
        self._robot.gripper_initialize()
        self._robot.gripper_goto(
            width=self.config.gripper_max_open,
            speed=self._gripper_speed,
            force=self._gripper_force,
            blocking=True
        )
        logger.info("===== [GRIPPER] Gripper initialized successfully.\n")
    
    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        logger.info(f"[ROBOT] Resetting {self._arm_side} arm...")
        self._robot.robot_go_home()
        if self.config.use_gripper:
            self._robot.gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=True
            )
        logger.info(f"===== [ROBOT] {self._arm_side} arm reset successfully =====\n")
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for single arm."""
        features = {}
        
        # Joint positions (6 DOF)
        for i in range(self._num_joints):
            features[f"joint_{i+1}.pos"] = float
        
        # Joint velocities
        for i in range(self._num_joints):
            features[f"joint_{i+1}.vel"] = float
        
        # End effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"ee_pose.{axis}"] = float
        
        # Gripper state
        if self.config.use_gripper:
            features["gripper_state_norm"] = float
            features["gripper_cmd_bin"] = float
        
        return features
    
    @property
    def action_features(self) -> dict[str, type]:
        """Return action features based on control mode."""
        if self.config.control_mode == "isoteleop":
            features = {}
            for i in range(self._num_joints):
                features[f"joint_{i+1}.pos"] = float
            if self.config.use_gripper:
                features["gripper_position"] = float
            return features
        elif self.config.control_mode == "oculus":
            features = {}
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"delta_ee_pose.{axis}"] = float
            if self.config.use_gripper:
                features["gripper_cmd_bin"] = float
            return features
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")
    
    def _handle_gripper(self, gripper_value: float, is_binary: bool = True) -> None:
        """Handle gripper control."""
        if not self.config.use_gripper:
            return
        
        if is_binary:
            gripper_position = gripper_value
        else:
            gripper_position = 0.0 if gripper_value < self.config.close_threshold else 1.0
        
        if self.config.gripper_reverse:
            gripper_position = 1 - gripper_position
        
        try:
            if gripper_position != self._last_gripper_position:
                self._robot.gripper_goto(
                    width=gripper_position * self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                )
                self._last_gripper_position = gripper_position
            
            gripper_state = self._robot.gripper_get_state()
            gripper_state_norm = max(0.0, min(1.0, gripper_state["width"] / self.config.gripper_max_open))
            if self.config.gripper_reverse:
                gripper_state_norm = 1 - gripper_state_norm
            self._gripper_position = gripper_state_norm
        except Exception as e:
            logger.warning(f"[GRIPPER] zerorpc error: {e}")
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.control_mode == "isoteleop":
            self._send_action_isoteleop(action)
        elif self.config.control_mode == "oculus":
            self._send_action_cartesian(action)
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")
        
        return action
    
    def _send_action_isoteleop(self, action: dict[str, Any]) -> None:
        """Send action in isoteleop mode (joint positions)."""
        target_joints = np.array([action[f"joint_{i+1}.pos"] for i in range(self._num_joints)])
        
        if not self.config.debug:
            try:
                joint_positions = self._robot.robot_get_joint_positions()
                max_delta = (np.abs(joint_positions - target_joints)).max()
                
                if max_delta > self.config.max_joint_delta:
                    logger.warning("MOVING TOO FAST! SLOW DOWN!")
                    steps = min(int(max_delta / 0.05), 100)
                    for jnt in np.linspace(joint_positions, target_joints, steps):
                        self._robot.robot_update_desired_joint_positions(jnt)
                        time.sleep(0.02)
                else:
                    self._robot.robot_update_desired_joint_positions(target_joints)
            except Exception as e:
                logger.warning(f"[ROBOT] isoteleop action failed: {e}")
                try:
                    self._robot.robot_start_joint_impedance_control()
                except Exception as e2:
                    logger.error(f"[ROBOT] Failed to restart controller: {e2}")
        
        if "gripper_position" in action:
            self._handle_gripper(action["gripper_position"], is_binary=False)
    
    def _send_action_cartesian(self, action: dict[str, Any]) -> None:
        """Send action in oculus mode (delta ee pose)."""
        # Check for reset request
        if action.get("reset_requested", False):
            logger.info(f"[ROBOT] Reset requested for {self._arm_side} arm...")
            try:
                self._robot.robot_go_home()
                self._robot.gripper_goto(
                    width=self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                    blocking=True
                )
                self._robot.robot_start_joint_impedance_control()
            except Exception as e:
                logger.warning(f"[ROBOT] Reset failed: {e}")
            return
        
        delta_ee_pose = np.array([action[f"delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]])
        
        # EMA smoothing
        if np.linalg.norm(delta_ee_pose) < 1e-6:
            self._smoothed_delta = None
        else:
            if self._smoothed_delta is None:
                self._smoothed_delta = delta_ee_pose.copy()
            else:
                alpha = self._smoothing_alpha
                self._smoothed_delta = alpha * delta_ee_pose + (1 - alpha) * self._smoothed_delta
            delta_ee_pose = self._smoothed_delta.copy()
        
        if not self.config.debug:
            import scipy.spatial.transform as st
            
            try:
                ee_pose = self._robot.robot_get_ee_pose()
            except Exception as e:
                logger.warning(f"[ROBOT] Failed to get ee pose: {e}")
                if "gripper_cmd_bin" in action:
                    self._handle_gripper(action["gripper_cmd_bin"], is_binary=True)
                return
            
            # Calculate position and rotation deltas
            position_delta = np.linalg.norm(delta_ee_pose[:3])
            rotation_delta = np.linalg.norm(delta_ee_pose[3:])
            
            max_position_step = 0.02
            max_rotation_step = 0.1
            
            position_steps = max(1, int(np.ceil(position_delta / max_position_step))) if position_delta > 0 else 1
            rotation_steps = max(1, int(np.ceil(rotation_delta / max_rotation_step))) if rotation_delta > 0 else 1
            num_steps = max(position_steps, rotation_steps)
            
            if num_steps > 1:
                for step in range(1, num_steps + 1):
                    alpha = step / num_steps
                    interpolated_delta = delta_ee_pose * alpha
                    
                    target_position = ee_pose[:3] + interpolated_delta[:3]
                    current_rot = st.Rotation.from_rotvec(ee_pose[3:])
                    delta_rot = st.Rotation.from_rotvec(interpolated_delta[3:])
                    target_rotation = delta_rot * current_rot
                    target_rotvec = target_rotation.as_rotvec()
                    target_ee_pose = np.concatenate([target_position, target_rotvec])
                    
                    try:
                        self._robot.robot_update_desired_ee_pose(target_ee_pose)
                    except Exception as e:
                        logger.warning(f"[ROBOT] zerorpc error during interpolation: {e}")
                        break
                    time.sleep(0.01)
            elif np.linalg.norm(delta_ee_pose) >= 0.01:
                target_position = ee_pose[:3] + delta_ee_pose[:3]
                current_rot = st.Rotation.from_rotvec(ee_pose[3:])
                delta_rot = st.Rotation.from_rotvec(delta_ee_pose[3:])
                target_rotation = delta_rot * current_rot
                target_rotvec = target_rotation.as_rotvec()
                target_ee_pose = np.concatenate([target_position, target_rotvec])
                
                try:
                    self._robot.robot_update_desired_ee_pose(target_ee_pose)
                except Exception as e:
                    logger.warning(f"[ROBOT] zerorpc error: {e}")
        
        if "gripper_cmd_bin" in action:
            self._handle_gripper(action["gripper_cmd_bin"], is_binary=True)
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            joint_position = self._robot.robot_get_joint_positions()
            joint_velocity = self._robot.robot_get_joint_velocities()
            ee_pose = self._robot.robot_get_ee_pose()
        except Exception as e:
            logger.warning(f"[ROBOT] zerorpc error in get_observation: {e}")
            if self._prev_observation is not None:
                return self._prev_observation
            else:
                raise
        
        obs_dict = {}
        
        # Joint positions and velocities
        for i in range(len(joint_position)):
            obs_dict[f"joint_{i+1}.pos"] = float(joint_position[i])
            obs_dict[f"joint_{i+1}.vel"] = float(joint_velocity[i])
        
        # End effector pose
        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs_dict[f"ee_pose.{axis}"] = float(ee_pose[i])
        
        # Gripper state
        if self.config.use_gripper:
            obs_dict["gripper_state_norm"] = self._gripper_position
            obs_dict["gripper_cmd_bin"] = self._last_gripper_position
        else:
            obs_dict["gripper_state_norm"] = None
            obs_dict["gripper_cmd_bin"] = None
        
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


class DobotDualArm(Robot):
    """
    Dual-arm Dobot Nova5 robot with two Robotiq 2F-85 grippers.
    Each arm has 6 DOF, total 12 DOF.
    """
    
    config_class = DobotDualArmConfig
    name = "dobot_dual_arm"
    
    def __init__(self, config: DobotDualArmConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self.config = config
        self._is_connected = False
        self._robot: Optional[DobotDualArmClient] = None
        self._prev_observation = None
        self._num_joints_per_arm = 6
        
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
        logger.info("[ROBOT] Connecting to Dobot Nova5 Dual-Arm System")
        logger.info("=" * 60)
        
        # Connect to both arms
        self._robot = self._check_dobot_connection()
        
        # Initialize grippers
        if self.config.use_gripper:
            self._check_grippers_connection()
        
        # Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")
        
        self.is_connected = True
        logger.info(f"[INFO] {self.name} initialization completed successfully.\n")
    
    def _check_dobot_connection(self) -> DobotDualArmClient:
        """Connect to both Dobot arms via zerorpc."""
        try:
            logger.info("\n===== [ROBOT] Connecting to Dobot dual-arm =====")
            
            robot = DobotDualArmClient(
                ip='127.0.0.1',
                left_port=self.config.left_arm_port,
                right_port=self.config.right_arm_port
            )
            robot.robot_start_joint_impedance_control()
            
            # Get joint positions for both arms
            left_joints = robot.left_arm.robot_get_joint_positions()
            right_joints = robot.right_arm.robot_get_joint_positions()
            
            if left_joints is not None and len(left_joints) == self._num_joints_per_arm:
                logger.info(f"[LEFT ARM] Joint positions: {[round(j, 4) for j in left_joints]}")
            if right_joints is not None and len(right_joints) == self._num_joints_per_arm:
                logger.info(f"[RIGHT ARM] Joint positions: {[round(j, 4) for j in right_joints]}")
            
            logger.info("===== [ROBOT] Dobot dual-arm connected successfully =====\n")
            return robot
            
        except Exception as e:
            logger.error("===== [ERROR] Failed to connect to Dobot dual-arm =====")
            logger.error(f"Exception: {e}\n")
            raise
    
    def _check_grippers_connection(self):
        """Initialize both Robotiq 2F-85 grippers."""
        logger.info("\n===== [GRIPPER] Initializing Robotiq 2F-85 grippers...")
        
        # Initialize left gripper
        self._robot.left_arm.gripper_initialize()
        self._robot.left_arm.gripper_goto(
            width=self.config.gripper_max_open,
            speed=self._gripper_speed,
            force=self._gripper_force,
            blocking=True
        )
        logger.info("[LEFT GRIPPER] Initialized successfully")
        
        # Initialize right gripper
        self._robot.right_arm.gripper_initialize()
        self._robot.right_arm.gripper_goto(
            width=self.config.gripper_max_open,
            speed=self._gripper_speed,
            force=self._gripper_force,
            blocking=True
        )
        logger.info("[RIGHT GRIPPER] Initialized successfully")
        logger.info("===== [GRIPPER] Both grippers initialized successfully =====\n")
    
    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        logger.info("[ROBOT] Resetting dual-arm system...")
        self._robot.robot_go_home()
        
        if self.config.use_gripper:
            self._robot.left_arm.gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=True
            )
            self._robot.right_arm.gripper_goto(
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
        
        # Left arm joint positions and velocities
        for i in range(self._num_joints_per_arm):
            features[f"left_joint_{i+1}.pos"] = float
            features[f"left_joint_{i+1}.vel"] = float
        
        # Right arm joint positions and velocities
        for i in range(self._num_joints_per_arm):
            features[f"right_joint_{i+1}.pos"] = float
            features[f"right_joint_{i+1}.vel"] = float
        
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
        """Return action features based on control mode."""
        if self.config.control_mode == "isoteleop":
            features = {}
            # Left arm joints
            for i in range(self._num_joints_per_arm):
                features[f"left_joint_{i+1}.pos"] = float
            # Right arm joints
            for i in range(self._num_joints_per_arm):
                features[f"right_joint_{i+1}.pos"] = float
            if self.config.use_gripper:
                features["left_gripper_position"] = float
                features["right_gripper_position"] = float
            return features
        elif self.config.control_mode == "oculus":
            features = {}
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
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")
    
    def _handle_gripper(self, arm_side: str, gripper_value: float, is_binary: bool = True) -> None:
        """Handle gripper control for specified arm."""
        if not self.config.use_gripper:
            return
        
        arm = self._robot.left_arm if arm_side == "left" else self._robot.right_arm
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
                arm.gripper_goto(
                    width=gripper_position * self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                )
                setattr(self, last_pos_attr, gripper_position)
            
            gripper_state = arm.gripper_get_state()
            gripper_state_norm = max(0.0, min(1.0, gripper_state["width"] / self.config.gripper_max_open))
            if self.config.gripper_reverse:
                gripper_state_norm = 1 - gripper_state_norm
            setattr(self, gripper_pos_attr, gripper_state_norm)
        except Exception as e:
            logger.warning(f"[{arm_side.upper()} GRIPPER] zerorpc error: {e}")
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.control_mode == "isoteleop":
            self._send_action_isoteleop(action)
        elif self.config.control_mode == "oculus":
            self._send_action_cartesian(action)
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")
        
        return action
    
    def _send_action_isoteleop(self, action: dict[str, Any]) -> None:
        """Send action in isoteleop mode (joint positions)."""
        # Left arm
        left_target_joints = np.array([
            action[f"left_joint_{i+1}.pos"] for i in range(self._num_joints_per_arm)
        ])
        # Right arm
        right_target_joints = np.array([
            action[f"right_joint_{i+1}.pos"] for i in range(self._num_joints_per_arm)
        ])
        
        if not self.config.debug:
            try:
                left_joints = self._robot.left_arm.robot_get_joint_positions()
                right_joints = self._robot.right_arm.robot_get_joint_positions()
                
                left_max_delta = (np.abs(left_joints - left_target_joints)).max()
                right_max_delta = (np.abs(right_joints - right_target_joints)).max()
                
                if left_max_delta > self.config.max_joint_delta or right_max_delta > self.config.max_joint_delta:
                    logger.warning("MOVING TOO FAST! SLOW DOWN!")
                
                self._robot.left_arm.robot_update_desired_joint_positions(left_target_joints)
                self._robot.right_arm.robot_update_desired_joint_positions(right_target_joints)
                
            except Exception as e:
                logger.warning(f"[ROBOT] isoteleop action failed: {e}")
        
        if "left_gripper_position" in action:
            self._handle_gripper("left", action["left_gripper_position"], is_binary=False)
        if "right_gripper_position" in action:
            self._handle_gripper("right", action["right_gripper_position"], is_binary=False)
    
    def _send_action_cartesian(self, action: dict[str, Any]) -> None:
        """Send action in oculus mode (delta ee pose)."""
        # Check for reset request
        if action.get("reset_requested", False):
            logger.info("[ROBOT] Reset requested for dual-arm system...")
            self._robot.robot_go_home()
            if self.config.use_gripper:
                self._robot.left_arm.gripper_goto(
                    width=self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                    blocking=True
                )
                self._robot.right_arm.gripper_goto(
                    width=self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                    blocking=True
                )
            self._robot.robot_start_joint_impedance_control()
            return
        
        # Get delta poses for both arms
        left_delta = np.array([
            action[f"left_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ])
        right_delta = np.array([
            action[f"right_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ])
        
        # EMA smoothing for both arms
        for arm_side, delta, smoothed_attr in [
            ("left", left_delta, "_left_smoothed_delta"),
            ("right", right_delta, "_right_smoothed_delta")
        ]:
            if np.linalg.norm(delta) < 1e-6:
                setattr(self, smoothed_attr, None)
            else:
                smoothed = getattr(self, smoothed_attr)
                if smoothed is None:
                    smoothed = delta.copy()
                else:
                    alpha = self._smoothing_alpha
                    smoothed = alpha * delta + (1 - alpha) * smoothed
                setattr(self, smoothed_attr, smoothed)
        
        if not self.config.debug:
            import scipy.spatial.transform as st
            
            for arm_side, delta, smoothed_attr in [
                ("left", left_delta, "_left_smoothed_delta"),
                ("right", right_delta, "_right_smoothed_delta")
            ]:
                arm = getattr(self._robot, f"{arm_side}_arm")
                smoothed = getattr(self, smoothed_attr)
                
                if smoothed is None or np.linalg.norm(smoothed) < 0.01:
                    continue
                
                try:
                    ee_pose = arm.robot_get_ee_pose()
                    
                    target_position = ee_pose[:3] + smoothed[:3]
                    current_rot = st.Rotation.from_rotvec(ee_pose[3:])
                    delta_rot = st.Rotation.from_rotvec(smoothed[3:])
                    target_rotation = delta_rot * current_rot
                    target_rotvec = target_rotation.as_rotvec()
                    target_ee_pose = np.concatenate([target_position, target_rotvec])
                    
                    arm.robot_update_desired_ee_pose(target_ee_pose)
                except Exception as e:
                    logger.warning(f"[{arm_side.upper()} ARM] zerorpc error: {e}")
        
        if "left_gripper_cmd_bin" in action:
            self._handle_gripper("left", action["left_gripper_cmd_bin"], is_binary=True)
        if "right_gripper_cmd_bin" in action:
            self._handle_gripper("right", action["right_gripper_cmd_bin"], is_binary=True)
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            left_joint_pos = self._robot.left_arm.robot_get_joint_positions()
            left_joint_vel = self._robot.left_arm.robot_get_joint_velocities()
            left_ee_pose = self._robot.left_arm.robot_get_ee_pose()
            
            right_joint_pos = self._robot.right_arm.robot_get_joint_positions()
            right_joint_vel = self._robot.right_arm.robot_get_joint_velocities()
            right_ee_pose = self._robot.right_arm.robot_get_ee_pose()
            
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
            obs_dict[f"left_joint_{i+1}.vel"] = float(left_joint_vel[i])
        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs_dict[f"left_ee_pose.{axis}"] = float(left_ee_pose[i])
        
        # Right arm observations
        for i in range(len(right_joint_pos)):
            obs_dict[f"right_joint_{i+1}.pos"] = float(right_joint_pos[i])
            obs_dict[f"right_joint_{i+1}.vel"] = float(right_joint_vel[i])
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

