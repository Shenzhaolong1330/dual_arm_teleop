"""
Example robot implementation template.

This class implements the Robot interface for a specific robot.
It handles:
- Connection/disconnection to robot hardware
- Sending actions (joint positions or delta EE pose)
- Getting observations (joint states, EE pose, camera images)
- Gripper control

Implementation Steps:
1. Inherit from Robot class
2. Set config_class and name class attributes
3. Implement all required methods:
   - connect(): Initialize connection to robot and cameras
   - disconnect(): Close all connections
   - send_action(action): Execute robot action
   - get_observation(): Read robot state and camera images
   - reset(): Move robot to home position
4. Define action_features and observation_features properties
"""
import logging
import time
from typing import Any, Dict

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError

from .config_example import ExampleRobotConfig
from .example_robot_interface_client import ExampleRobotInterfaceClient

logger = logging.getLogger(__name__)


class ExampleRobot(Robot):
    """
    Example robot implementation template.
    
    TODO: Replace with your robot's actual implementation.
    """
    
    config_class = ExampleRobotConfig
    name = "example_robot"
    
    def __init__(self, config: ExampleRobotConfig):
        """
        Initialize the robot.
        
        Implementation:
        - Call super().__init__(config)
        - Initialize cameras using make_cameras_from_configs()
        - Set up internal state variables:
          - _is_connected: connection status
          - _robot: interface client instance
          - _num_joints: number of joints
          - Gripper state variables
          - Smoothing/filtering variables
        """
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self.config = config
        self._is_connected = False
        self._robot = None  # Interface client
        self._prev_observation = None
        self._num_joints = config.num_joints
        
        # Gripper state
        self._gripper_force = config.gripper_force
        self._gripper_speed = config.gripper_speed
        self._gripper_position = 1.0  # 1.0 = open, 0.0 = closed
        self._last_gripper_position = 1.0
        
        # Action smoothing (EMA filter)
        self._smoothing_alpha = 0.4  # Smoothing coefficient (0-1)
        self._smoothed_delta = None
        
    # ==================== Connection Methods ====================
    
    def connect(self) -> None:
        """
        Connect to robot and cameras.
        
        Implementation:
        1. Check if already connected
        2. Connect to robot via interface client
        3. Initialize gripper if use_gripper is True
        4. Connect all cameras
        5. Set _is_connected = True
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        
        logger.info(f"\n===== [ROBOT] Connecting to {self.name} =====")
        
        # TODO: Implement robot connection
        # self._robot = ExampleRobotInterfaceClient(
        #     ip=self.config.robot_ip,
        #     port=self.config.robot_port
        # )
        # self._robot.start_control()
        
        # TODO: Initialize gripper if needed
        # if self.config.use_gripper:
        #     self._robot.gripper_initialize()
        
        # Connect cameras
        logger.info("===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected.")
        
        self._is_connected = True
        logger.info(f"===== [ROBOT] {self.name} connected successfully =====\n")


    def disconnect(self) -> None:
        """
        Disconnect from robot and cameras.
        
        Implementation:
        1. Check if connected
        2. Disconnect all cameras
        3. Close robot connection
        4. Set _is_connected = False
        """
        if not self.is_connected:
            return
        
        for cam in self.cameras.values():
            cam.disconnect()
        
        # TODO: Close robot connection
        # if self._robot:
        #     self._robot.close()
        
        self._is_connected = False
        logger.info(f"[INFO] {self.name} disconnected.")
    
    # ==================== Control Methods ====================
    
    def reset(self) -> None:
        """
        Reset robot to home position.
        
        Implementation:
        1. Move robot to predefined home pose
        2. Open gripper
        3. Start impedance control mode
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        logger.info(f"[ROBOT] Resetting {self.name} to home position...")
        
        # TODO: Implement reset
        # home_pose = np.array([0.4, 0.0, 0.4, 0.0, 0.0, 0.0])  # Example home pose
        # self._robot.move_to_ee_pose(pose=home_pose, time_to_go=5.0)
        # self._robot.gripper_goto(width=self.config.gripper_max_open, ...)
        # self._robot.start_impedance_control()
        
        logger.info(f"[ROBOT] {self.name} reset complete.")


    # ==================== Properties ====================
    
    @property
    def action_features(self) -> Dict[str, type]:
        """
        Define action space features.
        
        Returns dict mapping feature names to types.
        Format depends on control_mode:
        - oculus/spacemouse: delta_ee_pose.{axis}, gripper_cmd_bin
        - isoteleop: joint_{i}.pos, gripper_position
        """
        if self.config.control_mode in ["oculus", "spacemouse"]:
            features = {}
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"delta_ee_pose.{axis}"] = float
            if self.config.use_gripper:
                features["gripper_cmd_bin"] = float
            return features
        elif self.config.control_mode == "isoteleop":
            features = {}
            for i in range(self._num_joints):
                features[f"joint_{i+1}.pos"] = float
            if self.config.use_gripper:
                features["gripper_position"] = float
            return features
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send action to robot.
        
        Implementation:
        1. Check connection
        2. Route to appropriate method based on control_mode:
           - "oculus" or "spacemouse": _send_action_cartesian()
           - "isoteleop": _send_action_isoteleop()
        3. Handle gripper command
        4. Return the action dict
        
        Action dict format (oculus mode):
        - delta_ee_pose.x/y/z/rx/ry/rz: delta pose changes
        - gripper_cmd_bin: binary gripper command (0 or 1)
        - restart_requested: flag to restart controller
        - reset_requested: flag to reset to home position
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        if self.config.control_mode in ["oculus", "spacemouse"]:
            self._send_action_cartesian(action)
        elif self.config.control_mode == "isoteleop":
            self._send_action_isoteleop(action)
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")
        
        return action
    
    def _send_action_cartesian(self, action: Dict[str, Any]) -> None:
        """
        Send cartesian space action (delta EE pose).
        
        Implementation:
        1. Check for restart/reset requests
        2. Extract delta_ee_pose from action
        3. Apply EMA smoothing to reduce jitter
        4. Get current EE pose
        5. Compute target EE pose = current + delta
        6. Handle large motions with interpolation
        7. Send target pose to robot
        8. Handle gripper command
        """
        # TODO: Implement cartesian action
        # delta_ee_pose = np.array([action[f"delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]])
        # 
        # # Apply smoothing
        # if self._smoothed_delta is None:
        #     self._smoothed_delta = delta_ee_pose.copy()
        # else:
        #     self._smoothed_delta = self._smoothing_alpha * delta_ee_pose + (1 - self._smoothing_alpha) * self._smoothed_delta
        # 
        # # Get current pose and compute target
        # ee_pose = self._robot.get_ee_pose()
        # target_pose = ee_pose + self._smoothed_delta
        # 
        # # Send to robot
        # if not self.config.debug:
        #     self._robot.update_desired_ee_pose(target_pose)
        
        if "gripper_cmd_bin" in action:
            self._handle_gripper(action["gripper_cmd_bin"], is_binary=True)
    
    def _send_action_isoteleop(self, action: Dict[str, Any]) -> None:
        """
        Send joint space action (direct joint positions).
        
        Implementation:
        1. Extract target joint positions from action
        2. Get current joint positions
        3. Check for large jumps and interpolate if needed
        4. Send target positions to robot
        5. Handle gripper command
        """
        # TODO: Implement joint action
        # target_joints = np.array([action[f"joint_{i+1}.pos"] for i in range(self._num_joints)])
        # 
        # if not self.config.debug:
        #     current_joints = self._robot.get_joint_positions()
        #     max_delta = np.abs(current_joints - target_joints).max()
        #     
        #     if max_delta > 0.3:
        #         # Interpolate for large jumps
        #         steps = int(max_delta / 0.05)
        #         for jnt in np.linspace(current_joints, target_joints, steps):
        #             self._robot.update_desired_joint_positions(jnt)
        #             time.sleep(0.02)
        #     else:
        #         self._robot.update_desired_joint_positions(target_joints)
        
        if "gripper_position" in action:
            self._handle_gripper(action["gripper_position"], is_binary=False)
    
    def _handle_gripper(self, gripper_value: float, is_binary: bool = True) -> None:
        """
        Handle gripper control.
        
        Implementation:
        1. Convert value to normalized position (0-1)
        2. Apply gripper_reverse if configured
        3. Send gripper command if position changed
        4. Read and store gripper state
        """
        if not self.config.use_gripper:
            return
        
        # TODO: Implement gripper control
        # gripper_position = gripper_value if is_binary else (0.0 if gripper_value < self.config.close_threshold else 1.0)
        # 
        # if self.config.gripper_reverse:
        #     gripper_position = 1 - gripper_position
        # 
        # if gripper_position != self._last_gripper_position:
        #     self._robot.gripper_goto(
        #         width=gripper_position * self.config.gripper_max_open,
        #         speed=self._gripper_speed,
        #         force=self._gripper_force
        #     )
        #     self._last_gripper_position = gripper_position
        pass

    def get_observation(self) -> Dict[str, Any]:
        """
        Get current robot observation.
        
        Implementation:
        1. Check connection
        2. Read joint positions
        3. Read end-effector pose
        4. Read gripper state
        5. Capture camera images
        6. Return observation dict
        
        Observation dict format:
        - joint_{i}.pos: joint position for each joint
        - ee_pose.x/y/z/rx/ry/rz: end-effector pose
        - gripper_state_norm: normalized gripper position (0-1)
        - {camera_name}: camera image
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        
        obs_dict = {}
        
        # TODO: Read robot state
        # joint_positions = self._robot.get_joint_positions()
        # ee_pose = self._robot.get_ee_pose()
        # 
        # for i in range(len(joint_positions)):
        #     obs_dict[f"joint_{i+1}.pos"] = float(joint_positions[i])
        # 
        # for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        #     obs_dict[f"ee_pose.{axis}"] = float(ee_pose[i])
        
        # Gripper state
        if self.config.use_gripper:
            obs_dict["gripper_state_norm"] = self._gripper_position
            obs_dict["gripper_cmd_bin"] = self._last_gripper_position
        
        # Camera images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read()
        
        self._prev_observation = obs_dict
        return obs_dict
    
    @property
    def observation_features(self) -> Dict[str, Any]:
        """
        Define observation space features.
        
        Returns dict mapping feature names to types.
        Includes:
        - Joint positions
        - End-effector pose
        - Gripper state
        - Camera image dimensions
        """
        features = {}
        
        # Joint positions
        for i in range(self._num_joints):
            features[f"joint_{i+1}.pos"] = float
        
        # EE pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"ee_pose.{axis}"] = float
        
        # Gripper
        if self.config.use_gripper:
            features["gripper_state_norm"] = float
        
        # Cameras
        for cam_name, cam in self.cameras.items():
            features[cam_name] = (cam.height, cam.width, 3)
        
        return features
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value
    
    def is_calibrated(self) -> bool:
        return self.is_connected
    
    def calibrate(self) -> None:
        pass
    
    def configure(self) -> None:
        pass


# ==================== Main (for testing) ====================

if __name__ == "__main__":
    """Test the robot connection."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    config = ExampleRobotConfig(
        robot_ip="192.168.1.100",
        debug=True,
    )
    
    robot = ExampleRobot(config)
    robot.connect()
    
    # Test observation
    obs = robot.get_observation()
    print(f"Observation keys: {obs.keys()}")
    
    robot.disconnect()