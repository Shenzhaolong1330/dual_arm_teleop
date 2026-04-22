#!/usr/bin/env python

"""
Oculus Quest dual-arm teleoperation implementation.
Uses both Oculus controllers to control a dual-arm robot system.
Left controller -> Left arm, Right controller -> Right arm.
Only outputs delta EE pose, no IK implementation.
"""

import logging
from typing import Any, Dict, Optional

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_oculus_teleop import OculusTeleopConfig
from .oculus.oculus_dual_arm_robot import OculusDualArmRobot

logger = logging.getLogger(__name__)


class OculusTeleop(Teleoperator):
    """
    Dual-arm teleoperation using both Oculus Quest controllers.
    
    This teleoperation mode uses both Oculus Quest controllers to simultaneously
    control two robot arms in Cartesian space. The output is delta pose 
    (position and orientation changes) for both arms.
    
    Controls:
    - LG (Left Grip):    Must be pressed to enable left arm action recording
    - RG (Right Grip):   Must be pressed to enable right arm action recording
    - LTr (Left Trigger):  Controls left gripper  (0.0 = open, 1.0 = closed)
    - RTr (Right Trigger): Controls right gripper (0.0 = open, 1.0 = closed)
    - Left controller pose:  Controls left arm end-effector delta pose
    - Right controller pose: Controls right arm end-effector delta pose
    - A button: Request RIGHT arm reset
    - X button: Request LEFT arm reset
    """
    
    config_class = OculusTeleopConfig
    name = "OculusTeleop"
    
    def __init__(self, config: OculusTeleopConfig):
        super().__init__(config)
        self.cfg = config
        self._is_connected = False
        self.oculus_robot: Optional[OculusDualArmRobot] = None
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_connected
    
    @property
    def action_features(self) -> dict:
        """Return action features for dual-arm oculus mode (delta ee pose only)."""
        features = {}
        # Delta EE poses for both arms
        for arm in ["left", "right"]:
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"{arm}_delta_ee_pose.{axis}"] = float
        
        # Gripper commands
        if self.cfg.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        
        return features
    
    @property
    def feedback_features(self) -> dict:
        return {}
    
    def connect(self) -> None:
        """Connect to Oculus Quest."""
        if self._is_connected:
            logger.warning(f"{self.name} is already connected.")
            return
        
        logger.info(f"\n===== [TELEOP] Connecting to {self.name} =====")
        
        self.oculus_robot = OculusDualArmRobot(
            ip=self.cfg.ip,
            use_gripper=self.cfg.use_gripper,
            left_pose_scaler=self.cfg.left_pose_scaler,
            left_channel_signs=self.cfg.left_channel_signs,
            right_pose_scaler=self.cfg.right_pose_scaler,
            right_channel_signs=self.cfg.right_channel_signs,
            action_smoothing_alpha=self.cfg.action_smoothing_alpha,
        )
        
        self._is_connected = True
        logger.info(f"===== [TELEOP] {self.name} connected successfully =====")
        logger.info(f"[TELEOP] Oculus dual-arm connected at IP: {self.cfg.ip}")
    
    def disconnect(self) -> None:
        """Disconnect from Oculus Quest."""
        if not self._is_connected:
            return
        
        self.oculus_robot = None
        self._is_connected = False
        logger.info(f"[INFO] ===== {self.name} disconnected =====")
    
    def get_action(self) -> Dict[str, Any]:
        """Get the current action from the teleoperation device."""
        if not self._is_connected:
            raise RuntimeError(f"{self.name} is not connected.")
        return self._get_action_impl()
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """Get delta pose from both Oculus controllers."""
        if self.oculus_robot is None:
            raise RuntimeError("Oculus robot is not initialized.")
        
        # Get observations from OculusDualArmRobot
        obs = self.oculus_robot.get_observations()
        
        # Build action dict with delta EE poses
        action = {}
        
        # Delta EE poses
        for arm in ["left", "right"]:
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                key = f"{arm}_delta_ee_pose.{axis}"
                if key in obs:
                    action[key] = float(obs[key])
                else:
                    action[key] = 0.0
        
        # Gripper commands
        if self.cfg.use_gripper:
            action["left_gripper_cmd_bin"] = float(obs.get("left_gripper_cmd_bin", 1.0))
            action["right_gripper_cmd_bin"] = float(obs.get("right_gripper_cmd_bin", 1.0))
        
        # Reset request flag (for external use)
        action["left_arm_reset_requested"] = bool(obs.get("left_arm_reset_requested", False))
        action["right_arm_reset_requested"] = bool(obs.get("right_arm_reset_requested", False))
        action["reset_requested"] = obs.get("reset_requested", False)
        
        return action
    
    def calibrate(self) -> None:
        """Calibrate the teleoperation device. Default: no-op."""
        pass
    
    def configure(self) -> None:
        """Configure the teleoperation device. Default: no-op."""
        pass
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperation device. Default: no-op."""
        pass
    
    def is_reset_requested(self) -> bool:
        """Check if any arm reset was requested (A or X button pressed)."""
        if self.oculus_robot is None:
            return False
        return self.oculus_robot.is_reset_requested()


if __name__ == "__main__":
    import time
    
    # Test the OculusTeleop class
    config = OculusTeleopConfig(
        ip="192.168.110.62",
        use_gripper=True,
        left_pose_scaler=[0.5, 0.5],
        left_channel_signs=[1, 1, 1, 1, 1, 1],
        right_pose_scaler=[0.5, 0.5],
        right_channel_signs=[1, 1, 1, 1, 1, 1],
    )
    
    teleop = OculusTeleop(config)
    teleop.connect()
    
    print("===== Oculus Dual-Arm Teleop Test =====")
    print("Controls:")
    print("  - LG (Left Grip):    Press to enable LEFT arm action")
    print("  - RG (Right Grip):   Press to enable RIGHT arm action")
    print("  - LTr (Left Trigger):  Control LEFT gripper")
    print("  - RTr (Right Trigger): Control RIGHT gripper")
    print("  - A button: Request RIGHT arm reset")
    print("  - X button: Request LEFT arm reset")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            action = teleop.get_action()
            
            reset_flag = " [RESET]" if action.get("reset_requested", False) else ""
            
            print(f"\rL: X={action['left_delta_ee_pose.x']:+.4f} Y={action['left_delta_ee_pose.y']:+.4f} "
                  f"Z={action['left_delta_ee_pose.z']:+.4f} G={action['left_gripper_cmd_bin']:.2f} | "
                  f"R: X={action['right_delta_ee_pose.x']:+.4f} Y={action['right_delta_ee_pose.y']:+.4f} "
                  f"Z={action['right_delta_ee_pose.z']:+.4f} G={action['right_gripper_cmd_bin']:.2f}"
                  f"{reset_flag}    ", end="")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        teleop.disconnect()
