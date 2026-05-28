'''
Nero dual-arm robot interface client.
Connects to nero_interface_server via zerorpc.
'''


import logging
import numpy as np
import zerorpc
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


class NeroDualArmClient:
    """Client for dual-arm Nero robot."""
    
    def __init__(self, ip: str = '127.0.0.1', port: int = 4242):
        self.ip = ip
        self.port = port
        
        try:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect(f"tcp://{ip}:{port}")
            log.info(f"[CLIENT] Connected to {ip}:{port}")
        except Exception as e:
            log.error(f"[CLIENT] Connection failed: {e}")
            self.server = None
    
    # ==================== State Query ====================
    
    def left_robot_get_joint_positions(self) -> np.ndarray:
        """Get left arm joint positions (radians)."""
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.left_robot_get_joint_positions())
    
    def left_robot_get_joint_velocities(self) -> np.ndarray:
        """Get left arm joint velocities (rad/s)."""
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.left_robot_get_joint_velocities())
    
    def left_robot_get_arm_status(self) -> dict:
        """Get left arm status."""
        if self.server is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        return self.server.left_robot_get_arm_status()
    
    def right_robot_get_joint_positions(self) -> np.ndarray:
        """Get right arm joint positions (radians)."""
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.right_robot_get_joint_positions())
    
    def right_robot_get_joint_velocities(self) -> np.ndarray:
        """Get right arm joint velocities (rad/s)."""
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.right_robot_get_joint_velocities())
    
    def right_robot_get_arm_status(self) -> dict:
        """Get right arm status."""
        if self.server is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        return self.server.right_robot_get_arm_status()
    
    def left_robot_get_ee_pose(self) -> np.ndarray:
        """Get left arm EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return np.zeros(6)
        return np.array(self.server.left_robot_get_ee_pose())
    
    def right_robot_get_ee_pose(self) -> np.ndarray:
        """Get right arm EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return np.zeros(6)
        return np.array(self.server.right_robot_get_ee_pose())
    
    def right_robot_get_ee_pose(self) -> np.ndarray:
        """Get right arm EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return np.zeros(6)
        # Server returns meters and radians
        return np.array(self.server.right_robot_get_ee_pose())
    
    # ==================== MoveIt Motion ====================
    
    def left_robot_move_to_joint_positions(self, positions: np.ndarray, delta: bool = False):
        """Move left arm to joint positions (radians)."""
        if self.server is None:
            return
        self.server.left_robot_move_to_joint_positions(positions.tolist(), delta)
    
    def right_robot_move_to_joint_positions(self, positions: np.ndarray, delta: bool = False):
        """Move right arm to joint positions (radians)."""
        if self.server is None:
            return
        self.server.right_robot_move_to_joint_positions(positions.tolist(), delta)
    
    def left_robot_move_to_ee_pose(self, pose: np.ndarray, delta: bool = False):
        """Move left arm to EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return
        self.server.left_robot_move_to_ee_pose(pose.tolist(), delta)
    
    def right_robot_move_to_ee_pose(self, pose: np.ndarray, delta: bool = False):
        """Move right arm to EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return
        self.server.right_robot_move_to_ee_pose(pose.tolist(), delta)
    
    def dual_robot_move_to_ee_pose(self, left_pose: np.ndarray, right_pose: np.ndarray, delta: bool = False):
        """Move both arms to EE poses simultaneously."""
        if self.server is None:
            return
        self.server.dual_robot_move_to_ee_pose(left_pose.tolist(), right_pose.tolist(), delta)
    
    # ==================== Go Home ====================
    
    def left_robot_go_home(self):
        """Move left arm to home position."""
        if self.server is None:
            return
        self.server.left_robot_go_home()
    
    def right_robot_go_home(self):
        """Move right arm to home position."""
        if self.server is None:
            return
        self.server.right_robot_go_home()
    
    def robot_go_home(self):
        """Move both arms to home position."""
        if self.server is None:
            return
        self.server.robot_go_home()
    
    # ==================== ServoJ Control (Joint Servo) ====================
    
    def servo_j(self, robot_arm: str, joints: np.ndarray, delta: bool = False) -> bool:
        """
        Send ServoJ with joint angles.
        Args:
            robot_arm: 'left_robot' or 'right_robot'
            joints: Joint angles in RADIANS
            delta: False=absolute, True=relative
        """
        if self.server is None:
            return True
        return self.server.servo_j(robot_arm, joints.tolist(), delta)
    
    # ==================== ServoP Control (Pose Servo) ====================
    
    def servo_p(self, robot_arm: str, pose: np.ndarray, delta: bool = False) -> bool:
        """
        Send ServoP with target pose [x, y, z, rx, ry, rz] (m, radians).
        Args:
            robot_arm: 'left_robot' or 'right_robot'
            pose: Target pose in METERS and RADIANS
            delta: False=absolute, True=relative
        """
        if self.server is None:
            return True
        return self.server.servo_p(robot_arm, pose.tolist(), delta)
    
    def servo_p_OL(self, robot_arm: str, pose: np.ndarray, delta: bool = False) -> bool:
        """
        Send ServoP open loop with target pose [x, y, z, rx, ry, rz] (m, radians).
        Args:
            robot_arm: 'left_robot' or 'right_robot'
            pose: Target pose in METERS and RADIANS
            delta: False=absolute, True=relative
        """
        if self.server is None:
            return True
        return self.server.servo_p_OL(robot_arm, pose.tolist(), delta)
    
    # ==================== Gripper ========
    
    def left_gripper_goto(self, width: float, force: float):
        if self.server is None:
            return
        self.server.left_gripper_goto(width, force)
    
    def left_gripper_grasp(self, force: float = 1.0, width: float = 0.05):
        if self.server is None:
            return
        self.server.left_gripper_grasp(force, width)
    
    def left_gripper_get_state(self) -> dict:
        if self.server is None:
            return {"width": 0.0, "is_moving": False, "is_grasped": False}
        return self.server.left_gripper_get_state()
    
    def right_gripper_goto(self, width: float, force: float):
        if self.server is None:
            return
        self.server.right_gripper_goto(width, force)

    def right_gripper_grasp(self, force: float = 1.0, width: float = 0.05):
        if self.server is None:
            return
        self.server.right_gripper_grasp(force, width)    
    
    def right_gripper_get_state(self) -> dict:
        if self.server is None:
            return {"width": 0.0, "is_moving": False, "is_grasped": False}
        return self.server.right_gripper_get_state()
    
    # ==================== Utility ====================
    
    def stop(self, arm_name: str):
        """Stop specified arm.
        
        Args:
            arm_name: 'left_robot' or 'right_robot'
        """
        if self.server is None:
            log.warning("[CLIENT] Server not connected")
            return
        try:
            result = self.server.robot_stop(arm_name)
            log.info(f"[CLIENT] Stop command sent to {arm_name}: {result}")
        except Exception as e:
            log.error(f"[CLIENT] Failed to stop {arm_name}: {e}")
    
    def close(self):
        """Close connection."""
        if self.server is not None:
            try:
                self.server.robot_stop("left_robot")
                self.server.robot_stop("right_robot")
            except Exception as e:
                log.warning(f"[CLIENT] Error stopping robots: {e}")
            try:
                self.server.close()
            except Exception as e:
                log.debug(f"[CLIENT] Error closing server connection: {e}")
            self.server = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    client = NeroDualArmClient()
    
    # Test connection
    print("Testing connection...")
    left_joints = client.left_robot_get_joint_positions()
    right_joints = client.right_robot_get_joint_positions()
    print(f"Left joints (rad): {left_joints}")
    print(f"Right joints (rad): {right_joints}")
    
    left_pose = client.left_robot_get_ee_pose()
    print(f"Left pose (m, rad): {left_pose}")
    
    client.close()
    print("Done!")