'''
Dobot Nova5 dual-arm robot interface client.
Running on the user machine to connect to the dobot_interface_server via zerorpc.
Single port for dual-arm control.
'''

import logging
import numpy as np
import zerorpc
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


class DobotDualArmClient:
    """
    Client for dual-arm Dobot Nova5 robot.
    Connects to a single zerorpc server that controls both arms.
    """
    
    def __init__(self, ip: str = '127.0.0.1', port: int = 4242):
        """
        Initialize dual-arm client.
        
        Args:
            ip: Server IP address
            port: Server port (single port for both arms)
        """
        self.ip = ip
        self.port = port
        
        log.info("=" * 60)
        log.info("Initializing Dobot Dual-Arm Client")
        log.info("=" * 60)
        
        try:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect(f"tcp://{ip}:{port}")
            log.info(f"[DUAL-ARM] Connected to server at {ip}:{port}")
        except Exception as e:
            log.error(f"[DUAL-ARM] Failed to connect to server: {e}")
            self.server = None
        
        log.info("=" * 60)
        log.info("Dual-Arm Client Initialized")
        log.info("=" * 60)

    # ==================== Left Arm Interface ====================
    
    def left_robot_get_joint_positions(self) -> np.ndarray:
        """Get left arm joint positions as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            return np.array(self.server.left_robot_get_joint_positions())
        except Exception as e:
            log.error(f"[LEFT ARM] robot_get_joint_positions failed: {e}")
            return np.zeros(6)

    def left_robot_get_joint_velocities(self) -> np.ndarray:
        """Get left arm joint velocities as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            return np.array(self.server.left_robot_get_joint_velocities())
        except Exception as e:
            log.error(f"[LEFT ARM] robot_get_joint_velocities failed: {e}")
            return np.zeros(6)
    
    def left_robot_get_ee_pose(self) -> np.ndarray:
        """Get left arm end-effector pose [x, y, z, rx, ry, rz] as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            return np.array(self.server.left_robot_get_ee_pose())
        except Exception as e:
            log.error(f"[LEFT ARM] robot_get_ee_pose failed: {e}")
            return np.zeros(6)

    def left_robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
    ):
        """Move left arm to target joint positions."""
        if self.server is None:
            return
        try:
            self.server.left_robot_move_to_joint_positions(
                positions.tolist(), 
                time_to_go, 
                delta, 
                Kq.tolist() if Kq is not None else None, 
                Kqd.tolist() if Kqd is not None else None
            )
        except Exception as e:
            log.error(f"[LEFT ARM] robot_move_to_joint_positions failed: {e}")

    def left_robot_move_to_ee_pose(
        self,
        pose: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kx: np.ndarray = None,
        Kxd: np.ndarray = None,
        op_space_interp: bool = True,
        wait: bool = False,
    ):
        """Move left arm to target end-effector pose."""
        if self.server is None:
            return
        try:
            self.server.left_robot_move_to_ee_pose(
                pose.tolist(),
                time_to_go,
                delta,
                Kx.tolist() if Kx is not None else None,
                Kxd.tolist() if Kxd is not None else None,
                op_space_interp,
                wait,
            )
        except Exception as e:
            log.error(f"[LEFT ARM] robot_move_to_ee_pose failed: {e}")

    def left_robot_go_home(self):
        """Move left arm to home position."""
        if self.server is None:
            return
        try:
            self.server.left_robot_go_home()
        except Exception as e:
            log.error(f"[LEFT ARM] robot_go_home failed: {e}")

    # ==================== Left Gripper Interface ====================
    
    def left_gripper_initialize(self):
        """Initialize left gripper."""
        if self.server is None:
            log.error("[LEFT GRIPPER] Not connected to server")
            return
        try:
            self.server.left_gripper_initialize()
            log.info("[LEFT GRIPPER] Gripper initialized")
        except Exception as e:
            log.error(f"[LEFT GRIPPER] Failed to initialize gripper: {e}")

    def left_gripper_goto(
        self, 
        width: float, 
        speed: float, 
        force: float, 
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True
    ):
        """Move left gripper to specified width."""
        if self.server is None:
            return
        try:
            self.server.left_gripper_goto(width, speed, force, epsilon_inner, epsilon_outer, blocking)
        except Exception as e:
            log.error(f"[LEFT GRIPPER] gripper_goto failed: {e}")

    def left_gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        """Grasp with left gripper."""
        if self.server is None:
            return
        try:
            self.server.left_gripper_grasp(
                speed,
                force,
                grasp_width,
                epsilon_inner,
                epsilon_outer,
                blocking,
            )
        except Exception as e:
            log.error(f"[LEFT GRIPPER] gripper_grasp failed: {e}")

    def left_gripper_get_state(self) -> dict:
        """Get left gripper state."""
        if self.server is None:
            return {"width": 0.085, "is_moving": False, "is_grasped": False}
        try:
            return self.server.left_gripper_get_state()
        except Exception as e:
            log.error(f"[LEFT GRIPPER] gripper_get_state failed: {e}")
            return {"width": 0.085, "is_moving": False, "is_grasped": False}

    # ==================== Right Arm Interface ====================
    
    def right_robot_get_joint_positions(self) -> np.ndarray:
        """Get right arm joint positions as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            return np.array(self.server.right_robot_get_joint_positions())
        except Exception as e:
            log.error(f"[RIGHT ARM] robot_get_joint_positions failed: {e}")
            return np.zeros(6)

    def right_robot_get_joint_velocities(self) -> np.ndarray:
        """Get right arm joint velocities as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            return np.array(self.server.right_robot_get_joint_velocities())
        except Exception as e:
            log.error(f"[RIGHT ARM] robot_get_joint_velocities failed: {e}")
            return np.zeros(6)
    
    def right_robot_get_ee_pose(self) -> np.ndarray:
        """Get right arm end-effector pose [x, y, z, rx, ry, rz] as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            return np.array(self.server.right_robot_get_ee_pose())
        except Exception as e:
            log.error(f"[RIGHT ARM] robot_get_ee_pose failed: {e}")
            return np.zeros(6)

    def right_robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
    ):
        """Move right arm to target joint positions."""
        if self.server is None:
            return
        try:
            self.server.right_robot_move_to_joint_positions(
                positions.tolist(), 
                time_to_go, 
                delta, 
                Kq.tolist() if Kq is not None else None, 
                Kqd.tolist() if Kqd is not None else None
            )
        except Exception as e:
            log.error(f"[RIGHT ARM] robot_move_to_joint_positions failed: {e}")

    def right_robot_move_to_ee_pose(
        self,
        pose: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kx: np.ndarray = None,
        Kxd: np.ndarray = None,
        op_space_interp: bool = True,
        wait: bool = False,
    ):
        """Move right arm to target end-effector pose."""
        if self.server is None:
            return
        try:
            self.server.right_robot_move_to_ee_pose(
                pose.tolist(),
                time_to_go,
                delta,
                Kx.tolist() if Kx is not None else None,
                Kxd.tolist() if Kxd is not None else None,
                op_space_interp,
                wait,
            )
        except Exception as e:
            log.error(f"[RIGHT ARM] robot_move_to_ee_pose failed: {e}")

    def right_robot_go_home(self):
        """Move right arm to home position."""
        if self.server is None:
            return
        try:
            self.server.right_robot_go_home()
        except Exception as e:
            log.error(f"[RIGHT ARM] robot_go_home failed: {e}")

    # ==================== Right Gripper Interface ====================
    
    def right_gripper_initialize(self):
        """Initialize right gripper."""
        if self.server is None:
            log.error("[RIGHT GRIPPER] Not connected to server")
            return
        try:
            self.server.right_gripper_initialize()
            log.info("[RIGHT GRIPPER] Gripper initialized")
        except Exception as e:
            log.error(f"[RIGHT GRIPPER] Failed to initialize gripper: {e}")

    def right_gripper_goto(
        self, 
        width: float, 
        speed: float, 
        force: float, 
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True
    ):
        """Move right gripper to specified width."""
        if self.server is None:
            return
        try:
            self.server.right_gripper_goto(width, speed, force, epsilon_inner, epsilon_outer, blocking)
        except Exception as e:
            log.error(f"[RIGHT GRIPPER] gripper_goto failed: {e}")

    def right_gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        """Grasp with right gripper."""
        if self.server is None:
            return
        try:
            self.server.right_gripper_grasp(
                speed,
                force,
                grasp_width,
                epsilon_inner,
                epsilon_outer,
                blocking,
            )
        except Exception as e:
            log.error(f"[RIGHT GRIPPER] gripper_grasp failed: {e}")

    def right_gripper_get_state(self) -> dict:
        """Get right gripper state."""
        if self.server is None:
            return {"width": 0.085, "is_moving": False, "is_grasped": False}
        try:
            return self.server.right_gripper_get_state()
        except Exception as e:
            log.error(f"[RIGHT GRIPPER] gripper_get_state failed: {e}")
            return {"width": 0.085, "is_moving": False, "is_grasped": False}

    # ==================== Dual-Arm Convenience Methods ====================
    
    def robot_go_home(self):
        """Move dual arm to home position."""
        if self.server is None:
            return
        try:
            self.server.robot_go_home()
        except Exception as e:
            log.error(f"[DUAL ARM] robot_go_home failed: {e}")
    def gripper_initialize(self):
        """Initialize both grippers."""
        self.left_gripper_initialize()
        self.right_gripper_initialize()

    def dual_robot_move_to_ee_pose(self, left_pose, right_pose, time_to_go=None, delta=False, Kx=None, Kxd=None, op_space_interp=True, wait=False):
        """Move both arms to target poses simultaneously."""
        try:
            return self.server.dual_robot_move_to_ee_pose(left_pose, right_pose, time_to_go, delta, Kx, Kxd, op_space_interp, wait)
        except Exception as e:
            log.error("Failed to move both arms to EE poses: %s", str(e))
    
    def wait_for_left_arm(self):
        """Wait for left arm to complete current action."""
        try:
            return self.server.wait_for_left_arm()
        except Exception as e:
            log.error("Failed to wait for left arm: %s", str(e))
    
    def wait_for_right_arm(self):
        """Wait for right arm to complete current action."""
        try:
            return self.server.wait_for_right_arm()
        except Exception as e:
            log.error("Failed to wait for right arm: %s", str(e))
    
    def wait_for_all_arms(self):
        """Wait for both arms to complete current actions."""
        try:
            return self.server.wait_for_all_arms()
        except Exception as e:
            log.error("Failed to wait for all arms: %s", str(e))

    def close(self):
        """Close connection to server."""
        if self.server is not None:
            try:
                self.server.close()
            except:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Test dual-arm client (single port)
    client = DobotDualArmClient(ip="127.0.0.1", port=4242)
    
    # Initialize grippers
    client.gripper_initialize()
    
    # Test left gripper
    client.left_gripper_goto(width=0.04, speed=0.1, force=10.0)
    left_gripper_state = client.left_gripper_get_state()
    print(f"Left gripper state: {left_gripper_state}")
    
    # Test right gripper
    client.right_gripper_goto(width=0.04, speed=0.1, force=10.0)
    right_gripper_state = client.right_gripper_get_state()
    print(f"Right gripper state: {right_gripper_state}")
    
    # Get joint positions
    left_joints = client.left_robot_get_joint_positions()
    right_joints = client.right_robot_get_joint_positions()
    print(f"Left arm joints: {left_joints}")
    print(f"Right arm joints: {right_joints}")
    
    # Get EE poses
    left_pose = client.left_robot_get_ee_pose()
    right_pose = client.right_robot_get_ee_pose()
    print(f"Left arm EE pose: {left_pose}")
    print(f"Right arm EE pose: {right_pose}")