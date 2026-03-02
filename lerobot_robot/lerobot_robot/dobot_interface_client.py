'''
Dobot Nova5 dual-arm robot interface client.
Running on the user machine to connect to the dobot_interface_server via zerorpc.
'''

import logging
import numpy as np
import zerorpc
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


class DobotArmClient:
    """
    Client for a single Dobot Nova5 arm.
    Connects to the zerorpc server to control the robot.
    """
    
    def __init__(self, ip: str = '127.0.0.1', port: int = 4242, arm_side: str = "left"):
        """
        Initialize Dobot arm client.
        
        Args:
            ip: Server IP address
            port: Server port
            arm_side: "left" or "right" arm identifier
        """
        self.ip = ip
        self.port = port
        self.arm_side = arm_side
        
        try:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect(f"tcp://{ip}:{port}")
            log.info(f"[{arm_side.upper()} ARM] Connected to server at {ip}:{port}")
        except Exception as e:
            log.error(f"[{arm_side.upper()} ARM] Failed to connect to server: {e}")
            self.server = None

    # ==================== Gripper Interface ====================
    
    def gripper_initialize(self):
        """Initialize the gripper."""
        if self.server is None:
            log.error(f"[{self.arm_side.upper()} ARM] Not connected to server")
            return
        try:
            self.server.gripper_initialize()
            log.info(f"[{self.arm_side.upper()} ARM] Gripper initialized")
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] Failed to initialize gripper: {e}")

    def gripper_goto(
        self, 
        width: float, 
        speed: float, 
        force: float, 
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True
    ):
        """Move gripper to specified width."""
        if self.server is None:
            return
        try:
            self.server.gripper_goto(width, speed, force, epsilon_inner, epsilon_outer, blocking)
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] gripper_goto failed: {e}")

    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        """Grasp with gripper."""
        if self.server is None:
            return
        try:
            self.server.gripper_grasp(
                speed,
                force,
                grasp_width,
                epsilon_inner,
                epsilon_outer,
                blocking,
            )
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] gripper_grasp failed: {e}")

    def gripper_get_state(self) -> dict:
        """Get current gripper state."""
        if self.server is None:
            return {"width": 0.085, "is_moving": False, "is_grasped": False}
        try:
            return self.server.gripper_get_state()
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] gripper_get_state failed: {e}")
            return {"width": 0.085, "is_moving": False, "is_grasped": False}

    # ==================== Robot Interface ====================
    
    def robot_get_joint_positions(self) -> np.ndarray:
        """Get current joint positions as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            joint_positions = np.array(self.server.robot_get_joint_positions())
            return joint_positions
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_get_joint_positions failed: {e}")
            return np.zeros(6)

    def robot_get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            joint_velocities = np.array(self.server.robot_get_joint_velocities())
            return joint_velocities
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_get_joint_velocities failed: {e}")
            return np.zeros(6)
    
    def robot_get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose [x, y, z, rx, ry, rz] as numpy array."""
        if self.server is None:
            return np.zeros(6)
        try:
            pose = np.array(self.server.robot_get_ee_pose())
            return pose
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_get_ee_pose failed: {e}")
            return np.zeros(6)

    def robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
    ):
        """Move to target joint positions."""
        if self.server is None:
            return
        try:
            self.server.robot_move_to_joint_positions(
                positions.tolist(), 
                time_to_go, 
                delta, 
                Kq.tolist() if Kq is not None else None, 
                Kqd.tolist() if Kqd is not None else None
            )
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_move_to_joint_positions failed: {e}")

    def robot_go_home(self):
        """Move robot to home position."""
        if self.server is None:
            return
        try:
            self.server.robot_go_home()
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_go_home failed: {e}")

    def robot_move_to_ee_pose(
        self,
        pose: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kx: np.ndarray = None,
        Kxd: np.ndarray = None,
        op_space_interp: bool = True,
    ):
        """Move to target end-effector pose."""
        if self.server is None:
            return
        try:
            self.server.robot_move_to_ee_pose(
                pose.tolist(),
                time_to_go,
                delta,
                Kx.tolist() if Kx is not None else None,
                Kxd.tolist() if Kxd is not None else None,
                op_space_interp,
            )
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_move_to_ee_pose failed: {e}")

    def robot_start_joint_impedance_control(
        self, 
        Kq: np.ndarray = None, 
        Kqd: np.ndarray = None, 
        adaptive: bool = True,
    ):
        """Start joint impedance control mode."""
        if self.server is None:
            return
        try:
            self.server.robot_start_joint_impedance_control(
                Kq.tolist() if Kq is not None else None,
                Kqd.tolist() if Kqd is not None else None,
                adaptive,
            )
            log.info(f"[{self.arm_side.upper()} ARM] Joint impedance control started")
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_start_joint_impedance_control failed: {e}")

    def robot_start_cartesian_impedance_control(self, Kx: np.ndarray = None, Kxd: np.ndarray = None):
        """Start Cartesian impedance control mode."""
        if self.server is None:
            return
        try:
            self.server.robot_start_cartesian_impedance_control(
                Kx.tolist() if Kx is not None else None,
                Kxd.tolist() if Kxd is not None else None,
            )
            log.info(f"[{self.arm_side.upper()} ARM] Cartesian impedance control started")
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_start_cartesian_impedance_control failed: {e}")

    def robot_update_desired_joint_positions(self, positions: np.ndarray):
        """Update desired joint positions for impedance control."""
        if self.server is None:
            return
        try:
            self.server.robot_update_desired_joint_positions(positions.tolist())
        except Exception as e:
            log.warning(f"[{self.arm_side.upper()} ARM] robot_update_desired_joint_positions failed: {e}")

    def robot_update_desired_ee_pose(self, pose: np.ndarray):
        """Update desired end-effector pose for Cartesian control."""
        if self.server is None:
            return
        try:
            self.server.robot_update_desired_ee_pose(pose.tolist())
        except Exception as e:
            log.warning(f"[{self.arm_side.upper()} ARM] robot_update_desired_ee_pose failed: {e}")

    def robot_terminate_current_policy(self):
        """Terminate current control policy."""
        if self.server is None:
            return
        try:
            self.server.robot_terminate_current_policy()
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] robot_terminate_current_policy failed: {e}")

    def close(self):
        """Close connection to server."""
        if self.server is not None:
            try:
                self.server.close()
            except:
                pass


class DobotDualArmClient:
    """
    Client for dual-arm Dobot Nova5 robot.
    Manages connections to both left and right arm servers.
    """
    
    def __init__(
        self, 
        ip: str = '127.0.0.1', 
        left_port: int = 4242, 
        right_port: int = 4243
    ):
        """
        Initialize dual-arm client.
        
        Args:
            ip: Server IP address (both arms on same machine)
            left_port: Port for left arm server
            right_port: Port for right arm server
        """
        self.ip = ip
        self.left_port = left_port
        self.right_port = right_port
        
        log.info("=" * 60)
        log.info("Initializing Dobot Dual-Arm Client")
        log.info("=" * 60)
        
        # Initialize both arm clients
        self.left_arm = DobotArmClient(ip=ip, port=left_port, arm_side="left")
        self.right_arm = DobotArmClient(ip=ip, port=right_port, arm_side="right")
        
        log.info("=" * 60)
        log.info("Dual-Arm Client Initialized")
        log.info("=" * 60)

    def gripper_initialize(self):
        """Initialize both grippers."""
        self.left_arm.gripper_initialize()
        self.right_arm.gripper_initialize()

    def robot_go_home(self):
        """Move both arms to home position."""
        self.left_arm.robot_go_home()
        self.right_arm.robot_go_home()

    def robot_start_joint_impedance_control(
        self, 
        Kq: np.ndarray = None, 
        Kqd: np.ndarray = None, 
        adaptive: bool = True,
    ):
        """Start joint impedance control for both arms."""
        self.left_arm.robot_start_joint_impedance_control(Kq, Kqd, adaptive)
        self.right_arm.robot_start_joint_impedance_control(Kq, Kqd, adaptive)

    def robot_start_cartesian_impedance_control(self, Kx: np.ndarray = None, Kxd: np.ndarray = None):
        """Start Cartesian impedance control for both arms."""
        self.left_arm.robot_start_cartesian_impedance_control(Kx, Kxd)
        self.right_arm.robot_start_cartesian_impedance_control(Kx, Kxd)

    def close(self):
        """Close connections to both arm servers."""
        self.left_arm.close()
        self.right_arm.close()


# Legacy compatibility: DobotDualInterfaceClient maps to DobotDualArmClient
DobotDualInterfaceClient = DobotDualArmClient


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Test dual-arm client
    client = DobotDualArmClient(ip="127.0.0.1", left_port=4242, right_port=4243)
    
    # Initialize grippers
    client.gripper_initialize()
    
    # Test gripper
    client.left_arm.gripper_goto(width=0.04, speed=0.1, force=10.0)
    gripper_state = client.left_arm.gripper_get_state()
    print(f"Left gripper state: {gripper_state}")
    
    # Get joint positions
    left_joints = client.left_arm.robot_get_joint_positions()
    right_joints = client.right_arm.robot_get_joint_positions()
    print(f"Left arm joints: {left_joints}")
    print(f"Right arm joints: {right_joints}")
    
    # Get EE poses
    left_pose = client.left_arm.robot_get_ee_pose()
    right_pose = client.right_arm.robot_get_ee_pose()
    print(f"Left arm EE pose: {left_pose}")
    print(f"Right arm EE pose: {right_pose}")