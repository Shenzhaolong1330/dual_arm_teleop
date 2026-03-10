'''
Dobot Nova5 dual-arm robot interface server.
Runs locally and provides zerorpc server interface for dual-arm control.
The underlying Python wrapper handles ROS communication internally.
'''

import zerorpc
import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)


class DobotDualArmServer:
    """
    Dual-arm Dobot Nova5 server interface.
    Provides unified control for both arms via a single zerorpc server.
    The underlying Python wrapper handles ROS communication internally.
    """
    
    def __init__(self, gripper_enabled: bool = True):
        """
        Initialize dual-arm server.
        
        Args:
            gripper_enabled: Whether to enable grippers
        """
        self.gripper_enabled = gripper_enabled
        self.num_joints_per_arm = 6  # Dobot Nova5 has 6 DOF per arm
        
        # Import the underlying Python wrapper for Dobot ROS control
        # This wrapper handles all ROS communication internally
        try:
            from dobot_ros_wrapper import DobotMoveitWrapper
            self._robot_wrapper = DobotMoveitWrapper()
            log.info("[DUAL-ARM] Dobot ROS wrapper initialized successfully")
        except ImportError:
            log.warning("[DUAL-ARM] dobot_ros_wrapper not found, using mock mode")
            self._robot_wrapper = None
            self._init_mock_state()
        
        log.info("=" * 60)
        log.info("Dobot Nova5 Dual-Arm Server Initialized")
        log.info("=" * 60)
    
    def _init_mock_state(self):
        """Initialize mock state for testing without hardware."""
        self._mock_left_joints = np.zeros(self.num_joints_per_arm)
        self._mock_right_joints = np.zeros(self.num_joints_per_arm)
        self._mock_left_ee_pose = np.zeros(6)  # [x, y, z, rx, ry, rz]
        self._mock_right_ee_pose = np.zeros(6)
        self._mock_left_gripper = {"width": 0.085, "is_grasped": False}
        self._mock_right_gripper = {"width": 0.085, "is_grasped": False}
    
    # ==================== Left Arm Interface ====================
    
    def left_robot_get_joint_positions(self) -> list:
        """Get left arm joint positions."""
        if self._robot_wrapper is None:
            return self._mock_left_joints.tolist()
        return self._robot_wrapper.left_get_joint_positions()
    
    def left_robot_get_joint_velocities(self) -> list:
        """Get left arm joint velocities."""
        if self._robot_wrapper is None:
            return np.zeros(self.num_joints_per_arm).tolist()
        return self._robot_wrapper.left_get_joint_velocities()
    
    def left_robot_get_ee_pose(self) -> list:
        """Get left arm end-effector pose [x, y, z, rx, ry, rz]."""
        if self._robot_wrapper is None:
            return self._mock_left_ee_pose.tolist()
        return self._robot_wrapper.left_get_ee_pose()
    
    def left_robot_move_to_joint_positions(
        self,
        positions: list,
        time_to_go: float = None,
        delta: bool = False,
        Kq: list = None,
        Kqd: list = None,
    ):
        """Move left arm to target joint positions."""
        positions = np.array(positions)
        if self._robot_wrapper is None:
            if delta:
                self._mock_left_joints = self._mock_left_joints + positions
            else:
                self._mock_left_joints = positions
            return
        self._robot_wrapper.left_move_to_joint_positions(
            positions.tolist(), time_to_go, delta, Kq, Kqd
        )
    
    def left_robot_move_to_ee_pose(
        self,
        pose: list,
        time_to_go: float = None,
        delta: bool = False,
        Kx: list = None,
        Kxd: list = None,
        op_space_interp: bool = True,
        wait: bool = False,
    ):
        """Move left arm to target end-effector pose."""
        pose = np.array(pose)
        if self._robot_wrapper is None:
            if delta:
                self._mock_left_ee_pose = self._mock_left_ee_pose + pose
            else:
                self._mock_left_ee_pose = pose
            return
        # 关键：传递wait参数，让wrapper决定是否异步执行
        self._robot_wrapper.left_move_to_ee_pose(
            pose.tolist(), time_to_go, delta, Kx, Kxd, op_space_interp, wait
        )
    
    def left_robot_go_home(self):
        """Move left arm to home position."""
        if self._robot_wrapper is None:
            self._mock_left_joints = np.zeros(self.num_joints_per_arm)
            return
        self._robot_wrapper.left_go_home()
    
    # ==================== Left Gripper Interface ====================
    
    def left_gripper_initialize(self):
        """Initialize left gripper."""
        if not self.gripper_enabled:
            return
        if self._robot_wrapper is None:
            self._mock_left_gripper = {"width": 0.085, "is_grasped": False}
            return
        self._robot_wrapper.left_gripper_initialize()
    
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
        if not self.gripper_enabled:
            return
        if self._robot_wrapper is None:
            self._mock_left_gripper["width"] = width
            return
        self._robot_wrapper.left_gripper_goto(
            width, speed, force, epsilon_inner, epsilon_outer, blocking
        )
    
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
        if not self.gripper_enabled:
            return
        if self._robot_wrapper is None:
            self._mock_left_gripper["is_grasped"] = True
            return
        self._robot_wrapper.left_gripper_grasp(
            speed, force, grasp_width, epsilon_inner, epsilon_outer, blocking
        )
    
    def left_gripper_get_state(self) -> dict:
        """Get left gripper state."""
        if not self.gripper_enabled:
            return {"width": 0.085, "is_moving": False, "is_grasped": False}
        if self._robot_wrapper is None:
            return self._mock_left_gripper
        return self._robot_wrapper.left_gripper_get_state()
    
    # ==================== Right Arm Interface ====================
    
    def right_robot_get_joint_positions(self) -> list:
        """Get right arm joint positions."""
        if self._robot_wrapper is None:
            return self._mock_right_joints.tolist()
        return self._robot_wrapper.right_get_joint_positions()
    
    def right_robot_get_joint_velocities(self) -> list:
        """Get right arm joint velocities."""
        if self._robot_wrapper is None:
            return np.zeros(self.num_joints_per_arm).tolist()
        return self._robot_wrapper.right_get_joint_velocities()
    
    def right_robot_get_ee_pose(self) -> list:
        """Get right arm end-effector pose [x, y, z, rx, ry, rz]."""
        if self._robot_wrapper is None:
            return self._mock_right_ee_pose.tolist()
        return self._robot_wrapper.right_get_ee_pose()
    
    def right_robot_move_to_joint_positions(
        self,
        positions: list,
        time_to_go: float = None,
        delta: bool = False,
        Kq: list = None,
        Kqd: list = None,
    ):
        """Move right arm to target joint positions."""
        positions = np.array(positions)
        if self._robot_wrapper is None:
            if delta:
                self._mock_right_joints = self._mock_right_joints + positions
            else:
                self._mock_right_joints = positions
            return
        self._robot_wrapper.right_move_to_joint_positions(
            positions.tolist(), time_to_go, delta, Kq, Kqd
        )
    
    def right_robot_move_to_ee_pose(
        self,
        pose: list,
        time_to_go: float = None,
        delta: bool = False,
        Kx: list = None,
        Kxd: list = None,
        op_space_interp: bool = True,
        wait: bool = False,
    ):
        """Move right arm to target end-effector pose."""
        pose = np.array(pose)
        if self._robot_wrapper is None:
            if delta:
                self._mock_right_ee_pose = self._mock_right_ee_pose + pose
            else:
                self._mock_right_ee_pose = pose
            return
        # 关键：传递wait参数，让wrapper决定是否异步执行
        self._robot_wrapper.right_move_to_ee_pose(
            pose.tolist(), time_to_go, delta, Kx, Kxd, op_space_interp, wait
        )
    
    def right_robot_go_home(self):
        """Move right arm to home position."""
        if self._robot_wrapper is None:
            self._mock_right_joints = np.zeros(self.num_joints_per_arm)
            return
        self._robot_wrapper.right_go_home()
    
    
    # ==================== Right Gripper Interface ====================
    
    def right_gripper_initialize(self):
        """Initialize right gripper."""
        if not self.gripper_enabled:
            return
        if self._robot_wrapper is None:
            self._mock_right_gripper = {"width": 0.085, "is_grasped": False}
            return
        self._robot_wrapper.right_gripper_initialize()
    
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
        if not self.gripper_enabled:
            return
        if self._robot_wrapper is None:
            self._mock_right_gripper["width"] = width
            return
        self._robot_wrapper.right_gripper_goto(
            width, speed, force, epsilon_inner, epsilon_outer, blocking
        )
    
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
        if not self.gripper_enabled:
            return
        if self._robot_wrapper is None:
            self._mock_right_gripper["is_grasped"] = True
            return
        self._robot_wrapper.right_gripper_grasp(
            speed, force, grasp_width, epsilon_inner, epsilon_outer, blocking
        )
    
    def right_gripper_get_state(self) -> dict:
        """Get right gripper state."""
        if not self.gripper_enabled:
            return {"width": 0.085, "is_moving": False, "is_grasped": False}
        if self._robot_wrapper is None:
            return self._mock_right_gripper
        return self._robot_wrapper.right_gripper_get_state()
    
    # ==================== Dual-Arm Convenience Methods ====================
    
    def robot_go_home(self):
        """Move both arms to home position."""
        if self._robot_wrapper is None:
            import numpy as np
            self._mock_left_joints = np.zeros(self.num_joints_per_arm)
            self._mock_right_joints = np.zeros(self.num_joints_per_arm)
            return
        self._robot_wrapper.dual_go_home(wait=True)
    
    def gripper_initialize(self):
        """Initialize both grippers."""
        self.left_gripper_initialize()
        self.right_gripper_initialize()
    
    def dual_robot_move_to_ee_pose(
        self,
        left_pose: list,
        right_pose: list,
        time_to_go: float = None,
        delta: bool = False,
        Kx: list = None,
        Kxd: list = None,
        op_space_interp: bool = True,
        wait: bool = False,
    ):
        """Move both arms to target end-effector poses simultaneously."""
        import numpy as np
        left_pose = np.array(left_pose)
        right_pose = np.array(right_pose)
        
        if self._robot_wrapper is None:
            if delta:
                self._mock_left_ee_pose = self._mock_left_ee_pose + left_pose
                self._mock_right_ee_pose = self._mock_right_ee_pose + right_pose
            else:
                self._mock_left_ee_pose = left_pose
                self._mock_right_ee_pose = right_pose
            return
            
        self._robot_wrapper.dual_move_to_ee_pose(
            left_pose.tolist(), 
            right_pose.tolist(), 
            time_to_go, delta, Kx, Kxd, op_space_interp, wait
        )
    
    def wait_for_left_arm(self):
        """Wait for left arm to complete current action."""
        if self._robot_wrapper is None:
            return True
        return self._robot_wrapper.wait_for_arm('left')
    
    def wait_for_right_arm(self):
        """Wait for right arm to complete current action."""
        if self._robot_wrapper is None:
            return True
        return self._robot_wrapper.wait_for_arm('right')
    
    def wait_for_all_arms(self):
        """Wait for both arms to complete current actions."""
        if self._robot_wrapper is None:
            return True
        return self._robot_wrapper.wait_for_all_arms()

def start_server(port: int = 4242, gripper_enabled: bool = True):
    server = zerorpc.Server(DobotDualArmServer())
    server.bind(f"tcp://0.0.0.0:{port}")
    log.info(f"[DUAL-ARM SERVER] Started on port {port}")
    server.run()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=4242)
    parser.add_argument('--no-gripper', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    start_server(port=args.port, gripper_enabled=not args.no_gripper)
