"""
Example robot interface client template.

This client runs on the user machine and connects to the robot interface server
via zerorpc (RPC framework using ZeroMQ).

Implementation Steps:
1. Create zerorpc.Client instance
2. Connect to server using tcp://{ip}:{port}
3. Implement methods that call server methods via self.server.method_name()
4. Handle exceptions for network errors

Note: zerorpc uses msgpack for serialization, so numpy arrays must be converted
to lists before sending, and converted back to numpy arrays after receiving.
"""
import logging
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


class ExampleRobotInterfaceClient:
    """
    Example robot interface client using zerorpc.
    
    TODO: Replace with your robot's actual implementation.
    """
    
    def __init__(self, ip: str = "192.168.1.100", port: int = 4242):
        """
        Initialize and connect to the robot interface server.
        
        Implementation:
        1. Create zerorpc.Client with heartbeat for connection monitoring
        2. Connect to server using tcp://{ip}:{port}
        3. Handle connection errors gracefully
        """
        # TODO: Implement connection
        # try:
        #     self.server = zerorpc.Client(heartbeat=20)
        #     self.server.connect(f"tcp://{ip}:{port}")
        #     log.info("Connected to server")
        # except Exception as e:
        #     log.error(f"Failed to connect to server: {e}")
        pass
    
    # ==================== Gripper Methods ====================
    
    def gripper_initialize(self) -> None:
        """
        Initialize the gripper.
        
        Implementation:
        - Call server.gripper_initialize()
        - Handle connection errors
        """
        # TODO: Implement
        # try:
        #     self.server.gripper_initialize()
        #     log.info("Gripper initialized")
        # except Exception as e:
        #     log.error(f"Failed to initialize gripper: {e}")
        pass
    
    def gripper_goto(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True
    ) -> None:
        """
        Move gripper to specified width.
        
        Args:
            width: Target gripper width (meters)
            speed: Gripper speed (m/s)
            force: Gripper force (N)
            epsilon_inner: Inner tolerance
            epsilon_outer: Outer tolerance
            blocking: Whether to block until complete
        """
        # TODO: Implement
        # self.server.gripper_goto(width, speed, force, epsilon_inner, epsilon_outer, blocking)
        pass
    
    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ) -> None:
        """
        Grasp with specified parameters.
        
        Args:
            speed: Gripper speed (m/s)
            force: Gripper force (N)
            grasp_width: Width to grasp at
            epsilon_inner: Inner tolerance
            epsilon_outer: Outer tolerance
            blocking: Whether to block until complete
        """
        # TODO: Implement
        # self.server.gripper_grasp(speed, force, grasp_width, epsilon_inner, epsilon_outer, blocking)
        pass
    
    def gripper_get_state(self) -> Dict:
        """
        Get current gripper state.
        
        Returns:
            Dict with keys: width, is_grasping, temperature, etc.
        """
        # TODO: Implement
        # return self.server.gripper_get_state()
        return {"width": 0.0, "is_grasping": False}
    
    # ==================== Robot State Methods ====================
    
    def robot_get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.
        
        Returns:
            np.ndarray of joint positions (radians)
        
        Implementation:
        - Call server.robot_get_joint_positions() which returns a list
        - Convert list to numpy array
        """
        # TODO: Implement
        # joint_positions = np.array(self.server.robot_get_joint_positions())
        # return joint_positions
        return np.zeros(7)  # Placeholder
    
    def robot_get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities.
        
        Returns:
            np.ndarray of joint velocities (rad/s)
        """
        # TODO: Implement
        # joint_velocities = np.array(self.server.robot_get_joint_velocities())
        # return joint_velocities
        return np.zeros(7)  # Placeholder
    
    def robot_get_ee_pose(self) -> np.ndarray:
        """
        Get current end-effector pose.
        
        Returns:
            np.ndarray of [x, y, z, rx, ry, rz] (position + rotation vector)
        """
        # TODO: Implement
        # pose = np.array(self.server.robot_get_ee_pose())
        # return pose
        return np.zeros(6)  # Placeholder
    
    # ==================== Robot Control Methods ====================
    
    def robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
    ) -> None:
        """
        Move robot to target joint positions.
        
        Args:
            positions: Target joint positions (radians)
            time_to_go: Time to reach target (seconds)
            delta: Whether positions are relative to current
            Kq: Position gain matrix
            Kqd: Velocity gain matrix
        
        Implementation:
        - Convert numpy arrays to lists before sending
        - Call server.robot_move_to_joint_positions()
        """
        # TODO: Implement
        # self.server.robot_move_to_joint_positions(
        #     positions.tolist(),
        #     time_to_go,
        #     delta,
        #     Kq.tolist() if Kq is not None else None,
        #     Kqd.tolist() if Kqd is not None else None
        # )
        pass
    
    def robot_go_home(self) -> None:
        """
        Move robot to home position.
        """
        # TODO: Implement
        # self.server.robot_go_home()
        pass
    
    def robot_move_to_ee_pose(
        self,
        pose: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kx: np.ndarray = None,
        Kxd: np.ndarray = None,
        op_space_interp: bool = True,
    ) -> None:
        """
        Move robot end-effector to target pose.
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz]
            time_to_go: Time to reach target (seconds)
            delta: Whether pose is relative to current
            Kx: Position gain matrix
            Kxd: Velocity gain matrix
            op_space_interp: Whether to use operational space interpolation
        """
        # TODO: Implement
        # self.server.robot_move_to_ee_pose(
        #     pose.tolist(),
        #     time_to_go,
        #     delta,
        #     Kx.tolist() if Kx is not None else None,
        #     Kxd.tolist() if Kxd is not None else None,
        #     op_space_interp,
        # )
        pass
    
    def robot_start_joint_impedance_control(
        self,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
        adaptive: bool = True,
    ) -> None:
        """
        Start joint impedance control mode.
        
        Args:
            Kq: Position gain matrix
            Kqd: Velocity gain matrix
            adaptive: Whether to use adaptive gains
        """
        # TODO: Implement
        # self.server.robot_start_joint_impedance_control(
        #     Kq.tolist() if Kq is not None else None,
        #     Kqd.tolist() if Kqd is not None else None,
        #     adaptive,
        # )
        pass
    
    def robot_start_cartesian_impedance_control(
        self,
        Kx: np.ndarray,
        Kxd: np.ndarray
    ) -> None:
        """
        Start cartesian impedance control mode.
        
        Args:
            Kx: Position gain matrix
            Kxd: Velocity gain matrix
        """
        # TODO: Implement
        # self.server.robot_start_cartesian_impedance_control(
        #     Kx.tolist() if Kx is not None else None,
        #     Kxd.tolist() if Kxd is not None else None,
        # )
        pass
    
    def robot_update_desired_joint_positions(self, positions: np.ndarray) -> None:
        """
        Update desired joint positions in impedance control mode.
        
        Args:
            positions: Desired joint positions (radians)
        """
        # TODO: Implement
        # try:
        #     self.server.robot_update_desired_joint_positions(positions.tolist())
        # except Exception as e:
        #     log.warning(f"robot_update_desired_joint_positions failed: {e}")
        pass
    
    def robot_update_desired_ee_pose(self, pose: np.ndarray) -> None:
        """
        Update desired end-effector pose in impedance control mode.
        
        Args:
            pose: Desired pose [x, y, z, rx, ry, rz]
        """
        # TODO: Implement
        # try:
        #     self.server.robot_update_desired_ee_pose(pose.tolist())
        # except Exception as e:
        #     log.warning(f"robot_update_desired_ee_pose failed: {e}")
        pass
    
    def robot_terminate_current_policy(self) -> None:
        """
        Terminate the current control policy.
        """
        # TODO: Implement
        # self.server.robot_terminate_current_policy()
        pass
    
    def close(self) -> None:
        """
        Close the connection to the server.
        """
        # TODO: Implement
        # self.server.close()
        pass


# ==================== Main (for testing) ====================

if __name__ == "__main__":
    """Test the interface client."""
    logging.basicConfig(level=logging.INFO)
    
    client = ExampleRobotInterfaceClient(ip="192.168.1.100", port=4242)
    
    # Test gripper
    client.gripper_initialize()
    client.gripper_goto(width=0.06, speed=0.1, force=10.0)
    gripper_state = client.gripper_get_state()
    print(f"Gripper state: {gripper_state}")
    
    # Test robot
    client.robot_go_home()
    joint_positions = client.robot_get_joint_positions()
    print(f"Joint positions: {joint_positions}")
    
    client.close()