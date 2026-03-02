'''
Dobot Nova5 dual-arm robot interface server.
Runs locally and provides zerorpc server interface for ROS control.
Each arm is controlled via ROS, with Robotiq 2F-85 grippers via polymetis.
'''

import zerorpc
import rospy
import scipy.spatial.transform as st
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
from enum import Enum

# ROS message types for Dobot Nova5
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Twist
from std_msgs.msg import Header
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Polymetis for Robotiq gripper
from polymetis import GripperInterface

log = logging.getLogger(__name__)


class DobotArmServer:
    """
    Single Dobot Nova5 arm server interface using ROS.
    Provides joint control and end-effector pose control via ROS topics/services.
    """
    
    def __init__(self, arm_side: str = "left", gripper_enabled: bool = True):
        """
        Initialize Dobot arm server.
        
        Args:
            arm_side: "left" or "right" arm
            gripper_enabled: Whether to enable gripper control
        """
        self.arm_side = arm_side
        self.gripper_enabled = gripper_enabled
        self.num_joints = 6  # Dobot Nova5 has 6 DOF
        
        # ROS namespace for this arm
        self.ns = f"/dobot_{arm_side}_arm"
        
        # Joint names for Dobot Nova5
        self.joint_names = [
            f"{arm_side}_joint_1",
            f"{arm_side}_joint_2", 
            f"{arm_side}_joint_3",
            f"{arm_side}_joint_4",
            f"{arm_side}_joint_5",
            f"{arm_side}_joint_6",
        ]
        
        # Current state
        self._joint_positions = np.zeros(self.num_joints)
        self._joint_velocities = np.zeros(self.num_joints)
        self._ee_pose = np.zeros(6)  # [x, y, z, rx, ry, rz]
        self._gripper_state = {"width": 0.085, "is_grasped": False}
        
        # Control mode
        self._control_mode = "joint_impedance"
        self._desired_joint_positions = None
        self._desired_ee_pose = None
        
        # Initialize ROS node if not already initialized
        try:
            rospy.init_node(f"dobot_{arm_side}_server", anonymous=True)
        except rospy.exceptions.ROSInitException:
            pass  # Node already initialized
        
        self._connect_robot()
        
        if self.gripper_enabled:
            self._connect_gripper()
    
    def _connect_robot(self) -> bool:
        """Connect to Dobot robot via ROS."""
        try:
            # Subscribe to joint states
            self._joint_state_sub = rospy.Subscriber(
                f"{self.ns}/joint_states",
                JointState,
                self._joint_state_callback,
                queue_size=1
            )
            
            # Subscribe to end-effector pose
            self._ee_pose_sub = rospy.Subscriber(
                f"{self.ns}/ee_pose",
                PoseStamped,
                self._ee_pose_callback,
                queue_size=1
            )
            
            # Publisher for joint trajectory commands
            self._joint_traj_pub = rospy.Publisher(
                f"{self.ns}/joint_trajectory",
                JointTrajectory,
                queue_size=1
            )
            
            # Publisher for Cartesian velocity commands
            self._cartesian_vel_pub = rospy.Publisher(
                f"{self.ns}/cartesian_velocity",
                Twist,
                queue_size=1
            )
            
            log.info(f"[{self.arm_side.upper()} ARM] Connected to Dobot via ROS")
            return True
            
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] Failed to connect to Dobot: {e}")
            return False
    
    def _connect_gripper(self) -> bool:
        """Connect to Robotiq 2F-85 gripper via polymetis."""
        try:
            self.gripper = GripperInterface()
            log.info(f"[{self.arm_side.upper()} ARM] Connected to Robotiq 2F-85 gripper")
            return True
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] Failed to connect to gripper: {e}")
            return False
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        if len(msg.position) >= self.num_joints:
            self._joint_positions = np.array(msg.position[:self.num_joints])
            self._joint_velocities = np.array(msg.velocity[:self.num_joints]) if msg.velocity else np.zeros(self.num_joints)
    
    def _ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates."""
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, 
                         msg.pose.orientation.z, msg.pose.orientation.w])
        rot_vec = st.Rotation.from_quat(quat).as_rotvec()
        self._ee_pose = np.concatenate([pos, rot_vec])
    
    # ==================== Gripper Interface ====================
    
    def gripper_initialize(self):
        """Initialize the Robotiq gripper."""
        if not self.gripper_enabled:
            log.warning(f"[{self.arm_side.upper()} ARM] Gripper not enabled")
            return
        try:
            # Homing the gripper
            self.gripper.goto(width=0.085, speed=0.1, force=10.0, blocking=True)
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
        if not self.gripper_enabled:
            return
        try:
            self.gripper.goto(
                width=width,
                speed=speed,
                force=force,
                epsilon_inner=epsilon_inner,
                epsilon_outer=epsilon_outer,
                blocking=blocking,
            )
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
        if not self.gripper_enabled:
            return
        try:
            self.gripper.grasp(
                speed=speed,
                force=force,
                grasp_width=grasp_width,
                epsilon_inner=epsilon_inner,
                epsilon_outer=epsilon_outer,
                blocking=blocking,
            )
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] gripper_grasp failed: {e}")

    def gripper_get_state(self) -> dict:
        """Get current gripper state."""
        if not self.gripper_enabled:
            return {"width": 0.085, "is_moving": False, "is_grasped": False}
        try:
            state = self.gripper.get_state()
            return {
                "width": state.width,
                "is_moving": state.is_moving,
                "is_grasped": state.is_grasped,
                "prev_command_successful": state.prev_command_successful,
                "error_code": state.error_code,
            }
        except Exception as e:
            log.error(f"[{self.arm_side.upper()} ARM] gripper_get_state failed: {e}")
            return self._gripper_state

    # ==================== Robot Interface ====================
    
    def robot_get_joint_positions(self) -> list:
        """Get current joint positions."""
        return self._joint_positions.tolist()

    def robot_get_joint_velocities(self) -> list:
        """Get current joint velocities."""
        return self._joint_velocities.tolist()

    def robot_get_ee_pose(self) -> list:
        """Get current end-effector pose [x, y, z, rx, ry, rz]."""
        return self._ee_pose.tolist()

    def robot_move_to_joint_positions(
        self,
        positions: list,
        time_to_go: float = None,
        delta: bool = False,
        Kq: list = None,
        Kqd: list = None,
    ):
        """Move to target joint positions."""
        positions = np.array(positions)
        
        if delta:
            positions = self._joint_positions + positions
        
        if time_to_go is None:
            time_to_go = 2.0
        
        # Create JointTrajectory message
        traj = JointTrajectory()
        traj.header = Header()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.time_from_start = rospy.Duration(time_to_go)
        traj.points.append(point)
        
        self._joint_traj_pub.publish(traj)
        
        # Update desired positions
        self._desired_joint_positions = positions

    def robot_go_home(self):
        """Move robot to home position."""
        # Dobot Nova5 home position (adjust as needed)
        home_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.robot_move_to_joint_positions(home_positions, time_to_go=3.0)

    def robot_move_to_ee_pose(
        self,
        pose: list,
        time_to_go: float = None,
        delta: bool = False,
        Kx: list = None,
        Kxd: list = None,
        op_space_interp: bool = True,
    ):
        """Move to target end-effector pose."""
        pose = np.array(pose)
        
        if delta:
            pose = self._ee_pose + pose
        
        if time_to_go is None:
            time_to_go = 2.0
        
        # Convert pose to ROS message and publish
        # This assumes the Dobot ROS driver supports Cartesian commands
        # Implementation depends on specific ROS driver
        self._desired_ee_pose = pose
        
        log.info(f"[{self.arm_side.upper()} ARM] Moving to EE pose: {pose}")

    def robot_start_joint_impedance_control(self, Kq: list = None, Kqd: list = None, adaptive: bool = True):
        """Start joint impedance control mode."""
        self._control_mode = "joint_impedance"
        log.info(f"[{self.arm_side.upper()} ARM] Started joint impedance control")

    def robot_start_cartesian_impedance_control(self, Kx: list = None, Kxd: list = None):
        """Start Cartesian impedance control mode."""
        self._control_mode = "cartesian_impedance"
        log.info(f"[{self.arm_side.upper()} ARM] Started Cartesian impedance control")

    def robot_update_desired_joint_positions(self, positions: list):
        """Update desired joint positions for impedance control."""
        self._desired_joint_positions = np.array(positions)
        # Publish to appropriate ROS topic for impedance control
        # Implementation depends on specific Dobot ROS driver

    def robot_update_desired_ee_pose(self, pose: list):
        """Update desired end-effector pose for Cartesian control."""
        self._desired_ee_pose = np.array(pose)

    def robot_terminate_current_policy(self):
        """Terminate current control policy."""
        self._control_mode = "idle"
        log.info(f"[{self.arm_side.upper()} ARM] Terminated current policy")


class DobotDualArmServer:
    """
    Dual-arm Dobot Nova5 server interface.
    Manages both left and right arms with their grippers.
    """
    
    def __init__(self, left_port: int = 4242, right_port: int = 4243, gripper_enabled: bool = True):
        """
        Initialize dual-arm server.
        
        Args:
            left_port: Port for left arm zerorpc server
            right_port: Port for right arm zerorpc server
            gripper_enabled: Whether to enable grippers
        """
        self.left_port = left_port
        self.right_port = right_port
        
        log.info("=" * 60)
        log.info("Initializing Dobot Nova5 Dual-Arm Server")
        log.info("=" * 60)
        
        # Initialize both arms
        self.left_arm = DobotArmServer(arm_side="left", gripper_enabled=gripper_enabled)
        self.right_arm = DobotArmServer(arm_side="right", gripper_enabled=gripper_enabled)
        
        log.info("=" * 60)
        log.info("Dual-Arm Server Initialized Successfully")
        log.info("=" * 60)
    
    def start_servers(self):
        """Start zerorpc servers for both arms."""
        import threading
        
        def run_left_server():
            server = zerorpc.Server(self.left_arm)
            server.bind(f"tcp://0.0.0.0:{self.left_port}")
            log.info(f"[LEFT ARM] Server started on port {self.left_port}")
            server.run()
        
        def run_right_server():
            server = zerorpc.Server(self.right_arm)
            server.bind(f"tcp://0.0.0.0:{self.right_port}")
            log.info(f"[RIGHT ARM] Server started on port {self.right_port}")
            server.run()
        
        # Start servers in separate threads
        left_thread = threading.Thread(target=run_left_server, daemon=True)
        right_thread = threading.Thread(target=run_right_server, daemon=True)
        
        left_thread.start()
        right_thread.start()
        
        log.info("Both arm servers started")
        
        # Keep main thread alive
        rospy.spin()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dobot Nova5 Dual-Arm Server")
    parser.add_argument("--left-port", type=int, default=4242, help="Port for left arm")
    parser.add_argument("--right-port", type=int, default=4243, help="Port for right arm")
    parser.add_argument("--no-gripper", action="store_true", help="Disable grippers")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    server = DobotDualArmServer(
        left_port=args.left_port,
        right_port=args.right_port,
        gripper_enabled=not args.no_gripper
    )
    server.start_servers()