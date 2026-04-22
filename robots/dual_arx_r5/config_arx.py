"""
Configuration for ARX R5 dual-arm robot system.
Each arm has 6 DOF. Gripper is integrated (joint 7), controlled via RPC.
"""
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("arx_dual_arm")
@dataclass
class ArxDualArmConfig(RobotConfig):
    """Configuration for ARX R5 dual-arm robot.

    Communication: ZMQ+msgpack RPC to the ARX ROS2 bridge server.
    Control: ee_pose mode — 发送绝对末端位姿, 服务端控制器内置 IK.
    Gripper: Integrated, single float 0-1 via set_dual_ee_poses gripper param.
    """

    # Robot identification
    name: str = "arx_dual_arm"

    # Network (ZMQ+msgpack RPC server on robot-side Linux)
    robot_ip: str = "localhost"
    robot_port: int = 4242

    # Gripper (integrated, single 0-1 float via RPC)
    use_gripper: bool = True
    gripper_open_value: float = 0.0
    gripper_close_value: float = 1.0
    close_threshold: float = 0.5
    gripper_reverse: bool = False

    # Control
    control_mode: str = "oculus"
    debug: bool = True

    # Arm joints (6 per arm, gripper is separate)
    num_arm_joints: int = 6

    # 归零: connect 时平滑插值回初始零位
    go_home_on_connect: bool = True         # 连接时是否自动归零
    home_joint_positions: list = field(      # 零位关节角 (6 arm joints, rad)
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    home_steps: int = 50                    # 插值步数 (越多越平滑)
    home_step_interval: float = 0.02        # 每步间隔 (秒), 50 步 × 0.02s = 1s 总时长

    # Oculus delta ee_pose 低通滤波（EMA）
    enable_ee_action_filter: bool = True
    ee_action_filter_alpha_pos: float = 0.35
    ee_action_filter_alpha_rot: float = 0.25

    # Oculus delta ee_pose 死区（避免待机微抖持续下发）
    # 仅当 position/rotation 两个模长都低于阈值时才视为“静止”。
    enable_ee_action_deadband: bool = True
    ee_action_deadband_pos_norm: float = 0.00035  # m, 约 0.35 mm
    ee_action_deadband_rot_norm: float = 0.0025   # rad, 约 0.14 deg

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compatibility fields — accepted by create_robot_config() in run_record.py
    # but NOT used by ArxDualArm (ARX gripper has no separate server or Robotiq params)
    gripper_ip: str = "localhost"
    gripper_port: int = 4243
    gripper_max_open: float = 0.085
    gripper_force: float = 10.0
    gripper_speed: float = 0.1
    max_joint_velocity: float = 2.0
    max_ee_velocity: float = 0.5
