"""Configuration for a dual Flexiv Rizon4s setup controlled through Flexiv RDK."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("flexiv_dual_arm")
@dataclass
class FlexivDualArmConfig(RobotConfig):
    """Dual Flexiv Rizon4s robot configuration.

    Flexiv RDK connects by robot serial number, for example
    ``Rizon4s-123456``. Fill ``left_robot_sn`` and ``right_robot_sn`` in
    ``scripts/config/robots/flexiv_config.yaml`` before connecting hardware.
    """

    name: str = "flexiv_dual_arm"

    left_robot_sn: str = ""
    right_robot_sn: str = ""

    control_mode: str = "oculus"
    debug: bool = True

    use_gripper: bool = False
    left_gripper_name: str = ""
    right_gripper_name: str = ""
    left_tool_name: str = ""
    right_tool_name: str = ""
    switch_tool_on_connect: bool = True
    initialize_gripper_on_connect: bool = False
    gripper_min_width: float = 0.0
    gripper_reverse: bool = False
    close_threshold: float = 0.5
    gripper_max_open: float = 0.085
    gripper_force: float = 10.0
    gripper_speed: float = 0.2
    gripper_command_epsilon: float = 0.0005
    gripper_init_timeout_sec: float = 30.0
    gripper_init_settle_sec: float = 0.5
    stop_grippers_on_disconnect: bool = False
    action_debug: bool = True
    action_debug_every_n: int = 30
    timing_debug: bool = False
    timing_debug_every_n: int = 30
    timing_warn_ms: float = 33.0
    send_arms_parallel: bool = False
    use_cartesian_servo_thread: bool = False
    cartesian_servo_hz: float = 100.0
    cartesian_servo_alpha: float = 0.35

    enable_on_connect: bool = True
    clear_fault_on_connect: bool = True
    switch_cartesian_mode_on_connect: bool = True
    go_home_on_connect: bool = False
    reset_go_home: bool = False
    home_plan_name: str = "PLAN-Home"
    left_home_joints: list[float] = field(default_factory=list)
    right_home_joints: list[float] = field(default_factory=list)
    home_joint_max_vel: float = 1.0
    home_joint_max_acc: float = 2.0
    home_joint_tolerance: float = 0.01
    home_joint_timeout_sec: float = 20.0
    zero_ft_sensor_on_connect: bool = False

    max_cartesian_delta: float | None = 0.03
    max_rotation_delta: float | None = 0.06
    cartesian_max_linear_vel: float = 0.8
    cartesian_max_angular_vel: float = 1.2
    cartesian_max_linear_acc: float = 3.0
    cartesian_max_angular_acc: float = 6.0
    left_mount_raw_deg: float = 0.0
    right_mount_raw_deg: float = 0.0
    left_mount_roll_deg: float = 0.0
    right_mount_roll_deg: float = 0.0
    left_mount_pitch_deg: float = 0.0
    right_mount_pitch_deg: float = 0.0
    left_mount_yaw_deg: float = 0.0
    right_mount_yaw_deg: float = 0.0

    num_joints_per_arm: int = 7
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    camera_read_timeout_ms: int = 1000
    camera_warmup_attempts: int = 10
    camera_hardware_reset_on_connect: bool = False
    camera_hardware_reset_on_release: bool = False
    camera_reset_settle_sec: float = 6.0
    camera_reset_timeout_sec: float = 20.0
    save_depth_sidecar: bool = False
    save_ir_sidecar: bool = False
    save_rgbd_timestamps: bool = False

    # Compatibility fields accepted from the generic record config parser.
    robot_ip: str = "localhost"
    robot_port: int = 4242
    rpc_timeout_sec: float = 30.0
    open_grippers_on_connect: bool = False
    reset_opens_grippers: bool = False
    go_home_duration_sec: float | None = None
    go_home_rate_hz: float | None = None
