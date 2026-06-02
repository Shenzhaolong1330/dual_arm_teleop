from dataclasses import dataclass, field
from typing import List
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("oculus_teleop")
@dataclass
class OculusTeleopConfig(TeleoperatorConfig):
    """
    Configuration for dual-arm Oculus Quest teleoperation.
    Uses both Oculus controllers to control both arms simultaneously.
    Left controller -> Left arm, Right controller -> Right arm.
    """
    control_mode: str = "oculus"
    dual_arm: bool = True
    ip: str = "192.168.110.62"
    
    # Robot connection (for state feedback)
    robot_ip: str = "127.0.0.1"
    robot_port: int = 4242
    
    # Left controller (controls left arm)
    left_pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])
    left_channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    
    # Right controller (controls right arm)
    right_pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])
    right_channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])

    # Optional output-axis remapping after Oculus-to-robot conversion.
    # Values are output axes selecting from input axes: [1, 0, 2] swaps X/Y.
    position_axis_order: List[int] = field(default_factory=lambda: [0, 1, 2])
    rotation_axis_order: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    # Gripper control
    use_gripper: bool = True
    # Left gripper: Left Trigger (LTr)
    # Right gripper: Right Trigger (RTr)

    # Action smoothing for 6D delta pose per arm.
    # one_euro is adaptive and usually feels more responsive than EMA for delta actions.
    action_smoothing_method: str = "one_euro"
    action_smoothing_alpha: float = 0.35
    action_smoothing_freq: float = 30.0
    action_smoothing_min_cutoff: float = 1.2
    action_smoothing_beta: float = 0.4
    action_smoothing_d_cutoff: float = 1.0
    # Output-side filtering for noisy Oculus tracking. Deadband removes tiny
    # still-hand jitter; spike limits reject one-frame tracking jumps entirely.
    action_deadband_translation: float = 0.0
    action_deadband_rotation: float = 0.0
    action_spike_translation: float | None = None
    action_spike_rotation: float | None = None
    timing_debug: bool = False
    timing_debug_every_n: int = 30
    timing_warn_ms: float = 33.0

    # Mirror mode for operating while standing opposite the robot.
    # When enabled, controller-to-arm assignment is mirrored and the final
    # action remains expressed in the same robot-frame convention as normal.
    mirror_teleop: bool = False

    use_ik: bool = False
    servo_time: float = 0.017
    visualize_placo: bool = True
