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
    
    # Gripper control
    use_gripper: bool = True
    # Left gripper: Left Trigger (LTr)
    # Right gripper: Right Trigger (RTr)

    # Action smoothing (EMA) for 6D delta pose per arm.
    # 1.0 means no smoothing, smaller values increase smoothing.
    action_smoothing_alpha: float = 0.35

    # Mirror mode for operating while standing opposite the robot.
    # When enabled, controller-to-arm assignment is mirrored and the final
    # action remains expressed in the same robot-frame convention as normal.
    mirror_teleop: bool = False

    use_ik: bool = False
    servo_time: float = 0.017
    visualize_placo: bool = True
