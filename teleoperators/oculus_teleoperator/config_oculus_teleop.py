from dataclasses import dataclass, field
from typing import List
from lerobot.teleoperators.config import TeleoperatorConfig
from .processing_profiles import processing_profile_parameters, teleop_processing_metadata

_PROCESSING_DEFAULTS = processing_profile_parameters()

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
    processing_profile: str | None = None
    
    # Robot connection (for state feedback)
    robot_ip: str = "127.0.0.1"
    robot_port: int = 4242
    
    # Left controller (controls left arm)
    left_pose_scaler: List[float] = field(default_factory=lambda: list(_PROCESSING_DEFAULTS["left_pose_scaler"]))
    left_channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    
    # Right controller (controls right arm)
    right_pose_scaler: List[float] = field(default_factory=lambda: list(_PROCESSING_DEFAULTS["right_pose_scaler"]))
    right_channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])

    # Optional output-axis remapping after Oculus-to-robot conversion.
    # Values are output axes selecting from input axes: [1, 0, 2] swaps X/Y.
    position_axis_order: List[int] = field(default_factory=lambda: [0, 1, 2])
    rotation_axis_order: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    # Gripper control
    use_gripper: bool = True
    # The Quest reports a normalized trigger; teleop publishes the canonical
    # physical target aperture after applying these limits.
    gripper_min_width: float = 0.0
    gripper_max_open: float = 0.085
    # Left gripper: Left Trigger (LTr)
    # Right gripper: Right Trigger (RTr)
    # Trims noisy analog trigger endpoints before mapping trigger to gripper.
    gripper_trigger_deadzone: float = _PROCESSING_DEFAULTS["gripper_trigger_deadzone"]
    # 1.0 is linear; values below 1.0 close earlier, values above 1.0 close later.
    gripper_trigger_gamma: float = _PROCESSING_DEFAULTS["gripper_trigger_gamma"]

    # Action smoothing for 6D delta pose per arm.
    # one_euro is adaptive and usually feels more responsive than EMA for delta actions.
    action_smoothing_method: str = _PROCESSING_DEFAULTS["action_smoothing_method"]
    action_smoothing_alpha: float = _PROCESSING_DEFAULTS["action_smoothing_alpha"]
    action_smoothing_freq: float = _PROCESSING_DEFAULTS["action_smoothing_freq"]
    action_smoothing_min_cutoff: float = _PROCESSING_DEFAULTS["action_smoothing_min_cutoff"]
    action_smoothing_beta: float = _PROCESSING_DEFAULTS["action_smoothing_beta"]
    action_smoothing_d_cutoff: float = _PROCESSING_DEFAULTS["action_smoothing_d_cutoff"]
    action_missing_hold_frames: int = _PROCESSING_DEFAULTS["action_missing_hold_frames"]
    action_missing_decay: float = _PROCESSING_DEFAULTS["action_missing_decay"]
    # Output-side filtering for noisy Oculus tracking. Deadband removes tiny
    # still-hand jitter; spike limits reject one-frame tracking jumps entirely.
    action_deadband_translation: float = _PROCESSING_DEFAULTS["action_deadband_translation"]
    action_deadband_rotation: float = _PROCESSING_DEFAULTS["action_deadband_rotation"]
    action_spike_translation: float | None = _PROCESSING_DEFAULTS["action_spike_translation"]
    action_spike_rotation: float | None = _PROCESSING_DEFAULTS["action_spike_rotation"]
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

    def __post_init__(self):
        if self.processing_profile is not None:
            self.processing_metadata()  # Fail on conflicting named-profile values.

    def processing_metadata(self) -> dict:
        return teleop_processing_metadata(self)
