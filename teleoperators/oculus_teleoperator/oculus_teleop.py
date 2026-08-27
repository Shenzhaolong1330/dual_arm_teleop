#!/usr/bin/env python

"""
Oculus Quest dual-arm teleoperation implementation.
Uses both Oculus controllers to control a dual-arm robot system.
Left controller -> Left arm, Right controller -> Right arm.
Only outputs delta EE pose, no IK implementation.
"""

import logging
import time
from typing import Any, Dict, Optional

from lerobot.teleoperators.teleoperator import Teleoperator
from robots.dual_arm_schema import action_features as x_embodiment_action_features
from .config_oculus_teleop import OculusTeleopConfig
from .oculus.oculus_dual_arm_robot import OculusDualArmRobot

logger = logging.getLogger(__name__)


class OculusTeleop(Teleoperator):
    """
    Dual-arm teleoperation using both Oculus Quest controllers.
    
    This teleoperation mode uses both Oculus Quest controllers to simultaneously
    control two robot arms in Cartesian space. The output is delta pose 
    (position and orientation changes) for both arms.
    
    Controls:
    - LG (Left Grip):    Must be pressed to enable left arm action recording
    - RG (Right Grip):   Must be pressed to enable right arm action recording
    - LTr (Left Trigger):  Controls left gripper  (1.0 = open, 0.0 = closed)
    - RTr (Right Trigger): Controls right gripper (1.0 = open, 0.0 = closed)
    - Left controller pose:  Controls left arm end-effector delta pose
    - Right controller pose: Controls right arm end-effector delta pose
    - A button: Request robot reset
    """
    
    config_class = OculusTeleopConfig
    name = "OculusTeleop"
    
    def __init__(self, config: OculusTeleopConfig):
        super().__init__(config)
        self.cfg = config
        self._is_connected = False
        self.oculus_robot: Optional[OculusDualArmRobot] = None
        self._timing_debug_count = 0
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_connected
    
    @property
    def action_features(self) -> dict:
        """Return X-embodiment EE-delta and physical-width actions."""
        return x_embodiment_action_features(use_gripper=self.cfg.use_gripper)
    
    @property
    def feedback_features(self) -> dict:
        return {}
    
    def connect(self) -> None:
        """Connect to Oculus Quest."""
        if self._is_connected:
            logger.warning(f"{self.name} is already connected.")
            return
        
        logger.info(f"\n===== [TELEOP] Connecting to {self.name} =====")
        
        self.oculus_robot = OculusDualArmRobot(
            ip=self.cfg.ip,
            use_gripper=self.cfg.use_gripper,
            left_pose_scaler=self.cfg.left_pose_scaler,
            left_channel_signs=self.cfg.left_channel_signs,
            right_pose_scaler=self.cfg.right_pose_scaler,
            right_channel_signs=self.cfg.right_channel_signs,
            position_axis_order=self.cfg.position_axis_order,
            rotation_axis_order=self.cfg.rotation_axis_order,
            action_smoothing_method=self.cfg.action_smoothing_method,
            action_smoothing_alpha=self.cfg.action_smoothing_alpha,
            action_smoothing_freq=self.cfg.action_smoothing_freq,
            action_smoothing_min_cutoff=self.cfg.action_smoothing_min_cutoff,
            action_smoothing_beta=self.cfg.action_smoothing_beta,
            action_smoothing_d_cutoff=self.cfg.action_smoothing_d_cutoff,
            action_missing_hold_frames=self.cfg.action_missing_hold_frames,
            action_missing_decay=self.cfg.action_missing_decay,
            action_deadband_translation=self.cfg.action_deadband_translation,
            action_deadband_rotation=self.cfg.action_deadband_rotation,
            action_spike_translation=self.cfg.action_spike_translation,
            action_spike_rotation=self.cfg.action_spike_rotation,
            gripper_trigger_deadzone=self.cfg.gripper_trigger_deadzone,
            gripper_trigger_gamma=self.cfg.gripper_trigger_gamma,
            mirror_teleop=self.cfg.mirror_teleop,
        )
        
        self._is_connected = True
        logger.info(f"===== [TELEOP] {self.name} connected successfully =====")
        logger.info(f"[TELEOP] Oculus dual-arm connected at IP: {self.cfg.ip}")
        logger.info(f"[TELEOP] mirror_teleop={self.cfg.mirror_teleop}")
        logger.info(
            "[TELEOP] output_filter deadband_translation=%s deadband_rotation=%s "
            "spike_translation=%s spike_rotation=%s",
            self.cfg.action_deadband_translation,
            self.cfg.action_deadband_rotation,
            self.cfg.action_spike_translation,
            self.cfg.action_spike_rotation,
        )
    
    def disconnect(self) -> None:
        """Disconnect from Oculus Quest."""
        if not self._is_connected:
            return
        
        self.oculus_robot = None
        self._is_connected = False
        logger.info(f"[INFO] ===== {self.name} disconnected =====")
    
    def get_action(self) -> Dict[str, Any]:
        """Get the current action from the teleoperation device."""
        if not self._is_connected:
            raise RuntimeError(f"{self.name} is not connected.")
        return self._get_action_impl()
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """Get delta pose from both Oculus controllers."""
        if self.oculus_robot is None:
            raise RuntimeError("Oculus robot is not initialized.")
        
        # Get observations from OculusDualArmRobot
        action_start_t = time.perf_counter()
        obs = self.oculus_robot.get_observations()
        oculus_read_ms = (time.perf_counter() - action_start_t) * 1000.0
        
        # Build action dict with delta EE poses
        action = {}
        
        # Delta EE poses
        for arm in ["left", "right"]:
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                key = f"{arm}_delta_ee_pose.{axis}"
                if key in obs:
                    action[key] = float(obs[key])
                else:
                    action[key] = 0.0
        
        # Quest trigger values are normalized; the public action contract uses
        # physical target widths in metres, shared by Nero, Franka and Flexiv.
        if self.cfg.use_gripper:
            action["left_gripper_width"] = self._gripper_width_from_trigger(
                obs.get("left_gripper_cmd", obs.get("left_gripper_cmd_bin", 1.0))
            )
            action["right_gripper_width"] = self._gripper_width_from_trigger(
                obs.get("right_gripper_cmd", obs.get("right_gripper_cmd_bin", 1.0))
            )
        
        # Reset request flag (for external use)
        action["reset_requested"] = obs.get("reset_requested", False)
        for key in [
            "left_grip_pressed",
            "right_grip_pressed",
            "is_expert_override",
            "left_trigger_value",
            "right_trigger_value",
            "left_trigger_pressed",
            "right_trigger_pressed",
            "left_gripper_release_requested",
            "right_gripper_release_requested",
        ]:
            if key in obs:
                action[key] = obs[key]

        self._log_timing_debug(oculus_read_ms)
        
        return action

    def _gripper_width_from_trigger(self, value: Any) -> float:
        normalized = min(1.0, max(0.0, float(value)))
        minimum = max(0.0, float(self.cfg.gripper_min_width))
        maximum = max(minimum, float(self.cfg.gripper_max_open))
        return minimum + normalized * (maximum - minimum)

    def _log_timing_debug(self, total_ms: float) -> None:
        if not self.cfg.timing_debug:
            return

        self._timing_debug_count += 1
        every_n = max(1, int(self.cfg.timing_debug_every_n))
        warn_ms = max(0.0, float(self.cfg.timing_warn_ms))
        should_log = (
            self._timing_debug_count <= 5
            or self._timing_debug_count % every_n == 0
            or (warn_ms > 0.0 and total_ms >= warn_ms)
        )
        if not should_log:
            return

        log_fn = logger.warning if warn_ms > 0.0 and total_ms >= warn_ms else logger.info
        log_fn(
            "[OCULUS TIMING] step=%d get_action total_ms=%.1f",
            self._timing_debug_count,
            float(total_ms),
        )
    
    def calibrate(self) -> None:
        """Calibrate the teleoperation device. Default: no-op."""
        pass
    
    def configure(self) -> None:
        """Configure the teleoperation device. Default: no-op."""
        pass
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperation device. Default: no-op."""
        pass
    
    def is_reset_requested(self) -> bool:
        """Check if reset was requested (A button pressed)."""
        if self.oculus_robot is None:
            return False
        return self.oculus_robot.is_reset_requested()


if __name__ == "__main__":
    import time
    
    # Test the OculusTeleop class
    config = OculusTeleopConfig(
        ip="192.168.110.62",
        use_gripper=True,
        left_pose_scaler=[0.5, 0.5],
        left_channel_signs=[1, 1, 1, 1, 1, 1],
        right_pose_scaler=[0.5, 0.5],
        right_channel_signs=[1, 1, 1, 1, 1, 1],
    )
    
    teleop = OculusTeleop(config)
    teleop.connect()
    
    print("===== Oculus Dual-Arm Teleop Test =====")
    print("Controls:")
    print("  - LG (Left Grip):    Press to enable LEFT arm action")
    print("  - RG (Right Grip):   Press to enable RIGHT arm action")
    print("  - LTr (Left Trigger):  Control LEFT gripper")
    print("  - RTr (Right Trigger): Control RIGHT gripper")
    print("  - A button: Request robot reset")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            action = teleop.get_action()
            
            reset_flag = " [RESET]" if action.get("reset_requested", False) else ""
            
            print(f"\rL: X={action['left_delta_ee_pose.x']:+.4f} Y={action['left_delta_ee_pose.y']:+.4f} "
                  f"Z={action['left_delta_ee_pose.z']:+.4f} G={action['left_gripper_width']:.3f}m | "
                  f"R: X={action['right_delta_ee_pose.x']:+.4f} Y={action['right_delta_ee_pose.y']:+.4f} "
                  f"Z={action['right_delta_ee_pose.z']:+.4f} G={action['right_gripper_width']:.3f}m"
                  f"{reset_flag}    ", end="")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        teleop.disconnect()
