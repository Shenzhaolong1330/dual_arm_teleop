"""
Oculus Quest dual-arm robot controller.
Uses both left and right Oculus controllers to control a dual-arm robot system.

Default mode:
    Left controller  -> Left arm
    Right controller -> Right arm

Mirror mode:
    Left controller  -> Right arm, with mirrored pose deltas
    Right controller -> Left arm, with mirrored pose deltas
"""

from typing import Dict, Optional, Sequence
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from .oculus_reader.oculus_reader import OculusReader
from .robot import Robot

try:
    from OneEuroFilter import OneEuroFilter
except ImportError:
    OneEuroFilter = None


class OculusDualArmRobot(Robot):
    """
    A class representing dual Oculus Quest controllers for bimanual robot control.
    
    Controls:
    - LG (Left Grip): Must be pressed to enable left action recording
    - LTr (Left Trigger):  Controls left gripper  (0.0 = open, 1.0 = closed)
    - RG (Right Grip): Must be pressed to enable right action recording
    - RTr (Right Trigger): Controls right gripper (0.0 = open, 1.0 = closed)
    - Left controller pose:  Controls left arm end-effector delta pose
    - Right controller pose: Controls right arm end-effector delta pose
    - mirror_teleop: Swap controller-to-arm assignment and convert motion back
      to the canonical robot frame for opposite-side operation
    - A button: Request robot reset
    
    Coordinate Systems:
        Oculus: X(right), Y(up), Z(backward/towards user)
        Robot:  X(forward), Y(left), Z(up)
    
    Transformation matrix from Oculus to Robot:
        robot_x =  -oculus_z   (oculus backward -> robot forward)
        robot_y =  -oculus_x   (oculus right    -> robot left)
        robot_z =   oculus_y   (oculus up       -> robot up)
    """

    # Oculus -> Robot coordinate transform matrix (for position only)
    T_OCULUS_TO_ROBOT = np.array([
        [ 0.,  0., -1.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
    ])
    MIRROR_ACTION_SIGNS = np.array([-1., -1., 1., -1., -1., 1.])

    def __init__(
        self,
        ip: str = '192.168.110.62',
        use_gripper: bool = True,
        left_pose_scaler: Sequence[float] = [1.0, 1.0],
        left_channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
        right_pose_scaler: Sequence[float] = [1.0, 1.0],
        right_channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
        action_smoothing_alpha: float = 0.35,
        action_smoothing_method: str = "one_euro",
        action_smoothing_freq: float = 30.0,
        action_smoothing_min_cutoff: float = 1.2,
        action_smoothing_beta: float = 0.4,
        action_smoothing_d_cutoff: float = 1.0,
        mirror_teleop: bool = False,
    ):
        smoothing_method = str(action_smoothing_method).strip().lower()
        if smoothing_method not in {"one_euro", "ema", "none", "off", "raw"}:
            raise ValueError(
                "action_smoothing_method must be one of: one_euro, ema, none/off/raw"
            )
        if smoothing_method == "one_euro" and OneEuroFilter is None:
            raise ImportError(
                "action_smoothing_method='one_euro' requires the OneEuroFilter package. "
                "Install it with `pip install oneeurofilter` or reinstall this project."
            )

        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        self._mirror_teleop = bool(mirror_teleop)
        
        # Left arm configuration
        self._left_pose_scaler = left_pose_scaler
        self._left_channel_signs = left_channel_signs
        
        # Right arm configuration
        self._right_pose_scaler = right_pose_scaler
        self._right_channel_signs = right_channel_signs
        
        # State tracking - left arm
        self._left_prev_transform = None
        self._left_last_gripper_position = 1.0  # Default: open
        
        # State tracking - right arm
        self._right_prev_transform = None
        self._right_last_gripper_position = 1.0  # Default: open

        # Smoothing state (6D delta pose for each arm)
        self._action_smoothing_method = smoothing_method
        self._action_smoothing_alpha = float(action_smoothing_alpha)
        self._left_smoothed_delta = None
        self._right_smoothed_delta = None
        self._one_euro_freq = float(action_smoothing_freq)
        self._one_euro_min_cutoff = float(action_smoothing_min_cutoff)
        self._one_euro_beta = float(action_smoothing_beta)
        self._one_euro_d_cutoff = float(action_smoothing_d_cutoff)
        self._left_delta_filters = None
        self._right_delta_filters = None
        
        # Reset request
        self._reset_requested = False
        self._prev_a_pressed = False

    def _ema_smooth(self, current: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        """Apply EMA smoothing to a 6D delta vector."""
        alpha = max(0.0, min(1.0, self._action_smoothing_alpha))
        if prev is None or alpha >= 1.0:
            return current.copy()
        return alpha * current + (1.0 - alpha) * prev

    def _make_one_euro_filter(self):
        if OneEuroFilter is None:
            raise ImportError(
                "action_smoothing_method='one_euro' requires the OneEuroFilter package. "
                "Install it with `pip install oneeurofilter` or reinstall this project."
            )
        return OneEuroFilter(
            freq=self._one_euro_freq,
            mincutoff=self._one_euro_min_cutoff,
            beta=self._one_euro_beta,
            dcutoff=self._one_euro_d_cutoff,
        )

    def _make_one_euro_filter_bank(self):
        return [self._make_one_euro_filter() for _ in range(6)]

    def _one_euro_smooth(
        self,
        current: np.ndarray,
        filter_attr: str,
        timestamp: float,
    ) -> np.ndarray:
        filters = getattr(self, filter_attr)
        if filters is None:
            filters = self._make_one_euro_filter_bank()
            setattr(self, filter_attr, filters)

        filtered = np.empty(6, dtype=float)
        for idx, value in enumerate(current):
            filtered[idx] = float(filters[idx](float(value), timestamp))
        return filtered

    def _smooth_delta(self, current: np.ndarray, side: str, timestamp: float) -> np.ndarray:
        """Smooth a 6D delta vector for one arm."""
        if self._action_smoothing_method in {"none", "off", "raw"}:
            return current.copy()

        smoothed_attr = f"_{side}_smoothed_delta"
        if self._action_smoothing_method == "ema":
            smoothed = self._ema_smooth(current, getattr(self, smoothed_attr))
            setattr(self, smoothed_attr, smoothed.copy())
            return smoothed

        filter_attr = f"_{side}_delta_filters"
        return self._one_euro_smooth(current, filter_attr, timestamp)

    def _reset_smoothing(self, side: str) -> None:
        setattr(self, f"_{side}_smoothed_delta", None)
        setattr(self, f"_{side}_delta_filters", None)

    def _mirror_pose_delta(self, delta_pose: np.ndarray) -> np.ndarray:
        """Convert opposite-side operator motion back to the canonical robot frame."""
        return np.asarray(delta_pose, dtype=float) * self.MIRROR_ACTION_SIGNS

    def _get_trigger_value(self, buttons: Dict[str, object], analog_key: str, bool_key: str) -> float:
        value = buttons.get(analog_key, None)
        if value is None:
            return 1.0 if buttons.get(bool_key, False) else 0.0
        if isinstance(value, (tuple, list)) and len(value) > 0:
            return float(value[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return 1.0 if buttons.get(bool_key, False) else 0.0

    def num_dofs(self) -> int:
        # Each arm: 6 DOF pose + 1 gripper = 7, total = 14
        if self._use_gripper:
            return 14
        else:
            return 12

    def _compute_delta_pose(
        self, 
        current_transform: np.ndarray, 
        prev_transform: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute delta pose and map to robot coordinate system.
        Same coordinate transformation logic as single-arm OculusRobot.
        
        Returns: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz] in robot frame
        """
        if prev_transform is None:
            return np.zeros(6)
        
        # --- Position delta (in Oculus frame -> Robot frame via matrix) ---
        oculus_delta_pos = current_transform[:3, 3] - prev_transform[:3, 3]
        robot_delta_pos = self.T_OCULUS_TO_ROBOT @ oculus_delta_pos
        
        # --- Rotation delta (in Oculus frame) ---
        current_rot = current_transform[:3, :3]
        prev_rot = prev_transform[:3, :3]
        delta_rot_oculus = current_rot @ prev_rot.T
        oculus_delta_rotvec = R.from_matrix(delta_rot_oculus).as_rotvec()
        
        # --- Explicit axis mapping for rotation ---
        oculus_rx = oculus_delta_rotvec[0]
        oculus_ry = oculus_delta_rotvec[1]
        oculus_rz = oculus_delta_rotvec[2]
        
        robot_delta_rotvec = np.array([
            oculus_rz,   # robot roll
            oculus_rx,   # robot pitch
            oculus_ry,   # robot yaw
        ])
        
        return np.concatenate([robot_delta_pos, robot_delta_rotvec])

    def _apply_scaling(
        self,
        delta_pose: np.ndarray,
        pose_scaler: Sequence[float],
        channel_signs: Sequence[int],
    ) -> np.ndarray:
        """Apply scaling and channel signs to delta pose."""
        scaled = np.zeros(6)
        if len(pose_scaler) >= 2:
            position_scale = pose_scaler[0]
            orientation_scale = pose_scaler[1]
            
            scaled[0] = delta_pose[0] * position_scale * channel_signs[0]
            scaled[1] = delta_pose[1] * position_scale * channel_signs[1]
            scaled[2] = delta_pose[2] * position_scale * channel_signs[2]
            scaled[3] = delta_pose[3] * orientation_scale * channel_signs[3]
            scaled[4] = delta_pose[4] * orientation_scale * channel_signs[4]
            scaled[5] = delta_pose[5] * orientation_scale * channel_signs[5]
        else:
            scaled = delta_pose.copy()
        return scaled

    def get_action(self) -> np.ndarray:
        """
        Return actions for both arms.
        
        Output format (with gripper):
            [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_gripper,
             right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_gripper]
        
        Output format (without gripper):
            [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz,
             right_dx, right_dy, right_dz, right_drx, right_dry, right_drz]
        """
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()
        timestamp = time.monotonic()
        
        # Check grip buttons (both must be pressed for action)
        lg_pressed = buttons.get('LG', False)
        rg_pressed = buttons.get('RG', False)
        a_pressed = bool(buttons.get('A', False))
        
        self._reset_requested = a_pressed and not self._prev_a_pressed
        self._prev_a_pressed = a_pressed
        
        dof_per_arm = 7 if self._use_gripper else 6
        action = np.zeros(dof_per_arm * 2)
        left_delta_out = np.zeros(6)
        right_delta_out = np.zeros(6)
        
        # ========== Left arm (left controller) ==========
        if 'l' in transforms:
            left_transform = transforms['l']
            
            if lg_pressed:
                delta_left = self._compute_delta_pose(left_transform, self._left_prev_transform)
                scaled_left = self._apply_scaling(delta_left, self._left_pose_scaler, self._left_channel_signs)
                left_delta_out = self._smooth_delta(scaled_left, "left", timestamp)
                self._left_prev_transform = left_transform.copy()
            else:
                self._left_prev_transform = None
                self._reset_smoothing("left")
        else:
            self._left_prev_transform = None
            self._reset_smoothing("left")
        
        # ========== Right arm (right controller) ==========
        if 'r' in transforms:
            right_transform = transforms['r']
            
            if rg_pressed:
                delta_right = self._compute_delta_pose(right_transform, self._right_prev_transform)
                scaled_right = self._apply_scaling(delta_right, self._right_pose_scaler, self._right_channel_signs)
                right_delta_out = self._smooth_delta(scaled_right, "right", timestamp)
                self._right_prev_transform = right_transform.copy()
            else:
                self._right_prev_transform = None
                self._reset_smoothing("right")
        else:
            self._right_prev_transform = None
            self._reset_smoothing("right")
        
        # ========== Gripper control ==========
        if self._use_gripper:
            # Left gripper: Left Trigger
            lt_value = self._get_trigger_value(buttons, 'leftTrig', 'LTr')
            left_gripper = 1.0 - lt_value  # Invert: trigger pressed = closed (0.0)
            self._left_last_gripper_position = left_gripper
            
            # Right gripper: Right Trigger
            rt_value = self._get_trigger_value(buttons, 'rightTrig', 'RTr')
            right_gripper = 1.0 - rt_value  # Invert: trigger pressed = closed (0.0)
            self._right_last_gripper_position = right_gripper

        if self._mirror_teleop:
            left_delta_out, right_delta_out = (
                self._mirror_pose_delta(right_delta_out),
                self._mirror_pose_delta(left_delta_out),
            )
            if self._use_gripper:
                left_gripper, right_gripper = right_gripper, left_gripper

        action[0:6] = left_delta_out
        right_offset = dof_per_arm
        action[right_offset:right_offset + 6] = right_delta_out
        if self._use_gripper:
            action[6] = left_gripper
            action[13] = right_gripper
        
        return action

    def is_reset_requested(self) -> bool:
        """Check if reset was requested (A button pressed)."""
        return self._reset_requested

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Return the current robot observations for dual-arm system.
        
        Returns dict with keys:
            left_delta_ee_pose.{x,y,z,rx,ry,rz}
            right_delta_ee_pose.{x,y,z,rx,ry,rz}
            left_gripper_cmd_bin
            right_gripper_cmd_bin
            reset_requested
        """
        action_data = self.get_action()
        
        obs_dict = {}
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        
        dof_per_arm = 7 if self._use_gripper else 6
        
        # Left arm delta pose
        for i, axis in enumerate(axes):
            obs_dict[f"left_delta_ee_pose.{axis}"] = float(action_data[i])
        
        # Right arm delta pose
        right_offset = dof_per_arm
        for i, axis in enumerate(axes):
            obs_dict[f"right_delta_ee_pose.{axis}"] = float(action_data[right_offset + i])
        
        # Gripper positions
        if self._use_gripper:
            obs_dict["left_gripper_cmd_bin"] = float(action_data[6])
            obs_dict["right_gripper_cmd_bin"] = float(action_data[13])
        else:
            obs_dict["left_gripper_cmd_bin"] = None
            obs_dict["right_gripper_cmd_bin"] = None
        
        # Reset request flag
        obs_dict["reset_requested"] = self._reset_requested
        
        return obs_dict


if __name__ == "__main__":
    import time
    
    # Create dual-arm Oculus robot instance
    oculus = OculusDualArmRobot(
        ip='192.168.110.62',
        use_gripper=True,
        left_pose_scaler=[0.5, 0.5],
        left_channel_signs=[1, 1, 1, 1, 1, 1],
        right_pose_scaler=[0.5, 0.5],
        right_channel_signs=[1, 1, 1, 1, 1, 1],
        mirror_teleop=False,
    )
    
    print("===== Oculus Dual-Arm Robot Test =====")
    print("Controls:")
    print("  - LG (Left Grip):    Press to enable LEFT arm action")
    print("  - RG (Right Grip):   Press to enable RIGHT arm action")
    print("  - LTr (Left Trigger):  Control LEFT gripper")
    print("  - RTr (Right Trigger): Control RIGHT gripper")
    print("  - A button: Request robot reset")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            obs = oculus.get_observations()
            
            reset_flag = " [RESET]" if obs.get("reset_requested", False) else ""
            
            print(f"\rL: X={obs['left_delta_ee_pose.x']:+.4f} Y={obs['left_delta_ee_pose.y']:+.4f} "
                  f"Z={obs['left_delta_ee_pose.z']:+.4f} G={obs['left_gripper_cmd_bin']:.2f} | "
                  f"R: X={obs['right_delta_ee_pose.x']:+.4f} Y={obs['right_delta_ee_pose.y']:+.4f} "
                  f"Z={obs['right_delta_ee_pose.z']:+.4f} G={obs['right_gripper_cmd_bin']:.2f}"
                  f"{reset_flag}    ", end="")
            
            time.sleep(0.05)  # 20 Hz
            
    except KeyboardInterrupt:
        print("\n\n===== Test Ended =====")
