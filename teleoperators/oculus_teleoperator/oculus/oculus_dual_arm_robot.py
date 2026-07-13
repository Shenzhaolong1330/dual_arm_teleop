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

import logging
from typing import Dict, Optional, Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R

from .oculus_reader.oculus_reader import OculusReader
from .robot import Robot

logger = logging.getLogger(__name__)


class _LowPassFilter:
    def __init__(self) -> None:
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def apply(self, value: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        current = np.asarray(value, dtype=float)
        if self._prev is None:
            self._prev = current.copy()
            return current.copy()
        alpha_arr = np.asarray(alpha, dtype=float)
        filtered = alpha_arr * current + (1.0 - alpha_arr) * self._prev
        self._prev = filtered.copy()
        return filtered


class _OneEuroVectorFilter:
    def __init__(
        self,
        *,
        freq: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
        size: int = 6,
    ) -> None:
        self.freq = max(1e-6, float(freq))
        self.min_cutoff = max(1e-6, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.d_cutoff = max(1e-6, float(d_cutoff))
        self._value_filter = _LowPassFilter()
        self._derivative_filter = _LowPassFilter()
        self._prev_raw: np.ndarray | None = None
        self._size = int(size)

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, freq: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=float))
        te = 1.0 / max(1e-6, float(freq))
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        self._value_filter.reset()
        self._derivative_filter.reset()
        self._prev_raw = None

    def apply(self, value: np.ndarray) -> np.ndarray:
        current = np.asarray(value, dtype=float).reshape(self._size)
        if self._prev_raw is None:
            derivative = np.zeros(self._size, dtype=float)
        else:
            derivative = (current - self._prev_raw) * self.freq
        self._prev_raw = current.copy()

        filtered_derivative = self._derivative_filter.apply(
            derivative,
            self._alpha(self.d_cutoff, self.freq),
        )
        cutoff = self.min_cutoff + self.beta * np.abs(filtered_derivative)
        return self._value_filter.apply(current, self._alpha(cutoff, self.freq))


class OculusDualArmRobot(Robot):
    """
    A class representing dual Oculus Quest controllers for bimanual robot control.
    
    Controls:
    - LG (Left Grip): Must be pressed to enable left action recording
    - LTr (Left Trigger):  Controls left gripper  (1.0 = open, 0.0 = closed)
    - RG (Right Grip): Must be pressed to enable right action recording
    - RTr (Right Trigger): Controls right gripper (1.0 = open, 0.0 = closed)
    - Left controller pose:  Controls left arm end-effector delta pose
    - Right controller pose: Controls right arm end-effector delta pose
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
        position_axis_order: Sequence[int] = [0, 1, 2],
        rotation_axis_order: Sequence[int] = [0, 1, 2],
        action_smoothing_method: str = "ema",
        action_smoothing_alpha: float = 0.35,
        action_smoothing_freq: float = 30.0,
        action_smoothing_min_cutoff: float = 1.2,
        action_smoothing_beta: float = 0.4,
        action_smoothing_d_cutoff: float = 1.0,
        action_missing_hold_frames: int = 0,
        action_missing_decay: float = 0.5,
        action_deadband_translation: float = 0.0,
        action_deadband_rotation: float = 0.0,
        action_spike_translation: Optional[float] = None,
        action_spike_rotation: Optional[float] = None,
        gripper_trigger_deadzone: float = 0.02,
        gripper_trigger_gamma: float = 1.0,
        mirror_teleop: bool = False,
    ):
        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        self._mirror_teleop = bool(mirror_teleop)
        self._gripper_trigger_deadzone = float(np.clip(gripper_trigger_deadzone, 0.0, 0.45))
        self._gripper_trigger_gamma = self._positive_float(gripper_trigger_gamma, 1.0)
        
        # Left arm configuration
        self._left_pose_scaler = left_pose_scaler
        self._left_channel_signs = left_channel_signs
        
        # Right arm configuration
        self._right_pose_scaler = right_pose_scaler
        self._right_channel_signs = right_channel_signs
        self._position_axis_order = self._validate_axis_order(position_axis_order, "position_axis_order")
        self._rotation_axis_order = self._validate_axis_order(rotation_axis_order, "rotation_axis_order")
        
        # State tracking - left arm
        self._left_prev_transform = None
        self._left_last_gripper_position = 1.0  # Default: open
        
        # State tracking - right arm
        self._right_prev_transform = None
        self._right_last_gripper_position = 1.0  # Default: open

        # Output smoothing state (6D delta pose for each arm)
        self._action_smoothing_method = str(action_smoothing_method).strip().lower()
        if self._action_smoothing_method not in {"none", "ema", "one_euro"}:
            raise ValueError(
                "action_smoothing_method must be one of: none, ema, one_euro. "
                f"Got {action_smoothing_method!r}."
            )
        self._action_smoothing_alpha = float(action_smoothing_alpha)
        self._action_smoothing_freq = self._positive_float(action_smoothing_freq, 30.0)
        self._action_smoothing_min_cutoff = self._positive_float(action_smoothing_min_cutoff, 1.2)
        self._action_smoothing_beta = max(0.0, float(action_smoothing_beta))
        self._action_smoothing_d_cutoff = self._positive_float(action_smoothing_d_cutoff, 1.0)
        self._action_missing_hold_frames = max(0, int(action_missing_hold_frames))
        self._action_missing_decay = float(np.clip(action_missing_decay, 0.0, 1.0))
        self._action_deadband_translation = self._nonnegative_float(action_deadband_translation)
        self._action_deadband_rotation = self._nonnegative_float(action_deadband_rotation)
        self._action_spike_translation = self._optional_positive_float(action_spike_translation)
        self._action_spike_rotation = self._optional_positive_float(action_spike_rotation)
        self._left_smoothed_delta = None
        self._right_smoothed_delta = None
        self._left_one_euro = self._make_one_euro_filter()
        self._right_one_euro = self._make_one_euro_filter()
        self._left_missing_hold_count = 0
        self._right_missing_hold_count = 0
        self._delta_filter_warn_counts = {"left": 0, "right": 0}
        
        # Reset request
        self._reset_requested = False
        self._x_button_pressed = False
        self._y_button_pressed = False
        self._left_grip_pressed = False
        self._right_grip_pressed = False
        self._left_trigger_value = 0.0
        self._right_trigger_value = 0.0
        self._left_trigger_pressed = False
        self._right_trigger_pressed = False
        self._left_gripper_release_requested = False
        self._right_gripper_release_requested = False

    @staticmethod
    def _nonnegative_float(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return max(0.0, float(value))

    @staticmethod
    def _positive_float(value: Optional[float], default: float) -> float:
        if value is None:
            return float(default)
        value_float = float(value)
        return value_float if value_float > 0.0 else float(default)

    @staticmethod
    def _optional_positive_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value_float = float(value)
        return value_float if value_float > 0.0 else None

    @staticmethod
    def _validate_axis_order(axis_order: Sequence[int], name: str) -> tuple[int, int, int]:
        order = tuple(int(axis) for axis in axis_order)
        if len(order) != 3 or set(order) != {0, 1, 2}:
            raise ValueError(f"{name} must be a permutation of [0, 1, 2], got {list(axis_order)}")
        return order

    def _shape_trigger_value(self, value: float) -> float:
        """Map raw analog trigger travel into a stable [0, 1] control value."""
        raw = float(np.clip(value, 0.0, 1.0))
        deadzone = self._gripper_trigger_deadzone
        if raw <= deadzone:
            return 0.0

        upper = 1.0 - deadzone
        if raw >= upper:
            return 1.0

        span = max(1e-6, upper - deadzone)
        normalized = (raw - deadzone) / span
        shaped = normalized ** self._gripper_trigger_gamma
        return float(np.clip(shaped, 0.0, 1.0))

    def _remap_delta_axes(self, delta_pose: np.ndarray) -> np.ndarray:
        """Apply configurable XYZ/RPY axis ordering before scaling and signs."""
        remapped = np.asarray(delta_pose, dtype=float).copy()
        remapped[:3] = remapped[:3][list(self._position_axis_order)]
        remapped[3:] = remapped[3:][list(self._rotation_axis_order)]
        return remapped

    def _ema_smooth(self, current: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        """Apply EMA smoothing to a 6D delta vector."""
        alpha = max(0.0, min(1.0, self._action_smoothing_alpha))
        if prev is None or alpha >= 1.0:
            return current.copy()
        return alpha * current + (1.0 - alpha) * prev

    def _make_one_euro_filter(self) -> _OneEuroVectorFilter:
        return _OneEuroVectorFilter(
            freq=self._action_smoothing_freq,
            min_cutoff=self._action_smoothing_min_cutoff,
            beta=self._action_smoothing_beta,
            d_cutoff=self._action_smoothing_d_cutoff,
        )

    def _smooth_delta(self, side: str, current: np.ndarray) -> np.ndarray:
        if self._action_smoothing_method == "none":
            smoothed = np.asarray(current, dtype=float).copy()
        elif self._action_smoothing_method == "one_euro":
            filt = self._left_one_euro if side == "left" else self._right_one_euro
            smoothed = filt.apply(current)
        else:
            prev = self._left_smoothed_delta if side == "left" else self._right_smoothed_delta
            smoothed = self._ema_smooth(current, prev)

        if side == "left":
            self._left_smoothed_delta = smoothed.copy()
            self._left_missing_hold_count = 0
        else:
            self._right_smoothed_delta = smoothed.copy()
            self._right_missing_hold_count = 0
        return smoothed

    def _reset_delta_filter(self, side: str) -> None:
        if side == "left":
            self._left_prev_transform = None
            self._left_smoothed_delta = None
            self._left_one_euro.reset()
            self._left_missing_hold_count = 0
        else:
            self._right_prev_transform = None
            self._right_smoothed_delta = None
            self._right_one_euro.reset()
            self._right_missing_hold_count = 0

    def _hold_missing_delta(self, side: str) -> np.ndarray:
        if side == "left":
            last = self._left_smoothed_delta
            count = self._left_missing_hold_count
        else:
            last = self._right_smoothed_delta
            count = self._right_missing_hold_count

        if last is None or count >= self._action_missing_hold_frames:
            self._reset_delta_filter(side)
            return np.zeros(6, dtype=float)

        held = np.asarray(last, dtype=float) * (self._action_missing_decay ** (count + 1))
        if side == "left":
            self._left_missing_hold_count = count + 1
            self._left_smoothed_delta = held.copy()
        else:
            self._right_missing_hold_count = count + 1
            self._right_smoothed_delta = held.copy()
        return held

    def _mirror_pose_delta(self, delta_pose: np.ndarray) -> np.ndarray:
        """Convert opposite-side operator motion back to the canonical robot frame."""
        return np.asarray(delta_pose, dtype=float) * self.MIRROR_ACTION_SIGNS

    def _filter_delta_pose(self, side: str, delta_pose: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply deadband and reject single-frame tracking jumps."""
        filtered = np.asarray(delta_pose, dtype=float).copy()
        translation_norm = float(np.linalg.norm(filtered[:3]))
        rotation_norm = float(np.linalg.norm(filtered[3:]))

        translation_spike = (
            self._action_spike_translation is not None
            and translation_norm > self._action_spike_translation
        )
        rotation_spike = (
            self._action_spike_rotation is not None
            and rotation_norm > self._action_spike_rotation
        )
        if translation_spike or rotation_spike:
            self._delta_filter_warn_counts[side] += 1
            warn_count = self._delta_filter_warn_counts[side]
            if warn_count <= 5 or warn_count % 100 == 0:
                logger.warning(
                    "[TELEOP] Ignored %s controller tracking jump "
                    "(translation_norm=%.4f rotation_norm=%.4f "
                    "translation_limit=%s rotation_limit=%s raw=%s)",
                    side,
                    translation_norm,
                    rotation_norm,
                    self._action_spike_translation,
                    self._action_spike_rotation,
                    filtered.tolist(),
                )
            return np.zeros(6, dtype=float), True

        if (
            self._action_deadband_translation > 0.0
            and translation_norm < self._action_deadband_translation
        ):
            filtered[:3] = 0.0
        if (
            self._action_deadband_rotation > 0.0
            and rotation_norm < self._action_deadband_rotation
        ):
            filtered[3:] = 0.0

        return filtered, False

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
        
        # Check grip buttons (both must be pressed for action)
        lg_pressed = buttons.get('LG', False)
        rg_pressed = buttons.get('RG', False)
        a_pressed = buttons.get('A', False)
        x_pressed = buttons.get('X', False)
        y_pressed = buttons.get('Y', False)
        left_release_requested = bool(buttons.get('Y', False))
        right_release_requested = bool(buttons.get('B', False))
        
        self._reset_requested = bool(a_pressed)
        self._x_button_pressed = bool(x_pressed)
        self._y_button_pressed = bool(y_pressed)
        
        dof_per_arm = 7 if self._use_gripper else 6
        action = np.zeros(dof_per_arm * 2)
        left_delta_out = np.zeros(6)
        right_delta_out = np.zeros(6)
        left_trigger_value = 0.0
        right_trigger_value = 0.0
        left_trigger_pressed = False
        right_trigger_pressed = False
        left_gripper = self._left_last_gripper_position
        right_gripper = self._right_last_gripper_position
        
        # ========== Left arm (left controller) ==========
        if 'l' in transforms:
            left_transform = transforms['l']
            
            if lg_pressed:
                delta_left = self._compute_delta_pose(left_transform, self._left_prev_transform)
                delta_left = self._remap_delta_axes(delta_left)
                scaled_left = self._apply_scaling(delta_left, self._left_pose_scaler, self._left_channel_signs)
                filtered_left, left_spike_rejected = self._filter_delta_pose("left", scaled_left)
                if left_spike_rejected:
                    self._reset_delta_filter("left")
                    left_delta_out = np.zeros(6)
                else:
                    left_delta_out = self._smooth_delta("left", filtered_left)
                self._left_prev_transform = left_transform.copy()
            else:
                self._reset_delta_filter("left")
        else:
            if lg_pressed:
                left_delta_out = self._hold_missing_delta("left")
            else:
                self._reset_delta_filter("left")
        
        # ========== Right arm (right controller) ==========
        if 'r' in transforms:
            right_transform = transforms['r']
            
            if rg_pressed:
                delta_right = self._compute_delta_pose(right_transform, self._right_prev_transform)
                delta_right = self._remap_delta_axes(delta_right)
                scaled_right = self._apply_scaling(delta_right, self._right_pose_scaler, self._right_channel_signs)
                filtered_right, right_spike_rejected = self._filter_delta_pose("right", scaled_right)
                if right_spike_rejected:
                    self._reset_delta_filter("right")
                    right_delta_out = np.zeros(6)
                else:
                    right_delta_out = self._smooth_delta("right", filtered_right)
                self._right_prev_transform = right_transform.copy()
            else:
                self._reset_delta_filter("right")
        else:
            if rg_pressed:
                right_delta_out = self._hold_missing_delta("right")
            else:
                self._reset_delta_filter("right")
        
        # ========== Gripper control ==========
        if self._use_gripper:
            # Left gripper: Left Trigger
            left_trigger = buttons.get('leftTrig', (0.0,))
            if isinstance(left_trigger, tuple) and len(left_trigger) > 0:
                lt_raw_value = float(left_trigger[0])
            else:
                lt_raw_value = 0.0
            left_trigger_value = self._shape_trigger_value(lt_raw_value)
            left_trigger_pressed = (
                bool(buttons.get('LTr', False))
                or float(np.clip(lt_raw_value, 0.0, 1.0)) > self._gripper_trigger_deadzone
            )
            left_gripper = 1.0 - left_trigger_value  # Invert: trigger pressed = closed (0.0)
            self._left_last_gripper_position = left_gripper
            
            # Right gripper: Right Trigger
            right_trigger = buttons.get('rightTrig', (0.0,))
            if isinstance(right_trigger, tuple) and len(right_trigger) > 0:
                rt_raw_value = float(right_trigger[0])
            else:
                rt_raw_value = 0.0
            right_trigger_value = self._shape_trigger_value(rt_raw_value)
            right_trigger_pressed = (
                bool(buttons.get('RTr', False))
                or float(np.clip(rt_raw_value, 0.0, 1.0)) > self._gripper_trigger_deadzone
            )
            right_gripper = 1.0 - right_trigger_value  # Invert: trigger pressed = closed (0.0)
            self._right_last_gripper_position = right_gripper

        if self._mirror_teleop:
            left_delta_out, right_delta_out = (
                self._mirror_pose_delta(right_delta_out),
                self._mirror_pose_delta(left_delta_out),
            )
            self._left_grip_pressed = bool(rg_pressed)
            self._right_grip_pressed = bool(lg_pressed)
            left_release_requested, right_release_requested = (
                right_release_requested,
                left_release_requested,
            )
            if self._use_gripper:
                left_gripper, right_gripper = right_gripper, left_gripper
                left_trigger_value, right_trigger_value = right_trigger_value, left_trigger_value
                left_trigger_pressed, right_trigger_pressed = (
                    right_trigger_pressed,
                    left_trigger_pressed,
                )
        else:
            self._left_grip_pressed = bool(lg_pressed)
            self._right_grip_pressed = bool(rg_pressed)

        action[0:6] = left_delta_out
        right_offset = dof_per_arm
        action[right_offset:right_offset + 6] = right_delta_out
        if self._use_gripper:
            self._left_trigger_value = float(left_trigger_value)
            self._right_trigger_value = float(right_trigger_value)
            self._left_trigger_pressed = bool(left_trigger_pressed)
            self._right_trigger_pressed = bool(right_trigger_pressed)
            self._left_gripper_release_requested = bool(left_release_requested)
            self._right_gripper_release_requested = bool(right_release_requested)
            action[6] = left_gripper
            action[13] = right_gripper
        else:
            self._left_gripper_release_requested = bool(left_release_requested)
            self._right_gripper_release_requested = bool(right_release_requested)

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
            left_gripper_cmd
            right_gripper_cmd
            left_gripper_cmd_bin (legacy alias)
            right_gripper_cmd_bin (legacy alias)
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
            left_gripper = float(action_data[6])
            right_gripper = float(action_data[13])
            # Provide both key names so old and new pipelines keep working.
            obs_dict["left_gripper_cmd"] = left_gripper
            obs_dict["right_gripper_cmd"] = right_gripper
            obs_dict["left_gripper_cmd_bin"] = left_gripper
            obs_dict["right_gripper_cmd_bin"] = right_gripper

        else:
            obs_dict["left_gripper_cmd"] = None
            obs_dict["right_gripper_cmd"] = None
            obs_dict["left_gripper_cmd_bin"] = None
            obs_dict["right_gripper_cmd_bin"] = None

        obs_dict["left_grip_pressed"] = bool(self._left_grip_pressed)
        obs_dict["right_grip_pressed"] = bool(self._right_grip_pressed)
        obs_dict["is_expert_override"] = bool(
            self._left_grip_pressed or self._right_grip_pressed
        )
        obs_dict["left_trigger_value"] = float(self._left_trigger_value)
        obs_dict["right_trigger_value"] = float(self._right_trigger_value)
        obs_dict["left_trigger_pressed"] = bool(self._left_trigger_pressed)
        obs_dict["right_trigger_pressed"] = bool(self._right_trigger_pressed)
        obs_dict["left_gripper_release_requested"] = bool(self._left_gripper_release_requested)
        obs_dict["right_gripper_release_requested"] = bool(self._right_gripper_release_requested)
        obs_dict["x_button_pressed"] = bool(self._x_button_pressed)
        obs_dict["y_button_pressed"] = bool(self._y_button_pressed)
        
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
