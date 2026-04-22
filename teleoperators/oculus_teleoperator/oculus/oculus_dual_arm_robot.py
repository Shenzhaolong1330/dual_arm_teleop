"""
Oculus Quest dual-arm robot controller.
Uses both left and right Oculus controllers to control a dual-arm robot system.

Left controller  -> Left arm
Right controller -> Right arm
"""

from typing import Dict, Optional, Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R

from .oculus_reader.oculus_reader import OculusReader
from .robot import Robot


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
    - A button: Request RIGHT arm reset
    - X button: Request LEFT arm reset
    
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

    def __init__(
        self,
        ip: str = '192.168.110.62',
        use_gripper: bool = True,
        left_pose_scaler: Sequence[float] = [1.0, 1.0],
        left_channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
        right_pose_scaler: Sequence[float] = [1.0, 1.0],
        right_channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
        action_smoothing_alpha: float = 0.35,
    ):
        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        
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

        # EMA smoothing state (6D delta pose for each arm)
        self._action_smoothing_alpha = float(action_smoothing_alpha)
        self._left_smoothed_delta = None
        self._right_smoothed_delta = None
        
        # Reset request
        self._reset_requested = False
        self._left_arm_reset_requested = False
        self._right_arm_reset_requested = False

    def _ema_smooth(self, current: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        """Apply EMA smoothing to a 6D delta vector."""
        alpha = max(0.0, min(1.0, self._action_smoothing_alpha))
        if prev is None or alpha >= 1.0:
            return current.copy()
        return alpha * current + (1.0 - alpha) * prev

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
        self._right_arm_reset_requested = a_pressed
        self._left_arm_reset_requested = x_pressed
        self._reset_requested = a_pressed or x_pressed
        
        dof_per_arm = 7 if self._use_gripper else 6
        action = np.zeros(dof_per_arm * 2)
        
        # ========== Left arm (left controller) ==========
        if 'l' in transforms:
            left_transform = transforms['l']
            
            if lg_pressed:
                delta_left = self._compute_delta_pose(left_transform, self._left_prev_transform)
                scaled_left = self._apply_scaling(delta_left, self._left_pose_scaler, self._left_channel_signs)
                smoothed_left = self._ema_smooth(scaled_left, self._left_smoothed_delta)
                self._left_smoothed_delta = smoothed_left.copy()
                action[0:6] = smoothed_left
                self._left_prev_transform = left_transform.copy()
            else:
                self._left_prev_transform = None
                self._left_smoothed_delta = None
        else:
            self._left_prev_transform = None
            self._left_smoothed_delta = None
        
        # ========== Right arm (right controller) ==========
        if 'r' in transforms:
            right_transform = transforms['r']
            
            if rg_pressed:
                delta_right = self._compute_delta_pose(right_transform, self._right_prev_transform)
                scaled_right = self._apply_scaling(delta_right, self._right_pose_scaler, self._right_channel_signs)
                smoothed_right = self._ema_smooth(scaled_right, self._right_smoothed_delta)
                self._right_smoothed_delta = smoothed_right.copy()
                if self._use_gripper:
                    action[7:13] = smoothed_right
                else:
                    action[6:12] = smoothed_right
                self._right_prev_transform = right_transform.copy()
            else:
                self._right_prev_transform = None
                self._right_smoothed_delta = None
        else:
            self._right_prev_transform = None
            self._right_smoothed_delta = None
        
        # ========== Gripper control ==========
        if self._use_gripper:
            # Left gripper: Left Trigger
            left_trigger = buttons.get('leftTrig', (0.0,))
            if isinstance(left_trigger, tuple) and len(left_trigger) > 0:
                lt_value = left_trigger[0]
            else:
                lt_value = 0.0
            left_gripper = 1.0 - lt_value  # Invert: trigger pressed = closed (0.0)
            self._left_last_gripper_position = left_gripper
            action[6] = left_gripper
            
            # Right gripper: Right Trigger
            right_trigger = buttons.get('rightTrig', (0.0,))
            if isinstance(right_trigger, tuple) and len(right_trigger) > 0:
                rt_value = right_trigger[0]
            else:
                rt_value = 0.0
            right_gripper = 1.0 - rt_value  # Invert: trigger pressed = closed (0.0)
            self._right_last_gripper_position = right_gripper
            action[13] = right_gripper
        
        return action

    def is_reset_requested(self) -> bool:
        """Check if any arm reset was requested (A or X button pressed)."""
        return self._reset_requested

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Return the current robot observations for dual-arm system.
        
        Returns dict with keys:
            left_delta_ee_pose.{x,y,z,rx,ry,rz}
            right_delta_ee_pose.{x,y,z,rx,ry,rz}
            left_gripper_cmd_bin
            right_gripper_cmd_bin
            left_arm_reset_requested
            right_arm_reset_requested
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
        
        # Reset request flags
        obs_dict["left_arm_reset_requested"] = self._left_arm_reset_requested
        obs_dict["right_arm_reset_requested"] = self._right_arm_reset_requested
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
    print("  - A button: Request RIGHT arm reset")
    print("  - X button: Request LEFT arm reset")
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
