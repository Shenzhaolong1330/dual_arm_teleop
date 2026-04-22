"""
Placo IK solver wrapper for ARX R5 dual-arm robot.

Loads the dual_R5a.urdf and provides IK solving for delta EE poses
(from Oculus teleoperation) to joint positions.

URDF joint layout:
  - left_joint1~6, right_joint1~6: 12 revolute arm joints (actively controlled)
  - left_joint7/8, right_joint7/8: 4 prismatic gripper joints (locked in IK)
  - EE frames: left_link6, right_link6
"""

import logging
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_ARM_JOINT_NAMES = {
    "left": [f"left_joint{i}" for i in range(1, 7)],
    "right": [f"right_joint{i}" for i in range(1, 7)],
}

_EE_FRAME = {
    "left": "left_link6",
    "right": "right_link6",
}


class ArxR5IKSolver:
    """Placo-based IK solver for ARX R5 dual-arm URDF."""

    def __init__(self, urdf_path: str, servo_time: float = 0.017):
        self._urdf_path = str(Path(urdf_path).resolve())
        self._servo_time = servo_time
        self._last_left_q = np.zeros(6)
        self._last_right_q = np.zeros(6)

        self._setup_placo()
        self._setup_endeffector_tasks()
        self._setup_joints_regularization()

        logger.info(f"[IK] ArxR5IKSolver initialized from {self._urdf_path}")

    def _setup_placo(self):
        """Load URDF, create solver, build joint index map."""
        import placo  # Lazy import: placo is Linux-only, defer to avoid import errors on Windows
        self.placo_robot = placo.RobotWrapper(self._urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self._servo_time
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Build state.q index map using probe technique
        self._q_indices = {}
        for jname in self.placo_robot.joint_names():
            q_before = self.placo_robot.state.q.copy()
            self.placo_robot.set_joint(jname, 0.777)
            q_after = self.placo_robot.state.q.copy()
            self.placo_robot.set_joint(jname, 0.0)
            changed = np.where(np.abs(q_after - q_before) > 1e-10)[0]
            if len(changed) > 0:
                self._q_indices[jname] = int(changed[0])

        self.placo_robot.update_kinematics()

    def _setup_endeffector_tasks(self):
        """Configure frame tasks for both arm EE links."""
        self.effector_task = {}
        for side in ("left", "right"):
            link_name = _EE_FRAME[side]
            initial_pose = np.eye(4)
            task = self.solver.add_frame_task(link_name, initial_pose)
            task.configure(f"{side}_frame", "soft", 1.0)
            manip = self.solver.add_manipulability_task(link_name, "both", 1.0)
            manip.configure(f"{side}_manipulability", "soft", 1e-3)
            self.effector_task[side] = task

    def _setup_joints_regularization(self):
        """Lock non-arm joints (gripper prismatic joints) at zero."""
        arm_joint_names = set()
        for side in ("left", "right"):
            arm_joint_names.update(_ARM_JOINT_NAMES[side])

        joints_task = self.solver.add_joints_task()
        q0 = self.placo_robot.state.q
        non_arm = {}
        for jname in self.placo_robot.joint_names():
            if jname not in arm_joint_names and jname in self._q_indices:
                non_arm[jname] = q0[self._q_indices[jname]]
        joints_task.set_joints(non_arm)
        joints_task.configure("non_arm_regularization", "soft", 1e-4)

    def _set_joint(self, name: str, angle: float):
        idx = self._q_indices[name]
        self.placo_robot.state.q[idx] = angle

    def _get_joint(self, name: str) -> float:
        idx = self._q_indices[name]
        return self.placo_robot.state.q[idx]

    def sync_from_joint_positions(self, left_joints: np.ndarray, right_joints: np.ndarray):
        """Sync Placo model from real robot joint positions (6 per arm).
        Also updates the IK frame task targets to current FK poses
        and stores as fallback for IK failure recovery."""
        for i, jname in enumerate(_ARM_JOINT_NAMES["left"]):
            self._set_joint(jname, float(left_joints[i]))
        for i, jname in enumerate(_ARM_JOINT_NAMES["right"]):
            self._set_joint(jname, float(right_joints[i]))
        self.placo_robot.update_kinematics()

        # Update IK fallback with real state (avoids returning zeros on first IK failure)
        self._last_left_q = np.array(left_joints[:6], dtype=float)
        self._last_right_q = np.array(right_joints[:6], dtype=float)

        # Sync frame task targets to current FK poses
        for side in ("left", "right"):
            T_current = self.placo_robot.get_T_world_frame(_EE_FRAME[side])
            self.effector_task[side].T_world_frame = T_current

    def solve_delta_ee(
        self,
        left_delta: np.ndarray | None,
        right_delta: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply delta EE poses and solve IK.

        Args:
            left_delta: [dx, dy, dz, drx, dry, drz] in meters/radians, or None to hold.
            right_delta: same format.

        Returns:
            (left_q[6], right_q[6]) joint angles in radians.
        """
        for side, delta in [("left", left_delta), ("right", right_delta)]:
            if delta is None or np.linalg.norm(delta) < 1e-6:
                continue

            T_current = self.effector_task[side].T_world_frame
            target_pos = T_current[:3, 3] + delta[:3]
            current_rot = Rotation.from_matrix(T_current[:3, :3])
            # delta[3:] 是 rotvec (来自 oculus_dual_arm_robot 的 as_rotvec()), 不是 euler 角
            # 与 Franka 实现一致 (franka.py:361), 使用 from_rotvec
            delta_rot = Rotation.from_rotvec(delta[3:])
            # World-frame rotation: delta applied in world frame (左乘 = 全局坐标系旋转)
            target_rot = delta_rot * current_rot

            T_target = np.eye(4)
            T_target[:3, :3] = target_rot.as_matrix()
            T_target[:3, 3] = target_pos
            self.effector_task[side].T_world_frame = T_target

        try:
            self.solver.solve(True)
            self.placo_robot.update_kinematics()
            left_q, right_q = self.read_solved_joints()
            self._last_left_q = left_q.copy()
            self._last_right_q = right_q.copy()
            return left_q, right_q
        except RuntimeError as e:
            logger.warning(f"IK solver failed: {e}. Returning last good positions.")
            return self._last_left_q.copy(), self._last_right_q.copy()

    def read_solved_joints(self) -> tuple[np.ndarray, np.ndarray]:
        """Read current solved arm joint angles (6 per arm)."""
        left_q = np.array([self._get_joint(j) for j in _ARM_JOINT_NAMES["left"]])
        right_q = np.array([self._get_joint(j) for j in _ARM_JOINT_NAMES["right"]])
        return left_q, right_q


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    urdf = str(Path(__file__).parents[2] / "assets" / "arx_r5" / "dual_R5a.urdf")
    solver = ArxR5IKSolver(urdf)

    # Test: sync to zero config, read FK, apply small delta, verify
    zeros = np.zeros(6)
    solver.sync_from_joint_positions(zeros, zeros)
    left_q, right_q = solver.read_solved_joints()
    print(f"Initial left_q: {left_q}")
    print(f"Initial right_q: {right_q}")

    # Small delta
    delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_left, new_right = solver.solve_delta_ee(delta, delta)
    print(f"After delta left_q: {new_left}")
    print(f"After delta right_q: {new_right}")
