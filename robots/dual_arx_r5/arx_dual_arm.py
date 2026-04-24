"""
ARX R5 dual-arm robot implementation for dual_arm_teleop.

Each arm has 6 DOF. Gripper is integrated as joint 7 (controlled separately).
Communication via ZMQ+msgpack RPC to the ARX ROS2 bridge server.

Supports two control modes in send_action():
  1. Cartesian/Oculus mode: receives delta_ee_pose, computes absolute target,
     sends to server via set_dual_ee_poses (服务端内置 IK)
  2. Direct joint mode: receives joint positions directly (for replay)

Performance optimizations:
  - 服务端 IK: 不在客户端做 Placo IK, 而是发绝对 EE 位姿给服务端
    (与 Dobot/Franka 一致, send_action ~3ms vs 原来客户端 IK ~13ms)
  - 后台相机线程: 每台相机独立 daemon 线程持续 read(),
    get_observation() 取缓存帧 (避免 try_wait_for_frames 阻塞主循环)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config_arx import ArxDualArmConfig
from .arx_interface_client import ArxDualArmClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Joint mode (replay) 安全限幅: 每步最大变化量 (rad)
_MAX_JOINT_DELTA = 0.3

# VR 与双臂安装朝向补偿 (绕 z 轴):
# 左臂 x 相对 VR x 顺时针 90° -> -90°
# 右臂 x 相对 VR x 逆时针 90° -> +90°
_LEFT_ARM_YAW_COMP_DEG = 90.0
_RIGHT_ARM_YAW_COMP_DEG = -90.0


def _apply_delta_pose(current_pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply delta [dx,dy,dz,drx,dry,drz] to current pose [x,y,z,roll,pitch,yaw].

    Position: 向量加法
    Rotation: delta 是 rotation vector (来自 oculus_dual_arm_robot.as_rotvec()),
              在世界坐标系下左乘 (与原 arx_ik_solver.py:146 的 delta_rot * current_rot 一致)

    Args:
        current_pose: [x, y, z, roll, pitch, yaw], 当前 EE 位姿 (来自服务端 get_full_state)
        delta: [dx, dy, dz, drx, dry, drz], 增量 (position 为米, rotation 为 rotvec 弧度)

    Returns:
        [x, y, z, roll, pitch, yaw] 目标位姿, 格式与 current_pose 一致
    """
    target = np.zeros(6)

    # 位置: 直接加法
    target[:3] = current_pose[:3] + delta[:3]

    # 旋转: euler → Rotation, 左乘 delta rotvec, 转回 euler
    # 使用 'xyz' 内旋约定 (与 ARX ROS2 bridge 一致: roll=x, pitch=y, yaw=z)
    current_rot = Rotation.from_euler('xyz', current_pose[3:])
    delta_rot = Rotation.from_rotvec(delta[3:])
    target_rot = delta_rot * current_rot    # 世界坐标系左乘
    target[3:] = target_rot.as_euler('xyz')

    return target


class ArxDualArm(Robot):
    """ARX R5 dual-arm robot. 6 DOF per arm, integrated gripper."""

    config_class = ArxDualArmConfig
    name = "arx_dual_arm"

    def __init__(self, config: ArxDualArmConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.config = config
        self._is_connected = False
        self._client: Optional[ArxDualArmClient] = None
        self._num_arm_joints = config.num_arm_joints  # 6
        self._prev_observation = None

        # 缓存的关节位置 (用于 joint mode replay 安全限幅)
        self._cached_left_joints = np.zeros(self._num_arm_joints)
        self._cached_right_joints = np.zeros(self._num_arm_joints)

        # 缓存的 EE 位姿 (用于 cartesian mode: current + delta → target)
        # 格式: [x, y, z, roll, pitch, yaw], 从 get_full_state 的 end_pose 获取
        self._cached_left_ee_pose = np.zeros(6)
        self._cached_right_ee_pose = np.zeros(6)
        self._left_filtered_delta = np.zeros(6)
        self._right_filtered_delta = np.zeros(6)
        self._left_filter_initialized = False
        self._right_filter_initialized = False
        self._left_reset_latch = False
        self._right_reset_latch = False

        # Gripper state tracking
        self._last_left_gripper_cmd = 0.0
        self._last_right_gripper_cmd = 0.0
        self._left_gripper_state = 0.0
        self._right_gripper_state = 0.0

        # ============================================================
        # 后台相机读取 (避免 try_wait_for_frames 阻塞主循环)
        # ============================================================
        self._camera_threads: dict[str, threading.Thread] = {}
        self._camera_stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frames: dict[str, Any] = {}    # cam_name → latest np.ndarray

        # 可选 profiling hook: callback(stage_name: str, elapsed_ms: float)
        self._latency_hook: Optional[Callable[[str, float], None]] = None

    def set_latency_hook(self, hook: Optional[Callable[[str, float], None]]) -> None:
        """Attach/detach latency profiling hook."""
        self._latency_hook = hook

    def _record_latency(self, stage_name: str, elapsed_ms: float) -> None:
        hook = self._latency_hook
        if hook is None:
            return
        try:
            hook(stage_name, elapsed_ms)
        except Exception as e:
            logger.debug(f"[ROBOT] latency hook error on {stage_name}: {e}")

    @staticmethod
    def _normalize_alpha(alpha: float) -> float:
        try:
            alpha = float(alpha)
        except Exception:
            return 1.0
        return float(np.clip(alpha, 0.0, 1.0))

    def _apply_delta_filter(self, side: str, raw_delta: np.ndarray) -> np.ndarray:
        if not self.config.enable_ee_action_filter:
            return raw_delta

        # Oculus 在未按下 Grip 时会输出全 0，直接透传避免“松手后残余拖尾”。
        if np.linalg.norm(raw_delta) < 1e-9:
            if side == "left":
                self._left_filtered_delta = raw_delta.copy()
                self._left_filter_initialized = True
                return self._left_filtered_delta.copy()
            self._right_filtered_delta = raw_delta.copy()
            self._right_filter_initialized = True
            return self._right_filtered_delta.copy()

        alpha_pos = self._normalize_alpha(self.config.ee_action_filter_alpha_pos)
        alpha_rot = self._normalize_alpha(self.config.ee_action_filter_alpha_rot)
        alpha_vec = np.array([alpha_pos, alpha_pos, alpha_pos, alpha_rot, alpha_rot, alpha_rot], dtype=float)

        if side == "left":
            if not self._left_filter_initialized:
                self._left_filtered_delta = raw_delta.copy()
                self._left_filter_initialized = True
                return self._left_filtered_delta.copy()
            self._left_filtered_delta = alpha_vec * raw_delta + (1.0 - alpha_vec) * self._left_filtered_delta
            return self._left_filtered_delta.copy()

        if not self._right_filter_initialized:
            self._right_filtered_delta = raw_delta.copy()
            self._right_filter_initialized = True
            return self._right_filtered_delta.copy()
        self._right_filtered_delta = alpha_vec * raw_delta + (1.0 - alpha_vec) * self._right_filtered_delta
        return self._right_filtered_delta.copy()

    def _apply_delta_deadband(self, delta: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply deadband to one arm delta_ee.

        Returns:
            (delta_after_deadband, is_suppressed)
        """
        if not self.config.enable_ee_action_deadband:
            return delta, False

        pos_norm = float(np.linalg.norm(delta[:3]))
        rot_norm = float(np.linalg.norm(delta[3:]))
        pos_th = max(float(self.config.ee_action_deadband_pos_norm), 0.0)
        rot_th = max(float(self.config.ee_action_deadband_rot_norm), 0.0)
        suppressed = (pos_norm < pos_th) and (rot_norm < rot_th)
        if suppressed:
            return np.zeros(6, dtype=float), True
        return delta, False

    @staticmethod
    def _rotate_delta_about_z(delta: np.ndarray, yaw_deg: float) -> np.ndarray:
        """Rotate both translation and rotvec components around z-axis."""
        rot_z = Rotation.from_euler("z", yaw_deg, degrees=True)
        rotated = delta.copy()
        rotated[:3] = rot_z.apply(delta[:3])
        rotated[3:] = rot_z.apply(delta[3:])
        return rotated

    def _apply_arm_mount_compensation(self, side: str, delta: np.ndarray) -> np.ndarray:
        """Compensate fixed yaw offset between VR frame and each arm frame."""
        if side == "left":
            return self._rotate_delta_about_z(delta, _LEFT_ARM_YAW_COMP_DEG)
        if side == "right":
            return self._rotate_delta_about_z(delta, _RIGHT_ARM_YAW_COMP_DEG)
        return delta

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        logger.info("\n" + "=" * 60)
        logger.info("[ROBOT] Connecting to ARX R5 Dual-Arm System")
        logger.info("=" * 60)

        # 1. Connect RPC client
        logger.info(f"[ROBOT] Connecting to RPC server at {self.config.robot_ip}:{self.config.robot_port}")
        self._client = ArxDualArmClient(
            ip=self.config.robot_ip,
            port=self.config.robot_port,
        )
        if not self._client.system_connect(timeout=10.0):
            raise RuntimeError("Failed to connect to ARX RPC server")
        logger.info("[ROBOT] RPC connection established")

        # 2. Verify by reading initial state + 缓存关节和 EE 位姿
        state = self._client.get_full_state()
        if state is None:
            raise RuntimeError("Failed to read initial state from ARX RPC server")

        for side in ("left_arm", "right_arm"):
            if side in state:
                jp = state[side]["joint_positions"]
                ep = state[side]["end_pose"]
                logger.info(f"[{side.upper()}] joints: {[round(j, 4) for j in jp[:6]]}")
                logger.info(f"[{side.upper()}] ee_pose: {[round(e, 4) for e in ep]}")

        self._cached_left_joints = state["left_arm"]["joint_positions"][:self._num_arm_joints].copy()
        self._cached_right_joints = state["right_arm"]["joint_positions"][:self._num_arm_joints].copy()
        self._cached_left_ee_pose = np.array(state["left_arm"]["end_pose"][:6], dtype=float)
        self._cached_right_ee_pose = np.array(state["right_arm"]["end_pose"][:6], dtype=float)
        self._left_filter_initialized = False
        self._right_filter_initialized = False

        # 3. Connect cameras + 启动后台相机读取线程
        if self.cameras:
            logger.info("\n===== [CAM] Initializing Cameras =====")
            self._camera_stop_event.clear()
            for cam_name, cam in self.cameras.items():
                cam.connect()
                logger.info(f"[CAM] {cam_name} connected")
                # 后台线程持续 cam.read(), 缓存最新帧
                t = threading.Thread(
                    target=self._camera_read_loop,
                    args=(cam_name, cam),
                    name=f"cam_{cam_name}",
                    daemon=True,
                )
                t.start()
                self._camera_threads[cam_name] = t
            logger.info(f"[CAM] {len(self.cameras)} background camera threads started")
            logger.info("===== [CAM] Cameras Initialized =====\n")

        self.is_connected = True
        logger.info(f"[ROBOT] {self.name} initialization completed.\n")

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        # 1. 停止后台相机线程
        self._camera_stop_event.set()
        for name, t in self._camera_threads.items():
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning(f"[CAM] Camera thread {name} did not stop cleanly")
        self._camera_threads.clear()
        self._latest_frames.clear()

        # 2. 断开相机连接
        for cam in self.cameras.values():
            cam.disconnect()

        # 3. 断开 RPC 连接
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.warning(f"[ROBOT] Error during disconnect: {e}")
            self._client.close()
            self._client = None

        self.is_connected = False
        logger.info(f"[ROBOT] {self.name} disconnected")

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        logger.info("[ROBOT] Resetting dual-arm system...")

        # 刷新缓存
        state = self._client.get_full_state()
        if state is not None:
            self._cached_left_joints = state["left_arm"]["joint_positions"][:self._num_arm_joints].copy()
            self._cached_right_joints = state["right_arm"]["joint_positions"][:self._num_arm_joints].copy()
            self._cached_left_ee_pose = np.array(state["left_arm"]["end_pose"][:6], dtype=float)
            self._cached_right_ee_pose = np.array(state["right_arm"]["end_pose"][:6], dtype=float)
            self._left_filter_initialized = False
            self._right_filter_initialized = False
            self._left_reset_latch = False
            self._right_reset_latch = False

        # Open grippers
        if self.config.use_gripper:
            self._client.set_left_gripper(self.config.gripper_open_value)
            self._client.set_right_gripper(self.config.gripper_open_value)
            self._last_left_gripper_cmd = 0.0
            self._last_right_gripper_cmd = 0.0

        logger.info("[ROBOT] Reset complete\n")

    # ============================================================
    # Action
    # ============================================================

    def _home_joint_target(self) -> np.ndarray:
        home = np.asarray(self.config.home_joint_positions, dtype=float).reshape(-1)
        if home.size < self._num_arm_joints:
            logger.warning(
                f"[ROBOT] home_joint_positions only has {home.size} values, "
                f"padding to {self._num_arm_joints} with zeros."
            )
            padded = np.zeros(self._num_arm_joints, dtype=float)
            if home.size > 0:
                padded[:home.size] = home
            return padded
        if home.size > self._num_arm_joints:
            logger.warning(
                f"[ROBOT] home_joint_positions has {home.size} values, truncating to {self._num_arm_joints}."
            )
            return home[:self._num_arm_joints]
        return home

    def _smooth_reset_arms(self, reset_left: bool, reset_right: bool) -> None:
        if not reset_left and not reset_right:
            return
        if self._client is None:
            return

        start_left = self._cached_left_joints.copy()
        start_right = self._cached_right_joints.copy()
        state = self._client.get_full_state()
        if state is not None:
            start_left = state["left_arm"]["joint_positions"][:self._num_arm_joints].astype(float, copy=True)
            start_right = state["right_arm"]["joint_positions"][:self._num_arm_joints].astype(float, copy=True)

        home = self._home_joint_target()
        target_left = home.copy() if reset_left else start_left.copy()
        target_right = home.copy() if reset_right else start_right.copy()

        steps = max(int(self.config.home_steps), 1)
        step_interval = max(float(self.config.home_step_interval), 0.0)
        left_gripper = float(self._last_left_gripper_cmd)
        right_gripper = float(self._last_right_gripper_cmd)

        logger.info(
            "[ROBOT] Smooth reset request: "
            f"left={'ON' if reset_left else 'OFF'}, right={'ON' if reset_right else 'OFF'}, "
            f"steps={steps}, dt={step_interval:.4f}s"
        )

        for step in range(1, steps + 1):
            alpha = step / steps
            left_q = (1.0 - alpha) * start_left + alpha * target_left
            right_q = (1.0 - alpha) * start_right + alpha * target_right
            left_7 = np.append(left_q, left_gripper)
            right_7 = np.append(right_q, right_gripper)
            self._client.set_dual_joint_positions(left_7, right_7)
            if step_interval > 0.0 and step < steps:
                time.sleep(step_interval)

        self._cached_left_joints = target_left.copy()
        self._cached_right_joints = target_right.copy()
        self._left_filter_initialized = False
        self._right_filter_initialized = False

        # 尽快刷新 EE pose 缓存，避免 reset 后第一帧 delta 基于旧姿态叠加。
        final_state = self._client.get_full_state()
        if final_state is not None:
            self._cached_left_joints = final_state["left_arm"]["joint_positions"][:self._num_arm_joints].astype(
                float, copy=True
            )
            self._cached_right_joints = final_state["right_arm"]["joint_positions"][:self._num_arm_joints].astype(
                float, copy=True
            )
            self._cached_left_ee_pose = np.array(final_state["left_arm"]["end_pose"][:6], dtype=float)
            self._cached_right_ee_pose = np.array(final_state["right_arm"]["end_pose"][:6], dtype=float)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        legacy_reset = bool(action.get("reset_requested", False))
        left_reset_pressed = bool(action.get("left_arm_reset_requested", legacy_reset))
        right_reset_pressed = bool(action.get("right_arm_reset_requested", legacy_reset))
        left_reset_edge = left_reset_pressed and not self._left_reset_latch
        right_reset_edge = right_reset_pressed and not self._right_reset_latch
        motion_handled_by_reset = False

        if not self.config.debug and (left_reset_edge or right_reset_edge):
            self._smooth_reset_arms(left_reset_edge, right_reset_edge)
            motion_handled_by_reset = True

        if not self.config.debug and not motion_handled_by_reset:
            has_delta_ee = "left_delta_ee_pose.x" in action
            has_joints = all(f"left_joint_{i+1}.pos" in action for i in range(self._num_arm_joints))

            if has_delta_ee:
                self._send_action_cartesian(action)
            elif has_joints:
                self._send_action_joint(action)

        # Handle grippers
        if "left_gripper_cmd_bin" in action:
            self._handle_gripper("left", action["left_gripper_cmd_bin"])
        if "right_gripper_cmd_bin" in action:
            self._handle_gripper("right", action["right_gripper_cmd_bin"])

        self._left_reset_latch = left_reset_pressed
        self._right_reset_latch = right_reset_pressed

        return action

    def _send_action_cartesian(self, action: dict[str, Any]) -> None:
        """Oculus mode: delta_ee → 绝对目标位姿 → set_dual_ee_poses (服务端 IK).

        和 Dobot/Franka 一致的模式: 客户端只做加法, 服务端做 IK.
        send_action 总耗时 ~3ms (仅 RPC), 而不是之前客户端 IK 的 ~13ms.
        """
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        left_delta_raw = np.array([action.get(f"left_delta_ee_pose.{a}", 0.0) for a in axes], dtype=float)
        right_delta_raw = np.array([action.get(f"right_delta_ee_pose.{a}", 0.0) for a in axes], dtype=float)
        left_delta_raw = self._apply_arm_mount_compensation("left", left_delta_raw)
        right_delta_raw = self._apply_arm_mount_compensation("right", right_delta_raw)
        left_delta = self._apply_delta_filter("left", left_delta_raw)
        right_delta = self._apply_delta_filter("right", right_delta_raw)
        left_delta, left_suppressed = self._apply_delta_deadband(left_delta)
        right_delta, right_suppressed = self._apply_delta_deadband(right_delta)

        # 待机抖动: 双臂都在死区时不下发任何 EE 指令，避免误差累积。
        if left_suppressed and right_suppressed:
            return

        # 计算绝对目标位姿: current + delta (纯数学运算, <0.1ms)
        target_left = _apply_delta_pose(self._cached_left_ee_pose, left_delta)
        target_right = _apply_delta_pose(self._cached_right_ee_pose, right_delta)

        # 发送到服务端, 服务端内置 IK (fire-and-forget, ~3ms RPC)
        t_rpc = time.perf_counter()
        self._client.set_dual_ee_poses(
            target_left, target_right,
            self._last_left_gripper_cmd, self._last_right_gripper_cmd
        )
        self._record_latency("rpc_set_dual_ee_poses_ms", (time.perf_counter() - t_rpc) * 1e3)

        # 更新缓存 (下次 delta 基于本次目标位姿叠加)
        self._cached_left_ee_pose = target_left
        self._cached_right_ee_pose = target_right

    def _send_action_joint(self, action: dict[str, Any]) -> None:
        """Direct joint mode: joint positions → RPC (用于 replay)."""
        left_q = np.array([action[f"left_joint_{i+1}.pos"] for i in range(self._num_arm_joints)])
        right_q = np.array([action[f"right_joint_{i+1}.pos"] for i in range(self._num_arm_joints)])

        # 安全限幅
        left_q = self._clamp_joint_delta(left_q, self._cached_left_joints)
        right_q = self._clamp_joint_delta(right_q, self._cached_right_joints)

        left_7 = np.append(left_q, self._last_left_gripper_cmd)
        right_7 = np.append(right_q, self._last_right_gripper_cmd)

        self._client.set_dual_joint_positions(left_7, right_7)

        self._cached_left_joints = left_q.copy()
        self._cached_right_joints = right_q.copy()

    @staticmethod
    def _clamp_joint_delta(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Clamp per-joint change to _MAX_JOINT_DELTA (rad)."""
        delta = target - current
        clipped = np.clip(delta, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
        return current + clipped

    def _handle_gripper(self, side: str, value: float) -> None:
        if not self.config.use_gripper:
            return

        if value >= self.config.close_threshold:
            cmd = self.config.gripper_close_value
        else:
            cmd = self.config.gripper_open_value

        if self.config.gripper_reverse:
            cmd = 1.0 - cmd

        last_attr = f"_last_{side}_gripper_cmd"
        if cmd != getattr(self, last_attr):
            if side == "left":
                self._client.set_left_gripper(cmd)
            else:
                self._client.set_right_gripper(cmd)
            setattr(self, last_attr, cmd)

    # ============================================================
    # 后台相机读取线程
    # ============================================================

    def _camera_read_loop(self, cam_name: str, cam) -> None:
        """后台线程: 持续读取一台相机, 将最新帧缓存到 _latest_frames.

        LeRobot RealSenseCamera.read() 内部调用 try_wait_for_frames(timeout_ms=200),
        在主循环中可能阻塞 0-33ms (等下一帧). 后台线程持续读取,
        get_observation() 直接取缓存帧 (<0.1ms), 消除了帧同步阻塞风险.
        """
        logger.info(f"[CAM_THREAD] {cam_name} read loop started")
        while not self._camera_stop_event.is_set():
            try:
                frame = cam.read()
                with self._frame_lock:
                    self._latest_frames[cam_name] = frame
            except Exception as e:
                logger.warning(f"[CAM_THREAD] {cam_name} read error: {e}")
                self._camera_stop_event.wait(timeout=0.1)
        logger.info(f"[CAM_THREAD] {cam_name} read loop stopped")

    # ============================================================
    # Observation
    # ============================================================

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        # RPC: 获取机器人全状态
        t_rpc = time.perf_counter()
        try:
            state = self._client.get_full_state()
            if state is None:
                raise RuntimeError("get_full_state returned None")
        except Exception as e:
            logger.warning(f"[ROBOT] RPC error in get_observation: {e}")
            if self._prev_observation is not None:
                return self._prev_observation
            raise
        finally:
            self._record_latency("rpc_get_full_state_ms", (time.perf_counter() - t_rpc) * 1e3)

        # 缓存关节和 EE 位姿 (供 send_action 使用)
        self._cached_left_joints = state["left_arm"]["joint_positions"][:self._num_arm_joints].copy()
        self._cached_right_joints = state["right_arm"]["joint_positions"][:self._num_arm_joints].copy()
        self._cached_left_ee_pose = np.array(state["left_arm"]["end_pose"][:6], dtype=float)
        self._cached_right_ee_pose = np.array(state["right_arm"]["end_pose"][:6], dtype=float)

        obs = {}

        # Left arm joints (6, 1-indexed)
        left_jp = state["left_arm"]["joint_positions"]
        for i in range(self._num_arm_joints):
            obs[f"left_joint_{i+1}.pos"] = float(left_jp[i])

        # Right arm joints (6, 1-indexed)
        right_jp = state["right_arm"]["joint_positions"]
        for i in range(self._num_arm_joints):
            obs[f"right_joint_{i+1}.pos"] = float(right_jp[i])

        # End-effector poses
        left_ep = state["left_arm"]["end_pose"]
        right_ep = state["right_arm"]["end_pose"]
        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs[f"left_ee_pose.{axis}"] = float(left_ep[i])
            obs[f"right_ee_pose.{axis}"] = float(right_ep[i])

        # Gripper states
        if self.config.use_gripper:
            left_grip = state["left_arm"]["gripper"]
            right_grip = state["right_arm"]["gripper"]
            if self.config.gripper_reverse:
                left_grip = 1.0 - left_grip
                right_grip = 1.0 - right_grip
            self._left_gripper_state = left_grip
            self._right_gripper_state = right_grip
            obs["left_gripper_state_norm"] = self._left_gripper_state
            obs["right_gripper_state_norm"] = self._right_gripper_state
            obs["left_gripper_cmd_bin"] = self._last_left_gripper_cmd
            obs["right_gripper_cmd_bin"] = self._last_right_gripper_cmd

        # Camera images: 从后台线程缓存取最新帧 (<0.1ms)
        t_cam_total = time.perf_counter()
        if self.cameras:
            with self._frame_lock:
                for cam_name in self.cameras:
                    t_cam_one = time.perf_counter()
                    if cam_name in self._latest_frames:
                        obs[cam_name] = self._latest_frames[cam_name]
                    else:
                        # 后台线程还没读到第一帧, fallback 同步读
                        obs[cam_name] = self.cameras[cam_name].read()
                    self._record_latency(
                        f"camera_fetch_{cam_name}_ms",
                        (time.perf_counter() - t_cam_one) * 1e3,
                    )
        self._record_latency("camera_fetch_total_ms", (time.perf_counter() - t_cam_total) * 1e3)

        self._prev_observation = obs
        return obs

    # ========== Feature definitions ==========

    @property
    def _motors_ft(self) -> dict[str, type]:
        features = {}

        # Left arm joints
        for i in range(self._num_arm_joints):
            features[f"left_joint_{i+1}.pos"] = float

        # Right arm joints
        for i in range(self._num_arm_joints):
            features[f"right_joint_{i+1}.pos"] = float

        # Left arm end effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_ee_pose.{axis}"] = float

        # Right arm end effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_ee_pose.{axis}"] = float

        if self.config.use_gripper:
            features["left_gripper_state_norm"] = float
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_state_norm"] = float
            features["right_gripper_cmd_bin"] = float

        return features

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        if self.config.control_mode == "oculus":
            # Left arm delta pose
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"left_delta_ee_pose.{axis}"] = float

            # Right arm delta pose
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"right_delta_ee_pose.{axis}"] = float
        else:
            # Left arm joints
            for i in range(self._num_arm_joints):
                features[f"left_joint_{i+1}.pos"] = float

            # Right arm joints
            for i in range(self._num_arm_joints):
                features[f"right_joint_{i+1}.pos"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    # ========== Standard Robot interface ==========

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected

    def configure(self) -> None:
        pass
