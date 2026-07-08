"""LeRobot adapter for two Flexiv Rizon4s arms through Flexiv RDK."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_flexiv import FlexivDualArmConfig

logger = logging.getLogger(__name__)

AXES = ("x", "y", "z", "rx", "ry", "rz")
GRIPPER_WAIT_TOLERANCE_FLOOR = 0.001


def _as_np(values: Any, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float).reshape(-1)
    count = min(length, arr.size)
    if count:
        out[:count] = arr[:count]
    return out


def _clip_norm(values: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None or limit <= 0:
        return values
    norm = float(np.linalg.norm(values))
    if norm <= limit or norm < 1e-12:
        return values
    return values * (float(limit) / norm)


def _pose7_to_pose6(pose7: Any) -> np.ndarray:
    pose = _as_np(pose7, 7)
    rotvec = np.zeros(3, dtype=float)
    quat = _rdk_quat_wxyz_to_scipy_xyzw(pose[3:7])
    if np.linalg.norm(quat) > 1e-12:
        try:
            rotvec = Rotation.from_quat(quat).as_rotvec()
        except ValueError:
            rotvec = np.zeros(3, dtype=float)
    return np.concatenate([pose[:3], rotvec])


def _apply_delta_to_pose7(current_pose7: np.ndarray, delta6: np.ndarray) -> np.ndarray:
    target = np.asarray(current_pose7, dtype=float).copy()
    target[:3] += delta6[:3]

    current_quat = _rdk_quat_wxyz_to_scipy_xyzw(target[3:7])
    if np.linalg.norm(current_quat) < 1e-12:
        current_rot = Rotation.identity()
    else:
        current_rot = Rotation.from_quat(current_quat)
    target_rot = Rotation.from_rotvec(delta6[3:]) * current_rot
    target[3:7] = _scipy_quat_xyzw_to_rdk_wxyz(target_rot.as_quat())
    return target


def _rdk_quat_wxyz_to_scipy_xyzw(quat_wxyz: Any) -> np.ndarray:
    quat = _as_np(quat_wxyz, 4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _scipy_quat_xyzw_to_rdk_wxyz(quat_xyzw: Any) -> np.ndarray:
    quat = _as_np(quat_xyzw, 4)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


class FlexivDualArm(Robot):
    """Dual Flexiv Rizon4s adapter using Flexiv RDK Cartesian servo commands."""

    config_class = FlexivDualArmConfig
    name = "flexiv_dual_arm"

    def __init__(self, config: FlexivDualArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        self._flexivrdk = None
        self._left_robot = None
        self._right_robot = None
        self._left_gripper = None
        self._right_gripper = None
        self._left_tool = None
        self._right_tool = None
        self._left_robot_lock = threading.Lock()
        self._right_robot_lock = threading.Lock()
        self._num_joints_per_arm = int(config.num_joints_per_arm)
        self._prev_observation: dict[str, Any] | None = None
        self._cached_left_pose7 = np.zeros(7, dtype=float)
        self._cached_right_pose7 = np.zeros(7, dtype=float)
        self._cached_left_pose7[3] = 1.0
        self._cached_right_pose7[3] = 1.0
        self._servo_lock = threading.Lock()
        self._servo_stop_event = threading.Event()
        self._servo_thread: threading.Thread | None = None
        self._servo_left_target_pose7 = self._cached_left_pose7.copy()
        self._servo_right_target_pose7 = self._cached_right_pose7.copy()
        self._servo_left_command_pose7 = self._cached_left_pose7.copy()
        self._servo_right_command_pose7 = self._cached_right_pose7.copy()
        self._left_gripper_cmd = 1.0
        self._right_gripper_cmd = 1.0
        self._left_gripper_width: float | None = None
        self._right_gripper_width: float | None = None
        self._left_gripper_params: dict[str, float] | None = None
        self._right_gripper_params: dict[str, float] | None = None
        self._camera_stop_event = threading.Event()
        self._camera_threads: dict[str, threading.Thread] = {}
        self._connected_cameras: set[str] = set()
        self._frame_lock = threading.Lock()
        self._latest_frames: dict[str, Any] = {}
        self._global_frame_index = 0
        self._action_debug_count = 0
        self._timing_debug_counts: dict[str, int] = {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = bool(value)

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        if not self.config.left_robot_sn or not self.config.right_robot_sn:
            raise ValueError(
                "Flexiv robot serial numbers are required. Fill "
                "`left_robot_sn` and `right_robot_sn` in "
                "scripts/config/robots/flexiv_config.yaml."
            )

        try:
            import flexivrdk  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "flexivrdk is not installed in the active Python environment. "
                "Install it with: python -m pip install flexivrdk spdlog"
            ) from exc

        self._flexivrdk = flexivrdk
        logger.info("[FLEXIV] Connecting left arm: %s", self.config.left_robot_sn)
        self._left_robot = flexivrdk.Robot(self.config.left_robot_sn)
        logger.info("[FLEXIV] Connecting right arm: %s", self.config.right_robot_sn)
        self._right_robot = flexivrdk.Robot(self.config.right_robot_sn)

        for side, robot in (("left", self._left_robot), ("right", self._right_robot)):
            self._prepare_robot(side, robot)

        if self.config.go_home_on_connect:
            self._execute_home_for_sides(("left", "right"))
            self._stop_arms_for_sides(("left", "right"))

        if self._should_setup_grippers_on_connect():
            self._prepare_grippers()
            self._initialize_grippers()
            if self.config.go_home_on_connect:
                self.move_gripper_width(self.config.gripper_max_open, side="both", wait=True)

        for side, robot in (("left", self._left_robot), ("right", self._right_robot)):
            self._finish_prepare_robot(side, robot)

        self._refresh_cached_poses()
        if self.config.camera_hardware_reset_on_connect:
            self._hardware_reset_cameras("connect")
        self._connect_cameras()
        self.is_connected = True
        self._start_cartesian_servo_thread()
        logger.info("[FLEXIV] %s connected", self.name)

    def _prepare_robot(self, side: str, robot: Any) -> None:
        if self.config.clear_fault_on_connect and robot.fault():
            logger.warning("[FLEXIV] %s arm fault detected, clearing", side)
            if not robot.ClearFault():
                raise RuntimeError(f"Failed to clear {side} Flexiv arm fault.")

        if self.config.enable_on_connect:
            logger.info("[FLEXIV] Enabling %s arm", side)
            robot.Enable()
            while not robot.operational():
                time.sleep(0.2)
            logger.info("[FLEXIV] %s arm operational", side)

    def _finish_prepare_robot(self, side: str, robot: Any) -> None:
        if self.config.zero_ft_sensor_on_connect:
            self._zero_ft_sensor(side, robot)

        if self.config.switch_cartesian_mode_on_connect and not self.config.debug:
            try:
                robot.SwitchMode(self._flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to switch {side} Flexiv arm to NRT_CARTESIAN_MOTION_FORCE. "
                    "Make sure the arm is not touching anything while ZeroFTSensor runs. "
                    "If this is only a connection/reset smoke test, set "
                    "`switch_cartesian_mode_on_connect: false` in flexiv_config.yaml."
                ) from exc
            robot.SetForceControlAxis([False, False, False, False, False, False])

    def _should_setup_grippers_on_connect(self) -> bool:
        if not self.config.use_gripper or self.config.debug:
            return False
        if self.config.reset_go_home and not self.config.go_home_on_connect:
            return False
        return True

    def _prepare_grippers(self) -> None:
        if not self.config.use_gripper or self.config.debug:
            return

        for side, robot, gripper in (
            ("left", self._left_robot, self._left_gripper),
            ("right", self._right_robot, self._right_gripper),
        ):
            if gripper is not None:
                continue
            if robot is None:
                raise DeviceNotConnectedError(f"{side} Flexiv arm is not connected.")
            self._prepare_gripper(side, robot)

    def _prepare_gripper(self, side: str, robot: Any) -> None:
        gripper_name = (
            self.config.left_gripper_name if side == "left" else self.config.right_gripper_name
        )
        tool_name = self.config.left_tool_name if side == "left" else self.config.right_tool_name
        if not gripper_name:
            raise ValueError(
                f"`{side}_gripper_name` is required when `use_gripper: true`. "
                "Find the full name in Flexiv Elements -> Settings -> Device."
            )
        if self.config.switch_tool_on_connect and not tool_name:
            raise ValueError(
                f"`{side}_tool_name` is required when `switch_tool_on_connect: true`. "
                "Use the exact robot tool/TCP name from Flexiv Elements, or set "
                "`switch_tool_on_connect: false` to keep the controller's current tool."
            )

        logger.info("[FLEXIV] Enabling %s gripper: %s", side, gripper_name)
        gripper = self._flexivrdk.Gripper(robot)
        gripper.Enable(gripper_name)

        tool = None
        if self.config.switch_tool_on_connect:
            logger.info("[FLEXIV] Switching %s arm tool to %s", side, tool_name)
            tool = self._flexivrdk.Tool(robot)
            try:
                tool.Switch(tool_name)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to switch {side} Flexiv arm tool to {tool_name!r}. "
                    f"Keep `{side}_gripper_name` as the gripper device name from Flexiv "
                    f"Elements Settings -> Device. Set `{side}_tool_name` only to the exact "
                    "robot tool/TCP name if it differs, or set `switch_tool_on_connect: false` "
                    "if the correct tool is already selected on the controller."
                ) from exc

        try:
            params = gripper.params()
            self._set_cached_gripper_params(side, params)
            width = float(gripper.states().width)
            logger.info(
                "[FLEXIV] %s gripper params min_width=%.4f max_width=%.4f min_vel=%.4f max_vel=%.4f min_force=%.2f max_force=%.2f",
                side,
                float(params.min_width),
                float(params.max_width),
                float(params.min_vel),
                float(params.max_vel),
                float(params.min_force),
                float(params.max_force),
            )
        except Exception:  # noqa: BLE001
            width = float(self.config.gripper_max_open)

        if side == "left":
            self._left_gripper = gripper
            self._left_tool = tool
            self._left_gripper_width = width
        else:
            self._right_gripper = gripper
            self._right_tool = tool
            self._right_gripper_width = width

    def _initialize_grippers(self) -> None:
        if (
            not self.config.use_gripper
            or self.config.debug
            or not self.config.initialize_gripper_on_connect
        ):
            return

        grippers = {
            side: gripper
            for side, gripper in (("left", self._left_gripper), ("right", self._right_gripper))
            if gripper is not None
        }
        if not grippers:
            return

        def make_init_call(side: str, gripper: Any) -> tuple[str, Any]:
            def init() -> None:
                logger.info("[FLEXIV] Initializing %s gripper", side)
                gripper.Init()

            return side, init

        init_calls = tuple(
            make_init_call(side, gripper)
            for side, gripper in grippers.items()
        )
        if len(init_calls) > 1:
            self._run_parallel_robot_calls(init_calls)
        else:
            init_calls[0][1]()

        self._wait_grippers_idle_after_init(grippers)

    def _execute_home_plan(self, side: str, robot: Any) -> None:
        logger.info("[FLEXIV] Moving %s arm with plan %s", side, self.config.home_plan_name)
        robot.SwitchMode(self._flexivrdk.Mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan(self.config.home_plan_name)
        while robot.busy():
            time.sleep(0.2)

    def _home_joints_for_side(self, side: str) -> list[float]:
        joints = self.config.left_home_joints if side == "left" else self.config.right_home_joints
        return [float(value) for value in joints]

    def _execute_home(self, side: str, robot: Any) -> None:
        home_joints = self._home_joints_for_side(side)
        if home_joints:
            self._execute_joint_home(side, robot, home_joints)
        else:
            self._execute_home_plan(side, robot)

    def _execute_home_for_sides(self, sides: tuple[str, ...]) -> None:
        calls: list[tuple[str, Any]] = []
        for side in sides:
            robot = self._left_robot if side == "left" else self._right_robot
            if robot is None:
                raise DeviceNotConnectedError(f"{side} Flexiv arm is not connected.")
            calls.append((side, lambda side=side, robot=robot: self._execute_home(side, robot)))

        if self.config.send_arms_parallel and len(calls) > 1:
            self._run_parallel_robot_calls(tuple(calls))
            return

        for _, call in calls:
            call()

    def _stop_arms_for_sides(self, sides: tuple[str, ...]) -> None:
        calls: list[tuple[str, Any]] = []
        for side in sides:
            robot = self._left_robot if side == "left" else self._right_robot
            if robot is None:
                continue

            def stop(side: str = side, robot: Any = robot) -> None:
                if robot.fault():
                    logger.warning("[FLEXIV] %s arm faulted; skip Stop before gripper setup", side)
                    return
                if not robot.operational():
                    logger.warning(
                        "[FLEXIV] %s arm is not operational; skip Stop before gripper setup",
                        side,
                    )
                    return
                logger.info("[FLEXIV] Stopping %s arm before gripper setup", side)
                robot.Stop()

            calls.append((side, stop))

        if self.config.send_arms_parallel and len(calls) > 1:
            self._run_parallel_robot_calls(tuple(calls))
            return

        for _, call in calls:
            call()

    def _execute_joint_home(self, side: str, robot: Any, target_joints: list[float]) -> None:
        if len(target_joints) != self._num_joints_per_arm:
            raise ValueError(
                f"`{side}_home_joints` must contain {self._num_joints_per_arm} values, "
                f"got {len(target_joints)}."
            )

        logger.info("[FLEXIV] Moving %s arm to configured joint home", side)
        robot.SwitchMode(self._flexivrdk.Mode.NRT_JOINT_POSITION)
        zeros = [0.0] * self._num_joints_per_arm
        max_vel = [float(self.config.home_joint_max_vel)] * self._num_joints_per_arm
        max_acc = [float(self.config.home_joint_max_acc)] * self._num_joints_per_arm
        robot.SendJointPosition(target_joints, zeros, max_vel, max_acc)

        deadline = time.monotonic() + max(0.1, float(self.config.home_joint_timeout_sec))
        tolerance = max(0.0, float(self.config.home_joint_tolerance))
        target = np.asarray(target_joints, dtype=float)
        while time.monotonic() < deadline:
            current = _as_np(robot.states().q, self._num_joints_per_arm)
            if float(np.max(np.abs(current - target))) <= tolerance:
                logger.info("[FLEXIV] %s arm reached configured joint home", side)
                return
            time.sleep(0.1)
        logger.warning(
            "[FLEXIV] %s arm joint home timeout after %.1fs; continuing",
            side,
            float(self.config.home_joint_timeout_sec),
        )

    def _zero_ft_sensor(self, side: str, robot: Any) -> None:
        logger.info("[FLEXIV] Zeroing %s arm force-torque sensor", side)
        robot.SwitchMode(self._flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not robot.primitive_states()["terminated"]:
            time.sleep(0.2)

    def disconnect(self) -> None:
        if (
            not self.is_connected
            and self._left_robot is None
            and self._right_robot is None
            and self._left_gripper is None
            and self._right_gripper is None
            and not self._camera_threads
            and not self._connected_cameras
        ):
            return
        self._stop_cartesian_servo_thread()
        self._stop_cameras()
        if self.config.stop_grippers_on_disconnect:
            for side, gripper in (("left", self._left_gripper), ("right", self._right_gripper)):
                if gripper is not None:
                    try:
                        gripper.Stop()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[FLEXIV] %s gripper stop failed during disconnect: %s", side, exc)
        for side, robot in (("left", self._left_robot), ("right", self._right_robot)):
            if robot is not None:
                try:
                    if robot.fault():
                        logger.warning("[FLEXIV] %s arm faulted; skip Stop during disconnect", side)
                        continue
                    if not robot.operational():
                        logger.warning(
                            "[FLEXIV] %s arm is not operational; skip Stop during disconnect",
                            side,
                        )
                        continue
                    robot.Stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[FLEXIV] %s arm stop failed during disconnect: %s", side, exc)
        self._release_handles()
        logger.info("[FLEXIV] %s disconnected", self.name)

    def release(self) -> None:
        """Release camera and RDK Python handles without commanding robot Stop()."""
        if (
            not self.is_connected
            and self._left_robot is None
            and self._right_robot is None
            and self._left_gripper is None
            and self._right_gripper is None
            and not self._camera_threads
            and not self._connected_cameras
        ):
            return
        self._stop_cartesian_servo_thread()
        self._stop_cameras()
        if self.config.camera_hardware_reset_on_release:
            self._hardware_reset_cameras("release")
        self._release_handles()
        logger.info("[FLEXIV] %s released without arm Stop", self.name)

    def _release_handles(self) -> None:
        self._left_robot = None
        self._right_robot = None
        self._left_gripper = None
        self._right_gripper = None
        self._left_tool = None
        self._right_tool = None
        self._left_gripper_params = None
        self._right_gripper_params = None
        self._flexivrdk = None
        self.is_connected = False

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        self._stop_cartesian_servo_thread()
        if self.config.reset_go_home:
            self._execute_home_for_sides(("left", "right"))
            self._stop_arms_for_sides(("left", "right"))
            self._prepare_grippers()
            self._initialize_grippers()
            self.move_gripper_width(self.config.gripper_max_open, side="both", wait=True)
            if self.config.switch_cartesian_mode_on_connect and not self.config.debug:
                self._left_robot.SwitchMode(self._flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
                self._right_robot.SwitchMode(self._flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        self._refresh_cached_poses()
        self._start_cartesian_servo_thread()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        send_start_t = time.perf_counter()
        timing: dict[str, float] = {}
        if not self.config.debug and "left_delta_ee_pose.x" in action:
            cartesian_start_t = time.perf_counter()
            self._send_cartesian_delta(action)
            timing["cartesian_ms"] = (time.perf_counter() - cartesian_start_t) * 1000.0
        elif not self.config.debug and all(
            f"left_joint_{i + 1}.pos" in action for i in range(self._num_joints_per_arm)
        ):
            joint_start_t = time.perf_counter()
            self._send_joint_positions(action)
            timing["joint_ms"] = (time.perf_counter() - joint_start_t) * 1000.0

        if self.config.use_gripper:
            gripper_start_t = time.perf_counter()
            self._update_gripper_cache(action)
            timing["gripper_ms"] = (time.perf_counter() - gripper_start_t) * 1000.0
        self._log_action_debug(action)
        timing["total_ms"] = (time.perf_counter() - send_start_t) * 1000.0
        self._log_timing_debug("send_action", timing)
        return action

    def _send_cartesian_delta(self, action: dict[str, Any]) -> None:
        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        left_delta = self._apply_mount_rotation(
            left_delta,
            self._mount_raw_deg("left"),
            self.config.left_mount_pitch_deg,
            self.config.left_mount_yaw_deg,
        )
        right_delta = self._apply_mount_rotation(
            right_delta,
            self._mount_raw_deg("right"),
            self.config.right_mount_pitch_deg,
            self.config.right_mount_yaw_deg,
        )
        left_delta[:3] = _clip_norm(left_delta[:3], self.config.max_cartesian_delta)
        right_delta[:3] = _clip_norm(right_delta[:3], self.config.max_cartesian_delta)
        left_delta[3:] = _clip_norm(left_delta[3:], self.config.max_rotation_delta)
        right_delta[3:] = _clip_norm(right_delta[3:], self.config.max_rotation_delta)

        if self.config.use_cartesian_servo_thread:
            with self._servo_lock:
                target_left = _apply_delta_to_pose7(self._servo_left_target_pose7, left_delta)
                target_right = _apply_delta_to_pose7(self._servo_right_target_pose7, right_delta)
                self._servo_left_target_pose7 = target_left
                self._servo_right_target_pose7 = target_right
            self._cached_left_pose7 = target_left
            self._cached_right_pose7 = target_right
            return

        target_left = _apply_delta_to_pose7(self._cached_left_pose7, left_delta)
        target_right = _apply_delta_to_pose7(self._cached_right_pose7, right_delta)
        self._send_cartesian_pose_targets(target_left, target_right)
        self._cached_left_pose7 = target_left
        self._cached_right_pose7 = target_right

    def _send_cartesian_pose_targets(self, target_left: np.ndarray, target_right: np.ndarray) -> None:
        zero_cartesian = [0.0] * 6
        def send_left() -> None:
            with self._left_robot_lock:
                self._left_robot.SendCartesianMotionForce(
                    target_left.tolist(),
                    zero_cartesian,
                    zero_cartesian,
                    self.config.cartesian_max_linear_vel,
                    self.config.cartesian_max_angular_vel,
                    self.config.cartesian_max_linear_acc,
                    self.config.cartesian_max_angular_acc,
                )

        def send_right() -> None:
            with self._right_robot_lock:
                self._right_robot.SendCartesianMotionForce(
                    target_right.tolist(),
                    zero_cartesian,
                    zero_cartesian,
                    self.config.cartesian_max_linear_vel,
                    self.config.cartesian_max_angular_vel,
                    self.config.cartesian_max_linear_acc,
                    self.config.cartesian_max_angular_acc,
                )

        if self.config.send_arms_parallel:
            self._run_parallel_robot_calls((("left", send_left), ("right", send_right)))
        else:
            send_left()
            send_right()

    @staticmethod
    def _run_parallel_robot_calls(calls: tuple[tuple[str, Any], ...]) -> None:
        errors: list[tuple[str, BaseException]] = []
        lock = threading.Lock()

        def runner(side: str, fn: Any) -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append((side, exc))

        threads = [
            threading.Thread(target=runner, args=(side, fn), daemon=True)
            for side, fn in calls
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            side, exc = errors[0]
            raise RuntimeError(f"{side} Flexiv command failed during parallel send") from exc

    def _log_action_debug(self, action: dict[str, Any]) -> None:
        if not self.config.action_debug:
            return

        self._action_debug_count += 1
        every_n = max(1, int(self.config.action_debug_every_n))
        if self._action_debug_count > 5 and self._action_debug_count % every_n != 0:
            return

        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        left_mapped = self._apply_mount_rotation(
            left_delta,
            self._mount_raw_deg("left"),
            self.config.left_mount_pitch_deg,
            self.config.left_mount_yaw_deg,
        )
        right_mapped = self._apply_mount_rotation(
            right_delta,
            self._mount_raw_deg("right"),
            self.config.right_mount_pitch_deg,
            self.config.right_mount_yaw_deg,
        )
        left_grip = self._gripper_value_from_action(action, "left")
        right_grip = self._gripper_value_from_action(action, "right")
        logger.info(
            "[FLEXIV ACTION] step=%d raw_left_xyz=%.6f raw_right_xyz=%.6f "
            "mapped_left_xyz=%.6f mapped_right_xyz=%.6f left_rot=%.6f right_rot=%.6f "
            "left_grip=%s right_grip=%s target_left_xyz=[%.4f %.4f %.4f] target_right_xyz=[%.4f %.4f %.4f]",
            self._action_debug_count,
            float(np.linalg.norm(left_delta[:3])),
            float(np.linalg.norm(right_delta[:3])),
            float(np.linalg.norm(left_mapped[:3])),
            float(np.linalg.norm(right_mapped[:3])),
            float(np.linalg.norm(left_mapped[3:])),
            float(np.linalg.norm(right_mapped[3:])),
            None if left_grip is None else f"{left_grip:.3f}",
            None if right_grip is None else f"{right_grip:.3f}",
            float(self._cached_left_pose7[0]),
            float(self._cached_left_pose7[1]),
            float(self._cached_left_pose7[2]),
            float(self._cached_right_pose7[0]),
            float(self._cached_right_pose7[1]),
            float(self._cached_right_pose7[2]),
        )

    def _log_timing_debug(self, stage: str, timing: dict[str, float]) -> None:
        if not self.config.timing_debug:
            return

        count = self._timing_debug_counts.get(stage, 0) + 1
        self._timing_debug_counts[stage] = count
        every_n = max(1, int(self.config.timing_debug_every_n))
        total_ms = float(timing.get("total_ms", 0.0))
        warn_ms = max(0.0, float(self.config.timing_warn_ms))
        should_log = (
            count <= 5
            or count % every_n == 0
            or (warn_ms > 0.0 and total_ms >= warn_ms)
        )
        if not should_log:
            return

        details = " ".join(
            f"{key}={value:.1f}ms"
            for key, value in sorted(timing.items())
        )
        log_fn = logger.warning if warn_ms > 0.0 and total_ms >= warn_ms else logger.info
        log_fn(
            "[FLEXIV TIMING] step=%d %s %s parallel=%s",
            count,
            stage,
            details,
            self.config.send_arms_parallel,
        )

    def _reset_servo_targets_from_cached(self) -> None:
        with self._servo_lock:
            self._servo_left_target_pose7 = self._cached_left_pose7.copy()
            self._servo_right_target_pose7 = self._cached_right_pose7.copy()
            self._servo_left_command_pose7 = self._cached_left_pose7.copy()
            self._servo_right_command_pose7 = self._cached_right_pose7.copy()

    def _start_cartesian_servo_thread(self) -> None:
        if (
            self.config.debug
            or not self.config.use_cartesian_servo_thread
            or self._left_robot is None
            or self._right_robot is None
        ):
            return
        if self._servo_thread is not None and self._servo_thread.is_alive():
            return
        self._reset_servo_targets_from_cached()
        self._servo_stop_event.clear()
        self._servo_thread = threading.Thread(
            target=self._cartesian_servo_loop,
            name="flexiv_cartesian_servo",
            daemon=True,
        )
        self._servo_thread.start()
        logger.info(
            "[FLEXIV] Cartesian servo thread started hz=%.1f alpha=%.3f",
            float(self.config.cartesian_servo_hz),
            float(self.config.cartesian_servo_alpha),
        )

    def _stop_cartesian_servo_thread(self) -> None:
        self._servo_stop_event.set()
        thread = self._servo_thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("[FLEXIV] Cartesian servo thread did not stop cleanly")
        self._servo_thread = None

    def _cartesian_servo_loop(self) -> None:
        period_s = 1.0 / max(1.0, float(self.config.cartesian_servo_hz))
        alpha = float(np.clip(self.config.cartesian_servo_alpha, 0.01, 1.0))
        count = 0
        while not self._servo_stop_event.is_set():
            loop_start_t = time.perf_counter()
            with self._servo_lock:
                left_target = self._servo_left_target_pose7.copy()
                right_target = self._servo_right_target_pose7.copy()
                self._servo_left_command_pose7 = self._blend_pose7(
                    self._servo_left_command_pose7,
                    left_target,
                    alpha,
                )
                self._servo_right_command_pose7 = self._blend_pose7(
                    self._servo_right_command_pose7,
                    right_target,
                    alpha,
                )
                left_command = self._servo_left_command_pose7.copy()
                right_command = self._servo_right_command_pose7.copy()

            try:
                send_start_t = time.perf_counter()
                self._send_cartesian_pose_targets(left_command, right_command)
                send_ms = (time.perf_counter() - send_start_t) * 1000.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FLEXIV] Cartesian servo send failed: %s", exc)
                self._servo_stop_event.wait(timeout=0.05)
                continue

            count += 1
            if self.config.timing_debug:
                total_ms = (time.perf_counter() - loop_start_t) * 1000.0
                warn_ms = max(0.0, float(self.config.timing_warn_ms))
                every_n = max(1, int(self.config.timing_debug_every_n) * 3)
                if count <= 5 or count % every_n == 0 or (warn_ms > 0.0 and total_ms >= warn_ms):
                    log_fn = logger.warning if warn_ms > 0.0 and total_ms >= warn_ms else logger.info
                    log_fn(
                        "[FLEXIV SERVO] step=%d send_ms=%.1f total_ms=%.1f hz=%.1f alpha=%.3f",
                        count,
                        send_ms,
                        total_ms,
                        float(self.config.cartesian_servo_hz),
                        alpha,
                    )

            elapsed_s = time.perf_counter() - loop_start_t
            self._servo_stop_event.wait(timeout=max(0.0, period_s - elapsed_s))

    @staticmethod
    def _blend_pose7(current_pose7: np.ndarray, target_pose7: np.ndarray, alpha: float) -> np.ndarray:
        current = np.asarray(current_pose7, dtype=float).copy()
        target = np.asarray(target_pose7, dtype=float).copy()
        out = current.copy()
        out[:3] = current[:3] + alpha * (target[:3] - current[:3])

        current_quat = _rdk_quat_wxyz_to_scipy_xyzw(current[3:7])
        target_quat = _rdk_quat_wxyz_to_scipy_xyzw(target[3:7])
        if np.linalg.norm(current_quat) < 1e-12 or np.linalg.norm(target_quat) < 1e-12:
            out[3:7] = target[3:7]
            return out

        current_rot = Rotation.from_quat(current_quat)
        target_rot = Rotation.from_quat(target_quat)
        delta_rot = target_rot * current_rot.inv()
        step_rot = Rotation.from_rotvec(alpha * delta_rot.as_rotvec()) * current_rot
        out[3:7] = _scipy_quat_xyzw_to_rdk_wxyz(step_rot.as_quat())
        return out

    def _mount_raw_deg(self, side: str) -> float:
        raw_deg = float(getattr(self.config, f"{side}_mount_raw_deg", 0.0))
        roll_deg = float(getattr(self.config, f"{side}_mount_roll_deg", 0.0))
        return raw_deg if abs(raw_deg) >= 1e-12 else roll_deg

    @staticmethod
    def _apply_mount_rotation(
        delta: np.ndarray,
        raw_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> np.ndarray:
        if (
            abs(float(raw_deg)) < 1e-12
            and abs(float(pitch_deg)) < 1e-12
            and abs(float(yaw_deg)) < 1e-12
        ):
            return delta
        # Config angles describe the physical mount intuition. In this station's
        # teleop convention, mount X/Y/Z correspond to delta-frame Z/X/Y.
        # This keeps raw/roll=+45 equivalent to the previous yaw=+45 mapping.
        mount_rot = (
            Rotation.from_euler("y", float(yaw_deg), degrees=True)
            * Rotation.from_euler("x", float(pitch_deg), degrees=True)
            * Rotation.from_euler("z", float(raw_deg), degrees=True)
        )
        out = delta.copy()
        out[:3] = mount_rot.apply(out[:3])
        out[3:] = mount_rot.apply(out[3:])
        return out

    def _send_joint_positions(self, action: dict[str, Any]) -> None:
        left_q = [float(action[f"left_joint_{i + 1}.pos"]) for i in range(self._num_joints_per_arm)]
        right_q = [float(action[f"right_joint_{i + 1}.pos"]) for i in range(self._num_joints_per_arm)]
        zeros = [0.0] * self._num_joints_per_arm
        max_vel = [2.0] * self._num_joints_per_arm
        max_acc = [3.0] * self._num_joints_per_arm
        self._left_robot.SwitchMode(self._flexivrdk.Mode.NRT_JOINT_POSITION)
        self._right_robot.SwitchMode(self._flexivrdk.Mode.NRT_JOINT_POSITION)
        self._left_robot.SendJointPosition(left_q, zeros, max_vel, max_acc)
        self._right_robot.SendJointPosition(right_q, zeros, max_vel, max_acc)

    def _update_gripper_cache(self, action: dict[str, Any]) -> None:
        left = self._gripper_value_from_action(action, "left")
        right = self._gripper_value_from_action(action, "right")
        if left is not None:
            self._left_gripper_cmd = self._normalize_gripper(float(left))
            self._move_gripper_command_if_needed("left", self._left_gripper_cmd)
        if right is not None:
            self._right_gripper_cmd = self._normalize_gripper(float(right))
            self._move_gripper_command_if_needed("right", self._right_gripper_cmd)

    @staticmethod
    def _gripper_value_from_action(action: dict[str, Any], side: str) -> Any:
        for key in (f"{side}_gripper_cmd", f"{side}_gripper_cmd_bin"):
            value = action.get(key)
            if value is not None:
                return value
        return None

    def _normalize_gripper(self, value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        if self.config.gripper_reverse:
            value = 1.0 - value
        return value

    def _gripper_width_limits(self) -> tuple[float, float]:
        min_width = float(self.config.gripper_min_width)
        max_width = max(min_width, float(self.config.gripper_max_open))
        return min_width, max_width

    def _clip_gripper_width(self, width: float) -> float:
        min_width, max_width = self._gripper_width_limits()
        return float(np.clip(float(width), min_width, max_width))

    def _gripper_width_from_cmd(self, command: float) -> float:
        min_width, max_width = self._gripper_width_limits()
        command = float(np.clip(command, 0.0, 1.0))
        width = float(
            min_width
            + command * (max_width - min_width)
        )
        return self._clip_gripper_width(width)

    def _gripper_command_from_width(self, width: float) -> float:
        min_width, max_width = self._gripper_width_limits()
        span = max_width - min_width
        if span <= 1e-12:
            return 0.0
        return float(np.clip((float(width) - min_width) / span, 0.0, 1.0))

    def _gripper_state_norm_from_width(self, width: float) -> float:
        return self._gripper_command_from_width(width)

    def _observed_gripper_width(self, side: str, gripper: Any, fallback_command: float) -> float:
        try:
            width = float(gripper.states().width)
            self._set_cached_gripper_width(side, width)
            return width
        except Exception:  # noqa: BLE001
            cached_width = self._left_gripper_width if side == "left" else self._right_gripper_width
            if cached_width is not None:
                return float(cached_width)
            return self._gripper_width_from_cmd(fallback_command)

    def move_gripper_width(self, width_m: float, side: str = "both", wait: bool = True) -> None:
        if not self.config.use_gripper:
            return

        side = side.lower().strip()
        if side == "both":
            sides = ("left", "right")
        elif side in ("left", "right"):
            sides = (side,)
        else:
            raise ValueError("side must be 'left', 'right', or 'both'.")

        pending: dict[str, tuple[Any, float, float, float]] = {}
        move_calls: list[tuple[str, Any]] = []
        tolerance = max(0.0, float(self.config.gripper_command_epsilon))

        for current_side in sides:
            gripper = self._left_gripper if current_side == "left" else self._right_gripper
            if gripper is None:
                continue

            command = self._gripper_command_from_width(width_m)
            if current_side == "left":
                self._left_gripper_cmd = command
            else:
                self._right_gripper_cmd = command

            prepared = self._prepare_gripper_move(current_side, gripper, width_m)
            if prepared is None:
                continue
            target_width, velocity, force_limit = prepared
            width, is_moving, force = self._read_gripper_state(gripper)
            if width is not None and abs(width - target_width) <= tolerance and is_moving is False:
                logger.info(
                    "[FLEXIV] %s gripper already settled near target width=%.4f target=%.4f moving=%s",
                    current_side,
                    width,
                    target_width,
                    is_moving,
                )
                self._set_cached_gripper_width(current_side, width)
                continue

            logger.info(
                "[FLEXIV] %s gripper Move width=%.4f command=%.3f velocity=%.3f force=%.1f "
                "current_width=%s moving=%s current_force=%s",
                current_side,
                target_width,
                command,
                velocity,
                force_limit,
                "unknown" if width is None else f"{width:.4f}",
                is_moving,
                "unknown" if force is None else f"{force:.2f}",
            )
            pending[current_side] = (gripper, target_width, velocity, force_limit)

            def move(
                gripper: Any = gripper,
                target_width: float = target_width,
                velocity: float = velocity,
                force_limit: float = force_limit,
            ) -> None:
                gripper.Move(target_width, velocity, force_limit)

            move_calls.append((current_side, move))

        if len(move_calls) > 1:
            self._run_parallel_robot_calls(tuple(move_calls))
        elif move_calls:
            move_calls[0][1]()

        if wait and pending:
            self._wait_grippers_width(pending)
        elif not wait:
            for current_side, (_, target_width, _, _) in pending.items():
                self._set_cached_gripper_width(current_side, target_width)

    def _move_gripper_command_if_needed(self, side: str, command: float) -> None:
        self._move_gripper_to_width_if_needed(
            side,
            self._gripper_width_from_cmd(command),
            command=command,
        )

    @staticmethod
    def _params_to_dict(params: Any) -> dict[str, float]:
        return {
            "min_width": float(params.min_width),
            "max_width": float(params.max_width),
            "min_vel": float(params.min_vel),
            "max_vel": float(params.max_vel),
            "min_force": float(params.min_force),
            "max_force": float(params.max_force),
        }

    def _set_cached_gripper_params(self, side: str, params: Any) -> dict[str, float]:
        cached = self._params_to_dict(params)
        if side == "left":
            self._left_gripper_params = cached
        else:
            self._right_gripper_params = cached
        return cached

    def _get_gripper_params(self, side: str, gripper: Any) -> dict[str, float]:
        cached = self._left_gripper_params if side == "left" else self._right_gripper_params
        if cached is not None:
            return cached
        return self._set_cached_gripper_params(side, gripper.params())

    def _prepare_gripper_move(
        self,
        side: str,
        gripper: Any,
        width: float,
    ) -> tuple[float, float, float] | None:
        min_width, _ = self._gripper_width_limits()
        requested_width = max(min_width, float(width))
        target_width = requested_width
        try:
            params = self._get_gripper_params(side, gripper)
            target_width = float(np.clip(target_width, params["min_width"], params["max_width"]))
            requested_velocity = float(self.config.gripper_speed)
            velocity = float(np.clip(requested_velocity, params["min_vel"], params["max_vel"]))
            force_limit = float(np.clip(self.config.gripper_force, params["min_force"], params["max_force"]))
            if abs(velocity - requested_velocity) >= self.config.gripper_command_epsilon:
                logger.info(
                    "[FLEXIV] %s gripper velocity %.4f clipped by RDK range [%.4f, %.4f] to %.4f",
                    side,
                    requested_velocity,
                    params["min_vel"],
                    params["max_vel"],
                    velocity,
                )
            if abs(target_width - requested_width) >= self.config.gripper_command_epsilon:
                logger.warning(
                    "[FLEXIV] %s gripper target width %.4f clipped by hardware range "
                    "[%.4f, %.4f] to %.4f",
                    side,
                    requested_width,
                    params["min_width"],
                    params["max_width"],
                    target_width,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FLEXIV] %s gripper params unavailable before Move: %s", side, exc)
            velocity = float(self.config.gripper_speed)
            force_limit = float(self.config.gripper_force)
        return target_width, velocity, force_limit

    def _move_gripper_to_width_if_needed(
        self,
        side: str,
        width: float,
        command: float | None = None,
    ) -> None:
        gripper = self._left_gripper if side == "left" else self._right_gripper
        if gripper is None:
            return

        prepared = self._prepare_gripper_move(side, gripper, width)
        if prepared is None:
            return
        target_width, velocity, force_limit = prepared

        last_width = self._left_gripper_width if side == "left" else self._right_gripper_width
        if last_width is not None and abs(target_width - last_width) < self.config.gripper_command_epsilon:
            return

        logger.info(
            "[FLEXIV] %s gripper Move width=%.4f command=%s velocity=%.3f force=%.1f",
            side,
            target_width,
            "direct" if command is None else f"{float(command):.3f}",
            velocity,
            force_limit,
        )
        gripper.Move(target_width, velocity, force_limit)
        self._set_cached_gripper_width(side, target_width)

    @staticmethod
    def _read_gripper_state(gripper: Any) -> tuple[float | None, bool | None, float | None]:
        try:
            states = gripper.states()
            return (
                float(states.width),
                bool(states.is_moving),
                float(states.force),
            )
        except Exception:  # noqa: BLE001
            return None, None, None

    def _wait_grippers_idle_after_init(self, grippers: dict[str, Any]) -> None:
        timeout_sec = max(0.0, float(self.config.gripper_init_timeout_sec))
        settle_sec = max(0.0, float(self.config.gripper_init_settle_sec))
        deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
        next_log_time = {side: time.monotonic() + 1.0 for side in grippers}
        stable_since: dict[str, float | None] = {side: None for side in grippers}
        last_states: dict[str, tuple[float | None, bool | None, float | None]] = {
            side: (None, None, None) for side in grippers
        }
        settled: set[str] = set()

        logger.info(
            "[FLEXIV] Waiting for gripper init to settle sides=%s timeout=%.1fs settle=%.1fs",
            ",".join(grippers),
            timeout_sec,
            settle_sec,
        )
        while len(settled) < len(grippers):
            now = time.monotonic()
            for side, gripper in grippers.items():
                if side in settled:
                    continue

                width, is_moving, force = self._read_gripper_state(gripper)
                last_states[side] = (width, is_moving, force)

                if is_moving is False:
                    if stable_since[side] is None:
                        stable_since[side] = now
                    if now - stable_since[side] >= settle_sec:
                        logger.info(
                            "[FLEXIV] %s gripper init settled width=%s moving=%s force=%s",
                            side,
                            "unknown" if width is None else f"{width:.4f}",
                            is_moving,
                            "unknown" if force is None else f"{force:.2f}",
                        )
                        if width is not None:
                            self._set_cached_gripper_width(side, width)
                        settled.add(side)
                        continue
                else:
                    stable_since[side] = None

                if now >= next_log_time[side]:
                    logger.info(
                        "[FLEXIV] waiting for %s gripper init width=%s moving=%s force=%s",
                        side,
                        "unknown" if width is None else f"{width:.4f}",
                        is_moving,
                        "unknown" if force is None else f"{force:.2f}",
                    )
                    next_log_time[side] = now + 1.0

            if deadline is not None and now >= deadline:
                break

            time.sleep(0.05)

        if len(settled) == len(grippers):
            return

        pending_desc = ", ".join(
            f"{side}: width={'unknown' if width is None else f'{width:.4f}'} "
            f"moving={is_moving} force={'unknown' if force is None else f'{force:.2f}'}"
            for side, (width, is_moving, force) in last_states.items()
            if side not in settled
        )
        raise TimeoutError(
            f"gripper init did not settle within {timeout_sec:.1f}s: {pending_desc}"
        )

    def _wait_grippers_width(
        self,
        targets: dict[str, tuple[Any, float, float, float]],
    ) -> None:
        timeout_sec = max(0.0, float(self.config.gripper_init_timeout_sec))
        retry_interval_sec = max(1.0, float(self.config.gripper_init_settle_sec))
        tolerance = max(
            GRIPPER_WAIT_TOLERANCE_FLOOR,
            float(self.config.gripper_command_epsilon),
        )
        settle_sec = max(0.0, float(self.config.gripper_init_settle_sec))
        deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
        next_log_time = {side: time.monotonic() + 1.0 for side in targets}
        next_retry_time = {side: time.monotonic() + retry_interval_sec for side in targets}
        last_widths: dict[str, float | None] = {side: None for side in targets}
        stable_since: dict[str, float | None] = {side: None for side in targets}
        reached: set[str] = set()

        logger.info(
            "[FLEXIV] Waiting for gripper home sides=%s timeout=%.1fs tolerance=%.4f settle=%.1fs",
            ",".join(targets),
            timeout_sec,
            tolerance,
            settle_sec,
        )
        while len(reached) < len(targets):
            now = time.monotonic()
            for side, (gripper, target_width, velocity, force_limit) in targets.items():
                if side in reached:
                    continue

                width, is_moving, force = self._read_gripper_state(gripper)
                if width is not None:
                    last_widths[side] = width
                    near_target = abs(width - target_width) <= tolerance
                    settled = near_target and is_moving is False
                    if settled:
                        if stable_since[side] is None:
                            stable_since[side] = now
                        if now - stable_since[side] >= settle_sec:
                            logger.info(
                                "[FLEXIV] %s gripper settled at width=%.4f target=%.4f moving=%s",
                                side,
                                width,
                                target_width,
                                is_moving,
                            )
                            self._set_cached_gripper_width(side, width)
                            reached.add(side)
                            continue
                    else:
                        stable_since[side] = None

                    if near_target and is_moving is not False and now >= next_log_time[side]:
                        logger.info(
                            "[FLEXIV] %s gripper width near target but still moving width=%.4f target=%.4f moving=%s",
                            side,
                            width,
                            target_width,
                            is_moving,
                        )
                        next_log_time[side] = now + 1.0

                if (
                    is_moving is False
                    and width is not None
                    and abs(width - target_width) > tolerance
                    and now >= next_retry_time[side]
                ):
                    logger.warning(
                        "[FLEXIV] %s gripper stopped before home; retry Move "
                        "target=%.4f width=%.4f force=%s",
                        side,
                        target_width,
                        width,
                        "unknown" if force is None else f"{force:.2f}",
                    )
                    gripper.Move(target_width, velocity, force_limit)
                    next_retry_time[side] = now + retry_interval_sec

                if now >= next_log_time[side]:
                    logger.info(
                        "[FLEXIV] waiting for %s gripper home target=%.4f width=%s moving=%s force=%s",
                        side,
                        target_width,
                        "unknown" if width is None else f"{width:.4f}",
                        is_moving,
                        "unknown" if force is None else f"{force:.2f}",
                    )
                    next_log_time[side] = now + 1.0

            if deadline is not None and now >= deadline:
                break

            time.sleep(0.05)

        if len(reached) == len(targets):
            return

        pending = [
            (
                side,
                targets[side][1],
                last_widths.get(side),
            )
            for side in targets
            if side not in reached
        ]
        pending_desc = ", ".join(
            f"{side}: target={target_width:.4f} last="
            f"{'unknown' if width is None else f'{width:.4f}'}"
            for side, target_width, width in pending
        )
        logger.warning("[FLEXIV] gripper home wait timeout after %.1fs: %s", timeout_sec, pending_desc)
        raise TimeoutError(
            f"gripper home wait timeout after {timeout_sec:.1f}s: {pending_desc}"
        )

    def _set_cached_gripper_width(self, side: str, width: float) -> None:
        if side == "left":
            self._left_gripper_width = width
        else:
            self._right_gripper_width = width

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        try:
            obs_start_t = time.perf_counter()
            timing: dict[str, float] = {}
            obs = {}
            left_start_t = time.perf_counter()
            self._add_arm_observation(obs, "left", self._left_robot)
            timing["left_state_ms"] = (time.perf_counter() - left_start_t) * 1000.0
            right_start_t = time.perf_counter()
            self._add_arm_observation(obs, "right", self._right_robot)
            timing["right_state_ms"] = (time.perf_counter() - right_start_t) * 1000.0
            if self.config.save_rgbd_timestamps:
                obs["global_frame_index"] = self._next_global_frame_index()
                obs["robot_timestamp"] = time.time()
            camera_start_t = time.perf_counter()
            self._add_camera_observations(obs)
            timing["camera_ms"] = (time.perf_counter() - camera_start_t) * 1000.0
            timing["total_ms"] = (time.perf_counter() - obs_start_t) * 1000.0
            self._log_timing_debug("get_observation", timing)
            self._prev_observation = obs
            return obs
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FLEXIV] get_observation failed: %s", exc)
            if self._prev_observation is not None:
                return self._mark_reused_observation(dict(self._prev_observation))
            raise

    def _next_global_frame_index(self) -> int:
        frame_index = self._global_frame_index
        self._global_frame_index += 1
        return frame_index

    def _mark_reused_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self.config.save_rgbd_timestamps:
            return obs
        obs["global_frame_index"] = self._next_global_frame_index()
        obs["robot_timestamp"] = time.time()
        for cam_name in self.cameras:
            base_name = self._camera_base_name(cam_name)
            obs[f"{base_name}_rgbd_reused"] = True
        return obs

    def _add_arm_observation(self, obs: dict[str, Any], side: str, robot: Any) -> None:
        robot_lock = self._left_robot_lock if side == "left" else self._right_robot_lock
        with robot_lock:
            states = robot.states()
        joints = _as_np(getattr(states, "q", None), self._num_joints_per_arm)
        pose7 = _as_np(getattr(states, "tcp_pose", None), 7)
        if np.linalg.norm(pose7[3:7]) < 1e-12:
            pose7[3] = 1.0
        pose6 = _pose7_to_pose6(pose7)

        if side == "left":
            self._cached_left_pose7 = pose7.copy()
        else:
            self._cached_right_pose7 = pose7.copy()

        for index, value in enumerate(joints, start=1):
            obs[f"{side}_joint_{index}.pos"] = float(value)
        for index, axis in enumerate(AXES):
            obs[f"{side}_ee_pose.{axis}"] = float(pose6[index])

        if self.config.use_gripper:
            cmd = self._left_gripper_cmd if side == "left" else self._right_gripper_cmd
            gripper = self._left_gripper if side == "left" else self._right_gripper
            if gripper is not None:
                width = self._observed_gripper_width(side, gripper, cmd)
            else:
                width = self._gripper_width_from_cmd(cmd)
            obs[f"{side}_gripper_state_norm"] = self._gripper_state_norm_from_width(width)

    def _refresh_cached_poses(self) -> None:
        if self._left_robot is not None:
            with self._left_robot_lock:
                self._cached_left_pose7 = _as_np(self._left_robot.states().tcp_pose, 7)
        if self._right_robot is not None:
            with self._right_robot_lock:
                self._cached_right_pose7 = _as_np(self._right_robot.states().tcp_pose, 7)
        self._reset_servo_targets_from_cached()

    def _connect_cameras(self) -> None:
        if not self.cameras:
            return
        self._camera_stop_event.clear()
        warmed_cameras: list[tuple[str, Any]] = []
        try:
            for cam_name, cam in self.cameras.items():
                # LeRobot's default RealSense connect() warmup reads with a fixed
                # 200 ms timeout, which is too tight when three D435 pipelines start
                # together. Disable that warmup and use the configurable one below.
                cam.connect(warmup=False)
                self._connected_cameras.add(cam_name)
                self._warmup_camera(cam_name, cam)
                warmed_cameras.append((cam_name, cam))
                logger.info("[CAM] %s warmed up", cam_name)

            for cam_name, cam in warmed_cameras:
                thread = threading.Thread(
                    target=self._camera_read_loop,
                    args=(cam_name, cam),
                    name=f"flexiv_cam_{cam_name}",
                    daemon=True,
                )
                thread.start()
                self._camera_threads[cam_name] = thread
                logger.info("[CAM] %s connected", cam_name)
        except Exception:
            logger.warning("[CAM] Camera startup failed; closing opened RealSense pipelines")
            self._stop_cameras()
            raise

    def _configured_camera_serials(self) -> set[str]:
        serials: set[str] = set()
        for camera_cfg in self.config.cameras.values():
            serial = str(getattr(camera_cfg, "serial_number_or_name", "") or "").strip()
            if serial:
                serials.add(serial)
        return serials

    def _hardware_reset_cameras(self, reason: str) -> None:
        target_serials = self._configured_camera_serials()
        if not target_serials:
            return
        try:
            import pyrealsense2 as rs  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("[CAM] skip hardware reset on %s; pyrealsense2 import failed: %s", reason, exc)
            return

        try:
            devices = list(rs.context().query_devices())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CAM] skip hardware reset on %s; query_devices failed: %s", reason, exc)
            return

        reset_serials: set[str] = set()
        for device in devices:
            try:
                serial = str(device.get_info(rs.camera_info.serial_number))
            except Exception:  # noqa: BLE001
                serial = ""
            if serial not in target_serials:
                continue
            try:
                logger.info("[CAM] hardware reset %s on %s", serial, reason)
                device.hardware_reset()
                reset_serials.add(serial)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CAM] hardware reset failed for %s on %s: %s", serial, reason, exc)

        missing_serials = sorted(target_serials - reset_serials)
        if missing_serials:
            logger.warning(
                "[CAM] hardware reset on %s could not see configured camera serial(s): %s. "
                "If a camera is wedged in USB/UVC, run tools-reset-rs --mode sysfs --all with sudo.",
                reason,
                missing_serials,
            )
        if not reset_serials:
            return

        time.sleep(max(float(self.config.camera_reset_settle_sec), 0.0))
        self._wait_for_realsense_serials(reset_serials, timeout_sec=self.config.camera_reset_timeout_sec)

    @staticmethod
    def _wait_for_realsense_serials(serials: set[str], timeout_sec: float) -> None:
        try:
            import pyrealsense2 as rs  # noqa: PLC0415
        except ImportError:
            return
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while time.monotonic() < deadline:
            try:
                current = {
                    str(device.get_info(rs.camera_info.serial_number))
                    for device in rs.context().query_devices()
                }
            except Exception:  # noqa: BLE001
                current = set()
            if serials.issubset(current):
                return
            time.sleep(0.5)
        logger.warning("[CAM] timed out waiting for RealSense serial(s) after reset: %s", sorted(serials))

    def _warmup_camera(self, cam_name: str, cam: Any) -> None:
        attempts = max(int(self.config.camera_warmup_attempts), 1)
        timeout_ms = max(int(self.config.camera_read_timeout_ms), 200)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                frame = self._capture_camera_frame(cam)
                with self._frame_lock:
                    self._latest_frames[cam_name] = frame
                if attempt > 1:
                    logger.info("[CAM] %s warmed up after %d attempts", cam_name, attempt)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.1)
        raise RuntimeError(
            f"Camera {cam_name} did not produce a frame after {attempts} warmup "
            f"attempts with timeout_ms={timeout_ms}. Close realsense-viewer and "
            "other camera users, then replug this RealSense if needed."
        ) from last_error

    def stop_cameras(self) -> None:
        """Stop camera reader threads and release RealSense pipelines."""
        self._stop_cameras()

    def _stop_cameras(self) -> None:
        if not self.cameras:
            return
        self._camera_stop_event.set()
        join_timeout = max(float(self.config.camera_read_timeout_ms) / 1000.0 + 1.0, 2.0)
        for cam_name, thread in list(self._camera_threads.items()):
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                logger.warning("[CAM] %s thread did not stop cleanly", cam_name)
        self._camera_threads.clear()
        with self._frame_lock:
            self._latest_frames.clear()
        for cam_name, cam in self.cameras.items():
            is_connected_attr = getattr(cam, "is_connected", False)
            is_connected = bool(is_connected_attr() if callable(is_connected_attr) else is_connected_attr)
            if cam_name not in self._connected_cameras and not is_connected:
                continue
            try:
                cam.disconnect()
                logger.info("[CAM] %s disconnected", cam_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CAM] %s disconnect failed: %s", cam_name, exc)
            finally:
                self._connected_cameras.discard(cam_name)

    def _camera_read_loop(self, cam_name: str, cam: Any) -> None:
        timeout_ms = max(int(self.config.camera_read_timeout_ms), 200)
        while not self._camera_stop_event.is_set():
            try:
                frame = self._capture_camera_frame(cam, timeout_ms=timeout_ms)
                with self._frame_lock:
                    self._latest_frames[cam_name] = frame
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CAM] %s read failed: %s", cam_name, exc)
                self._mark_latest_camera_frame_reused(cam_name)
                self._camera_stop_event.wait(timeout=0.1)

    def _capture_camera_frame(self, cam: Any, timeout_ms: int | None = None) -> Any:
        timeout = max(int(timeout_ms or self.config.camera_read_timeout_ms), 200)
        needs_rgbd_frame = self.config.save_depth_sidecar or self.config.save_ir_sidecar or self.config.save_rgbd_timestamps
        if needs_rgbd_frame and hasattr(cam, "read_rgbd_ir"):
            return cam.read_rgbd_ir(timeout_ms=timeout)
        return cam.read(timeout_ms=timeout)

    def _mark_latest_camera_frame_reused(self, cam_name: str) -> None:
        with self._frame_lock:
            frame = self._latest_frames.get(cam_name)
            if isinstance(frame, dict):
                reused = dict(frame)
                reused["reused"] = True
                self._latest_frames[cam_name] = reused

    def _add_camera_observations(self, obs: dict[str, Any]) -> None:
        if not self.cameras:
            return
        with self._frame_lock:
            latest_frames = {cam_name: self._latest_frames.get(cam_name) for cam_name in self.cameras}
        for cam_name, cam in self.cameras.items():
            frame = latest_frames.get(cam_name)
            if frame is None:
                frame = self._capture_camera_frame(cam)
            self._add_camera_frame_observation(obs, cam_name, frame)

    @staticmethod
    def _camera_base_name(cam_name: str) -> str:
        if cam_name.endswith("_rgb"):
            return cam_name.removesuffix("_rgb")
        if cam_name.endswith("_image"):
            return cam_name.removesuffix("_image")
        return cam_name

    def _add_camera_frame_observation(self, obs: dict[str, Any], cam_name: str, frame: Any) -> None:
        if not isinstance(frame, dict):
            obs[cam_name] = frame
            return

        base_name = self._camera_base_name(cam_name)
        obs[cam_name] = frame["rgb"]

        if self.config.save_depth_sidecar:
            if frame.get("depth") is None:
                raise RuntimeError(f"Camera {cam_name} did not provide a depth frame.")
            obs[f"sidecar.{base_name}_depth"] = frame["depth"]

        if self.config.save_ir_sidecar:
            if frame.get("left_ir") is None or frame.get("right_ir") is None:
                raise RuntimeError(f"Camera {cam_name} did not provide both IR frames.")
            obs[f"sidecar.{base_name}_left_ir"] = frame["left_ir"]
            obs[f"sidecar.{base_name}_right_ir"] = frame["right_ir"]

        if self.config.save_rgbd_timestamps:
            obs[f"{base_name}_rgbd_timestamp"] = float(frame["timestamp"])
            obs[f"{base_name}_rgbd_reused"] = bool(frame.get("reused", False))

    @property
    def _motors_ft(self) -> dict[str, type]:
        features = {}
        for side in ("left", "right"):
            for index in range(self._num_joints_per_arm):
                features[f"{side}_joint_{index + 1}.pos"] = float
            for axis in AXES:
                features[f"{side}_ee_pose.{axis}"] = float
            if self.config.use_gripper:
                features[f"{side}_gripper_state_norm"] = float
        return features

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        if self.config.control_mode == "oculus":
            for side in ("left", "right"):
                for axis in AXES:
                    features[f"{side}_delta_ee_pose.{axis}"] = float
        else:
            for side in ("left", "right"):
                for index in range(self._num_joints_per_arm):
                    features[f"{side}_joint_{index + 1}.pos"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd"] = float
            features["right_gripper_cmd"] = float
        return features

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def dataset_extra_features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {}
        for cam_name, cam in self.cameras.items():
            base_name = self._camera_base_name(cam_name)
            shape = (int(cam.height), int(cam.width))
            if self.config.save_depth_sidecar:
                features[f"sidecar.{base_name}_depth"] = {
                    "dtype": "uint16",
                    "shape": shape,
                    "names": ["height", "width"],
                }
            if self.config.save_ir_sidecar:
                features[f"sidecar.{base_name}_left_ir"] = {
                    "dtype": "uint8",
                    "shape": shape,
                    "names": ["height", "width"],
                }
                features[f"sidecar.{base_name}_right_ir"] = {
                    "dtype": "uint8",
                    "shape": shape,
                    "names": ["height", "width"],
                }
            if self.config.save_rgbd_timestamps:
                features[f"{base_name}_rgbd_timestamp"] = {
                    "dtype": "float64",
                    "shape": (1,),
                    "names": None,
                }
                features[f"{base_name}_rgbd_reused"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }

        if self.config.save_rgbd_timestamps:
            features["global_frame_index"] = {"dtype": "int64", "shape": (1,), "names": None}
            features["robot_timestamp"] = {"dtype": "float64", "shape": (1,), "names": None}

        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    def calibrate(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def configure(self) -> None:
        pass
