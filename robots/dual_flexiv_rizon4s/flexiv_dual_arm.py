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

from robots.dual_arm_schema import (
    AXES,
    action_features as x_embodiment_action_features,
    clip_gripper_width,
    finite_gripper_value,
    observation_features as x_embodiment_observation_features,
    width_from_normalized_command,
)
from .config_flexiv import FlexivDualArmConfig
from .gripper_aperture import GripperApertureController
from .pose_frames import BaseToWorldTransform

logger = logging.getLogger(__name__)

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
        if config.gripper_control_mode not in {"move", "force_feedback"}:
            raise ValueError("gripper_control_mode must be 'move' or 'force_feedback'.")
        for field in ("gripper_feedback_force_limit", "gripper_feedback_timeout_sec"):
            if not np.isfinite(getattr(config, field)) or getattr(config, field) <= 0:
                raise ValueError(f"{field} must be positive and finite.")
        self._gripper_feedback: dict[str, GripperApertureController] = {}
        self._gripper_feedback_lock = threading.Lock()
        if (not np.isfinite(config.gripper_min_command_interval_sec)
                or config.gripper_min_command_interval_sec < 0):
            raise ValueError("gripper_min_command_interval_sec must be finite and nonnegative.")
        if (config.gripper_continuous_speed is not None
                and (not np.isfinite(config.gripper_continuous_speed)
                     or config.gripper_continuous_speed <= 0)):
            raise ValueError("gripper_continuous_speed must be positive and finite, or None.")
        if config.gripper_init_strategy not in {"always", "if_needed"}:
            raise ValueError("gripper_init_strategy must be 'always' or 'if_needed'.")
        if not np.isfinite(config.gripper_ready_timeout_sec) or config.gripper_ready_timeout_sec <= 0:
            raise ValueError("gripper_ready_timeout_sec must be positive and finite.")
        if config.observation_pose_frame not in {"base", "world"}:
            raise ValueError("observation_pose_frame must be 'base' or 'world'.")
        self._observation_transforms: dict[str, BaseToWorldTransform] = {}
        if config.observation_pose_frame == "world":
            for side in ("left", "right"):
                field_name = f"{side}_base_pose_in_world"
                extrinsic = getattr(config, field_name)
                if extrinsic is None:
                    raise ValueError(
                        f"{field_name} is required for observation_pose_frame='world'. "
                        "Provide T_world_base as [x, y, z, qw, qx, qy, qz]."
                    )
                self._observation_transforms[side] = BaseToWorldTransform(extrinsic, name=field_name)
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
        self._actual_left_pose7: np.ndarray | None = None
        self._actual_right_pose7: np.ndarray | None = None
        self._servo_lock = threading.Lock()
        self._servo_stop_event = threading.Event()
        self._servo_thread: threading.Thread | None = None
        self._servo_error: Exception | None = None
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
        self._frame_lock = threading.Lock()
        self._latest_frames: dict[str, Any] = {}
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
        self._left_robot = flexivrdk.Robot(
            self.config.left_robot_sn, self.config.network_interface_whitelist
        )
        logger.info("[FLEXIV] Connecting right arm: %s", self.config.right_robot_sn)
        self._right_robot = flexivrdk.Robot(
            self.config.right_robot_sn, self.config.network_interface_whitelist
        )

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
        self._connect_cameras()
        self.is_connected = True
        self._start_cartesian_servo_thread()
        logger.info("[FLEXIV] %s connected", self.name)

    def _prepare_robot(self, side: str, robot: Any) -> None:
        timeout = float(self.config.enable_timeout_sec)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("enable_timeout_sec must be positive and finite.")
        if self.config.clear_fault_on_connect and robot.fault():
            logger.warning("[FLEXIV] %s arm fault detected, clearing", side)
            if not robot.ClearFault():
                raise RuntimeError(f"Failed to clear {side} Flexiv arm fault.")

        if self.config.enable_on_connect:
            if robot.operational():
                logger.info("[FLEXIV] %s arm already operational; skip redundant Enable", side)
                return
            logger.info("[FLEXIV] Enabling %s arm", side)
            try:
                robot.Enable()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to enable {side} Flexiv arm: {exc}. "
                    f"Status after failure: {self._connection_status(robot)}. "
                    "The enable request may have reached the controller even if its reply was lost. "
                    "No automatic retry was sent. Check the robot Ethernet link, ARP/IP conflicts, "
                    "and the controller's Remote/Ethernet mode before retrying."
                ) from exc
            deadline = time.monotonic() + timeout
            while not robot.operational():
                status = self._connection_status(robot)
                if status.get("connected") is False or status.get("fault") is True:
                    raise RuntimeError(f"{side} Flexiv arm lost readiness while enabling: {status}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{side} Flexiv arm did not become operational within {timeout:.1f}s "
                        f"after Enable returned: {status}"
                    )
                time.sleep(0.2)
            logger.info("[FLEXIV] %s arm operational", side)

    @staticmethod
    def _connection_status(robot: Any) -> dict[str, Any]:
        status = {}
        for name in ("connected", "operational", "operational_status", "fault",
                     "estop_released", "recovery", "reduced", "mode"):
            method = getattr(robot, name, None)
            if not callable(method):
                continue
            try:
                value = method()
                status[name] = value if isinstance(value, (bool, int, float)) else str(value)
            except Exception as exc:  # Status reporting must not mask the original failure.
                status[name] = f"unavailable: {exc}"
        return status

    def _finish_prepare_robot(self, side: str, robot: Any) -> None:
        if self.config.debug:
            return
        # The 6DoF F/T controller requires calibration before entering this
        # mode, even when all force-control axes will subsequently be disabled.
        # Run after tool/gripper setup, which may invalidate calibration.
        if self.config.zero_ft_sensor_on_connect or self.config.switch_cartesian_mode_on_connect:
            self._zero_ft_sensor(side, robot)

        if self.config.switch_cartesian_mode_on_connect:
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

        if self.config.gripper_init_strategy == "if_needed":
            grippers = self._grippers_requiring_init(grippers)
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

    def _grippers_requiring_init(self, grippers: dict[str, Any]) -> dict[str, Any]:
        """Verify normal opening before repeating GN01's slow calibration.

        RDK 1.9 does not expose an initialized flag. A measured, stopped gripper
        at the opening target is sufficient for this startup operation. Never
        infer readiness from commanded-width caches or a file from a prior run.
        Delivery failures propagate; only a delivered move that fails to reach
        its target falls back to Init (as on an uncalibrated GN01 after power-on).
        """
        tolerance = max(GRIPPER_WAIT_TOLERANCE_FLOOR, self.config.gripper_command_epsilon)
        pending: dict[str, tuple[Any, float]] = {}
        calls: list[tuple[str, Any]] = []
        for side, gripper in grippers.items():
            target, velocity, force = self._prepare_gripper_move(
                side, gripper, self.config.gripper_max_open,
            )
            width, moving, _ = self._read_gripper_state(gripper)
            if width is not None and abs(width - target) <= tolerance and moving is False:
                self._set_cached_gripper_width(side, width)
                logger.info("[FLEXIV] %s gripper already open; skip redundant Init", side)
                continue
            pending[side] = (gripper, target)
            calls.append((side, lambda g=gripper, t=target, v=velocity, f=force: g.Move(t, v, f)))
        if calls:
            self._run_parallel_robot_calls(tuple(calls))
        deadline = time.monotonic() + self.config.gripper_ready_timeout_sec
        while pending:
            for side, (gripper, target) in list(pending.items()):
                width, moving, _ = self._read_gripper_state(gripper)
                if width is not None and abs(width - target) <= tolerance and moving is False:
                    self._set_cached_gripper_width(side, width)
                    logger.info("[FLEXIV] %s gripper opened normally; skip redundant Init", side)
                    del pending[side]
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        return {side: gripper for side, (gripper, _) in pending.items()}

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
        try:
            self._close_gripper_feedback()
        except Exception as exc:
            logger.warning("[FLEXIV] aperture feedback stop failed during disconnect: %s", exc)
        if (
            not self.is_connected
            and self._left_robot is None
            and self._right_robot is None
            and self._left_gripper is None
            and self._right_gripper is None
        ):
            return
        self._stop_cartesian_servo_thread()
        if self._camera_threads:
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
        self._left_robot = None
        self._right_robot = None
        self._left_gripper = None
        self._right_gripper = None
        self._left_tool = None
        self._right_tool = None
        self._left_gripper_params = None
        self._right_gripper_params = None
        self.is_connected = False
        logger.info("[FLEXIV] %s disconnected", self.name)

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        self._close_gripper_feedback()
        self._stop_cartesian_servo_thread()
        if self.config.reset_go_home:
            self._execute_home_for_sides(("left", "right"))
            self._stop_arms_for_sides(("left", "right"))
            self._prepare_grippers()
            self._initialize_grippers()
            self.move_gripper_width(self.config.gripper_max_open, side="both", wait=True)
            if self.config.switch_cartesian_mode_on_connect and not self.config.debug:
                for side, robot in (("left", self._left_robot), ("right", self._right_robot)):
                    self._finish_prepare_robot(side, robot)
        self._refresh_cached_poses()
        self._start_cartesian_servo_thread()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        self._check_cartesian_servo_error()

        sent_action = dict(action)
        send_start_t = time.perf_counter()
        timing: dict[str, float] = {}
        has_cartesian = any(f"{side}_delta_ee_pose.{axis}" in action for side in ("left", "right") for axis in AXES)
        if self.config.debug and has_cartesian:
            for side in ("left", "right"):
                for axis in AXES:
                    sent_action[f"{side}_delta_ee_pose.{axis}"] = 0.0
        elif has_cartesian:
            cartesian_start_t = time.perf_counter()
            self._send_cartesian_delta(sent_action)
            timing["cartesian_ms"] = (time.perf_counter() - cartesian_start_t) * 1000.0
        elif not self.config.debug and all(
            f"left_joint_{i + 1}.pos" in sent_action for i in range(self._num_joints_per_arm)
        ):
            joint_start_t = time.perf_counter()
            self._send_joint_positions(sent_action)
            timing["joint_ms"] = (time.perf_counter() - joint_start_t) * 1000.0

        if self.config.use_gripper:
            gripper_start_t = time.perf_counter()
            self._update_gripper_cache(sent_action)
            timing["gripper_ms"] = (time.perf_counter() - gripper_start_t) * 1000.0
        self._log_action_debug(sent_action)
        timing["total_ms"] = (time.perf_counter() - send_start_t) * 1000.0
        self._log_timing_debug("send_action", timing)
        return sent_action

    def _send_cartesian_delta(self, action: dict[str, Any]) -> None:
        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        if not np.all(np.isfinite(left_delta)) or not np.all(np.isfinite(right_delta)):
            raise ValueError("Flexiv Cartesian actions must be finite")
        left_delta = self._apply_mount_yaw(left_delta, self.config.left_mount_yaw_deg)
        right_delta = self._apply_mount_yaw(right_delta, self.config.right_mount_yaw_deg)
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
        else:
            target_left = _apply_delta_to_pose7(self._cached_left_pose7, left_delta)
            target_right = _apply_delta_to_pose7(self._cached_right_pose7, right_delta)
            self._send_cartesian_pose_targets(target_left, target_right)
            self._cached_left_pose7 = target_left
            self._cached_right_pose7 = target_right

        # Labels describe the effective target increment, not the low-level
        # servo's interpolation substeps. Undo only the hardware basis change;
        # otherwise replay would apply mount_yaw twice.
        for side, base_delta in (("left", left_delta), ("right", right_delta)):
            public_delta = self._apply_mount_yaw(base_delta, -getattr(self.config, f"{side}_mount_yaw_deg"))
            for index, axis in enumerate(AXES):
                action[f"{side}_delta_ee_pose.{axis}"] = float(public_delta[index])

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
            raise RuntimeError(f"{side} Flexiv command failed during parallel send: {exc}") from exc

    def _log_action_debug(self, action: dict[str, Any]) -> None:
        if not self.config.action_debug:
            return

        self._action_debug_count += 1
        every_n = max(1, int(self.config.action_debug_every_n))
        if self._action_debug_count > 5 and self._action_debug_count % every_n != 0:
            return

        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in AXES], dtype=float)
        left_mapped = self._apply_mount_yaw(left_delta, self.config.left_mount_yaw_deg)
        right_mapped = self._apply_mount_yaw(right_delta, self.config.right_mount_yaw_deg)
        left_grip = self._gripper_width_from_action(action, "left")
        right_grip = self._gripper_width_from_action(action, "right")
        left_error = (float(np.linalg.norm(self._cached_left_pose7[:3] - self._actual_left_pose7[:3]))
                      if self._actual_left_pose7 is not None else None)
        right_error = (float(np.linalg.norm(self._cached_right_pose7[:3] - self._actual_right_pose7[:3]))
                       if self._actual_right_pose7 is not None else None)
        logger.info(
            "[FLEXIV ACTION] step=%d raw_left_xyz=%.6f raw_right_xyz=%.6f "
            "mapped_left_xyz=%.6f mapped_right_xyz=%.6f left_rot=%.6f right_rot=%.6f "
            "left_grip=%s right_grip=%s target_left_xyz=[%.4f %.4f %.4f] target_right_xyz=[%.4f %.4f %.4f] "
            "left_tracking_error_m=%s right_tracking_error_m=%s servo_alive=%s",
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
            None if left_error is None else f"{left_error:.4f}",
            None if right_error is None else f"{right_error:.4f}",
            self._servo_thread is not None and self._servo_thread.is_alive(),
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
        self._servo_error = None
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
                # Surface a failed hardware send to the recording thread. Do not
                # keep accepting goals while one arm has stopped following.
                self._servo_error = exc
                self._servo_stop_event.set()
                logger.exception("[FLEXIV] Cartesian servo stopped after send failure: %s", exc)
                return

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

    def _check_cartesian_servo_error(self) -> None:
        if self.config.use_cartesian_servo_thread and self._servo_error is not None:
            raise RuntimeError(f"Flexiv Cartesian servo stopped: {self._servo_error}") from self._servo_error

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

    @staticmethod
    def _apply_mount_yaw(delta: np.ndarray, yaw_deg: float) -> np.ndarray:
        if abs(float(yaw_deg)) < 1e-12:
            return delta
        rot_z = Rotation.from_euler("z", float(yaw_deg), degrees=True)
        out = delta.copy()
        out[:3] = rot_z.apply(out[:3])
        out[3:] = rot_z.apply(out[3:])
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
        if not self.config.use_gripper:
            return
        calls: list[tuple[str, Any]] = []
        for side in ("left", "right"):
            width = self._gripper_width_from_action(action, side)
            if width is None:
                continue

            def send(side: str = side, width: float = width) -> None:
                target = self._move_gripper_to_width_if_needed(side, width)
                setattr(self, f"_{side}_gripper_cmd", self._gripper_command_from_width(target))
                action[f"{side}_gripper_width"] = target

            calls.append((side, send))
        # Move blocks for delivery, not finger arrival. Overlap controllers but
        # join before returning so labels remain delivered, clipped targets.
        if self.config.send_arms_parallel and len(calls) > 1:
            self._run_parallel_robot_calls(tuple(calls))
        else:
            for _, send in calls:
                send()

    def _gripper_width_from_action(self, action: dict[str, Any], side: str) -> float | None:
        width = action.get(f"{side}_gripper_width")
        if width is not None:
            return self._clip_gripper_width(float(width))

        # Legacy normalized aliases are still accepted for manual compatibility,
        # but never emitted by the public X-embodiment schema.
        for key in (f"{side}_gripper_cmd", f"{side}_gripper_cmd_bin"):
            value = action.get(key)
            if value is not None:
                return width_from_normalized_command(
                    float(value),
                    self.config.gripper_min_width,
                    self.config.gripper_max_open,
                    reverse=self.config.gripper_reverse,
                )
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
        return clip_gripper_width(
            width,
            self.config.gripper_min_width,
            self.config.gripper_max_open,
        )

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

    def move_gripper_width(self, width_m: float, side: str = "both", wait: bool = True) -> None:
        self._close_gripper_feedback()
        self._left_gripper_target_width = None
        self._right_gripper_target_width = None
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
            raise RuntimeError(f"{side} gripper hardware limits unavailable") from exc
        return target_width, velocity, force_limit

    def _close_gripper_feedback(self) -> None:
        with self._gripper_feedback_lock:
            controllers = list(self._gripper_feedback.values())
            self._gripper_feedback.clear()
        errors = []
        for controller in controllers:
            try:
                controller.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Could not stop GN01 aperture feedback") from errors[0]

    def _move_gripper_to_width_if_needed(
        self,
        side: str,
        width: float,
        command: float | None = None,
    ) -> float:
        gripper = self._left_gripper if side == "left" else self._right_gripper
        if gripper is None:
            raise RuntimeError(f"{side} gripper is unavailable")

        prepared = self._prepare_gripper_move(side, gripper, width)
        if prepared is None:
            raise RuntimeError(f"{side} gripper target could not be prepared")
        target_width, velocity, force_limit = map(finite_gripper_value, prepared)

        if self.config.gripper_control_mode == "force_feedback":
            with self._gripper_feedback_lock:
                controller = self._gripper_feedback.get(side)
                if controller is None:
                    params = self._get_gripper_params(side, gripper)
                    free_force = min(self.config.gripper_feedback_force_limit, force_limit)
                    if params["min_force"] > -free_force or params["max_force"] < free_force:
                        raise ValueError("GN01 feedback requires signed direct-force support")
                    controller = GripperApertureController(
                        gripper,
                        min_width=max(self.config.gripper_min_width, params["min_width"]),
                        max_width=min(self.config.gripper_max_open, params["max_width"]),
                        speed=velocity, force_limit=force_limit,
                        free_force_limit=free_force,
                        timeout=self.config.gripper_feedback_timeout_sec,
                    )
                    self._gripper_feedback[side] = controller
            target_width = controller.update(target_width)
            setattr(self, f"_{side}_gripper_target_width", target_width)
            self._set_cached_gripper_width(side, target_width)
            return target_width

        if self.config.gripper_continuous_speed is not None:
            params = self._get_gripper_params(side, gripper)
            min_width = max(self.config.gripper_min_width, params["min_width"])
            max_width = min(self.config.gripper_max_open, params["max_width"])
            epsilon = self.config.gripper_command_epsilon
            if min_width + epsilon < target_width < max_width - epsilon:
                velocity = min(velocity, max(params["min_vel"], self.config.gripper_continuous_speed))

        last_width = getattr(self, f"_{side}_gripper_target_width", None)
        if last_width is not None and abs(target_width - last_width) < self.config.gripper_command_epsilon:
            return float(last_width)
        last_send = getattr(self, f"_{side}_gripper_last_send_time", None)
        if (last_width is not None and last_send is not None
                and time.monotonic() - last_send < self.config.gripper_min_command_interval_sec):
            # No queue: the next recording frame supplies the newest desired
            # width. Return the last delivered target for truthful action labels.
            return float(last_width)

        logger.debug(
            "[FLEXIV] %s gripper Move width=%.4f command=%s velocity=%.3f force=%.1f",
            side,
            target_width,
            "direct" if command is None else f"{float(command):.3f}",
            velocity,
            force_limit,
        )
        gripper.Move(target_width, velocity, force_limit)
        setattr(self, f"_{side}_gripper_target_width", target_width)
        setattr(self, f"_{side}_gripper_last_send_time", time.monotonic())
        self._set_cached_gripper_width(side, target_width)
        return target_width

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
        tolerance = max(
            GRIPPER_WAIT_TOLERANCE_FLOOR,
            float(self.config.gripper_command_epsilon),
        )
        deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
        last_widths: dict[str, float | None] = {side: None for side in targets}
        reached: set[str] = set()

        logger.info(
            "[FLEXIV] Waiting for gripper home sides=%s timeout=%.1fs tolerance=%.4f",
            ",".join(targets),
            timeout_sec,
            tolerance,
        )
        while len(reached) < len(targets):
            now = time.monotonic()
            for side, (gripper, target_width, _velocity, _force_limit) in targets.items():
                if side in reached:
                    continue

                width, is_moving, _force = self._read_gripper_state(gripper)
                if width is not None:
                    last_widths[side] = width
                    if abs(width - target_width) <= tolerance and is_moving is False:
                        logger.info(
                            "[FLEXIV] %s gripper reached width=%.4f target=%.4f",
                            side,
                            width,
                            target_width,
                        )
                        self._set_cached_gripper_width(side, width)
                        reached.add(side)

            if len(reached) == len(targets):
                return
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
        self._check_cartesian_servo_error()
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
                return self._prev_observation
            raise

    def _add_arm_observation(self, obs: dict[str, Any], side: str, robot: Any) -> None:
        robot_lock = self._left_robot_lock if side == "left" else self._right_robot_lock
        with robot_lock:
            states = robot.states()
        pose7 = _as_np(getattr(states, "tcp_pose", None), 7)
        if np.linalg.norm(pose7[3:7]) < 1e-12:
            pose7[3] = 1.0
        setattr(self, f"_actual_{side}_pose7", pose7.copy())

        # Motion/servo targets must stay in the RDK base frame. Transform only
        # the public observation, after caching the unmodified SDK pose.
        if side == "left":
            self._cached_left_pose7 = pose7.copy()
        else:
            self._cached_right_pose7 = pose7.copy()

        transform = self._observation_transforms.get(side)
        observed_pose7 = transform.apply(pose7) if transform is not None else pose7
        pose6 = _pose7_to_pose6(observed_pose7)
        for index, axis in enumerate(AXES):
            obs[f"{side}_ee_pose.{axis}"] = float(pose6[index])

        if self.config.use_gripper:
            cmd = self._left_gripper_cmd if side == "left" else self._right_gripper_cmd
            gripper = self._left_gripper if side == "left" else self._right_gripper
            width = None
            if gripper is not None:
                try:
                    width = self._clip_gripper_width(float(gripper.states().width))
                except Exception:  # noqa: BLE001
                    pass
            if width is None:
                width = self._left_gripper_width if side == "left" else self._right_gripper_width
            if width is None:
                width = self._gripper_width_from_cmd(cmd)
            width = self._clip_gripper_width(width)
            self._set_cached_gripper_width(side, width)
            obs[f"{side}_gripper_width"] = width

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
        for cam_name, cam in self.cameras.items():
            # LeRobot's default RealSense connect() warmup reads with a fixed
            # 200 ms timeout, which is too tight when three D435 pipelines start
            # together. Disable that warmup and use the configurable one below.
            cam.connect(warmup=False)
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

    def _warmup_camera(self, cam_name: str, cam: Any) -> None:
        attempts = max(int(self.config.camera_warmup_attempts), 1)
        timeout_ms = max(int(self.config.camera_read_timeout_ms), 200)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                frame = cam.read(timeout_ms=timeout_ms)
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

    def _stop_cameras(self) -> None:
        self._camera_stop_event.set()
        for cam_name, thread in self._camera_threads.items():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("[CAM] %s thread did not stop cleanly", cam_name)
        self._camera_threads.clear()
        self._latest_frames.clear()
        for cam in self.cameras.values():
            cam.disconnect()

    def _camera_read_loop(self, cam_name: str, cam: Any) -> None:
        timeout_ms = max(int(self.config.camera_read_timeout_ms), 200)
        while not self._camera_stop_event.is_set():
            try:
                frame = cam.read(timeout_ms=timeout_ms)
                with self._frame_lock:
                    self._latest_frames[cam_name] = frame
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CAM] %s read failed: %s", cam_name, exc)
                self._camera_stop_event.wait(timeout=0.1)

    def _add_camera_observations(self, obs: dict[str, Any]) -> None:
        if not self.cameras:
            return
        with self._frame_lock:
            for cam_name in self.cameras:
                if cam_name in self._latest_frames:
                    obs[cam_name] = self._latest_frames[cam_name]
                else:
                    obs[cam_name] = self.cameras[cam_name].read(
                        timeout_ms=max(int(self.config.camera_read_timeout_ms), 200)
                    )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            name: feature
            for name, feature in x_embodiment_observation_features(
                {}, use_gripper=self.config.use_gripper
            ).items()
        }

    @property
    def action_features(self) -> dict[str, type]:
        return x_embodiment_action_features(use_gripper=self.config.use_gripper)

    @property
    def observation_features(self) -> dict[str, Any]:
        return x_embodiment_observation_features(
            self.cameras,
            use_gripper=self.config.use_gripper,
        )

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
