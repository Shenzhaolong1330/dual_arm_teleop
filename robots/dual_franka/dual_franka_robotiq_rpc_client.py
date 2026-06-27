#!/usr/bin/env python3
"""ROS-free ZeroRPC client for dual_franka_robotiq_rpc_server.

This module intentionally has no ROS imports. It can be run from a machine that
only has Python and zerorpc installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional


def _json_loads(value: str) -> Any:
    if value == '-':
        value = sys.stdin.read()
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _print_json(value: Any, *, pretty: bool = True) -> None:
    indent = 2 if pretty else None
    print(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=indent,
            sort_keys=pretty,
            allow_nan=False,
        )
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float_vector(value: Any, length: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f'{name} must be a sequence of {length} finite floats.')
    try:
        value_length = len(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a sequence of {length} finite floats.') from exc
    if value_length < length:
        raise ValueError(f'{name} must contain at least {length} values, got {value_length}.')

    result = [float(item) for item in value[:length]]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f'{name} must contain only finite values, got {result!r}.')
    return result


def _normalize_quat_xyzw(quat: Sequence[float], name: str = 'quaternion') -> tuple[float, float, float, float]:
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError(f'{name} has near-zero norm.')
    return x / norm, y / norm, z / norm, w / norm


def _rotvec_to_quat_xyzw(rotvec: Sequence[float]) -> tuple[float, float, float, float]:
    rx, ry, rz = float(rotvec[0]), float(rotvec[1]), float(rotvec[2])
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    scale = math.sin(angle * 0.5) / angle
    return _normalize_quat_xyzw((rx * scale, ry * scale, rz * scale, math.cos(angle * 0.5)))


def _quat_to_rotvec(quat: Sequence[float]) -> list[float]:
    x, y, z, w = _normalize_quat_xyzw(quat)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w

    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half <= 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(sin_half, w)
    scale = angle / sin_half
    return [x * scale, y * scale, z * scale]


def _quat_inverse_xyzw(quat: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = _normalize_quat_xyzw(quat)
    return -x, -y, -z, w


def _quat_multiply_xyzw(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = _normalize_quat_xyzw(first, 'first quaternion')
    x2, y2, z2, w2 = _normalize_quat_xyzw(second, 'second quaternion')
    return _normalize_quat_xyzw(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def _pose_from_side_observation(observation: Mapping[str, Any], side: str) -> list[float]:
    side_state = _as_mapping(observation.get(side))
    robot_state = _as_mapping(side_state.get('robot_state')) or side_state

    if 'end_pose' in side_state:
        return _as_float_vector(side_state.get('end_pose'), 6, f'{side}.end_pose')
    if 'end_pose' in robot_state:
        return _as_float_vector(robot_state.get('end_pose'), 6, f'{side}.robot_state.end_pose')

    eef_pose = _as_mapping(robot_state.get('eef_pose'))
    if not eef_pose:
        raise ValueError(f'Observation for {side} does not include end_pose or eef_pose.')

    position = _as_float_vector(eef_pose.get('position'), 3, f'{side}.eef_pose.position')
    quat = _as_float_vector(eef_pose.get('orientation_xyzw'), 4, f'{side}.eef_pose.orientation_xyzw')
    return position + _quat_to_rotvec(quat)


def _absolute_target_to_delta(current_pose: Sequence[float], target_pose: Sequence[float]) -> list[float]:
    current = _as_float_vector(current_pose, 6, 'current_pose')
    target = _as_float_vector(target_pose, 6, 'target_pose')
    translation = [target[index] - current[index] for index in range(3)]

    current_quat = _rotvec_to_quat_xyzw(current[3:])
    target_quat = _rotvec_to_quat_xyzw(target[3:])
    delta_quat = _quat_multiply_xyzw(target_quat, _quat_inverse_xyzw(current_quat))
    return translation + _quat_to_rotvec(delta_quat)


def _pose_from_delta(current_pose: Sequence[float], delta_pose: Sequence[float]) -> list[float]:
    current = _as_float_vector(current_pose, 6, 'current_pose')
    delta = _as_float_vector(delta_pose, 6, 'delta_pose')
    position = [current[index] + delta[index] for index in range(3)]
    current_quat = _rotvec_to_quat_xyzw(current[3:])
    delta_quat = _rotvec_to_quat_xyzw(delta[3:])
    target_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    return position + _quat_to_rotvec(target_quat)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _s_curve_fraction(value: float) -> float:
    """Quintic smoothstep with zero velocity and acceleration at both ends."""
    u = _clamp01(value)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _quat_slerp_xyzw(
    start_quat: Sequence[float],
    target_quat: Sequence[float],
    fraction: float,
) -> tuple[float, float, float, float]:
    x0, y0, z0, w0 = _normalize_quat_xyzw(start_quat, 'start quaternion')
    x1, y1, z1, w1 = _normalize_quat_xyzw(target_quat, 'target quaternion')
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot

    fraction = _clamp01(fraction)
    if dot > 0.9995:
        return _normalize_quat_xyzw(
            (
                x0 + fraction * (x1 - x0),
                y0 + fraction * (y1 - y0),
                z0 + fraction * (z1 - z0),
                w0 + fraction * (w1 - w0),
            ),
            'interpolated quaternion',
        )

    theta_0 = math.acos(_clamp01(dot))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * fraction
    sin_theta = math.sin(theta)
    scale_start = math.cos(theta) - dot * sin_theta / sin_theta_0
    scale_target = sin_theta / sin_theta_0
    return _normalize_quat_xyzw(
        (
            scale_start * x0 + scale_target * x1,
            scale_start * y0 + scale_target * y1,
            scale_start * z0 + scale_target * z1,
            scale_start * w0 + scale_target * w1,
        ),
        'interpolated quaternion',
    )


def _interpolate_pose(
    start_pose: Sequence[float],
    target_pose: Sequence[float],
    fraction: float,
) -> list[float]:
    start = _as_float_vector(start_pose, 6, 'start_pose')
    target = _as_float_vector(target_pose, 6, 'target_pose')
    fraction = _clamp01(fraction)
    position = [start[index] + fraction * (target[index] - start[index]) for index in range(3)]
    quat = _quat_slerp_xyzw(
        _rotvec_to_quat_xyzw(start[3:]),
        _rotvec_to_quat_xyzw(target[3:]),
        fraction,
    )
    return position + _quat_to_rotvec(quat)


def _positive_float_or_none(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be a positive finite value, got {value!r}.')
    return value


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be a positive finite value, got {value!r}.')
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be a non-negative finite value, got {value!r}.')
    return value


def _ceil_div_motion(distance: float, limit: Optional[float]) -> int:
    if limit is None:
        return 1
    if distance <= 1e-12:
        return 1
    return max(1, int(math.ceil(distance / limit)))


def _motion_norms(delta_pose: Sequence[float]) -> tuple[float, float]:
    delta = _as_float_vector(delta_pose, 6, 'delta_pose')
    translation_norm = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
    rotation_norm = math.sqrt(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5])
    return translation_norm, rotation_norm


def _plan_smooth_absolute_trajectory(
    left_start_pose: Sequence[float],
    right_start_pose: Sequence[float],
    left_target_pose: Sequence[float],
    right_target_pose: Sequence[float],
    *,
    duration_sec: Optional[float] = None,
    rate_hz: float = 50.0,
    max_translation_speed: float = 0.05,
    max_rotation_speed: float = 0.4,
    max_translation_step: Optional[float] = 0.003,
    max_rotation_step: Optional[float] = 0.025,
    min_duration_sec: float = 0.5,
    max_steps: int = 3000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left_start = _as_float_vector(left_start_pose, 6, 'left_start_pose')
    right_start = _as_float_vector(right_start_pose, 6, 'right_start_pose')
    left_target = _as_float_vector(left_target_pose, 6, 'left_target_pose')
    right_target = _as_float_vector(right_target_pose, 6, 'right_target_pose')
    rate_hz = _positive_float(rate_hz, 'rate_hz')
    max_translation_speed = _positive_float(max_translation_speed, 'max_translation_speed')
    max_rotation_speed = _positive_float(max_rotation_speed, 'max_rotation_speed')
    max_translation_step = _positive_float_or_none(max_translation_step, 'max_translation_step')
    max_rotation_step = _positive_float_or_none(max_rotation_step, 'max_rotation_step')
    min_duration_sec = _nonnegative_float(min_duration_sec, 'min_duration_sec')
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f'max_steps must be positive, got {max_steps!r}.')

    left_total_delta = _absolute_target_to_delta(left_start, left_target)
    right_total_delta = _absolute_target_to_delta(right_start, right_target)
    left_translation, left_rotation = _motion_norms(left_total_delta)
    right_translation, right_rotation = _motion_norms(right_total_delta)
    max_translation = max(left_translation, right_translation)
    max_rotation = max(left_rotation, right_rotation)

    metadata: dict[str, Any] = {
        'left_start_pose': left_start,
        'right_start_pose': right_start,
        'left_target_pose': left_target,
        'right_target_pose': right_target,
        'left_total_delta': left_total_delta,
        'right_total_delta': right_total_delta,
        'max_translation_m': max_translation,
        'max_rotation_rad': max_rotation,
        'rate_hz': rate_hz,
        'profile': 'quintic_s_curve',
    }

    if max_translation <= 1e-9 and max_rotation <= 1e-9:
        metadata.update({'steps': 0, 'duration_sec': 0.0, 'period_sec': 0.0})
        return [], metadata

    # Quintic smoothstep has peak normalized speed 1.875. Account for that so
    # commanded waypoint deltas and speed limits stay conservative near mid-path.
    s_curve_peak_speed = 1.875
    speed_duration = max(
        s_curve_peak_speed * max_translation / max_translation_speed,
        s_curve_peak_speed * max_rotation / max_rotation_speed,
    )
    requested_duration = _positive_float_or_none(duration_sec, 'duration_sec')
    if requested_duration is None:
        requested_duration = max(min_duration_sec, speed_duration)
    else:
        requested_duration = max(min_duration_sec, requested_duration)

    step_limited_steps = max(
        _ceil_div_motion(s_curve_peak_speed * max_translation, max_translation_step),
        _ceil_div_motion(s_curve_peak_speed * max_rotation, max_rotation_step),
    )
    duration = max(requested_duration, step_limited_steps / rate_hz)
    steps = max(1, int(math.ceil(duration * rate_hz)))
    if steps > max_steps:
        raise ValueError(
            f'Smooth trajectory requires {steps} steps at {rate_hz:g} Hz, exceeding max_steps={max_steps}. '
            'Increase max_steps, increase speed/step limits, or choose a nearer target.'
        )
    period = duration / steps

    trajectory: list[dict[str, Any]] = []
    previous_left = left_start
    previous_right = right_start
    for index in range(1, steps + 1):
        path_fraction = _s_curve_fraction(index / steps)
        left_waypoint = _interpolate_pose(left_start, left_target, path_fraction)
        right_waypoint = _interpolate_pose(right_start, right_target, path_fraction)
        left_delta = _absolute_target_to_delta(previous_left, left_waypoint)
        right_delta = _absolute_target_to_delta(previous_right, right_waypoint)
        trajectory.append(
            {
                'index': index,
                'path_fraction': path_fraction,
                'left_pose': left_waypoint,
                'right_pose': right_waypoint,
                'left_delta': left_delta,
                'right_delta': right_delta,
            }
        )
        previous_left = left_waypoint
        previous_right = right_waypoint

    metadata.update({'steps': steps, 'duration_sec': duration, 'period_sec': period})
    return trajectory, metadata


def _pose_error_report(
    left_current_pose: Sequence[float],
    right_current_pose: Sequence[float],
    left_target_pose: Sequence[float],
    right_target_pose: Sequence[float],
) -> dict[str, Any]:
    left_error = _absolute_target_to_delta(left_current_pose, left_target_pose)
    right_error = _absolute_target_to_delta(right_current_pose, right_target_pose)
    left_translation, left_rotation = _motion_norms(left_error)
    right_translation, right_rotation = _motion_norms(right_error)
    return {
        'left_error_delta': left_error,
        'right_error_delta': right_error,
        'left_translation_error_m': left_translation,
        'right_translation_error_m': right_translation,
        'left_rotation_error_rad': left_rotation,
        'right_rotation_error_rad': right_rotation,
        'max_translation_error_m': max(left_translation, right_translation),
        'max_rotation_error_rad': max(left_rotation, right_rotation),
    }


def _pose_error_within_tolerance(
    error: Mapping[str, Any],
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> bool:
    return (
        float(error.get('max_translation_error_m', math.inf)) <= position_tolerance_m
        and float(error.get('max_rotation_error_rad', math.inf)) <= rotation_tolerance_rad
    )


def _motion_action_from_delta(delta_pose: Sequence[float]) -> dict[str, dict[str, list[Any]]]:
    return {
        'motion': {
            'translation': list(delta_pose[:3]),
            'rotation_rotvec': list(delta_pose[3:]),
        }
    }


def _looks_like_missing_rpc_method(exc: BaseException) -> bool:
    message = f'{type(exc).__name__}: {exc}'.lower()
    if 'dual_robot_move_to_ee_pose' not in message:
        return isinstance(exc, AttributeError)
    missing_method_markers = (
        'attributeerror',
        'has no attribute',
        'no such method',
        'unknown method',
        'method not found',
        'not found',
    )
    return any(marker in message for marker in missing_method_markers)


def _connect(server: str, timeout: float):
    try:
        import zerorpc
    except ImportError as exc:
        raise SystemExit(
            'zerorpc is not installed. This client does not need ROS2, but it '
            'does need ZeroRPC:\n'
            '  python3 -m pip install --user zerorpc gevent pyzmq'
        ) from exc

    client = zerorpc.Client(timeout=timeout)
    client.connect(server)
    return client


def _side(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        'l': 'left_arm',
        'left': 'left_arm',
        'left_arm': 'left_arm',
        'r': 'right_arm',
        'right': 'right_arm',
        'right_arm': 'right_arm',
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            'side must be one of: left, left_arm, right, right_arm'
        )
    return aliases[normalized]


def _side_or_both(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ('both', 'all'):
        return 'both'
    return _side(value)


class DualFrankaRobotiqRpcClient:
    """Reusable Python client for dual_franka_robotiq_rpc_server.py."""

    def __init__(
        self,
        ip: str = '127.0.0.1',
        port: int = 4242,
        timeout: float = 30.0,
        server: Optional[str] = None,
    ) -> None:
        self.server = server or f'tcp://{ip}:{int(port)}'
        self.timeout = float(timeout)
        self._client = _connect(self.server, self.timeout)

    def _call(self, name: str, *args):
        return getattr(self._client, name)(*args)

    def close(self) -> None:
        self._client.close()

    def ping(self):
        return self._call('ping')

    def reset(self):
        return self._call('reset')

    def step(self, action: Optional[dict[str, Any]] = None):
        return self._call('step', action)

    def get_observation(self):
        return self._call('get_observation')

    def get_full_state(self):
        return self.get_observation()

    def get_home(self):
        return self._call('get_home')

    def set_home_current(self, side: str = 'both'):
        return self._call('set_home_current', _side_or_both(side))

    def save_home_current(self, side: str = 'both'):
        return self._call('save_home_current', _side_or_both(side))

    def go_home(
        self,
        side: str = 'both',
        duration_sec: Optional[float] = None,
        rate_hz: Optional[float] = None,
    ):
        return self._call('go_home', _side_or_both(side), duration_sec, rate_hz)

    def command_gripper(
        self,
        side: str = 'left_arm',
        command: Optional[dict[str, Any]] = None,
    ):
        return self._call('command_gripper', _side(side), command or {})

    def open_gripper(self, side: str = 'left_arm'):
        return self._call('open_gripper', _side(side))

    def close_gripper(self, side: str = 'left_arm'):
        return self._call('close_gripper', _side(side))

    def reactivate_gripper(self, side: str = 'left_arm'):
        return self._call('reactivate_gripper', _side(side))

    def left_gripper_initialize(self):
        return self.reactivate_gripper('left_arm')

    def right_gripper_initialize(self):
        return self.reactivate_gripper('right_arm')

    def gripper_initialize(self):
        return {
            'left': self.left_gripper_initialize(),
            'right': self.right_gripper_initialize(),
        }

    def left_gripper_goto(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 10.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        del epsilon_inner, epsilon_outer, blocking
        return self.command_gripper(
            'left_arm',
            {'width': float(width), 'max_velocity': float(speed), 'max_effort': float(force)},
        )

    def right_gripper_goto(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 10.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        del epsilon_inner, epsilon_outer, blocking
        return self.command_gripper(
            'right_arm',
            {'width': float(width), 'max_velocity': float(speed), 'max_effort': float(force)},
        )

    def left_gripper_get_state(self) -> dict[str, Any]:
        obs = self.get_observation()
        return self._gripper_state_from_observation(obs.get('left_arm', {}) if isinstance(obs, dict) else {})

    def right_gripper_get_state(self) -> dict[str, Any]:
        obs = self.get_observation()
        return self._gripper_state_from_observation(obs.get('right_arm', {}) if isinstance(obs, dict) else {})

    @staticmethod
    def _gripper_state_from_observation(side_obs: dict[str, Any]) -> dict[str, Any]:
        grip = side_obs.get('gripper', {}) if isinstance(side_obs, dict) else {}
        if not isinstance(grip, dict):
            grip = {'position': grip}
        return grip

    def set_left_gripper(self, normalized_close: float):
        return self.command_gripper('left_arm', {'normalized': float(normalized_close)})

    def set_right_gripper(self, normalized_close: float):
        return self.command_gripper('right_arm', {'normalized': float(normalized_close)})

    def dual_robot_move_to_ee_pose(
        self,
        left_delta,
        right_delta,
        delta: bool = True,
        wait: bool = False,
        *,
        smooth: Optional[bool] = None,
        duration_sec: Optional[float] = None,
        rate_hz: float = 50.0,
        max_translation_speed: float = 0.05,
        max_rotation_speed: float = 0.4,
        max_translation_step: Optional[float] = 0.003,
        max_rotation_step: Optional[float] = 0.025,
        min_duration_sec: float = 0.5,
        max_steps: int = 3000,
        sleep: bool = True,
        settle_time_sec: float = 0.4,
        position_tolerance_m: float = 0.0015,
        rotation_tolerance_rad: float = 0.01,
        max_correction_iters: int = 1,
    ):
        """Move both end-effectors through the server Cartesian pose API.

        ``delta=True`` keeps the legacy 6D torso-frame delta semantics. When
        ``delta=False``, the two pose inputs are absolute torso-frame targets:
        ``[x, y, z, rx, ry, rz]``.

        Absolute targets default to a client-side smooth trajectory: position is
        linearly interpolated in torso frame, orientation uses quaternion slerp,
        and path progress follows a quintic S-curve. Set ``smooth=False`` for
        the old single-RPC behavior.
        """
        left_pose = _as_float_vector(left_delta, 6, 'left_delta')
        right_pose = _as_float_vector(right_delta, 6, 'right_delta')
        if smooth is None:
            smooth = not bool(delta)
        if smooth:
            return self._smooth_dual_robot_move_to_ee_pose(
                left_pose,
                right_pose,
                delta=delta,
                wait=wait,
                duration_sec=duration_sec,
                rate_hz=rate_hz,
                max_translation_speed=max_translation_speed,
                max_rotation_speed=max_rotation_speed,
                max_translation_step=max_translation_step,
                max_rotation_step=max_rotation_step,
                min_duration_sec=min_duration_sec,
                max_steps=max_steps,
                sleep=sleep,
                settle_time_sec=settle_time_sec,
                position_tolerance_m=position_tolerance_m,
                rotation_tolerance_rad=rotation_tolerance_rad,
                max_correction_iters=max_correction_iters,
            )

        try:
            return self._call(
                'dual_robot_move_to_ee_pose',
                left_pose,
                right_pose,
                bool(delta),
                bool(wait),
            )
        except Exception as exc:  # noqa: BLE001
            if not _looks_like_missing_rpc_method(exc):
                raise
        return self._legacy_dual_robot_move_to_ee_pose(left_pose, right_pose, delta=delta, wait=wait)

    def _smooth_dual_robot_move_to_ee_pose(
        self,
        left_pose: Sequence[float],
        right_pose: Sequence[float],
        *,
        delta: bool,
        wait: bool,
        duration_sec: Optional[float],
        rate_hz: float,
        max_translation_speed: float,
        max_rotation_speed: float,
        max_translation_step: Optional[float],
        max_rotation_step: Optional[float],
        min_duration_sec: float,
        max_steps: int,
        sleep: bool,
        settle_time_sec: float,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        max_correction_iters: int,
    ):
        # Smooth trajectory streaming is synchronous by design; ``wait`` is kept
        # for API compatibility with older callers.
        del wait
        observation = self.get_observation()
        if not isinstance(observation, Mapping):
            raise RuntimeError(f'Unexpected observation payload: {type(observation)!r}')

        left_start = _pose_from_side_observation(observation, 'left_arm')
        right_start = _pose_from_side_observation(observation, 'right_arm')
        if delta:
            left_target = _pose_from_delta(left_start, left_pose)
            right_target = _pose_from_delta(right_start, right_pose)
        else:
            left_target = _as_float_vector(left_pose, 6, 'left_pose')
            right_target = _as_float_vector(right_pose, 6, 'right_pose')

        settle_time_sec = _nonnegative_float(settle_time_sec, 'settle_time_sec')
        position_tolerance_m = _positive_float(position_tolerance_m, 'position_tolerance_m')
        rotation_tolerance_rad = _positive_float(rotation_tolerance_rad, 'rotation_tolerance_rad')
        max_correction_iters = int(max_correction_iters)
        if max_correction_iters < 0:
            raise ValueError(f'max_correction_iters must be non-negative, got {max_correction_iters!r}.')

        trajectory, metadata = _plan_smooth_absolute_trajectory(
            left_start,
            right_start,
            left_target,
            right_target,
            duration_sec=duration_sec,
            rate_hz=rate_hz,
            max_translation_speed=max_translation_speed,
            max_rotation_speed=max_rotation_speed,
            max_translation_step=max_translation_step,
            max_rotation_step=max_rotation_step,
            min_duration_sec=min_duration_sec,
            max_steps=max_steps,
        )

        def stream(trajectory_items: list[dict[str, Any]], trajectory_metadata: Mapping[str, Any]) -> Any:
            result: Any = None
            deadline = time.monotonic()
            for waypoint in trajectory_items:
                action = {
                    'left_arm': _motion_action_from_delta(waypoint['left_delta']),
                    'right_arm': _motion_action_from_delta(waypoint['right_delta']),
                }
                result = self.step(action)
                if sleep and waypoint['index'] < trajectory_metadata['steps']:
                    deadline += float(trajectory_metadata['period_sec'])
                    time.sleep(max(0.0, deadline - time.monotonic()))
            return result

        last_result: Any = None
        correction_reports: list[dict[str, Any]] = []
        last_result = stream(trajectory, metadata)

        if sleep and settle_time_sec > 0.0:
            time.sleep(settle_time_sec)

        final_error: dict[str, Any] | None = None
        for correction_index in range(max_correction_iters + 1):
            current_observation = self.get_observation()
            if not isinstance(current_observation, Mapping):
                raise RuntimeError(f'Unexpected observation payload: {type(current_observation)!r}')
            left_current = _pose_from_side_observation(current_observation, 'left_arm')
            right_current = _pose_from_side_observation(current_observation, 'right_arm')
            final_error = _pose_error_report(left_current, right_current, left_target, right_target)
            final_error['correction_index'] = correction_index
            correction_reports.append(final_error)
            if _pose_error_within_tolerance(final_error, position_tolerance_m, rotation_tolerance_rad):
                break
            if correction_index >= max_correction_iters:
                break

            correction_trajectory, correction_metadata = _plan_smooth_absolute_trajectory(
                left_current,
                right_current,
                left_target,
                right_target,
                duration_sec=None,
                rate_hz=rate_hz,
                max_translation_speed=max_translation_speed,
                max_rotation_speed=max_rotation_speed,
                max_translation_step=max_translation_step,
                max_rotation_step=max_rotation_step,
                min_duration_sec=min_duration_sec,
                max_steps=max_steps,
            )
            correction_metadata['correction_index'] = correction_index + 1
            correction_reports[-1]['correction_trajectory'] = {
                'steps': correction_metadata['steps'],
                'duration_sec': correction_metadata['duration_sec'],
                'period_sec': correction_metadata['period_sec'],
            }
            last_result = stream(correction_trajectory, correction_metadata)
            if sleep and settle_time_sec > 0.0:
                time.sleep(settle_time_sec)

        return {
            'ok': final_error is None
            or _pose_error_within_tolerance(final_error, position_tolerance_m, rotation_tolerance_rad),
            'trajectory': metadata,
            'final_error': final_error,
            'corrections': correction_reports,
            'tolerances': {
                'position_tolerance_m': position_tolerance_m,
                'rotation_tolerance_rad': rotation_tolerance_rad,
                'settle_time_sec': settle_time_sec,
                'max_correction_iters': max_correction_iters,
            },
            'last_step': last_result,
        }

    def _legacy_dual_robot_move_to_ee_pose(
        self,
        left_delta: Sequence[float],
        right_delta: Sequence[float],
        delta: bool,
        wait: bool,
    ):
        del wait
        if not delta:
            observation = self.get_observation()
            if not isinstance(observation, Mapping):
                raise RuntimeError(f'Unexpected observation payload: {type(observation)!r}')

            left_current_pose = _pose_from_side_observation(observation, 'left_arm')
            right_current_pose = _pose_from_side_observation(observation, 'right_arm')
            left_delta = _absolute_target_to_delta(left_current_pose, left_delta)
            right_delta = _absolute_target_to_delta(right_current_pose, right_delta)

        action = {
            'left_arm': _motion_action_from_delta(left_delta),
            'right_arm': _motion_action_from_delta(right_delta),
        }
        return self.step(action)


FrankaDualArmClient = DualFrankaRobotiqRpcClient


def _add_motion_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f'--{prefix}-dx', type=float, default=0.0)
    parser.add_argument(f'--{prefix}-dy', type=float, default=0.0)
    parser.add_argument(f'--{prefix}-dz', type=float, default=0.0)
    parser.add_argument(f'--{prefix}-rx', type=float, default=0.0)
    parser.add_argument(f'--{prefix}-ry', type=float, default=0.0)
    parser.add_argument(f'--{prefix}-rz', type=float, default=0.0)


def _add_gripper_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f'--{prefix}-gripper-normalized', type=float)
    parser.add_argument(f'--{prefix}-gripper-position', type=float)
    parser.add_argument(f'--{prefix}-gripper-width', type=float)
    parser.add_argument(f'--{prefix}-open', action='store_true')
    parser.add_argument(f'--{prefix}-close', action='store_true')
    parser.add_argument(f'--{prefix}-max-effort', type=float)
    parser.add_argument(f'--{prefix}-max-velocity', type=float)


def _get(args: argparse.Namespace, name: str) -> Any:
    return getattr(args, name.replace('-', '_'))


def _motion_from_args(args: argparse.Namespace, prefix: str) -> Optional[dict[str, Any]]:
    translation = [
        _get(args, f'{prefix}-dx'),
        _get(args, f'{prefix}-dy'),
        _get(args, f'{prefix}-dz'),
    ]
    rotation_rotvec = [
        _get(args, f'{prefix}-rx'),
        _get(args, f'{prefix}-ry'),
        _get(args, f'{prefix}-rz'),
    ]
    if not any(abs(v) > 0.0 for v in translation + rotation_rotvec):
        return None
    return {
        'translation': translation,
        'rotation_rotvec': rotation_rotvec,
    }


def _gripper_from_args(args: argparse.Namespace, prefix: str) -> Optional[dict[str, Any]]:
    command: dict[str, Any] = {}
    fields = (
        ('normalized', f'{prefix}-gripper-normalized'),
        ('position', f'{prefix}-gripper-position'),
        ('width', f'{prefix}-gripper-width'),
        ('max_effort', f'{prefix}-max-effort'),
        ('max_velocity', f'{prefix}-max-velocity'),
    )
    for command_field, arg_name in fields:
        value = _get(args, arg_name)
        if value is not None:
            command[command_field] = value
    if _get(args, f'{prefix}-open'):
        command['open'] = True
    if _get(args, f'{prefix}-close'):
        command['close'] = True
    return command or None


def _build_step_action(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    if args.action_json is not None:
        return args.action_json

    action: dict[str, Any] = {}
    for prefix, side in (('left', 'left_arm'), ('right', 'right_arm')):
        side_action: dict[str, Any] = {}
        motion = _motion_from_args(args, prefix)
        gripper = _gripper_from_args(args, prefix)
        if motion is not None:
            side_action['motion'] = motion
        if gripper is not None:
            side_action['gripper'] = gripper
        if side_action:
            action[side] = side_action
    return action or None


def _build_gripper_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command_json is not None:
        return args.command_json

    command: dict[str, Any] = {}
    for name in ('normalized', 'position', 'width', 'max_effort', 'max_velocity'):
        value = getattr(args, name)
        if value is not None:
            command[name] = value
    if args.open:
        command['open'] = True
    if args.close:
        command['close'] = True
    if not command:
        raise SystemExit('No gripper command was provided.')
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--server',
        default='tcp://127.0.0.1:4242',
        help='ZeroRPC server endpoint, e.g. tcp://192.168.1.20:4242',
    )
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--compact', action='store_true', help='print one-line JSON')

    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('ping')
    subparsers.add_parser('reset')
    subparsers.add_parser('obs')
    subparsers.add_parser('home')

    set_home = subparsers.add_parser('set-home-current')
    set_home.add_argument('side', nargs='?', type=_side_or_both, default='both')

    save_home = subparsers.add_parser('save-home-current')
    save_home.add_argument('side', nargs='?', type=_side_or_both, default='both')

    go_home = subparsers.add_parser('go-home')
    go_home.add_argument('side', nargs='?', type=_side_or_both, default='both')
    go_home.add_argument('--duration', type=float)
    go_home.add_argument('--rate', type=float)

    recover = subparsers.add_parser('recover')
    recover.add_argument('side', nargs='?', type=_side_or_both, default='both')

    step = subparsers.add_parser('step')
    step.add_argument(
        '--action-json',
        type=_json_loads,
        help='raw action JSON, or "-" to read JSON from stdin',
    )
    _add_motion_args(step, 'left')
    _add_motion_args(step, 'right')
    _add_gripper_args(step, 'left')
    _add_gripper_args(step, 'right')

    raw_step = subparsers.add_parser('raw-step')
    raw_step.add_argument('action_json', type=_json_loads)

    gripper = subparsers.add_parser('gripper')
    gripper.add_argument('side', type=_side)
    gripper.add_argument('--command-json', type=_json_loads)
    gripper.add_argument('--normalized', type=float)
    gripper.add_argument('--position', type=float)
    gripper.add_argument('--width', type=float)
    gripper.add_argument('--open', action='store_true')
    gripper.add_argument('--close', action='store_true')
    gripper.add_argument('--max-effort', type=float)
    gripper.add_argument('--max-velocity', type=float)

    open_cmd = subparsers.add_parser('open')
    open_cmd.add_argument('side', nargs='?', type=_side, default='left_arm')

    close_cmd = subparsers.add_parser('close')
    close_cmd.add_argument('side', nargs='?', type=_side, default='left_arm')

    reactivate = subparsers.add_parser('reactivate')
    reactivate.add_argument('side', nargs='?', type=_side, default='left_arm')
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    client = _connect(args.server, args.timeout)
    try:
        if args.command == 'ping':
            result = client.ping()
        elif args.command == 'reset':
            result = client.reset()
        elif args.command == 'obs':
            result = client.get_observation()
        elif args.command == 'home':
            result = client.get_home()
        elif args.command == 'set-home-current':
            result = client.set_home_current(args.side)
        elif args.command == 'save-home-current':
            result = client.save_home_current(args.side)
        elif args.command == 'go-home':
            result = client.go_home(args.side, args.duration, args.rate)
        elif args.command == 'recover':
            result = client.recover_robot(args.side)
        elif args.command == 'step':
            result = client.step(_build_step_action(args))
        elif args.command == 'raw-step':
            result = client.step(args.action_json)
        elif args.command == 'gripper':
            result = client.command_gripper(args.side, _build_gripper_command(args))
        elif args.command == 'open':
            result = client.open_gripper(args.side)
        elif args.command == 'close':
            result = client.close_gripper(args.side)
        elif args.command == 'reactivate':
            result = client.reactivate_gripper(args.side)
        else:
            raise SystemExit(f'unknown command: {args.command}')
    finally:
        client.close()

    _print_json(result, pretty=not args.compact)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
