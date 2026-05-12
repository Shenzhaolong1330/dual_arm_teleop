#!/usr/bin/env python3
"""ZeroRPC server for DualFrankaRobotiqEnv.

The server owns one ROS2 node and one DualFrankaRobotiqEnv. ROS callbacks are
spun in a background executor thread, while ZeroRPC exposes a small synchronous
API for remote clients.
"""

from __future__ import annotations

import argparse
import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from exp_env_interact.dual_franka_robotiq_env import DualFrankaRobotiqEnv


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected boolean, got {value!r}')


def _none_if_empty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def to_jsonable(value: Any) -> Any:
    """Convert common ROS/env Python objects into ZeroRPC-friendly values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return to_jsonable(value.tolist())
    if hasattr(value, 'item'):
        return to_jsonable(value.item())
    if hasattr(value, 'as_quat'):
        return to_jsonable(value.as_quat())
    return str(value)


class DualFrankaRobotiqRpcApi:
    """Synchronous ZeroRPC API wrapper around DualFrankaRobotiqEnv."""

    def __init__(self, node: Node, env: DualFrankaRobotiqEnv) -> None:
        self._node = node
        self._env = env
        self._lock = threading.RLock()
        self._last_obs: dict[str, Any] = {}
        self._step_count = 0

    def ping(self) -> dict[str, Any]:
        return {
            'ok': True,
            'node_name': self._node.get_name(),
            'step_count': self._step_count,
        }

    def reset(self) -> Any:
        with self._lock:
            self._last_obs = self._env.reset()
            self._step_count = 0
            return to_jsonable(self._last_obs)

    def step(self, action: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        with self._lock:
            obs, reward, done, info = self._env.step(action)
            self._last_obs = obs
            self._step_count += 1
            return to_jsonable(
                {
                    'observation': obs,
                    'reward': reward,
                    'done': done,
                    'info': info,
                    'step_count': self._step_count,
                }
            )

    def get_observation(self) -> Any:
        with self._lock:
            if hasattr(self._env, 'get_observation'):
                self._last_obs = self._env.get_observation()
            return to_jsonable(self._last_obs)

    def get_home(self) -> dict[str, Any]:
        with self._lock:
            return to_jsonable(self._env.get_home())

    def set_home_current(self, side: str = 'both') -> dict[str, Any]:
        with self._lock:
            result = self._env.set_home_current(side)
            if hasattr(self._env, 'get_observation'):
                self._last_obs = self._env.get_observation()
                result['observation'] = self._last_obs
            return to_jsonable(result)

    def save_home_current(self, side: str = 'both') -> dict[str, Any]:
        with self._lock:
            if not hasattr(self._env, 'save_home_current'):
                return {'ok': False, 'error': 'environment does not support save_home_current'}
            result = self._env.save_home_current(side)
            if hasattr(self._env, 'get_observation'):
                self._last_obs = self._env.get_observation()
                result['observation'] = self._last_obs
            return to_jsonable(result)

    def go_home(
        self,
        side: str = 'both',
        duration_sec: Optional[float] = None,
        rate_hz: Optional[float] = None,
    ) -> dict[str, Any]:
        with self._lock:
            result = self._env.go_home(
                sides=side,
                duration_sec=duration_sec,
                rate_hz=rate_hz,
            )
            if hasattr(self._env, 'get_observation'):
                self._last_obs = self._env.get_observation()
                result['observation'] = self._last_obs
            return to_jsonable(result)

    def recover_robot(self, side: str = 'both') -> dict[str, Any]:
        with self._lock:
            result = self._env.recover_errors(side)
            if hasattr(self._env, 'get_observation'):
                self._last_obs = self._env.get_observation()
                result['observation'] = self._last_obs
            return to_jsonable(result)

    def command_gripper(
        self,
        side: str = 'left_arm',
        command: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        side = side or 'left_arm'
        if side in ('left', 'l'):
            side = 'left_arm'
        elif side in ('right', 'r'):
            side = 'right_arm'
        if side not in ('left_arm', 'right_arm'):
            return {'ok': False, 'error': f'unknown side: {side}'}
        return self.step({side: {'gripper': command or {}}})

    def open_gripper(self, side: str = 'left_arm') -> dict[str, Any]:
        return self.command_gripper(side, {'open': True})

    def close_gripper(self, side: str = 'left_arm') -> dict[str, Any]:
        return self.command_gripper(side, {'close': True})

    def reactivate_gripper(self, side: str = 'left_arm') -> dict[str, Any]:
        side = side or 'left_arm'
        if side in ('left', 'l'):
            client = getattr(self._env, '_left_gripper', None)
        elif side in ('right', 'r'):
            client = getattr(self._env, '_right_gripper', None)
        elif side == 'left_arm':
            client = getattr(self._env, '_left_gripper', None)
        elif side == 'right_arm':
            client = getattr(self._env, '_right_gripper', None)
        else:
            return {'ok': False, 'error': f'unknown side: {side}'}
        if client is None:
            return {'ok': False, 'error': f'no gripper client for {side}'}
        with self._lock:
            ok = client.reactivate()
        if not ok:
            return {'ok': False, 'error': f'activation service is not ready for {side}'}
        return {'ok': True}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='tcp://0.0.0.0:4242')
    parser.add_argument('--node-name', default='dual_franka_robotiq_rpc_server')
    parser.add_argument('--debug', type=_str_to_bool, default=True)
    parser.add_argument('--wait-for-state-timeout-sec', type=float, default=0.0)
    parser.add_argument('--torso-extrinsics-yaml', default='')
    parser.add_argument('--home-pose-yaml', default='')
    parser.add_argument('--go-home-duration-sec', type=float, default=5.0)
    parser.add_argument('--go-home-rate-hz', type=float, default=50.0)
    parser.add_argument('--left-controller-manager', default='')
    parser.add_argument('--right-controller-manager', default='')
    parser.add_argument('--left-cartesian-controller', default='cartesian_impedance_pose_topic_controller')
    parser.add_argument('--right-cartesian-controller', default='cartesian_impedance_pose_topic_controller')
    parser.add_argument('--left-joint-home-controller', default='joint_home_controller')
    parser.add_argument('--right-joint-home-controller', default='joint_home_controller')
    parser.add_argument('--left-joint-home-action-name', default='')
    parser.add_argument('--right-joint-home-action-name', default='')
    parser.add_argument('--joint-home-switch-timeout-sec', type=float, default=5.0)
    parser.add_argument('--joint-home-action-server-timeout-sec', type=float, default=5.0)
    parser.add_argument('--joint-home-result-timeout-extra-sec', type=float, default=5.0)
    parser.add_argument('--left-error-recovery-action-name', default='')
    parser.add_argument('--right-error-recovery-action-name', default='')
    parser.add_argument('--left-topic-namespace', default='left_arm')
    parser.add_argument('--right-topic-namespace', default='right_arm')

    parser.add_argument('--left-gripper-enabled', type=_str_to_bool, default=True)
    parser.add_argument('--right-gripper-enabled', type=_str_to_bool, default=True)
    parser.add_argument('--left-gripper-command-mode', default='forward')
    parser.add_argument('--right-gripper-command-mode', default='forward')
    parser.add_argument(
        '--left-gripper-command-topic',
        default='/left_gripper/robotiq_gripper_controller/commands',
    )
    parser.add_argument(
        '--right-gripper-command-topic',
        default='/right_gripper/robotiq_gripper_controller/commands',
    )
    parser.add_argument(
        '--left-gripper-action-name',
        default='/left_gripper/robotiq_gripper_controller/gripper_cmd',
    )
    parser.add_argument(
        '--right-gripper-action-name',
        default='/right_gripper/robotiq_gripper_controller/gripper_cmd',
    )
    parser.add_argument('--left-gripper-joint-state-topic', default='/joint_states')
    parser.add_argument('--right-gripper-joint-state-topic', default='/joint_states')
    parser.add_argument('--left-gripper-joint-name', default='robotiq_85_left_knuckle_joint')
    parser.add_argument(
        '--right-gripper-joint-name',
        default='robotiq_85_left_knuckle_joint',
    )
    parser.add_argument(
        '--left-gripper-activation-service',
        default='/left_gripper/robotiq_activation_controller/reactivate_gripper',
    )
    parser.add_argument(
        '--right-gripper-activation-service',
        default='/right_gripper/robotiq_activation_controller/reactivate_gripper',
    )
    parser.add_argument('--gripper-closed-position', type=float, default=0.7929)
    parser.add_argument('--gripper-open-position', type=float, default=0.0)
    parser.add_argument('--gripper-open-width', type=float, default=0.085)
    parser.add_argument('--gripper-default-max-effort', type=float, default=50.0)
    parser.add_argument('--gripper-default-max-velocity', type=float, default=0.5)
    parser.add_argument('--activate-grippers-on-reset', type=_str_to_bool, default=False)
    return parser


def serve(args: argparse.Namespace) -> None:
    try:
        import zerorpc
    except ImportError as exc:
        raise SystemExit(
            'zerorpc is not installed. Install it with:\n'
            '  python3 -m pip install -r '
            '/home/deepcybo/Desktop/interface_ros2/src/exp_env_interact/requirements-rpc.txt'
        ) from exc

    rclpy.init()
    node = Node(args.node_name)
    executor = MultiThreadedExecutor(num_threads=2)
    env = None
    spin_thread = None
    try:
        env = DualFrankaRobotiqEnv(
            node,
            debug=args.debug,
            wait_for_state_timeout_sec=args.wait_for_state_timeout_sec,
            torso_extrinsics_yaml=_none_if_empty(args.torso_extrinsics_yaml),
            home_pose_yaml=_none_if_empty(args.home_pose_yaml),
            go_home_duration_sec=args.go_home_duration_sec,
            go_home_rate_hz=args.go_home_rate_hz,
            left_controller_manager=_none_if_empty(args.left_controller_manager),
            right_controller_manager=_none_if_empty(args.right_controller_manager),
            left_cartesian_controller=args.left_cartesian_controller,
            right_cartesian_controller=args.right_cartesian_controller,
            left_joint_home_controller=args.left_joint_home_controller,
            right_joint_home_controller=args.right_joint_home_controller,
            left_joint_home_action_name=_none_if_empty(args.left_joint_home_action_name),
            right_joint_home_action_name=_none_if_empty(args.right_joint_home_action_name),
            joint_home_switch_timeout_sec=args.joint_home_switch_timeout_sec,
            joint_home_action_server_timeout_sec=args.joint_home_action_server_timeout_sec,
            joint_home_result_timeout_extra_sec=args.joint_home_result_timeout_extra_sec,
            left_error_recovery_action_name=_none_if_empty(args.left_error_recovery_action_name),
            right_error_recovery_action_name=_none_if_empty(args.right_error_recovery_action_name),
            left_topic_namespace=args.left_topic_namespace,
            right_topic_namespace=args.right_topic_namespace,
            left_gripper_enabled=args.left_gripper_enabled,
            right_gripper_enabled=args.right_gripper_enabled,
            left_gripper_command_mode=args.left_gripper_command_mode,
            right_gripper_command_mode=args.right_gripper_command_mode,
            left_gripper_command_topic=args.left_gripper_command_topic,
            right_gripper_command_topic=args.right_gripper_command_topic,
            left_gripper_action_name=args.left_gripper_action_name,
            right_gripper_action_name=args.right_gripper_action_name,
            left_gripper_joint_state_topic=args.left_gripper_joint_state_topic,
            right_gripper_joint_state_topic=args.right_gripper_joint_state_topic,
            left_gripper_joint_name=args.left_gripper_joint_name,
            right_gripper_joint_name=args.right_gripper_joint_name,
            left_gripper_activation_service=_none_if_empty(
                args.left_gripper_activation_service
            ),
            right_gripper_activation_service=_none_if_empty(
                args.right_gripper_activation_service
            ),
            gripper_closed_position=args.gripper_closed_position,
            gripper_open_position=args.gripper_open_position,
            gripper_open_width=args.gripper_open_width,
            gripper_default_max_effort=args.gripper_default_max_effort,
            gripper_default_max_velocity=args.gripper_default_max_velocity,
            activate_grippers_on_reset=args.activate_grippers_on_reset,
        )

        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        api = DualFrankaRobotiqRpcApi(node, env)
        server = zerorpc.Server(api)
        server.bind(args.bind)
        node.get_logger().info(f'DualFrankaRobotiq ZeroRPC server listening on {args.bind}')
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if env is not None:
            env.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = _build_parser()
    # launch_ros appends ROS-specific arguments (for example "--ros-args") to
    # every Node executable. This server owns its ROS node internally and uses
    # argparse for its transport/env configuration, so ignore those ROS args.
    args, _unknown_ros_args = parser.parse_known_args()
    serve(args)


if __name__ == '__main__':
    main()
