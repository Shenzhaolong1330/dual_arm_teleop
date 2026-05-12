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
    ):
        del wait
        if not delta:
            raise NotImplementedError('Only delta=True is supported by this RPC server adapter.')
        action = {
            'left_arm': {
                'motion': {
                    'translation': list(left_delta[:3]),
                    'rotation_rotvec': list(left_delta[3:]),
                }
            },
            'right_arm': {
                'motion': {
                    'translation': list(right_delta[:3]),
                    'rotation_rotvec': list(right_delta[3:]),
                }
            },
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
