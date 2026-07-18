"""Read-only real-time monitor for the two Flexiv arm/gripper force signals.

This tool intentionally does not use the production ``FlexivDualArm`` adapter.
That adapter also owns cameras, teleoperation, motion-mode setup, homing, and
dataset observations.  The monitor only constructs RDK ``Robot`` and
``Gripper`` handles, enables the configured gripper devices, and reads state.

The only arm signal read here is
``robot.states().ext_wrench_in_tcp_raw`` in ``[fx, fy, fz, mx, my, mz]`` order.
The only gripper signal read here is ``gripper.states().force``.  Values are
only converted to ``float`` for logging/validation; they are not filtered,
clipped, normalized, biased, or sign-changed.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml


WRENCH_FIELD = "ext_wrench_in_tcp_raw"
GRIPPER_FORCE_FIELD = "force"
WRENCH_LABELS = ("fx", "fy", "fz", "mx", "my", "mz")

RERUN_WAVEFORM_GROUPS = (
    ("left TCP force [N]", "flexiv/left/tcp/force", WRENCH_LABELS[:3]),
    ("left TCP moment [Nm]", "flexiv/left/tcp/moment", WRENCH_LABELS[3:]),
    ("right TCP force [N]", "flexiv/right/tcp/force", WRENCH_LABELS[:3]),
    ("right TCP moment [Nm]", "flexiv/right/tcp/moment", WRENCH_LABELS[3:]),
    ("gripper force [N]", "flexiv/gripper", ("left_force", "right_force")),
)

RERUN_ENTITY_PATHS = {
    "left_tcp_raw": tuple(
        f"flexiv/left/tcp/{'force' if index < 3 else 'moment'}/{label}"
        for index, label in enumerate(WRENCH_LABELS)
    ),
    "right_tcp_raw": tuple(
        f"flexiv/right/tcp/{'force' if index < 3 else 'moment'}/{label}"
        for index, label in enumerate(WRENCH_LABELS)
    ),
    "left_gripper_force": "flexiv/gripper/left_force",
    "right_gripper_force": "flexiv/gripper/right_force",
}


class MonitorError(RuntimeError):
    """Expected configuration, connection, read, or visualization failure."""


class SampleReadError(MonitorError):
    """A required RDK state field could not be read or validated."""


@dataclass(frozen=True)
class MonitorConfig:
    """The hardware identifiers read from the existing Flexiv YAML file."""

    source_path: Path
    left_robot_sn: str
    right_robot_sn: str
    left_gripper_name: str
    right_gripper_name: str


@dataclass(frozen=True)
class ForceSample:
    """One unmodified pair of arm/gripper force measurements."""

    timestamp: float
    left_tcp_raw: tuple[float, float, float, float, float, float]
    left_gripper_force: float
    right_tcp_raw: tuple[float, float, float, float, float, float]
    right_gripper_force: float

    def rerun_scalars(self) -> tuple[tuple[str, float], ...]:
        """Return the fourteen scalar entity/value pairs in fixed signal order."""

        return (
            *tuple(zip(RERUN_ENTITY_PATHS["left_tcp_raw"], self.left_tcp_raw)),
            (RERUN_ENTITY_PATHS["left_gripper_force"], self.left_gripper_force),
            *tuple(zip(RERUN_ENTITY_PATHS["right_tcp_raw"], self.right_tcp_raw)),
            (RERUN_ENTITY_PATHS["right_gripper_force"], self.right_gripper_force),
        )


@dataclass
class ConnectedDevices:
    """RDK handles owned by the monitor, with no command-oriented cleanup."""

    left_robot: Any | None = None
    right_robot: Any | None = None
    left_gripper: Any | None = None
    right_gripper: Any | None = None

    def close(self) -> None:
        """Drop RDK handles without calling Stop or any other robot command."""

        self.left_gripper = None
        self.right_gripper = None
        self.left_robot = None
        self.right_robot = None


def _default_robot_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "robots" / "flexiv_config.yaml"


def _required_config_string(robot_cfg: Mapping[str, Any], key: str, path: Path) -> str:
    value = robot_cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MonitorError(f"{path}: robot.{key} must be a non-empty string")
    return value.strip()


def load_monitor_config(config_path: Path) -> MonitorConfig:
    """Load only the existing Flexiv serial/name fields needed by this tool."""

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise MonitorError(f"cannot read robot config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MonitorError(f"invalid YAML in robot config {config_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise MonitorError(f"{config_path}: top-level YAML value must be a mapping")
    robot_cfg = document.get("robot")
    if not isinstance(robot_cfg, Mapping):
        raise MonitorError(f"{config_path}: missing top-level robot mapping")

    return MonitorConfig(
        source_path=config_path,
        left_robot_sn=_required_config_string(robot_cfg, "left_robot_sn", config_path),
        right_robot_sn=_required_config_string(robot_cfg, "right_robot_sn", config_path),
        left_gripper_name=_required_config_string(robot_cfg, "left_gripper_name", config_path),
        right_gripper_name=_required_config_string(robot_cfg, "right_gripper_name", config_path),
    )


def _load_flexivrdk() -> Any:
    try:
        import flexivrdk  # noqa: PLC0415
    except ImportError as exc:
        raise MonitorError(
            "flexivrdk is not installed in the active Python environment; "
            "run this tool from the dual_arm_teleop environment"
        ) from exc
    return flexivrdk


def _configure_robot(robot: Any, side: str, *, clear_fault: bool, enable_robot: bool) -> None:
    """Perform only explicitly requested dangerous robot operations."""

    if clear_fault:
        try:
            has_fault = bool(robot.fault())
        except Exception as exc:  # noqa: BLE001
            raise MonitorError(f"{side} robot fault status read failed: {exc}") from exc
        if has_fault:
            try:
                cleared = robot.ClearFault()
            except Exception as exc:  # noqa: BLE001
                raise MonitorError(f"{side} robot ClearFault failed: {exc}") from exc
            if not cleared:
                raise MonitorError(f"{side} robot ClearFault returned failure")

    if enable_robot:
        try:
            robot.Enable()
        except Exception as exc:  # noqa: BLE001
            raise MonitorError(f"{side} robot Enable failed: {exc}") from exc


def connect_devices(
    config: MonitorConfig,
    *,
    rdk_module: Any | None = None,
    clear_fault: bool = False,
    enable_robot: bool = False,
) -> ConnectedDevices:
    """Construct direct RDK handles and select each configured gripper device.

    Flexiv RDK's ``Robot`` constructor connects by the configured serial number.
    No camera, teleoperator, policy, dataset, tool, motion mode, Home, Move,
    Grasp, Init, or ZeroFTSensor object/action is created here.
    """

    rdk = _load_flexivrdk() if rdk_module is None else rdk_module
    devices = ConnectedDevices()
    try:
        for side, serial in (
            ("left", config.left_robot_sn),
            ("right", config.right_robot_sn),
        ):
            try:
                robot = rdk.Robot(serial)
            except Exception as exc:  # noqa: BLE001
                raise MonitorError(
                    f"{side} robot connection failed for serial {serial!r}: {exc}"
                ) from exc
            setattr(devices, f"{side}_robot", robot)
            _configure_robot(
                robot,
                side,
                clear_fault=clear_fault,
                enable_robot=enable_robot,
            )

        for side, robot, device_name in (
            ("left", devices.left_robot, config.left_gripper_name),
            ("right", devices.right_robot, config.right_gripper_name),
        ):
            try:
                gripper = rdk.Gripper(robot)
                gripper.Enable(device_name)
            except Exception as exc:  # noqa: BLE001
                raise MonitorError(
                    f"{side} gripper Enable failed for device {device_name!r}: {exc}"
                ) from exc
            setattr(devices, f"{side}_gripper", gripper)
    except Exception:
        devices.close()
        raise

    return devices


def _read_wrench(state: Any, side: str) -> tuple[float, float, float, float, float, float]:
    field_name = WRENCH_FIELD
    try:
        raw_values = getattr(state, field_name)
    except AttributeError as exc:
        raise SampleReadError(f"{side} robot field {field_name!r} is missing") from exc
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(f"{side} robot field {field_name!r} read failed: {exc}") from exc

    try:
        values = list(raw_values)
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(
            f"{side} robot field {field_name!r} is not an iterable six-dimensional value"
        ) from exc
    if len(values) != 6:
        raise SampleReadError(
            f"{side} robot field {field_name!r} must have length 6, got {len(values)}"
        )

    converted: list[float] = []
    for index, value in enumerate(values):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise SampleReadError(
                f"{side} robot field {field_name!r}[{index}] is not numeric: {value!r}"
            ) from exc
        if not math.isfinite(numeric):
            raise SampleReadError(
                f"{side} robot field {field_name!r}[{index}] is not finite: {numeric!r}"
            )
        converted.append(numeric)

    return tuple(converted)  # type: ignore[return-value]


def _read_gripper_force(state: Any, side: str) -> float:
    field_name = GRIPPER_FORCE_FIELD
    try:
        raw_value = getattr(state, field_name)
    except AttributeError as exc:
        raise SampleReadError(f"{side} gripper field {field_name!r} is missing") from exc
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(f"{side} gripper field {field_name!r} read failed: {exc}") from exc

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise SampleReadError(
            f"{side} gripper field {field_name!r} is not numeric: {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise SampleReadError(
            f"{side} gripper field {field_name!r} is not finite: {value!r}"
        )
    return value


def read_force_sample(devices: ConnectedDevices, timestamp: float | None = None) -> ForceSample:
    """Read each arm state and each gripper state exactly once."""

    try:
        left_robot_state = devices.left_robot.states()
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(f"left robot field {WRENCH_FIELD!r} state read failed: {exc}") from exc
    left_tcp_raw = _read_wrench(left_robot_state, "left")

    try:
        right_robot_state = devices.right_robot.states()
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(f"right robot field {WRENCH_FIELD!r} state read failed: {exc}") from exc
    right_tcp_raw = _read_wrench(right_robot_state, "right")

    try:
        left_gripper_state = devices.left_gripper.states()
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(f"left gripper field {GRIPPER_FORCE_FIELD!r} state read failed: {exc}") from exc
    left_gripper_force = _read_gripper_force(left_gripper_state, "left")

    try:
        right_gripper_state = devices.right_gripper.states()
    except Exception as exc:  # noqa: BLE001
        raise SampleReadError(f"right gripper field {GRIPPER_FORCE_FIELD!r} state read failed: {exc}") from exc
    right_gripper_force = _read_gripper_force(right_gripper_state, "right")

    return ForceSample(
        timestamp=time.time() if timestamp is None else float(timestamp),
        left_tcp_raw=left_tcp_raw,
        left_gripper_force=left_gripper_force,
        right_tcp_raw=right_tcp_raw,
        right_gripper_force=right_gripper_force,
    )


def format_sample(sample: ForceSample) -> str:
    """Format a complete two-arm sample without changing any measurement value."""

    left = "[" + ", ".join(repr(value) for value in sample.left_tcp_raw) + "]"
    right = "[" + ", ".join(repr(value) for value in sample.right_tcp_raw) + "]"
    return (
        f"t={sample.timestamp:.6f} "
        f"left_tcp_raw={left} left_gripper_force={sample.left_gripper_force!r} "
        f"right_tcp_raw={right} right_gripper_force={sample.right_gripper_force!r}"
    )


class RerunLogger:
    """Small adapter that records exactly one Rerun scalar per signal."""

    def __init__(self, rerun_module: Any):
        self._rerun = rerun_module
        self._closed = False

    def log_sample(self, sample: ForceSample, sequence: int) -> None:
        try:
            self._rerun.set_time_sequence("sample", int(sequence))
        except Exception as exc:  # noqa: BLE001
            raise MonitorError(f"Rerun sequence timeline update failed: {exc}") from exc

        for entity_path, value in sample.rerun_scalars():
            try:
                self._rerun.log(entity_path, self._rerun.Scalars(value))
            except Exception as exc:  # noqa: BLE001
                raise MonitorError(
                    f"Rerun logging failed for scalar {entity_path!r}: {exc}"
                ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutdown = getattr(self._rerun, "rerun_shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Rerun shutdown failed: {exc}", file=sys.stderr)


def build_rerun_blueprint(rerun_module: Any) -> Any:
    """Build five visible time-series panels for the fourteen scalar entities."""

    blueprint = rerun_module.blueprint
    views = [
        blueprint.TimeSeriesView(
            origin=origin,
            contents="$origin/**",
            name=title,
        )
        for title, origin, _labels in RERUN_WAVEFORM_GROUPS
    ]
    return blueprint.Blueprint(
        blueprint.Vertical(*views, name="Flexiv force feedback"),
        auto_layout=False,
        auto_views=False,
    )


def start_rerun() -> RerunLogger:
    """Start the local Rerun viewer using the installed SDK's spawn API."""

    try:
        import rerun as rr  # noqa: PLC0415
    except ImportError as exc:
        raise MonitorError(
            "Rerun is unavailable; install the current rerun-sdk or use --no-rerun"
        ) from exc

    try:
        rr.init(
            "flexiv_force_feedback_monitor",
            spawn=True,
            default_blueprint=build_rerun_blueprint(rr),
        )
    except Exception as exc:  # noqa: BLE001
        shutdown = getattr(rr, "rerun_shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:  # noqa: BLE001
                pass
        raise MonitorError(f"failed to start Rerun viewer: {exc}") from exc
    return RerunLogger(rr)


def sample_and_report(
    devices: ConnectedDevices,
    *,
    sequence: int,
    print_sample: bool,
    rerun_logger: RerunLogger | None,
    emit: Callable[[str], None] = print,
) -> ForceSample:
    """Read one sample, optionally print it, and log all fourteen Rerun scalars."""

    sample = read_force_sample(devices)
    if print_sample:
        emit(format_sample(sample))
    if rerun_logger is not None:
        rerun_logger.log_sample(sample, sequence)
    return sample


def validate_sampling_options(frequency: float, duration: float | None, print_every: int) -> None:
    """Validate loop options before any RDK or Rerun handle is created."""

    if not math.isfinite(frequency) or frequency <= 0:
        raise MonitorError(f"frequency must be a finite positive number, got {frequency!r}")
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise MonitorError(f"duration must be finite and non-negative, got {duration!r}")
    if print_every <= 0:
        raise MonitorError(f"print-every must be a positive integer, got {print_every!r}")


def run_monitor(
    devices: ConnectedDevices,
    *,
    frequency: float,
    duration: float | None,
    print_every: int,
    rerun_logger: RerunLogger | None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> int:
    """Run the fixed-rate read loop until duration expires or Ctrl+C is raised."""

    validate_sampling_options(frequency, duration, print_every)

    period = 1.0 / frequency
    next_sample_at = monotonic_clock()
    end_at = None if duration is None else next_sample_at + duration
    sample_count = 0

    while end_at is None or monotonic_clock() < end_at:
        sample_count += 1
        sample_and_report(
            devices,
            sequence=sample_count - 1,
            print_sample=(sample_count % print_every == 0),
            rerun_logger=rerun_logger,
            emit=emit,
        )

        next_sample_at += period
        delay = next_sample_at - monotonic_clock()
        if delay > 0:
            sleep(delay)
        else:
            next_sample_at = monotonic_clock()

    return sample_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Flexiv dual-arm TCP raw wrench and signed gripper force "
            "with optional Rerun time-series plots."
        )
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=_default_robot_config_path(),
        help="Existing Flexiv robot YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=30.0,
        help="Sampling frequency in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds; omit to run until Ctrl+C (default: continuous)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one complete sample every N samples (default: %(default)s)",
    )
    rerun_group = parser.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--rerun",
        dest="rerun",
        action="store_true",
        help="Spawn the Rerun viewer with five time-series panels (default)",
    )
    rerun_group.add_argument(
        "--no-rerun",
        dest="rerun",
        action="store_false",
        help="Disable Rerun and print/read only",
    )
    parser.set_defaults(rerun=True)
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="DANGEROUS: explicitly call Robot.Enable(); default never does this",
    )
    parser.add_argument(
        "--clear-fault",
        action="store_true",
        help="DANGEROUS: explicitly clear a reported robot fault; default never does this",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    devices: ConnectedDevices | None = None
    rerun_logger: RerunLogger | None = None
    try:
        config = load_monitor_config(args.robot_config.expanduser().resolve())
        validate_sampling_options(args.frequency, args.duration, args.print_every)
        if args.rerun:
            rerun_logger = start_rerun()
        devices = connect_devices(
            config,
            clear_fault=args.clear_fault,
            enable_robot=args.enable_robot,
        )
        print(
            f"[INFO] monitoring config={config.source_path} "
            f"left={config.left_robot_sn} right={config.right_robot_sn}; "
            "TCP raw units=[N, N, N, Nm, Nm, Nm], gripper force unit=N; "
            "no motion command, ZeroFTSensor, camera, teleoperator, policy, or dataset is used"
        )
        run_monitor(
            devices,
            frequency=args.frequency,
            duration=args.duration,
            print_every=args.print_every,
            rerun_logger=rerun_logger,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received; stopping read loop without robot/gripper motion commands.", file=sys.stderr)
        return 130
    except MonitorError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if devices is not None:
            devices.close()
        if rerun_logger is not None:
            rerun_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
