"""Pure validation helpers for Flexiv raw force feedback signals.

This module deliberately has no RDK or LeRobot dependencies. The production
robot adapter and the read-only force monitor both use these helpers so the
signal source, component order, finite-value checks, and error semantics stay
identical.
"""

from __future__ import annotations

import math
from typing import Any


WRENCH_SOURCE = "robot.states().ext_wrench_in_tcp_raw"
WRENCH_FIELD = "ext_wrench_in_tcp_raw"
WRENCH_FRAME = "tcp"
WRENCH_ORDER = ("fx", "fy", "fz", "mx", "my", "mz")
WRENCH_UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
GRIPPER_FORCE_SOURCE = "gripper.states().force"
GRIPPER_FORCE_FIELD = "force"
GRIPPER_FORCE_UNIT = "N"


class FlexivRawSignalReadError(ValueError):
    """Raised when a required raw force signal is absent or invalid."""


def _finite_float(value: Any, *, source: str, component: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FlexivRawSignalReadError(
            f"{source} component {component!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise FlexivRawSignalReadError(
            f"{source} component {component!r} is not finite: {number!r}"
        )
    return number


def read_ext_wrench_in_tcp_raw(
    robot_states: Any,
    *,
    side: str,
) -> tuple[float, float, float, float, float, float]:
    """Read exactly ``[fx, fy, fz, mx, my, mz]`` from one RobotStates snapshot."""

    try:
        raw_values = getattr(robot_states, WRENCH_FIELD)
    except AttributeError as exc:
        raise FlexivRawSignalReadError(
            f"{side} {WRENCH_SOURCE} field {WRENCH_FIELD!r} is missing"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise FlexivRawSignalReadError(
            f"{side} {WRENCH_SOURCE} field {WRENCH_FIELD!r} read failed: {exc}"
        ) from exc

    if isinstance(raw_values, (str, bytes)):
        raise FlexivRawSignalReadError(
            f"{side} {WRENCH_SOURCE} field {WRENCH_FIELD!r} must be an iterable of 6 numeric values"
        )
    try:
        values = list(raw_values)
    except Exception as exc:  # noqa: BLE001
        raise FlexivRawSignalReadError(
            f"{side} {WRENCH_SOURCE} field {WRENCH_FIELD!r} must be an iterable of length 6"
        ) from exc
    if len(values) != len(WRENCH_ORDER):
        raise FlexivRawSignalReadError(
            f"{side} {WRENCH_SOURCE} field {WRENCH_FIELD!r} must have length 6, got {len(values)}"
        )

    converted = [
        _finite_float(value, source=f"{side} {WRENCH_SOURCE}", component=component)
        for component, value in zip(WRENCH_ORDER, values, strict=True)
    ]
    return tuple(converted)  # type: ignore[return-value]


def read_gripper_force(gripper_states: Any, *, side: str) -> float:
    """Read the signed force from one GripperStates snapshot without fallback."""

    try:
        raw_value = getattr(gripper_states, GRIPPER_FORCE_FIELD)
    except AttributeError as exc:
        raise FlexivRawSignalReadError(
            f"{side} gripper field {GRIPPER_FORCE_FIELD!r} is missing "
            f"(source: {GRIPPER_FORCE_SOURCE})"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise FlexivRawSignalReadError(
            f"{side} {GRIPPER_FORCE_SOURCE} field {GRIPPER_FORCE_FIELD!r} read failed: {exc}"
        ) from exc

    return _finite_float(
        raw_value,
        source=f"{side} {GRIPPER_FORCE_SOURCE}",
        component=GRIPPER_FORCE_FIELD,
    )
