"""Canonical X-embodiment schema shared by supported dual-arm adapters.

The schema intentionally contains only task-space state and RGB observations:

* observation: absolute 6D EE pose, physical gripper width (metres), and RGB cameras;
* action: 6D EE delta and effective commanded gripper target width (metres).

Keeping these names and units in one place prevents robot-specific datasets from
silently drifting apart.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from scripts.utils.dataset_utils import GRIPPER_ACTION_SEMANTICS_KEY, GRIPPER_ACTION_SEMANTICS


AXES = ("x", "y", "z", "rx", "ry", "rz")
SIDES = ("left", "right")
GRIPPER_WIDTH_KEYS = ("left_gripper_width", "right_gripper_width")
CAMERA_KEYS = ("left_wrist_image", "right_wrist_image", "head_image")


def action_feature_names(*, use_gripper: bool = True) -> tuple[str, ...]:
    """Return the ordered X-embodiment dual-arm action names."""

    names = tuple(
        f"{side}_delta_ee_pose.{axis}"
        for side in SIDES
        for axis in AXES
    )
    if use_gripper:
        names += tuple(f"{side}_gripper_width" for side in SIDES)
    return names


def observation_state_feature_names(*, use_gripper: bool = True) -> tuple[str, ...]:
    """Return ordered low-dimensional observation names (all values in SI units)."""

    names = tuple(
        f"{side}_ee_pose.{axis}"
        for side in SIDES
        for axis in AXES
    )
    if use_gripper:
        names += tuple(f"{side}_gripper_width" for side in SIDES)
    return names


def action_features(*, use_gripper: bool = True) -> dict[str, type]:
    """Return the canonical action feature mapping expected by LeRobot."""

    return {name: float for name in action_feature_names(use_gripper=use_gripper)}


def observation_features(
    cameras: Mapping[str, Any],
    *,
    use_gripper: bool = True,
) -> dict[str, Any]:
    """Return canonical low-dimensional and available RGB camera features.

    Recording configuration always provides the three names in ``CAMERA_KEYS``.
    Keeping this helper permissive for an empty camera mapping preserves offline
    adapter tests and explicit no-camera debug sessions.
    """

    features: dict[str, Any] = {
        name: float for name in observation_state_feature_names(use_gripper=use_gripper)
    }
    for name, camera in cameras.items():
        features[name] = (camera.height, camera.width, 3)
    return features


def gripper_width_limits(
    min_width: float | None,
    max_width: float,
) -> tuple[float, float]:
    """Return ordered physical gripper limits in metres."""

    minimum = max(0.0, finite_gripper_value(0.0 if min_width is None else min_width))
    return minimum, max(minimum, finite_gripper_value(max_width))


def clip_gripper_width(width: float, min_width: float | None, max_width: float) -> float:
    """Clamp a physical gripper width to configured hardware limits."""

    minimum, maximum = gripper_width_limits(min_width, max_width)
    return max(minimum, min(maximum, finite_gripper_value(width)))


def width_from_normalized_command(
    command: float,
    min_width: float | None,
    max_width: float,
    *,
    reverse: bool = False,
) -> float:
    """Convert a legacy [0, 1] gripper command into a physical width."""

    normalized = max(0.0, min(1.0, finite_gripper_value(command)))
    if reverse:
        normalized = 1.0 - normalized
    minimum, maximum = gripper_width_limits(min_width, max_width)
    return minimum + normalized * (maximum - minimum)


def metric_gripper_keys(features: Mapping[str, Any]) -> tuple[str, ...]:
    names = features.get("action", {}).get("names") or []
    return tuple(key for key in GRIPPER_WIDTH_KEYS if key in names)


def command_target_action(action, sent_action, features) -> dict[str, Any]:
    """Keep arm labels unchanged, but require effective targets from send_action."""
    result = dict(action)
    for key in metric_gripper_keys(features):
        if not isinstance(sent_action, Mapping) or key not in sent_action:
            raise ValueError(f"send_action did not return effective target {key}")
        result[key] = finite_gripper_value(sent_action[key])
    return result


def finite_gripper_value(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gripper command/limit must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("Gripper command/limit must be finite")
    return value


def validate_recording_semantics(info, features) -> None:
    if metric_gripper_keys(features) and info.get(GRIPPER_ACTION_SEMANTICS_KEY) != GRIPPER_ACTION_SEMANTICS:
        raise ValueError("Cannot resume old/unknown gripper action semantics; use a new dataset name/root")
