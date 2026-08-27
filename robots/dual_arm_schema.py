"""Canonical X-embodiment schema shared by supported dual-arm adapters.

The schema intentionally contains only task-space state and RGB observations:

* observation: absolute 6D EE pose, physical gripper width (metres), and RGB cameras;
* action: 6D EE delta and target gripper width for the next observation.

Keeping these names and units in one place prevents robot-specific datasets from
silently drifting apart.
"""

from __future__ import annotations

from typing import Any, Mapping


AXES = ("x", "y", "z", "rx", "ry", "rz")
SIDES = ("left", "right")
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

    minimum = max(0.0, float(min_width or 0.0))
    return minimum, max(minimum, float(max_width))


def clip_gripper_width(width: float, min_width: float | None, max_width: float) -> float:
    """Clamp a physical gripper width to configured hardware limits."""

    minimum, maximum = gripper_width_limits(min_width, max_width)
    return max(minimum, min(maximum, float(width)))


def width_from_normalized_command(
    command: float,
    min_width: float | None,
    max_width: float,
    *,
    reverse: bool = False,
) -> float:
    """Convert a legacy [0, 1] gripper command into a physical width."""

    normalized = max(0.0, min(1.0, float(command)))
    if reverse:
        normalized = 1.0 - normalized
    minimum, maximum = gripper_width_limits(min_width, max_width)
    return minimum + normalized * (maximum - minimum)
