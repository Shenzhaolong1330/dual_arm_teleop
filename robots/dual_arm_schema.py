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

from scripts.utils.dataset_utils import (
    ACTION_SEMANTICS, ACTION_SEMANTICS_KEY,
    GRIPPER_ACTION_SEMANTICS_KEY, GRIPPER_ACTION_SEMANTICS,
)


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


def cartesian_action_keys(features: Mapping[str, Any]) -> tuple[str, ...]:
    names = features.get("action", {}).get("names") or []
    return tuple(key for key in action_feature_names(use_gripper=False) if key in names)


def command_target_action(action, sent_action, features) -> dict[str, Any]:
    """Record adapter-returned effective targets in the public action frame.

    Fail before saving if a target is missing or nonfinite. Controller
    interpolation and measured motion are separate from these target deltas.
    """
    result = dict(action)
    for key in cartesian_action_keys(features) + metric_gripper_keys(features):
        if not isinstance(sent_action, Mapping) or key not in sent_action:
            raise ValueError(f"send_action did not return effective target {key}")
        try:
            value = float(sent_action[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Effective target {key} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"Effective target {key} must be finite")
        result[key] = value
    return result


def finite_gripper_value(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gripper command/limit must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("Gripper command/limit must be finite")
    return value


def recording_semantics(features, robot_config=None, teleop_config=None) -> dict[str, Any]:
    """Version changed arm labels and persist the explicitly configured frame."""
    info: dict[str, Any] = {}
    if cartesian_action_keys(features):
        info[ACTION_SEMANTICS_KEY] = ACTION_SEMANTICS
    if metric_gripper_keys(features):
        info[GRIPPER_ACTION_SEMANTICS_KEY] = GRIPPER_ACTION_SEMANTICS
    frame = getattr(robot_config, "observation_pose_frame", None)
    if frame in ("base", "world"):
        info["observation_pose_frame"] = frame
        info["action_pose_frame"] = "world" if hasattr(robot_config, "left_mount_yaw_deg") else frame
        info["orientation_representation"] = "rotvec"
        if frame == "world":
            extrinsics = {
                side: getattr(robot_config, f"{side}_base_pose_in_world", None)
                for side in SIDES
            }
            if all(value is not None for value in extrinsics.values()):
                info["base_poses_in_world"] = extrinsics
    if teleop_config is not None:
        info["quest_processing"] = teleop_config.processing_metadata()
    if robot_config is not None and hasattr(robot_config, "physical_calibration"):
        from robots.physical_alignment import physical_metadata
        info.update(physical_metadata(robot_config))
    return info


def validate_recording_semantics(info, features, expected_semantics=None) -> None:
    if metric_gripper_keys(features) and info.get(GRIPPER_ACTION_SEMANTICS_KEY) != GRIPPER_ACTION_SEMANTICS:
        raise ValueError("Cannot resume old/unknown gripper action semantics; use a new dataset name/root")
    expected = recording_semantics(features) if expected_semantics is None else expected_semantics
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"Cannot resume different/unknown {key}; use a new dataset name/root")
