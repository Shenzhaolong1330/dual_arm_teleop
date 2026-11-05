"""Versioned Quest response parameters, independent of installation geometry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


CROSS_EMBODIMENT_PROFILE = "cross_embodiment_v1"
_PARAMETERS = {
    "left_pose_scaler": [1.0, 1.0],
    "right_pose_scaler": [1.0, 1.0],
    "action_smoothing_method": "one_euro",
    "action_smoothing_alpha": 0.25,  # Used only if a custom profile selects EMA.
    "action_smoothing_freq": 30.0,
    "action_smoothing_min_cutoff": 0.8,
    "action_smoothing_beta": 0.25,
    "action_smoothing_d_cutoff": 1.0,
    "action_missing_hold_frames": 2,
    "action_missing_decay": 0.5,
    "action_deadband_translation": 0.0002,
    "action_deadband_rotation": 0.003,
    "action_spike_translation": 0.05,
    "action_spike_rotation": 0.1,
    "gripper_trigger_deadzone": 0.02,
    "gripper_trigger_gamma": 1.0,
}
PROCESSING_PARAMETER_KEYS = tuple(_PARAMETERS)
MAPPING_KEYS = (
    "left_channel_signs", "right_channel_signs", "position_axis_order",
    "rotation_axis_order", "mirror_teleop",
)


def processing_profile_parameters(name: str = CROSS_EMBODIMENT_PROFILE) -> dict[str, Any]:
    if name != CROSS_EMBODIMENT_PROFILE:
        raise ValueError(f"Unknown Quest processing_profile: {name!r}")
    return deepcopy(_PARAMETERS)


def resolve_processing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a named profile, rejecting per-robot response drift.

    Geometry, routing, connection details and physical gripper limits are not
    profile parameters. Explicitly use processing_profile=null for custom work.
    """
    resolved = deepcopy(dict(config))
    name = resolved.get("processing_profile")
    if name is None:
        return resolved
    parameters = processing_profile_parameters(name)
    for key, expected in parameters.items():
        if key in resolved and resolved[key] != expected:
            raise ValueError(f"Quest processing_profile {name!r} conflicts with {key}; remove the override or select a custom profile")
    if "pose_scaler" in resolved and resolved["pose_scaler"] != [1.0, 1.0]:
        raise ValueError(f"Quest processing_profile {name!r} conflicts with pose_scaler")
    resolved.update(parameters)
    return resolved


def teleop_processing_metadata(config) -> dict[str, Any]:
    parameters = {key: deepcopy(getattr(config, key)) for key in PROCESSING_PARAMETER_KEYS}
    name = config.processing_profile
    if name is not None:
        resolve_processing_config({"processing_profile": name, **parameters})
    return {
        "profile": name or "custom",
        "parameters": parameters,
        "mapping": {key: deepcopy(getattr(config, key)) for key in MAPPING_KEYS},
        "gripper_width_limits_m": [float(config.gripper_min_width), float(config.gripper_max_open)],
    }
