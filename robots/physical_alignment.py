"""Explicit physical units and calibration provenance for cross-body datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PHYSICAL_VERSION = "x_embodiment_physical_v2"
TCP_DEFINITION = "finger_midpoint_x_closing_z_approach_v1"
WORLD_CONVENTION = "base_midpoint_x_forward_y_left_z_up_v1"
EMBODIMENT_IDS = {"franka_dual_arm": 0, "nero_dual_arm": 1, "flexiv_dual_arm": 2}


def physical_metadata(config):
    """Unverified tools remain collectable, but cannot pass cross-body merge."""
    calibration = getattr(config, "physical_calibration", None) or {}
    if not isinstance(calibration, dict):
        raise ValueError("physical_calibration must be a mapping")
    contract = {
        "version": PHYSICAL_VERSION,
        "position_unit": "m",
        "orientation": "absolute_rotvec_rad",
        "rotation_delta": "spatial_left_multiply_rad",
        "translation_delta": "accepted_target_position_difference_m_per_step",
        "gripper_state": "feedback_derived_total_aperture_m",
        "gripper_action": "commanded_total_aperture_m",
        "gripper_width_definition": "total_inner_finger_separation",
        "gripper_opening_direction": "zero_closed_larger_more_open",
        "sides": "robot_left_then_robot_right",
        "world_convention": calibration.get("world_convention", WORLD_CONVENTION),
        "tcp_definition": calibration.get("tcp_definition", "native_tcp_unverified"),
    }
    runtime = {
        "robot_type": config.name,
        "status": calibration.get("status", "unverified"),
        "source": calibration.get("source", "controller configuration; TCP qualification pending"),
        "gripper_limits_m": [getattr(config, "gripper_min_width", 0.0), config.gripper_max_open],
        "gripper_feedback_width_tolerance_m": config.gripper_feedback_width_tolerance_m,
        "gripper_feedback_model": ("prefer_measured_width_m_else_robotiq_register_linear_aperture_estimate"
                                   if config.name == "franka_dual_arm" else "sdk_width_m"),
        "calibration": calibration,
    }
    for side in ("left", "right"):
        for field in ("base_pose_in_world", "tcp_in_native", "tool_name"):
            value = getattr(config, f"{side}_{field}", None)
            if value is not None:
                runtime[f"{side}_{field}"] = value
    runtime["control_parameters"] = {
        name: getattr(config, name)
        for name in (
            "gripper_force", "gripper_speed", "gripper_control_mode",
            "gripper_feedback_force_limit", "gripper_feedback_timeout_sec",
            "max_cartesian_delta", "max_rotation_delta", "use_cartesian_servo_thread",
            "cartesian_servo_hz", "cartesian_servo_alpha",
        ) if hasattr(config, name)
    }
    protocol = getattr(config, "cartesian_protocol", None)
    if protocol is not None:
        runtime["cartesian_protocol"] = protocol
        runtime["expected_server_tcp_offset_rpy"] = config.expected_server_tcp_offset_rpy
        if protocol != "v2":
            runtime["status"] = "legacy_unverified"
    encoded = json.dumps(runtime, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    runtime["sha256"] = hashlib.sha256(encoded).hexdigest()
    return {"physical_contract": contract, "physical_calibration": runtime}


def require_cross_body_alignment(info):
    contract = info.get("physical_contract") or {}
    calibration = info.get("physical_calibration") or {}
    if (contract.get("version") != PHYSICAL_VERSION
            or contract.get("tcp_definition") != TCP_DEFINITION
            or calibration.get("status") not in {"model_based", "measured"}
            or not calibration.get("calibration", {}).get("evidence")):
        raise ValueError("Cross-body merge requires a qualified TCP and calibration evidence; native/old unknown data cannot be merged")


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema") != "dual_arm_physical_calibration_v1":
        raise ValueError("Unknown physical calibration schema")
    return data
