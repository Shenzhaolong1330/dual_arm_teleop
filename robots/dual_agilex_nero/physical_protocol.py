"""Hardware-free, versioned Nero wire contract and server receipt helpers.

Copy this module next to nero_interface_server.py on the NUC. Importing it does
not import the SDK, open CAN, enable joints, or start the RPC service.
"""
from __future__ import annotations

import time
from functools import wraps
import numpy as np
from scipy.spatial.transform import Rotation

PROTOCOL = "nero.cartesian-target.v2"


def vector6(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected six finite Cartesian coordinates")
    return value


def rpy_pose_to_rotvec(value):
    value = vector6(value)
    return np.r_[value[:3], Rotation.from_euler("xyz", value[3:]).as_rotvec()]


def rotvec_pose_to_rpy(value):
    value = vector6(value)
    return np.r_[value[:3], Rotation.from_rotvec(value[3:]).as_euler("xyz")]


def pose_delta(before, after):
    before, after = vector6(before), vector6(after)
    rotation = Rotation.from_rotvec(after[3:]) * Rotation.from_rotvec(before[3:]).inv()
    return np.r_[after[:3] - before[:3], rotation.as_rotvec()]


def tool_pose(value, tcp_in_native, *, inverse=False):
    """Change the tracked tool point/axes, including the translation lever arm."""
    value = vector6(value)
    tool = np.asarray(tcp_in_native, dtype=float)
    if tool.shape != (7,) or not np.all(np.isfinite(tool)) or np.linalg.norm(tool[3:]) < 1e-12:
        raise ValueError("TCP offset must be [x,y,z,qw,qx,qy,qz]")
    offset_r = Rotation.from_quat(tool[[4, 5, 6, 3]])
    offset_p = tool[:3]
    if inverse:
        offset_p = -offset_r.inv().apply(offset_p)
        offset_r = offset_r.inv()
    rotation = Rotation.from_rotvec(value[3:])
    return np.r_[value[:3] + rotation.apply(offset_p), (rotation * offset_r).as_rotvec()]


def integrate_pose(reference, delta):
    reference, delta = vector6(reference), vector6(delta)
    rotation = Rotation.from_rotvec(delta[3:]) * Rotation.from_rotvec(reference[3:])
    return np.r_[reference[:3] + delta[:3], rotation.as_rotvec()]


def serialized_cartesian(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        # ZeroRPC uses greenlets. A regular threading.RLock would not serialize
        # different greenlets on the same OS thread.
        from gevent.lock import RLock
        if not hasattr(self, "_physical_lock"):
            self._physical_lock = RLock()
        with self._physical_lock:
            return method(self, *args, **kwargs)
    return call


def validate_receipt(receipt):
    if not isinstance(receipt, dict) or receipt.get("protocol") != PROTOCOL:
        raise RuntimeError("Nero v2 effective-target receipt is required; update the NUC server")
    if type(receipt.get("accepted")) is not bool:
        raise RuntimeError("Nero receipt must have an explicit boolean outcome")
    if receipt.get("accepted") is not True:
        if receipt.get("reason") != "rejected":
            raise RuntimeError("Nero command outcome is unknown; frame must not be recorded")
        return np.zeros(6)
    before = vector6(receipt.get("reference_pose"))
    after = vector6(receipt.get("target_pose"))
    effective = vector6(receipt.get("effective_delta"))
    if not np.allclose(pose_delta(before, after), effective, atol=1e-8, rtol=0):
        raise RuntimeError("Nero returned an inconsistent Cartesian target receipt")
    return effective


class PhysicalProtocolMixin:
    """Methods installed on the audited NUC server, retaining its legacy API."""

    def get_teleop_protocol(self):
        return {
            "protocol": PROTOCOL,
            "pose_frame": "arm_base",
            "orientation": "rotvec",
            "rotation_delta": "spatial_left_multiply",
            "legacy_orientation": "extrinsic_xyz_rpy",
            "position_unit": "m",
            "rotation_unit": "rad",
            "tcp_offset_rpy": list(self.tcp_offset),
        }

    def get_ee_pose_v2(self, side):
        if side not in ("left", "right"):
            raise ValueError("Unknown Nero arm")
        robot = getattr(self, f"{side}_robot")
        if robot is None:
            raise RuntimeError(f"{side} arm is unavailable")
        robot.set_tcp_offset(self.tcp_offset)
        feedback = robot.get_tcp_pose()
        if feedback is None:
            raise RuntimeError(f"{side} TCP feedback is unavailable")
        return {
            "protocol": PROTOCOL,
            "pose": rpy_pose_to_rotvec(feedback.msg).tolist(),
            "feedback_timestamp": float(feedback.timestamp),
            "server_monotonic_ns": time.monotonic_ns(),
        }

    def _physical_prepare_delta(self, robot_arm, legacy_pose):
        request = getattr(self, "_physical_request", None)
        if request is None:
            return legacy_pose
        side = robot_arm.split('_')[0]
        current = getattr(self, f"{side}_cur_pose")
        if current is None:
            solver = getattr(self, f"{side}_ik_solver")
            current = solver.fk_pose(solver.state.q_prev)
        native_reference = rpy_pose_to_rotvec(current)
        reference = tool_pose(native_reference, request["tcp"])
        target = integrate_pose(reference, request["delta"])
        native_target = tool_pose(target, request["tcp"], inverse=True)
        return rotvec_pose_to_rpy(pose_delta(native_reference, native_target))

    @serialized_cartesian
    def servo_p_OL_v2(self, robot_arm, pose, tcp_in_native=None):
        # Keep the legacy safety limits and IK behavior. The instrumented
        # implementation captures the reference AFTER any idle re-sync and the
        # accepted target AFTER clipping, attenuation and orientation freezing.
        if robot_arm not in ("left_robot", "right_robot"):
            raise ValueError("Unknown Nero arm")
        delta = vector6(pose)
        tcp = [0, 0, 0, 1, 0, 0, 0] if tcp_in_native is None else tcp_in_native
        tool_pose(np.zeros(6), tcp)  # validate before any control call
        self._physical_receipt = None
        self._physical_outcome = "unknown"
        self._physical_request = {"delta": delta, "tcp": tcp}
        try:
            accepted = self.servo_p_OL(robot_arm, [0.0] * 6, delta=True)
        finally:
            self._physical_request = None
        result = {
            "protocol": PROTOCOL,
            "accepted": accepted is True,
            "robot_arm": robot_arm,
            "server_monotonic_ns": time.monotonic_ns(),
            "clock_domain": "nero-nuc-monotonic",
        }
        if accepted is not True:
            return {**result, "reason": self._physical_outcome}
        if self._physical_receipt is None:
            raise RuntimeError("Server receipt hook missing; command outcome is unconfirmed")
        before, after = self._physical_receipt
        before, after = rpy_pose_to_rotvec(before), rpy_pose_to_rotvec(after)
        before, after = tool_pose(before, tcp), tool_pose(after, tcp)
        return {
            **result,
            "reference_pose": before.tolist(),
            "target_pose": after.tolist(),
            "effective_delta": pose_delta(before, after).tolist(),
        }
