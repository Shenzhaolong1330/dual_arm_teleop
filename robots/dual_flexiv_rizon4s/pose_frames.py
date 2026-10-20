"""Rigid transforms for Flexiv TCP observations.

``base_pose_in_world`` describes T_world_base, with translation in metres and
a quaternion in RDK order [x, y, z, qw, qx, qy, qz]. Applying it computes
T_world_tcp = T_world_base @ T_base_tcp. Orientation is composed as rotations,
never by adding Euler angles or rotating the absolute rotation vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _validate_pose7(value: Any, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be [x, y, z, qw, qx, qy, qz].") from exc
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain exactly seven finite numbers.")
    if np.linalg.norm(pose[3:]) < 1e-12:
        raise ValueError(f"{name} must contain a nonzero quaternion.")
    return pose


class BaseToWorldTransform:
    """An explicit, validated extrinsic for one arm's RDK WORLD/base frame."""

    def __init__(self, base_pose_in_world: Any, *, name: str):
        pose = _validate_pose7(base_pose_in_world, name)
        self._translation = pose[:3].copy()
        self._rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])

    def apply(self, tcp_pose_in_base: Any) -> np.ndarray:
        """Return a new world-frame TCP pose in RDK quaternion order."""
        pose = _validate_pose7(tcp_pose_in_base, "tcp_pose_in_base")
        position = self._rotation.apply(pose[:3]) + self._translation
        rotation = self._rotation * Rotation.from_quat(pose[[4, 5, 6, 3]])
        quaternion_xyzw = rotation.as_quat()
        return np.concatenate([position, quaternion_xyzw[[3, 0, 1, 2]]])
