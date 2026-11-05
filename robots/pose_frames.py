"""Shared rigid transforms. Extrinsics are [x,y,z,qw,qx,qy,qz], in metres."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers.") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly {size} finite numbers.")
    return result


class BaseToWorldTransform:
    """An explicit T_world_base, independent of the underlying robot SDK."""

    def __init__(self, base_pose_in_world: Any, *, name: str):
        pose = _finite_vector(base_pose_in_world, 7, name)
        if np.linalg.norm(pose[3:]) < 1e-12:
            raise ValueError(f"{name} must contain a nonzero quaternion.")
        self._translation = pose[:3].copy()
        self._rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])

    def apply(self, tcp_pose_in_base: Any) -> np.ndarray:
        """Return T_world_tcp in [x,y,z,qw,qx,qy,qz] order."""
        pose = _finite_vector(tcp_pose_in_base, 7, "tcp_pose_in_base")
        if np.linalg.norm(pose[3:]) < 1e-12:
            raise ValueError("tcp_pose_in_base must contain a nonzero quaternion.")
        position = self._rotation.apply(pose[:3]) + self._translation
        rotation = self._rotation * Rotation.from_quat(pose[[4, 5, 6, 3]])
        quaternion = rotation.as_quat()
        return np.concatenate([position, quaternion[[3, 0, 1, 2]]])

    def apply_pose6(self, pose_in_base: Any) -> np.ndarray:
        """Transform an absolute [x,y,z,rotvec] pose into world coordinates."""
        pose = _finite_vector(pose_in_base, 6, "pose_in_base")
        position = self._rotation.apply(pose[:3]) + self._translation
        rotation = self._rotation * Rotation.from_rotvec(pose[3:])
        return np.concatenate([position, rotation.as_rotvec()])

    def delta_to_world(self, delta_in_base: Any) -> np.ndarray:
        delta = _finite_vector(delta_in_base, 6, "delta_in_base")
        return np.concatenate([self._rotation.apply(delta[:3]), self._rotation.apply(delta[3:])])

    def delta_to_base(self, delta_in_world: Any) -> np.ndarray:
        delta = _finite_vector(delta_in_world, 6, "delta_in_world")
        inverse = self._rotation.inv()
        return np.concatenate([inverse.apply(delta[:3]), inverse.apply(delta[3:])])
