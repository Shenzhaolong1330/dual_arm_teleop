"""Offline TCP pivot calibration and world-frame fitting. Never opens hardware.

pivot input: {"poses": [[x,y,z,rx,ry,rz], ...]} at a stationary contact point.
world input: {"source_points": [[x,y,z], ...], "world_points": [[x,y,z], ...]}.
Both use metres. Rotations in pivot poses are absolute rotation vectors, radians.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _points(value, dimensions, minimum):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != dimensions or len(p) < minimum or not np.all(np.isfinite(p)):
        raise ValueError(f"Need at least {minimum} finite {dimensions}D samples")
    return p


def pivot_calibration(poses, max_rms_m=0.002):
    poses = _points(poses, 6, 6)
    rotations = Rotation.from_rotvec(poses[:, 3:]).as_matrix()
    A = np.concatenate([rotations, -np.broadcast_to(np.eye(3), rotations.shape)], axis=2).reshape(-1, 6)
    b = -poses[:, :3].reshape(-1)
    solution, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    if rank != 6 or condition > 1e4:
        raise ValueError("Pivot samples are unobservable; rotate around multiple independent axes")
    residual = (A @ solution - b).reshape(-1, 3)
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if rms > max_rms_m:
        raise ValueError(f"Pivot residual {rms:.6f} m exceeds {max_rms_m:.6f} m")
    return {"tcp_translation_in_native_m": solution[:3].tolist(),
            "stationary_point_in_world_m": solution[3:].tolist(),
            "rms_m": rms, "condition_number": condition,
            "orientation_calibrated": False, "sample_count": len(poses)}


def world_calibration(source, target, max_rms_m=0.002):
    source, target = _points(source, 3, 3), _points(target, 3, 3)
    if source.shape != target.shape:
        raise ValueError("Source and world points must be paired")
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    if np.linalg.matrix_rank(a, tol=1e-8) < 2 or np.linalg.matrix_rank(b, tol=1e-8) < 2:
        raise ValueError("World calibration needs non-collinear points")
    u, _, vt = np.linalg.svd(a.T @ b)
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ fix @ u.T
    t = target.mean(axis=0) - r @ source.mean(axis=0)
    residual = source @ r.T + t - target
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if rms > max_rms_m:
        raise ValueError(f"World residual {rms:.6f} m exceeds {max_rms_m:.6f} m")
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    return {"pose_in_world": np.r_[t, q].tolist(), "rms_m": rms, "sample_count": len(source)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["pivot", "world"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rms-mm", type=float, default=2.0)
    a = p.parse_args()
    if not np.isfinite(a.max_rms_mm) or a.max_rms_mm <= 0:
        raise ValueError("RMS tolerance must be positive")
    data = json.loads(a.input.read_text())
    if a.mode == "pivot":
        result = pivot_calibration(data["poses"], a.max_rms_mm / 1000)
    else:
        result = world_calibration(data["source_points"], data["world_points"], a.max_rms_mm / 1000)
    result.update(method=a.mode, input=str(a.input.resolve()), hardware_connected=False)
    # A calibration tool must never silently overwrite previous measurements.
    with a.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
