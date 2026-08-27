"""Eye-in-hand calibration for wrist cameras using ChArUco observations."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .charuco_calibration import (
        _detect_charuco,
        _estimate_board_pose,
        _make_board,
        _read_board_spec,
        _read_realsense_frame,
        _save_debug_image,
        _start_realsense,
    )
except ModuleNotFoundError:
    from charuco_calibration import (
        _detect_charuco,
        _estimate_board_pose,
        _make_board,
        _read_board_spec,
        _read_realsense_frame,
        _save_debug_image,
        _start_realsense,
    )


ARM_CAMERA_NAMES = {
    "left": "left_wrist_image",
    "right": "right_wrist_image",
}
HEAD_CAMERA_NAME = "head_image"

HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _print_sample_report(label: str, used_samples: list[dict[str, Any]], rejected_samples: list[dict[str, Any]]) -> None:
    print(f"{label}: used samples ({len(used_samples)})")
    if used_samples:
        for sample in used_samples:
            details = [f"id={sample.get('id')}"]
            if sample.get("image"):
                details.append(f"image={sample['image']}")
            if "charuco_corners" in sample:
                details.append(f"corners={sample['charuco_corners']}")
            print("  + " + ", ".join(details))
    else:
        print("  + none")

    print(f"{label}: rejected samples ({len(rejected_samples)})")
    if rejected_samples:
        for sample in rejected_samples:
            details = [f"id={sample.get('id')}", f"reason={sample.get('reason')}"]
            if sample.get("image"):
                details.append(f"image={sample['image']}")
            if "charuco_corners" in sample:
                details.append(f"corners={sample['charuco_corners']}")
            print("  - " + ", ".join(details))
    else:
        print("  - none")


def _excluded_sample_ids(args: argparse.Namespace) -> set[str]:
    return {str(sample_id) for sample_id in getattr(args, "exclude_sample", [])}


def _parse_key_value(values: list[str] | None, valid_keys: set[str], name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected {name} as KEY=VALUE, got: {value}")
        key, item = value.split("=", 1)
        key = key.strip()
        item = item.strip()
        if key not in valid_keys:
            raise ValueError(f"Unknown {name} key {key!r}. Expected one of {sorted(valid_keys)}.")
        parsed[key] = item
    return parsed


def _parse_pose_axes(value: str) -> list[str]:
    axes = [axis.strip() for axis in value.split(",") if axis.strip()]
    expected = {"x", "y", "z", "rx", "ry", "rz"}
    if len(axes) != 6 or set(axes) != expected:
        raise ValueError("--pose-axes must contain exactly x,y,z,rx,ry,rz in the raw pose order.")
    return axes


def _pose_to_transform(
    pose: list[float] | np.ndarray,
    pose_axes: str,
    rotation_type: str,
    euler_order: str,
) -> np.ndarray:
    values = np.asarray(pose, dtype=float).reshape(-1)
    if values.size < 6:
        raise ValueError(f"Expected 6D pose, got {values.tolist()}")
    by_axis = {axis: float(values[i]) for i, axis in enumerate(_parse_pose_axes(pose_axes))}
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [by_axis["x"], by_axis["y"], by_axis["z"]]
    rotation_values = np.array([by_axis["rx"], by_axis["ry"], by_axis["rz"]], dtype=float)
    if rotation_type == "rotvec":
        rotation, _ = cv2.Rodrigues(rotation_values.reshape(3, 1))
    elif rotation_type == "euler":
        rotation = Rotation.from_euler(euler_order, rotation_values).as_matrix()
    else:
        raise ValueError(f"Unsupported rotation type: {rotation_type}")
    transform[:3, :3] = rotation
    return transform


def _matrix_to_json(transform: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in transform.tolist()]


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def _rt_to_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3] @ b[:3, :3].T
    value = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.degrees(math.acos(value))


def _pose_from_transform(transform: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return np.concatenate([rvec.reshape(3), transform[:3, 3].reshape(3)])


def _transform_from_pose(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    rotation, _ = cv2.Rodrigues(np.asarray(pose[:3], dtype=np.float64).reshape(3, 1))
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(pose[3:6], dtype=np.float64).reshape(3)
    return transform


def _transform_error_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = _invert_transform(a) @ b
    rvec, _ = cv2.Rodrigues(delta[:3, :3])
    return np.concatenate([delta[:3, 3].reshape(3), rvec.reshape(3)])


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.stack([t[:3, 3] for t in transforms], axis=0)
    rotations = np.stack([t[:3, :3] for t in transforms], axis=0)
    mean_rotation = rotations.mean(axis=0)
    u, _s, vh = np.linalg.svd(mean_rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translations.mean(axis=0)
    return transform


def _alternating_eye_to_hand_initial_guess(
    base_from_gripper: list[np.ndarray],
    camera_from_target: list[np.ndarray],
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    gripper_from_target = np.eye(4, dtype=np.float64)
    base_from_camera = np.eye(4, dtype=np.float64)
    for _ in range(iterations):
        base_from_camera_candidates = [
            b_t_g @ gripper_from_target @ _invert_transform(c_t_target)
            for b_t_g, c_t_target in zip(base_from_gripper, camera_from_target)
        ]
        base_from_camera = _mean_transform(base_from_camera_candidates)
        gripper_from_target_candidates = [
            _invert_transform(b_t_g) @ base_from_camera @ c_t_target
            for b_t_g, c_t_target in zip(base_from_gripper, camera_from_target)
        ]
        gripper_from_target = _mean_transform(gripper_from_target_candidates)
    return base_from_camera, gripper_from_target


def _solve_eye_to_hand(
    base_from_gripper: list[np.ndarray],
    camera_from_target: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import least_squares

    base_from_camera0, gripper_from_target0 = _alternating_eye_to_hand_initial_guess(
        base_from_gripper,
        camera_from_target,
    )
    x0 = np.concatenate([
        _pose_from_transform(base_from_camera0),
        _pose_from_transform(gripper_from_target0),
    ])

    def residual(params: np.ndarray) -> np.ndarray:
        base_from_camera = _transform_from_pose(params[:6])
        gripper_from_target = _transform_from_pose(params[6:12])
        values: list[np.ndarray] = []
        for b_t_g, c_t_target in zip(base_from_gripper, camera_from_target):
            base_from_target_robot = b_t_g @ gripper_from_target
            base_from_target_camera = base_from_camera @ c_t_target
            values.append(_transform_error_vector(base_from_target_robot, base_from_target_camera))
        return np.concatenate(values)

    result = least_squares(
        residual,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=3000,
    )
    return _transform_from_pose(result.x[:6]), _transform_from_pose(result.x[6:12])


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def _next_sample_index(samples: list[dict[str, Any]], images_dir: Path) -> int:
    ids = [
        int(sample["id"])
        for sample in samples
        if str(sample.get("id", "")).isdigit()
    ]
    for path in images_dir.glob("*/*.png"):
        frame_id = path.stem.split("_", 1)[0]
        if frame_id.isdigit():
            ids.append(int(frame_id))
    return max(ids, default=-1) + 1


def _load_samples(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"samples": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("samples", [])
    return data


def _connect_robot(robot_type: str, ip: str, port: int, timeout: float) -> Any:
    if robot_type == "nero_dual_arm":
        from robots.dual_agilex_nero.nero_interface_client import NeroDualArmClient

        return NeroDualArmClient(ip=ip, port=port)
    if robot_type == "franka_dual_arm":
        from robots.dual_franka.dual_franka_robotiq_rpc_client import FrankaDualArmClient

        return FrankaDualArmClient(ip=ip, port=port, timeout=timeout)
    raise ValueError(f"Unsupported robot type: {robot_type}")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _franka_ee_pose_from_side(side_state: Mapping[str, Any]) -> np.ndarray:
    robot_state = _as_mapping(side_state.get("robot_state")) if side_state.get("robot_state") is not None else side_state
    if "end_pose" in side_state:
        return np.asarray(side_state.get("end_pose"), dtype=float).reshape(-1)[:6]
    if "end_pose" in robot_state:
        return np.asarray(robot_state.get("end_pose"), dtype=float).reshape(-1)[:6]

    eef_pose = _as_mapping(robot_state.get("eef_pose"))
    if eef_pose:
        position = np.asarray(eef_pose.get("position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)[:3]
        quat = np.asarray(eef_pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float).reshape(-1)[:4]
        rotvec = np.zeros(3, dtype=float)
        if np.linalg.norm(quat) > 1e-12:
            rotvec = Rotation.from_quat(quat).as_rotvec()
        return np.concatenate([position, rotvec])
    return np.zeros(6, dtype=float)


def _read_nero_arm_pose(robot: Any, arm: str) -> list[float]:
    if arm == "left":
        pose = robot.left_robot_get_ee_pose()
    elif arm == "right":
        pose = robot.right_robot_get_ee_pose()
    else:
        raise ValueError(f"Unsupported arm: {arm}")
    return np.asarray(pose, dtype=float).reshape(-1)[:6].tolist()


def _read_franka_arm_pose(robot: Any, arm: str) -> list[float]:
    observation = robot.get_observation()
    if not isinstance(observation, Mapping):
        raise RuntimeError(f"Unexpected Franka observation type: {type(observation)!r}")
    side_state = _as_mapping(observation.get(f"{arm}_arm") or observation.get(arm))
    if not side_state:
        raise RuntimeError(f"Franka observation does not contain {arm}_arm state.")
    return _franka_ee_pose_from_side(side_state).tolist()


def _read_arm_pose(robot: Any, robot_type: str, arm: str) -> list[float]:
    if robot_type == "nero_dual_arm":
        return _read_nero_arm_pose(robot, arm)
    if robot_type == "franka_dual_arm":
        return _read_franka_arm_pose(robot, arm)
    raise ValueError(f"Unsupported robot type: {robot_type}")


def _check_robot_rpc(robot: Any, robot_type: str) -> None:
    ping = getattr(robot, "ping", None)
    if callable(ping):
        try:
            ping()
        except Exception as exc:
            raise RuntimeError(
                f"{robot_type} RPC is not responding. Check that the robot ZeroRPC server is running "
                f"and listening on the configured IP/port."
            ) from exc


def cmd_capture_eye_in_hand(args: argparse.Namespace) -> None:
    arms = args.arm or ["left", "right"]
    camera_serials = _parse_key_value(args.camera, set(ARM_CAMERA_NAMES), "camera")
    missing = [arm for arm in arms if arm not in camera_serials]
    if missing:
        raise ValueError(f"Missing --camera mapping for arm(s): {missing}")

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    samples_path = out_dir / "samples.json"
    data = _load_samples(samples_path)
    samples: list[dict[str, Any]] = data["samples"]
    idx = _next_sample_index(samples, images_dir)

    robot = _connect_robot(args.robot_type, args.robot_ip, args.robot_port, args.rpc_timeout_sec)
    _check_robot_rpc(robot, args.robot_type)
    pipelines: dict[str, Any] = {}
    try:
        for arm in arms:
            serial = camera_serials[arm]
            pipeline, _profile = _start_realsense(serial, args.width, args.height, args.fps)
            pipelines[arm] = pipeline
            print(f"Started {arm} wrist camera {ARM_CAMERA_NAMES[arm]}: {serial}")

        metadata = {
            "robot_type": args.robot_type,
            "robot_ip": args.robot_ip,
            "robot_port": args.robot_port,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "pose_axes": args.pose_axes,
            "rotation_type": args.rotation_type,
            "euler_order": args.euler_order,
            "cameras": {
                ARM_CAMERA_NAMES[arm]: {"arm": arm, "serial": camera_serials[arm]}
                for arm in arms
            },
            "notes": "For eye-in-hand calibration, keep the ChArUco board fixed while moving the wrist camera.",
        }
        data["metadata"] = metadata
        _write_json(samples_path, data)

        print("Keep the ChArUco board fixed. Move one or both wrists to a new pose, wait until motion stops.")
        print("Press ENTER to capture images and EE poses, or type q then ENTER to finish.")
        while True:
            user = input(f"[{idx:04d}] hand-eye capture> ").strip().lower()
            if user in {"q", "quit", "exit"}:
                break
            if args.settle_sec > 0:
                time.sleep(args.settle_sec)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            sample: dict[str, Any] = {
                "id": f"{idx:04d}",
                "timestamp": stamp,
                "arms": {},
            }
            for arm in arms:
                raw_pose = _read_arm_pose(robot, args.robot_type, arm)
                gripper2base = _pose_to_transform(
                    raw_pose,
                    args.pose_axes,
                    args.rotation_type,
                    args.euler_order,
                )
                image = _read_realsense_frame(pipelines[arm], warmup=args.warmup)
                camera_name = ARM_CAMERA_NAMES[arm]
                image_path = images_dir / camera_name / f"{idx:04d}_{stamp}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), image)
                sample["arms"][arm] = {
                    "camera_name": camera_name,
                    "image": str(image_path.relative_to(out_dir)),
                    "ee_pose_raw": raw_pose,
                    "transform_base_from_gripper": _matrix_to_json(gripper2base),
                }
            samples.append(sample)
            _write_json(samples_path, data)
            print(f"Saved sample {idx:04d}")
            idx += 1
    finally:
        for pipeline in pipelines.values():
            pipeline.stop()
        close = getattr(robot, "close", None)
        if callable(close):
            close()


def cmd_capture_eye_to_hand(args: argparse.Namespace) -> None:
    camera_map = _parse_key_value(args.camera, {"head"}, "camera")
    if "head" not in camera_map:
        raise ValueError("Missing --camera head=SERIAL mapping.")

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    samples_path = out_dir / "samples.json"
    data = _load_samples(samples_path)
    samples: list[dict[str, Any]] = data["samples"]
    idx = _next_sample_index(samples, images_dir)

    robot = _connect_robot(args.robot_type, args.robot_ip, args.robot_port, args.rpc_timeout_sec)
    _check_robot_rpc(robot, args.robot_type)
    pipeline = None
    try:
        pipeline, _profile = _start_realsense(camera_map["head"], args.width, args.height, args.fps)
        print(f"Started head camera {HEAD_CAMERA_NAME}: {camera_map['head']}")

        metadata = {
            "calibration_type": "eye_to_hand",
            "robot_type": args.robot_type,
            "robot_ip": args.robot_ip,
            "robot_port": args.robot_port,
            "arm": args.arm,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "pose_axes": args.pose_axes,
            "rotation_type": args.rotation_type,
            "euler_order": args.euler_order,
            "camera": {
                "name": HEAD_CAMERA_NAME,
                "serial": camera_map["head"],
            },
            "notes": (
                "For eye-to-hand calibration, rigidly mount the ChArUco board to the selected gripper, "
                "then move the gripper while the head camera stays fixed."
            ),
        }
        data["metadata"] = metadata
        _write_json(samples_path, data)

        print("Rigidly attach the ChArUco board to the selected gripper/end-effector.")
        print("Move the arm to a new pose where the fixed head camera sees the board clearly.")
        print("Press ENTER to capture the head image and EE pose, or type q then ENTER to finish.")
        while True:
            user = input(f"[{idx:04d}] eye-to-hand capture> ").strip().lower()
            if user in {"q", "quit", "exit"}:
                break
            if args.settle_sec > 0:
                time.sleep(args.settle_sec)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            raw_pose = _read_arm_pose(robot, args.robot_type, args.arm)
            gripper2base = _pose_to_transform(
                raw_pose,
                args.pose_axes,
                args.rotation_type,
                args.euler_order,
            )
            image = _read_realsense_frame(pipeline, warmup=args.warmup)
            image_path = images_dir / HEAD_CAMERA_NAME / f"{idx:04d}_{stamp}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), image)
            samples.append(
                {
                    "id": f"{idx:04d}",
                    "timestamp": stamp,
                    "arm": args.arm,
                    "camera_name": HEAD_CAMERA_NAME,
                    "image": str(image_path.relative_to(out_dir)),
                    "ee_pose_raw": raw_pose,
                    "transform_base_from_gripper": _matrix_to_json(gripper2base),
                }
            )
            _write_json(samples_path, data)
            print(f"Saved sample {idx:04d}")
            idx += 1
    finally:
        if pipeline is not None:
            pipeline.stop()
        close = getattr(robot, "close", None)
        if callable(close):
            close()


def _sample_arm_image(out_dir: Path, sample: dict[str, Any], arm: str) -> Path | None:
    arm_data = sample.get("arms", {}).get(arm)
    if not arm_data:
        return None
    image = arm_data.get("image")
    if not image:
        return None
    return out_dir / image


def _sample_arm_pose(
    sample: dict[str, Any],
    arm: str,
    pose_axes: str,
    rotation_type: str,
    euler_order: str,
    force_raw_pose: bool,
) -> np.ndarray | None:
    arm_data = sample.get("arms", {}).get(arm)
    if not arm_data:
        return None
    transform = arm_data.get("transform_base_from_gripper")
    if transform is not None and not force_raw_pose:
        return np.asarray(transform, dtype=np.float64)
    raw_pose = arm_data.get("ee_pose_raw")
    if raw_pose is None:
        return None
    return _pose_to_transform(raw_pose, pose_axes, rotation_type, euler_order)


def _calibrate_one_arm(args: argparse.Namespace, data: dict[str, Any], arm: str) -> dict[str, Any]:
    capture_dir = Path(args.capture_dir)
    spec = _read_board_spec(Path(args.board))
    board = _make_board(spec)
    metadata = data.get("metadata", {})
    pose_axes = args.pose_axes or metadata.get("pose_axes", "x,y,z,rx,ry,rz")
    rotation_type = args.rotation_type or metadata.get("rotation_type", "rotvec")
    euler_order = args.euler_order or metadata.get("euler_order", "xyz")
    force_raw_pose = args.pose_axes is not None or args.rotation_type is not None or args.euler_order is not None
    camera_name = ARM_CAMERA_NAMES[arm]
    intrinsics_path = Path(args.intrinsics_dir) / f"{camera_name}_intrinsics.npz"
    camera_matrix, dist_coeffs = _load_intrinsics(intrinsics_path)

    r_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    r_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []
    used_samples: list[dict[str, Any]] = []
    rejected_samples: list[dict[str, Any]] = []
    excluded_samples = _excluded_sample_ids(args)

    for sample in data.get("samples", []):
        if str(sample.get("id")) in excluded_samples:
            continue
        image_path = _sample_arm_image(capture_dir, sample, arm)
        gripper2base = _sample_arm_pose(sample, arm, pose_axes, rotation_type, euler_order, force_raw_pose)
        if image_path is None or gripper2base is None:
            continue
        image_relpath = str(image_path.relative_to(capture_dir))
        image = cv2.imread(str(image_path))
        if image is None:
            rejected_samples.append({"id": sample.get("id"), "image": image_relpath, "reason": "image_read_failed"})
            continue
        ok, rvec, tvec, count, marker_corners, marker_ids, corners, ids = _estimate_board_pose(
            image,
            board,
            spec,
            camera_matrix,
            dist_coeffs,
            args.min_corners,
        )
        if args.debug_dir:
            _save_debug_image(
                image,
                Path(args.debug_dir) / arm / image_path.name,
                marker_corners,
                marker_ids,
                corners,
                ids,
            )
        if not ok or rvec is None or tvec is None:
            rejected_samples.append(
                {
                    "id": sample.get("id"),
                    "image": image_relpath,
                    "reason": "not_enough_charuco_corners",
                    "charuco_corners": int(count),
                }
            )
            continue
        target2cam = np.eye(4, dtype=np.float64)
        target2cam[:3, :3], _ = cv2.Rodrigues(rvec)
        target2cam[:3, 3] = tvec.reshape(3)
        r_gripper2base.append(gripper2base[:3, :3])
        t_gripper2base.append(gripper2base[:3, 3].reshape(3, 1))
        r_target2cam.append(target2cam[:3, :3])
        t_target2cam.append(target2cam[:3, 3].reshape(3, 1))
        used_samples.append({"id": sample.get("id"), "image": image_relpath, "charuco_corners": int(count)})

    _print_sample_report(f"{arm} ({camera_name})", used_samples, rejected_samples)
    if len(used_samples) < args.min_samples:
        raise RuntimeError(
            f"{arm}: only {len(used_samples)} valid samples, need at least {args.min_samples}. "
            f"Rejected {len(rejected_samples)} samples."
        )

    method = HAND_EYE_METHODS[args.method]
    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    gripper_from_camera = _rt_to_transform(r_cam2gripper, t_cam2gripper)
    camera_from_gripper = _invert_transform(gripper_from_camera)

    target_in_base: list[np.ndarray] = []
    for rg, tg, rt, tt in zip(r_gripper2base, t_gripper2base, r_target2cam, t_target2cam):
        base_from_gripper = _rt_to_transform(rg, tg)
        camera_from_target = _rt_to_transform(rt, tt)
        base_from_target = base_from_gripper @ gripper_from_camera @ camera_from_target
        target_in_base.append(base_from_target)
    mean_target = _mean_transform(target_in_base)
    translation_errors = [
        float(np.linalg.norm(transform[:3, 3] - mean_target[:3, 3]))
        for transform in target_in_base
    ]
    rotation_errors = [_rotation_error_deg(transform, mean_target) for transform in target_in_base]

    return {
        "arm": arm,
        "camera_name": camera_name,
        "method": args.method,
        "valid_samples": len(used_samples),
        "excluded_samples": sorted(excluded_samples),
        "rejected_samples": rejected_samples,
        "used_samples": used_samples,
        "pose_axes": pose_axes,
        "rotation_type": rotation_type,
        "euler_order": euler_order,
        "transform_gripper_from_camera": _matrix_to_json(gripper_from_camera),
        "transform_camera_from_gripper": _matrix_to_json(camera_from_gripper),
        "estimated_transform_base_from_target_mean": _matrix_to_json(mean_target),
        "target_static_residual": {
            "translation_mean_m": float(np.mean(translation_errors)),
            "translation_std_m": float(np.std(translation_errors)),
            "translation_max_m": float(np.max(translation_errors)),
            "rotation_mean_deg": float(np.mean(rotation_errors)),
            "rotation_std_deg": float(np.std(rotation_errors)),
            "rotation_max_deg": float(np.max(rotation_errors)),
        },
    }


def cmd_calibrate_eye_in_hand(args: argparse.Namespace) -> None:
    arms = args.arm or ["left", "right"]
    capture_dir = Path(args.capture_dir)
    samples_path = capture_dir / "samples.json"
    data = _load_samples(samples_path)
    results = {
        "calibration_type": "eye_in_hand",
        "board": asdict(_read_board_spec(Path(args.board))),
        "capture_dir": str(capture_dir),
        "arms": {},
    }
    for arm in arms:
        result = _calibrate_one_arm(args, data, arm)
        results["arms"][arm] = result
        residual = result["target_static_residual"]
        print(
            f"{arm}: valid={result['valid_samples']}, "
            f"target residual mean={residual['translation_mean_m']:.4f}m/"
            f"{residual['rotation_mean_deg']:.3f}deg, "
            f"max={residual['translation_max_m']:.4f}m/{residual['rotation_max_deg']:.3f}deg"
        )

    output_dir = Path(args.output_dir)
    _write_json(output_dir / "eye_in_hand.json", results)
    print(f"Wrote hand-eye result: {output_dir / 'eye_in_hand.json'}")


def _sample_eye_to_hand_pose(
    sample: dict[str, Any],
    pose_axes: str,
    rotation_type: str,
    euler_order: str,
    force_raw_pose: bool,
) -> np.ndarray | None:
    transform = sample.get("transform_base_from_gripper")
    if transform is not None and not force_raw_pose:
        return np.asarray(transform, dtype=np.float64)
    raw_pose = sample.get("ee_pose_raw")
    if raw_pose is None:
        return None
    return _pose_to_transform(raw_pose, pose_axes, rotation_type, euler_order)


def cmd_calibrate_eye_to_hand(args: argparse.Namespace) -> None:
    capture_dir = Path(args.capture_dir)
    samples_path = capture_dir / "samples.json"
    data = _load_samples(samples_path)
    metadata = data.get("metadata", {})
    pose_axes = args.pose_axes or metadata.get("pose_axes", "x,y,z,rx,ry,rz")
    rotation_type = args.rotation_type or metadata.get("rotation_type", "rotvec")
    euler_order = args.euler_order or metadata.get("euler_order", "xyz")
    force_raw_pose = args.pose_axes is not None or args.rotation_type is not None or args.euler_order is not None
    arm = args.arm or metadata.get("arm", "left")
    camera_name = args.camera_name

    spec = _read_board_spec(Path(args.board))
    board = _make_board(spec)
    camera_matrix, dist_coeffs = _load_intrinsics(Path(args.intrinsics_dir) / f"{camera_name}_intrinsics.npz")

    base_from_gripper: list[np.ndarray] = []
    camera_from_target: list[np.ndarray] = []
    used_samples: list[dict[str, Any]] = []
    rejected_samples: list[dict[str, Any]] = []
    excluded_samples = _excluded_sample_ids(args)

    for sample in data.get("samples", []):
        if str(sample.get("id")) in excluded_samples:
            continue
        if sample.get("arm", arm) != arm:
            continue
        image = sample.get("image")
        if not image:
            continue
        image_path = capture_dir / image
        image_relpath = str(Path(image))
        gripper2base = _sample_eye_to_hand_pose(sample, pose_axes, rotation_type, euler_order, force_raw_pose)
        if gripper2base is None:
            rejected_samples.append({"id": sample.get("id"), "image": image_relpath, "reason": "missing_gripper_pose"})
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            rejected_samples.append({"id": sample.get("id"), "image": image_relpath, "reason": "image_read_failed"})
            continue
        ok, rvec, tvec, count, marker_corners, marker_ids, corners, ids = _estimate_board_pose(
            frame,
            board,
            spec,
            camera_matrix,
            dist_coeffs,
            args.min_corners,
        )
        if args.debug_dir:
            _save_debug_image(
                frame,
                Path(args.debug_dir) / camera_name / image_path.name,
                marker_corners,
                marker_ids,
                corners,
                ids,
            )
        if not ok or rvec is None or tvec is None:
            rejected_samples.append(
                {
                    "id": sample.get("id"),
                    "image": image_relpath,
                    "reason": "not_enough_charuco_corners",
                    "charuco_corners": int(count),
                }
            )
            continue
        target2camera = np.eye(4, dtype=np.float64)
        target2camera[:3, :3], _ = cv2.Rodrigues(rvec)
        target2camera[:3, 3] = tvec.reshape(3)
        base_from_gripper.append(gripper2base)
        camera_from_target.append(target2camera)
        used_samples.append({"id": sample.get("id"), "image": image_relpath, "charuco_corners": int(count)})

    _print_sample_report(f"head/base ({camera_name}, arm={arm})", used_samples, rejected_samples)
    if len(used_samples) < args.min_samples:
        raise RuntimeError(
            f"head/base: only {len(used_samples)} valid samples, need at least {args.min_samples}. "
            f"Rejected {len(rejected_samples)} samples."
        )

    base_from_camera, gripper_from_target = _solve_eye_to_hand(base_from_gripper, camera_from_target)
    camera_from_base = _invert_transform(base_from_camera)
    target_from_gripper = _invert_transform(gripper_from_target)

    residual_translations: list[float] = []
    residual_rotations: list[float] = []
    for b_t_g, c_t_target in zip(base_from_gripper, camera_from_target):
        base_from_target_robot = b_t_g @ gripper_from_target
        base_from_target_camera = base_from_camera @ c_t_target
        residual_translations.append(float(np.linalg.norm(base_from_target_robot[:3, 3] - base_from_target_camera[:3, 3])))
        residual_rotations.append(_rotation_error_deg(base_from_target_robot, base_from_target_camera))

    result = {
        "calibration_type": "eye_to_hand",
        "robot_type": metadata.get("robot_type", args.robot_type),
        "arm": arm,
        "camera_name": camera_name,
        "board": asdict(spec),
        "pose_axes": pose_axes,
        "rotation_type": rotation_type,
        "euler_order": euler_order,
        "valid_samples": len(used_samples),
        "excluded_samples": sorted(excluded_samples),
        "used_samples": used_samples,
        "rejected_samples": rejected_samples,
        "transform_base_from_camera": _matrix_to_json(base_from_camera),
        "transform_camera_from_base": _matrix_to_json(camera_from_base),
        "estimated_transform_gripper_from_target": _matrix_to_json(gripper_from_target),
        "estimated_transform_target_from_gripper": _matrix_to_json(target_from_gripper),
        "residual": {
            "translation_mean_m": float(np.mean(residual_translations)),
            "translation_std_m": float(np.std(residual_translations)),
            "translation_max_m": float(np.max(residual_translations)),
            "rotation_mean_deg": float(np.mean(residual_rotations)),
            "rotation_std_deg": float(np.std(residual_rotations)),
            "rotation_max_deg": float(np.max(residual_rotations)),
        },
    }

    output_dir = Path(args.output_dir)
    _write_json(output_dir / "head_to_base.json", result)
    residual = result["residual"]
    print(
        f"head->base: valid={len(used_samples)}, "
        f"residual mean={residual['translation_mean_m']:.4f}m/{residual['rotation_mean_deg']:.3f}deg, "
        f"max={residual['translation_max_m']:.4f}m/{residual['rotation_max_deg']:.3f}deg"
    )
    print(f"Wrote head/base result: {output_dir / 'head_to_base.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture-eye-in-hand",
        help="Capture wrist-camera images with robot EE poses for eye-in-hand calibration.",
    )
    capture.add_argument("--output-dir", default="calibration/hand_eye_capture")
    capture.add_argument("--robot-type", choices=["franka_dual_arm", "nero_dual_arm"], default="franka_dual_arm")
    capture.add_argument("--robot-ip", default="172.16.0.1")
    capture.add_argument("--robot-port", type=int, default=4242)
    capture.add_argument("--rpc-timeout-sec", type=float, default=30.0)
    capture.add_argument("--arm", action="append", choices=sorted(ARM_CAMERA_NAMES), default=None)
    capture.add_argument("--camera", action="append", required=True, help="Arm camera mapping, e.g. left=SERIAL.")
    capture.add_argument("--width", type=int, default=424)
    capture.add_argument("--height", type=int, default=240)
    capture.add_argument("--fps", type=int, default=30)
    capture.add_argument("--warmup", type=int, default=4)
    capture.add_argument("--settle-sec", type=float, default=0.0)
    capture.add_argument(
        "--pose-axes",
        default="x,y,z,rx,ry,rz",
        help="Raw 6D robot pose axis order. Franka uses x,y,z,rx,ry,rz; Nero may need x,y,z,rz,ry,rx.",
    )
    capture.add_argument("--rotation-type", choices=["rotvec", "euler"], default="rotvec")
    capture.add_argument("--euler-order", default="xyz")
    capture.set_defaults(func=cmd_capture_eye_in_hand)

    capture_head = subparsers.add_parser(
        "capture-eye-to-hand",
        help="Capture head-camera images with one EE pose for fixed-camera-to-base calibration.",
    )
    capture_head.add_argument("--output-dir", default="calibration/head_eye_capture")
    capture_head.add_argument("--robot-type", choices=["franka_dual_arm", "nero_dual_arm"], default="franka_dual_arm")
    capture_head.add_argument("--robot-ip", default="172.16.0.1")
    capture_head.add_argument("--robot-port", type=int, default=4242)
    capture_head.add_argument("--rpc-timeout-sec", type=float, default=30.0)
    capture_head.add_argument("--arm", choices=sorted(ARM_CAMERA_NAMES), default="left")
    capture_head.add_argument("--camera", action="append", required=True, help="Head camera mapping: head=SERIAL.")
    capture_head.add_argument("--width", type=int, default=424)
    capture_head.add_argument("--height", type=int, default=240)
    capture_head.add_argument("--fps", type=int, default=30)
    capture_head.add_argument("--warmup", type=int, default=4)
    capture_head.add_argument("--settle-sec", type=float, default=0.0)
    capture_head.add_argument("--pose-axes", default="x,y,z,rx,ry,rz")
    capture_head.add_argument("--rotation-type", choices=["rotvec", "euler"], default="rotvec")
    capture_head.add_argument("--euler-order", default="xyz")
    capture_head.set_defaults(func=cmd_capture_eye_to_hand)

    calibrate = subparsers.add_parser(
        "calibrate-eye-in-hand",
        help="Calibrate camera-to-gripper transforms from captured ChArUco samples.",
    )
    calibrate.add_argument("--capture-dir", default="calibration/hand_eye_capture")
    calibrate.add_argument("--board", required=True)
    calibrate.add_argument("--intrinsics-dir", default="calibration/hand_eye_result")
    calibrate.add_argument("--output-dir", default="calibration/hand_eye_result")
    calibrate.add_argument("--arm", action="append", choices=sorted(ARM_CAMERA_NAMES), default=None)
    calibrate.add_argument("--method", choices=sorted(HAND_EYE_METHODS), default="tsai")
    calibrate.add_argument("--min-corners", type=int, default=12)
    calibrate.add_argument("--min-samples", type=int, default=12)
    calibrate.add_argument("--exclude-sample", action="append", default=[], help="Sample id to ignore during calibration.")
    calibrate.add_argument("--debug-dir", default=None)
    calibrate.add_argument("--pose-axes", default=None)
    calibrate.add_argument("--rotation-type", choices=["rotvec", "euler"], default=None)
    calibrate.add_argument("--euler-order", default=None)
    calibrate.set_defaults(func=cmd_calibrate_eye_in_hand)

    calibrate_head = subparsers.add_parser(
        "calibrate-eye-to-hand",
        help="Estimate fixed head camera to robot base from a board rigidly mounted on the gripper.",
    )
    calibrate_head.add_argument("--capture-dir", default="calibration/head_eye_capture")
    calibrate_head.add_argument("--board", required=True)
    calibrate_head.add_argument("--intrinsics-dir", default="calibration/head_eye_result")
    calibrate_head.add_argument("--output-dir", default="calibration/head_eye_result")
    calibrate_head.add_argument("--robot-type", choices=["franka_dual_arm", "nero_dual_arm"], default="franka_dual_arm")
    calibrate_head.add_argument("--arm", choices=sorted(ARM_CAMERA_NAMES), default=None)
    calibrate_head.add_argument("--camera-name", default=HEAD_CAMERA_NAME)
    calibrate_head.add_argument("--min-corners", type=int, default=12)
    calibrate_head.add_argument("--min-samples", type=int, default=12)
    calibrate_head.add_argument("--exclude-sample", action="append", default=[], help="Sample id to ignore during calibration.")
    calibrate_head.add_argument("--debug-dir", default=None)
    calibrate_head.add_argument("--pose-axes", default=None)
    calibrate_head.add_argument("--rotation-type", choices=["rotvec", "euler"], default=None)
    calibrate_head.add_argument("--euler-order", default=None)
    calibrate_head.set_defaults(func=cmd_calibrate_eye_to_hand)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
