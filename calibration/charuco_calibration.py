"""ChArUco tools for printing boards and calibrating RealSense camera rigs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


DEFAULT_CAMERA_NAMES = ("head_image", "left_wrist_image", "right_wrist_image")
DEFAULT_ARUCO_DICT = "DICT_4X4_50"


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary: str = DEFAULT_ARUCO_DICT


def _cv2() -> Any:
    global cv2
    if cv2 is None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenCV is missing. Install opencv-contrib-python.") from exc
        cv2 = cv2_module
    return cv2


def _np() -> Any:
    global np
    if np is None:
        try:
            import numpy as np_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("NumPy is missing. Install the project dependencies.") from exc
        np = np_module
    return np


def _aruco() -> Any:
    cv2_module = _cv2()
    if not hasattr(cv2_module, "aruco"):
        raise RuntimeError("OpenCV aruco module is missing. Install opencv-contrib-python.")
    return cv2_module.aruco


def _get_dictionary(name: str) -> Any:
    aruco = _aruco()
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def _make_board(spec: BoardSpec) -> Any:
    aruco = _aruco()
    dictionary = _get_dictionary(spec.dictionary)
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_length_m,
            spec.marker_length_m,
            dictionary,
        )
    return aruco.CharucoBoard_create(
        spec.squares_x,
        spec.squares_y,
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )


def _draw_board(board: Any, size_px: tuple[int, int]) -> np.ndarray:
    if hasattr(board, "generateImage"):
        return board.generateImage(size_px, marginSize=0, borderBits=1)
    return board.draw(size_px, marginSize=0, borderBits=1)


def _detector_parameters() -> Any:
    aruco = _aruco()
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    return aruco.DetectorParameters_create()


def _detect_markers(gray: np.ndarray, dictionary: Any, parameters: Any) -> tuple[Any, Any, Any]:
    aruco = _aruco()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(gray)
    return aruco.detectMarkers(gray, dictionary, parameters=parameters)


def _interpolate_charuco(
    gray: np.ndarray,
    board: Any,
    marker_corners: Any,
    marker_ids: Any,
) -> tuple[int, Any, Any]:
    aruco = _aruco()
    return aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)


def _read_board_spec(path: Path) -> BoardSpec:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BoardSpec(
        squares_x=int(data["squares_x"]),
        squares_y=int(data["squares_y"]),
        square_length_m=float(data["square_length_m"]),
        marker_length_m=float(data["marker_length_m"]),
        dictionary=str(data.get("dictionary", DEFAULT_ARUCO_DICT)),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _print_frame_report(label: str, used_frames: list[dict[str, Any]], rejected_frames: list[dict[str, Any]]) -> None:
    print(f"{label}: used frames ({len(used_frames)})")
    if used_frames:
        for frame in used_frames:
            details = [f"file={frame.get('file')}"]
            if "set_id" in frame:
                details.append(f"set={frame['set_id']}")
            if "charuco_corners" in frame:
                details.append(f"corners={frame['charuco_corners']}")
            print("  + " + ", ".join(details))
    else:
        print("  + none")

    print(f"{label}: rejected frames ({len(rejected_frames)})")
    if rejected_frames:
        for frame in rejected_frames:
            details = [f"file={frame.get('file')}", f"reason={frame.get('reason')}"]
            if "set_id" in frame:
                details.append(f"set={frame['set_id']}")
            if "charuco_corners" in frame:
                details.append(f"corners={frame['charuco_corners']}")
            print("  - " + ", ".join(details))
    else:
        print("  - none")


def _load_image(path: Path) -> np.ndarray:
    cv2_module = _cv2()
    image = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _image_size(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    cv2_module = _cv2()
    return cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)


def _save_debug_image(
    image: np.ndarray,
    path: Path,
    marker_corners: Any,
    marker_ids: Any,
    charuco_corners: Any,
    charuco_ids: Any,
) -> None:
    aruco = _aruco()
    debug = image.copy()
    if marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids) > 0:
        aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    _cv2().imwrite(str(path), debug)


def _detect_charuco(
    image: np.ndarray,
    board: Any,
    spec: BoardSpec,
) -> tuple[int, Any, Any, Any, Any]:
    gray = _to_gray(image)
    dictionary = _get_dictionary(spec.dictionary)
    marker_corners, marker_ids, rejected = _detect_markers(gray, dictionary, _detector_parameters())
    if marker_ids is None or len(marker_ids) == 0:
        return 0, None, None, marker_corners, marker_ids
    count, charuco_corners, charuco_ids = _interpolate_charuco(gray, board, marker_corners, marker_ids)
    return int(count), charuco_corners, charuco_ids, marker_corners, marker_ids


def cmd_create_board(args: argparse.Namespace) -> None:
    np_module = _np()
    spec = BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_mm / 1000.0,
        marker_length_m=args.marker_mm / 1000.0,
        dictionary=args.dictionary,
    )
    board = _make_board(spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_w_px = round(args.page_width_mm / 25.4 * args.dpi)
    page_h_px = round(args.page_height_mm / 25.4 * args.dpi)
    board_w_px = round(args.squares_x * args.square_mm / 25.4 * args.dpi)
    board_h_px = round(args.squares_y * args.square_mm / 25.4 * args.dpi)
    if board_w_px > page_w_px or board_h_px > page_h_px:
        raise ValueError("Board is larger than the requested page. Reduce squares or square size.")

    board_image = _draw_board(board, (board_w_px, board_h_px))
    page = np_module.full((page_h_px, page_w_px), 255, dtype=np_module.uint8)
    x0 = (page_w_px - board_w_px) // 2
    y0 = (page_h_px - board_h_px) // 2
    page[y0 : y0 + board_h_px, x0 : x0 + board_w_px] = board_image

    name = f"charuco_{args.squares_x}x{args.squares_y}_{args.square_mm:g}mm"
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    spec_path = out_dir / f"{name}.json"
    _cv2().imwrite(str(png_path), page)

    try:
        from PIL import Image

        image = Image.fromarray(page)
        image.save(str(pdf_path), "PDF", resolution=args.dpi)
    except Exception as exc:
        print(f"[WARN] Failed to write PDF with Pillow: {exc}")
        pdf_path = None

    _write_json(
        spec_path,
        {
            **asdict(spec),
            "square_length_mm": args.square_mm,
            "marker_length_mm": args.marker_mm,
            "page_width_mm": args.page_width_mm,
            "page_height_mm": args.page_height_mm,
            "dpi": args.dpi,
            "print_scale": "Print at 100% / actual size. Do not fit to page.",
        },
    )
    print(f"Wrote board PNG: {png_path}")
    if pdf_path is not None:
        print(f"Wrote board PDF: {pdf_path}")
    print(f"Wrote board spec: {spec_path}")
    print("Print the PDF at 100% / actual size, then measure one square to confirm its size.")


def _parse_camera_arg(values: list[str]) -> dict[str, str]:
    cameras: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=SERIAL camera argument, got: {value}")
        name, serial = value.split("=", 1)
        cameras[name.strip()] = serial.strip()
    return cameras


def _start_realsense(serial: str, width: int, height: int, fps: int) -> tuple[Any, Any]:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    return pipeline, profile


def _read_realsense_frame(pipeline: Any, warmup: int = 0) -> np.ndarray:
    frame = None
    for _ in range(max(1, warmup + 1)):
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
    if frame is None:
        raise RuntimeError("No color frame received from RealSense.")
    return _np().asanyarray(frame.get_data())


def cmd_capture(args: argparse.Namespace) -> None:
    cameras = _parse_camera_arg(args.camera)
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "cameras.json", {"cameras": cameras, "width": args.width, "height": args.height, "fps": args.fps})

    pipelines: dict[str, Any] = {}
    try:
        for name, serial in cameras.items():
            pipeline, _profile = _start_realsense(serial, args.width, args.height, args.fps)
            pipelines[name] = pipeline
            print(f"Started {name}: {serial}")

        print("Hold the printed board where all cameras can see it.")
        print("Press ENTER to capture a synchronized set, or type q then ENTER to finish.")
        existing_ids: list[int] = []
        for camera_dir in raw_dir.iterdir() if raw_dir.exists() else []:
            if not camera_dir.is_dir():
                continue
            for path in camera_dir.glob("*.png"):
                frame_id = path.stem.split("_", 1)[0]
                if frame_id.isdigit():
                    existing_ids.append(int(frame_id))
        idx = max(existing_ids, default=-1) + 1
        while True:
            user = input(f"[{idx:04d}] capture> ").strip().lower()
            if user in {"q", "quit", "exit"}:
                break
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for name, pipeline in pipelines.items():
                image = _read_realsense_frame(pipeline, warmup=args.warmup)
                path = raw_dir / name / f"{idx:04d}_{stamp}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                _cv2().imwrite(str(path), image)
            print(f"Saved set {idx:04d}")
            idx += 1
    finally:
        for pipeline in pipelines.values():
            pipeline.stop()


def _iter_camera_images(images_dir: Path, camera_name: str) -> list[Path]:
    paths = sorted((images_dir / camera_name).glob("*.png"))
    paths.extend(sorted((images_dir / camera_name).glob("*.jpg")))
    paths.extend(sorted((images_dir / camera_name).glob("*.jpeg")))
    return paths


def cmd_calibrate_intrinsics(args: argparse.Namespace) -> None:
    spec = _read_board_spec(Path(args.board))
    board = _make_board(spec)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"board": asdict(spec), "cameras": {}}
    for camera_name in args.camera_name:
        all_corners = []
        all_ids = []
        image_size = None
        accepted = 0
        used_frames: list[dict[str, Any]] = []
        rejected_frames: list[dict[str, Any]] = []
        for path in _iter_camera_images(images_dir, camera_name):
            image = _load_image(path)
            image_size = _image_size(image)
            count, corners, ids, marker_corners, marker_ids = _detect_charuco(image, board, spec)
            relpath = str(path.relative_to(images_dir))
            set_id = path.stem.split("_", 1)[0]
            if args.debug_dir:
                _save_debug_image(
                    image,
                    Path(args.debug_dir) / "intrinsics" / camera_name / path.name,
                    marker_corners,
                    marker_ids,
                    corners,
                    ids,
                )
            if count >= args.min_corners:
                all_corners.append(corners)
                all_ids.append(ids)
                accepted += 1
                used_frames.append({"file": relpath, "set_id": set_id, "charuco_corners": int(count)})
            else:
                rejected_frames.append(
                    {
                        "file": relpath,
                        "set_id": set_id,
                        "reason": "not_enough_charuco_corners",
                        "charuco_corners": int(count),
                    }
                )

        if image_size is None:
            raise RuntimeError(f"No images found for camera: {camera_name}")
        _print_frame_report(camera_name, used_frames, rejected_frames)
        if len(all_corners) < args.min_frames:
            raise RuntimeError(
                f"{camera_name}: only {len(all_corners)} valid frames, need at least {args.min_frames}."
            )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = _aruco().calibrateCameraCharuco(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None,
        )
        np_module = _np()
        np_module.savez(
            out_dir / f"{camera_name}_intrinsics.npz",
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=np_module.array(image_size),
            rms=np_module.array(rms),
        )
        results["cameras"][camera_name] = {
            "rms_reprojection_error_px": float(rms),
            "valid_frames": int(accepted),
            "used_frames": used_frames,
            "rejected_frames": rejected_frames,
            "image_size": list(image_size),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        }
        print(f"{camera_name}: rms={rms:.4f}px, valid_frames={accepted}")

    _write_json(out_dir / "intrinsics.json", results)
    print(f"Wrote intrinsics: {out_dir / 'intrinsics.json'}")


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _np().load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def _board_object_points(board: Any, ids: np.ndarray) -> np.ndarray:
    np_module = _np()
    chessboard_corners = getattr(board, "chessboardCorners", None)
    if chessboard_corners is None and hasattr(board, "getChessboardCorners"):
        chessboard_corners = board.getChessboardCorners()
    if chessboard_corners is None:
        raise RuntimeError("Cannot read ChArUco board chessboard corners from OpenCV board object.")
    ids_flat = ids.reshape(-1).astype(int)
    return np_module.asarray(chessboard_corners, dtype=np_module.float32)[ids_flat]


def _estimate_board_pose(
    image: np.ndarray,
    board: Any,
    spec: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_corners: int,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, int, Any, Any, Any, Any]:
    count, corners, ids, marker_corners, marker_ids = _detect_charuco(image, board, spec)
    if ids is None or corners is None or count < min_corners:
        return False, None, None, count, marker_corners, marker_ids, corners, ids
    object_points = _board_object_points(board, ids)
    ok, rvec, tvec = _cv2().solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    return bool(ok), rvec, tvec, count, marker_corners, marker_ids, corners, ids


def _rt_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = _cv2().Rodrigues(rvec)
    transform = _np().eye(4, dtype=_np().float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return transform


def _matrix_to_rt(transform: np.ndarray) -> tuple[list[list[float]], list[float]]:
    return transform[:3, :3].tolist(), transform[:3, 3].tolist()


def _rotation_angle(transform: np.ndarray) -> float:
    trace = _np().trace(transform[:3, :3])
    value = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(value))


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    np_module = _np()
    translations = np_module.stack([t[:3, 3] for t in transforms], axis=0)
    rotations = np_module.stack([t[:3, :3] for t in transforms], axis=0)
    mean_rotation = rotations.mean(axis=0)
    u, _s, vh = np_module.linalg.svd(mean_rotation)
    rotation = u @ vh
    if np_module.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    out = np_module.eye(4, dtype=np_module.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translations.mean(axis=0)
    return out


def cmd_calibrate_rig(args: argparse.Namespace) -> None:
    spec = _read_board_spec(Path(args.board))
    board = _make_board(spec)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = {
        name: _load_intrinsics(Path(args.intrinsics_dir) / f"{name}_intrinsics.npz")
        for name in args.camera_name
    }
    base = args.base_camera
    if base not in intrinsics:
        raise ValueError(f"Base camera {base!r} is not in --camera-name.")

    per_set_poses: dict[str, dict[str, np.ndarray]] = {}
    per_camera_used: dict[str, list[dict[str, Any]]] = {name: [] for name in args.camera_name}
    per_camera_rejected: dict[str, list[dict[str, Any]]] = {name: [] for name in args.camera_name}
    for camera_name in args.camera_name:
        camera_matrix, dist_coeffs = intrinsics[camera_name]
        for path in _iter_camera_images(images_dir, camera_name):
            set_id = path.stem.split("_", 1)[0]
            image = _load_image(path)
            ok, rvec, tvec, _count, marker_corners, marker_ids, corners, ids = _estimate_board_pose(
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
                    Path(args.debug_dir) / "rig" / camera_name / path.name,
                    marker_corners,
                    marker_ids,
                    corners,
                    ids,
                )
            if ok and rvec is not None and tvec is not None:
                per_set_poses.setdefault(set_id, {})[camera_name] = _rt_to_matrix(rvec, tvec)
                per_camera_used[camera_name].append(
                    {
                        "file": str(path.relative_to(images_dir)),
                        "set_id": set_id,
                        "charuco_corners": int(_count),
                    }
                )
            else:
                per_camera_rejected[camera_name].append(
                    {
                        "file": str(path.relative_to(images_dir)),
                        "set_id": set_id,
                        "reason": "not_enough_charuco_corners",
                        "charuco_corners": int(_count),
                    }
                )

    for camera_name in args.camera_name:
        _print_frame_report(
            f"rig pose {camera_name}",
            per_camera_used[camera_name],
            per_camera_rejected[camera_name],
        )

    relatives: dict[str, list[np.ndarray]] = {name: [] for name in args.camera_name if name != base}
    relative_used_sets: dict[str, list[str]] = {name: [] for name in relatives}
    for set_id, poses in sorted(per_set_poses.items()):
        if base not in poses:
            continue
        base_from_board = poses[base]
        board_from_base = _np().linalg.inv(base_from_board)
        for name in relatives:
            if name not in poses:
                continue
            camera_from_board = poses[name]
            camera_from_base = camera_from_board @ board_from_base
            relatives[name].append(camera_from_base)
            relative_used_sets[name].append(set_id)

    result: dict[str, Any] = {"base_camera": base, "board": asdict(spec), "transforms": {}}
    for name, transforms in relatives.items():
        if len(transforms) < args.min_pairs:
            raise RuntimeError(f"{name}: only {len(transforms)} valid shared board poses, need {args.min_pairs}.")
        mean = _mean_transform(transforms)
        np_module = _np()
        rot_errors = [_rotation_angle(t @ np_module.linalg.inv(mean)) for t in transforms]
        trans_errors = [float(np_module.linalg.norm(t[:3, 3] - mean[:3, 3])) for t in transforms]
        rotation, translation = _matrix_to_rt(mean)
        result["transforms"][name] = {
            "parent": base,
            "child": name,
            "transform_child_from_parent": mean.tolist(),
            "rotation_matrix": rotation,
            "translation_m": translation,
            "valid_pairs": len(transforms),
            "used_sets": relative_used_sets[name],
            "translation_std_m": float(np_module.std(trans_errors)),
            "rotation_std_deg": float(np_module.std(rot_errors)),
        }
        print(
            f"{base} -> {name}: pairs={len(transforms)}, "
            f"translation={translation}, trans_std={np_module.std(trans_errors):.5f}m, "
            f"rot_std={np_module.std(rot_errors):.3f}deg"
        )
        print(f"{base} -> {name}: used shared sets={', '.join(relative_used_sets[name])}")

    _write_json(out_dir / "rig_extrinsics.json", result)
    print(f"Wrote rig extrinsics: {out_dir / 'rig_extrinsics.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-board", help="Generate printable ChArUco board PNG/PDF.")
    create.add_argument("--output-dir", default="calibration/charuco_board")
    create.add_argument("--squares-x", type=int, default=7)
    create.add_argument("--squares-y", type=int, default=5)
    create.add_argument("--square-mm", type=float, default=28.0)
    create.add_argument("--marker-mm", type=float, default=21.0)
    create.add_argument("--dictionary", default=DEFAULT_ARUCO_DICT)
    create.add_argument("--page-width-mm", type=float, default=210.0)
    create.add_argument("--page-height-mm", type=float, default=297.0)
    create.add_argument("--dpi", type=int, default=300)
    create.set_defaults(func=cmd_create_board)

    capture = subparsers.add_parser("capture", help="Capture ChArUco images from RealSense cameras.")
    capture.add_argument("--output-dir", default="calibration/charuco_capture")
    capture.add_argument("--camera", action="append", required=True, help="Camera mapping NAME=SERIAL.")
    capture.add_argument("--width", type=int, default=424)
    capture.add_argument("--height", type=int, default=240)
    capture.add_argument("--fps", type=int, default=30)
    capture.add_argument("--warmup", type=int, default=4)
    capture.set_defaults(func=cmd_capture)

    intr = subparsers.add_parser("calibrate-intrinsics", help="Calibrate each camera from captured images.")
    intr.add_argument("--board", required=True)
    intr.add_argument("--images-dir", default="calibration/charuco_capture/images")
    intr.add_argument("--output-dir", default="calibration/charuco_result")
    intr.add_argument("--camera-name", action="append", default=None)
    intr.add_argument("--min-corners", type=int, default=12)
    intr.add_argument("--min-frames", type=int, default=12)
    intr.add_argument("--debug-dir", default=None)
    intr.set_defaults(func=cmd_calibrate_intrinsics)

    rig = subparsers.add_parser("calibrate-rig", help="Estimate relative extrinsics between calibrated cameras.")
    rig.add_argument("--board", required=True)
    rig.add_argument("--images-dir", default="calibration/charuco_capture/images")
    rig.add_argument("--intrinsics-dir", default="calibration/charuco_result")
    rig.add_argument("--output-dir", default="calibration/charuco_result")
    rig.add_argument("--camera-name", action="append", default=None)
    rig.add_argument("--base-camera", default="head_image")
    rig.add_argument("--min-corners", type=int, default=12)
    rig.add_argument("--min-pairs", type=int, default=8)
    rig.add_argument("--debug-dir", default=None)
    rig.set_defaults(func=cmd_calibrate_rig)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "camera_name") and args.camera_name is None:
        args.camera_name = list(DEFAULT_CAMERA_NAMES)
    args.func(args)


if __name__ == "__main__":
    main()
