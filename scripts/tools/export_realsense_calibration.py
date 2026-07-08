#!/usr/bin/env python
"""Export RealSense intrinsics/extrinsics for RGB-D/IR stereo validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_WIDTH = 424
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 30
SCHEMA_VERSION = 1
PROFILE_NOTE = (
    "Default standalone profile is 424x240@30 to match the earlier recorded RGB-D/IR sidecar data. "
    "For Fast-FoundationStereo ONNX/TensorRT deployment, prefer inputs divisible by 32, for example "
    "640x480, or use explicit padding/resize before inference."
)
BASELINE_NOTE = (
    "For RealSense left/right IR stereo, use baseline_m_abs_x when the stereo pair is rectified "
    "and the left-to-right translation is mostly along x. baseline_m_norm is also exported as a "
    "geometry sanity check and fallback when the extrinsics are not x-dominant."
)


def _import_rs():
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 is required. Activate the dual_arm_teleop conda env or install pyrealsense2."
        ) from exc
    return rs


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.split(".")[-1] if "." in text else text


def _safe_get_info(device: Any, camera_info: Any) -> str:
    try:
        return str(device.get_info(camera_info))
    except Exception:  # noqa: BLE001
        return ""


def device_info(device: Any, rs: Any | None = None) -> dict[str, str]:
    rs = rs or _import_rs()
    return {
        "serial": _safe_get_info(device, rs.camera_info.serial_number),
        "name": _safe_get_info(device, rs.camera_info.name),
        "firmware": _safe_get_info(device, rs.camera_info.firmware_version),
        "usb_type": _safe_get_info(device, rs.camera_info.usb_type_descriptor),
        "physical_port": _safe_get_info(device, rs.camera_info.physical_port),
        "product_id": _safe_get_info(device, rs.camera_info.product_id),
        "product_line": _safe_get_info(device, rs.camera_info.product_line),
    }


def _video_profiles(device: Any) -> list[Any]:
    profiles: list[Any] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                profiles.append(profile.as_video_stream_profile())
    return profiles


def _stream_index(profile: Any) -> int:
    try:
        return int(profile.stream_index())
    except Exception:  # noqa: BLE001
        return -1


def _stream_name(profile: Any) -> str:
    try:
        return str(profile.stream_name())
    except Exception:  # noqa: BLE001
        try:
            return _enum_name(profile.stream_type())
        except Exception:  # noqa: BLE001
            return "unknown"


def _profile_summary(profile: Any) -> dict[str, Any]:
    return {
        "stream": _stream_name(profile),
        "stream_index": _stream_index(profile),
        "width": int(profile.width()),
        "height": int(profile.height()),
        "fps": int(profile.fps()),
        "format": _enum_name(profile.format()),
    }


def _format_profile_candidate(profile: Any) -> str:
    summary = _profile_summary(profile)
    index = summary["stream_index"]
    index_text = "" if index < 0 else f"[{index}]"
    return (
        f"{summary['stream']}{index_text} "
        f"{summary['width']}x{summary['height']}@{summary['fps']} "
        f"{summary['format']}"
    )


def _required_stream_specs(rs: Any) -> tuple[dict[str, Any], ...]:
    return (
        {"key": "color", "stream": rs.stream.color, "index": -1, "format": rs.format.rgb8},
        {"key": "depth", "stream": rs.stream.depth, "index": -1, "format": rs.format.z16},
        {"key": "infrared1", "stream": rs.stream.infrared, "index": 1, "format": rs.format.y8},
        {"key": "infrared2", "stream": rs.stream.infrared, "index": 2, "format": rs.format.y8},
    )


def _profile_matches_spec(profile: Any, spec: Mapping[str, Any], width: int, height: int, fps: int) -> bool:
    try:
        if profile.stream_type() != spec["stream"]:
            return False
        if int(spec["index"]) >= 0 and _stream_index(profile) != int(spec["index"]):
            return False
        return (
            int(profile.width()) == int(width)
            and int(profile.height()) == int(height)
            and int(profile.fps()) == int(fps)
            and profile.format() == spec["format"]
        )
    except Exception:  # noqa: BLE001
        return False


def _same_stream_candidates(profiles: list[Any], spec: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for profile in profiles:
        try:
            if profile.stream_type() != spec["stream"]:
                continue
            if int(spec["index"]) >= 0 and _stream_index(profile) != int(spec["index"]):
                continue
            candidates.append(_format_profile_candidate(profile))
        except Exception:  # noqa: BLE001
            continue
    return sorted(set(candidates))


def validate_required_profiles(device: Any, width: int, height: int, fps: int, rs: Any | None = None) -> None:
    rs = rs or _import_rs()
    profiles = _video_profiles(device)
    missing: list[str] = []
    for spec in _required_stream_specs(rs):
        if any(_profile_matches_spec(profile, spec, width, height, fps) for profile in profiles):
            continue
        candidates = _same_stream_candidates(profiles, spec)
        candidate_text = "\n    ".join(candidates[:40]) if candidates else "(no candidates)"
        if len(candidates) > 40:
            candidate_text += f"\n    ... {len(candidates) - 40} more"
        missing.append(
            f"- {spec['key']} requested {width}x{height}@{fps} "
            f"{_enum_name(spec['format'])}; supported candidates:\n    {candidate_text}"
        )
    if missing:
        info = device_info(device, rs)
        raise RuntimeError(
            "Requested RealSense stream profile is unavailable for "
            f"serial={info.get('serial')} name={info.get('name')}:\n" + "\n".join(missing)
        )


def _intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": fx,
        "fy": fy,
        "ppx": ppx,
        "ppy": ppy,
        "cx": ppx,
        "cy": ppy,
        "distortion_model": _enum_name(intrinsics.model),
        "coeffs": [float(v) for v in intrinsics.coeffs],
        "K": [
            [fx, 0.0, ppx],
            [0.0, fy, ppy],
            [0.0, 0.0, 1.0],
        ],
    }


def _video_stream_to_dict(profile: Any) -> dict[str, Any]:
    data = _profile_summary(profile)
    data["intrinsics"] = _intrinsics_to_dict(profile.get_intrinsics())
    return data


def _extrinsics_to_dict(source: Any, target: Any) -> dict[str, Any]:
    extrinsics = source.get_extrinsics_to(target)
    rotation = [float(v) for v in extrinsics.rotation]
    translation = [float(v) for v in extrinsics.translation]
    return {
        "rotation": rotation,
        "translation_m": translation,
        "rotation_matrix_row_major": [
            rotation[0:3],
            rotation[3:6],
            rotation[6:9],
        ],
    }


def _baseline_from_extrinsics(extrinsics: Mapping[str, Any]) -> dict[str, Any]:
    translation = [float(v) for v in extrinsics["translation_m"]]
    norm = math.sqrt(sum(v * v for v in translation))
    abs_x = abs(translation[0]) if translation else 0.0
    x_dominance = abs_x / norm if norm > 0.0 else 0.0
    if abs_x > 0.0 and x_dominance >= 0.95:
        recommended_key = "baseline_m_abs_x"
        recommended_value = abs_x
        recommendation_reason = "left-to-right translation is x-dominant"
    else:
        recommended_key = "baseline_m_norm"
        recommended_value = norm
        recommendation_reason = "left-to-right translation is not x-dominant"
    return {
        "baseline_m_abs_x": abs_x,
        "baseline_m_norm": norm,
        "x_dominance_ratio": x_dominance,
        "recommended_baseline_key": recommended_key,
        "recommended_baseline_m": recommended_value,
        "recommendation_reason": recommendation_reason,
        "note": BASELINE_NOTE,
    }


def _depth_scale(profile: Any) -> float | None:
    try:
        return float(profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception:  # noqa: BLE001
        return None


def calibration_from_pipeline_profile(
    profile: Any,
    logical_camera: str | None = None,
    requested_profile: Mapping[str, Any] | None = None,
    rs: Any | None = None,
) -> dict[str, Any]:
    rs = rs or _import_rs()
    device = profile.get_device()
    streams = {
        "color": profile.get_stream(rs.stream.color).as_video_stream_profile(),
        "depth": profile.get_stream(rs.stream.depth).as_video_stream_profile(),
        "infrared1": profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
        "infrared2": profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile(),
    }
    extrinsics = {
        "infrared1_to_infrared2": _extrinsics_to_dict(streams["infrared1"], streams["infrared2"]),
        "infrared2_to_infrared1": _extrinsics_to_dict(streams["infrared2"], streams["infrared1"]),
        "infrared1_to_color": _extrinsics_to_dict(streams["infrared1"], streams["color"]),
        "color_to_infrared1": _extrinsics_to_dict(streams["color"], streams["infrared1"]),
        "depth_to_color": _extrinsics_to_dict(streams["depth"], streams["color"]),
        "color_to_depth": _extrinsics_to_dict(streams["color"], streams["depth"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "logical_camera": logical_camera,
        "device": device_info(device, rs),
        "requested_profile": dict(requested_profile or {}),
        "profile_note": PROFILE_NOTE,
        "depth_scale_m_per_unit": _depth_scale(profile),
        "streams": {name: _video_stream_to_dict(stream) for name, stream in streams.items()},
        "extrinsics": extrinsics,
        "baseline": _baseline_from_extrinsics(extrinsics["infrared1_to_infrared2"]),
    }


def _enable_required_streams(config: Any, rs: Any, width: int, height: int, fps: int) -> None:
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)


def collect_device_calibration(
    serial: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    rs: Any | None = None,
) -> dict[str, Any]:
    rs = rs or _import_rs()
    ctx = rs.context()
    devices = {device_info(device, rs)["serial"]: device for device in ctx.query_devices()}
    if serial not in devices:
        available = ", ".join(sorted(devices)) or "(none)"
        raise RuntimeError(f"RealSense serial {serial!r} was not found. Available serials: {available}")

    device = devices[serial]
    validate_required_profiles(device, width=width, height=height, fps=fps, rs=rs)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    _enable_required_streams(config, rs, width=width, height=height, fps=fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to start RealSense serial={serial} with color/depth/IR profile "
            f"{width}x{height}@{fps}. Close other camera users and verify bandwidth/profile support."
        ) from exc

    try:
        return calibration_from_pipeline_profile(
            profile,
            logical_camera=None,
            requested_profile={"width": int(width), "height": int(height), "fps": int(fps)},
            rs=rs,
        )
    finally:
        pipeline.stop()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _calibration_filename(serial: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in serial)
    return f"realsense_{safe}.json"


def export_calibration_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    manifest_path: Path | None = None,
    session_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser()
    written: dict[str, Path] = {}
    manifest_cameras: dict[str, Any] = {}
    for logical_name, payload in payloads.items():
        serial = str(payload.get("device", {}).get("serial") or logical_name)
        path = output_dir / _calibration_filename(serial)
        write_json(path, payload)
        written[logical_name] = path
        manifest_cameras[logical_name] = {
            **dict(payload),
            "calibration_file": str(path),
        }

    if manifest_path is not None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now_iso(),
            "profile_note": PROFILE_NOTE,
            "baseline_note": BASELINE_NOTE,
            "session": dict(session_metadata or {}),
            "calibration_output_dir": str(output_dir),
            "cameras": manifest_cameras,
        }
        write_json(Path(manifest_path).expanduser(), manifest)
        written["manifest"] = Path(manifest_path).expanduser()
    return written


def export_connected_realsense_calibrations(
    cameras: Mapping[str, Any],
    output_dir: Path,
    manifest_path: Path | None = None,
    session_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    rs = _import_rs()
    payloads: dict[str, dict[str, Any]] = {}
    for logical_name, camera in cameras.items():
        profile = getattr(camera, "rs_profile", None)
        if profile is None:
            continue
        requested_profile = {
            "width": getattr(camera, "capture_width", getattr(camera, "width", None)),
            "height": getattr(camera, "capture_height", getattr(camera, "height", None)),
            "fps": getattr(camera, "fps", None),
        }
        payloads[str(logical_name)] = calibration_from_pipeline_profile(
            profile,
            logical_camera=str(logical_name),
            requested_profile=requested_profile,
            rs=rs,
        )
    if not payloads:
        raise RuntimeError("No connected RealSense camera profiles were found on the robot object.")
    return export_calibration_payloads(
        payloads,
        output_dir=output_dir,
        manifest_path=manifest_path,
        session_metadata=session_metadata,
    )


def _query_devices(rs: Any) -> list[Any]:
    return list(rs.context().query_devices())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="append", default=None, help="RealSense serial to export. May be repeated.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Stream width. Default: 424.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Stream height. Default: 240.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Stream FPS. Default: 30.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for realsense_<serial>.json. Default: /tmp.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional aggregate JSON manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rs = _import_rs()
    devices = _query_devices(rs)
    infos = [device_info(device, rs) for device in devices]
    print(f"Detected {len(infos)} RealSense device(s):")
    for info in infos:
        print(
            "  serial={serial} name={name} firmware={firmware} usb={usb_type}".format(
                **info
            )
        )

    if not infos:
        raise SystemExit("No RealSense devices found.")

    requested_serials = args.serial or [info["serial"] for info in infos]
    known_serials = {info["serial"] for info in infos}
    missing_serials = sorted(set(requested_serials) - known_serials)
    if missing_serials:
        raise SystemExit(
            f"Requested RealSense serial(s) not found: {missing_serials}. "
            f"Available: {sorted(known_serials)}"
        )

    output_dir = args.output_dir or Path("/tmp")
    payloads: dict[str, dict[str, Any]] = {}
    for serial in requested_serials:
        payloads[serial] = collect_device_calibration(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            rs=rs,
        )

    written = export_calibration_payloads(
        payloads,
        output_dir=output_dir,
        manifest_path=args.manifest_path,
        session_metadata={
            "source": "standalone_cli",
            "requested_profile": {"width": args.width, "height": args.height, "fps": args.fps},
        },
    )

    print("Exported RealSense calibration:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    print(PROFILE_NOTE)
    print(BASELINE_NOTE)


if __name__ == "__main__":
    main()
