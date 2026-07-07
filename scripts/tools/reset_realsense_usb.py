"""Reset Intel RealSense USB devices without physically replugging them."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INTEL_VENDOR_ID = "8086"
REALSENSE_PRODUCT_IDS = {
    "0ad1",
    "0ad2",
    "0ad3",
    "0ad4",
    "0ad5",
    "0b07",
    "0b3a",
    "0b5c",
    "0b64",
}


@dataclass(frozen=True)
class UsbDevice:
    path: Path
    busnum: str
    devnum: str
    vendor: str
    product: str
    usb_serial: str
    usb_name: str
    speed: str
    camera_serial: str = ""
    camera_name: str = ""
    camera_product_id: str = ""
    camera_physical_port: str = ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _is_realsense_usb_device(path: Path) -> bool:
    vendor = _read_text(path / "idVendor").lower()
    product = _read_text(path / "idProduct").lower()
    name = _read_text(path / "product").lower()
    if vendor != INTEL_VENDOR_ID:
        return False
    return product in REALSENSE_PRODUCT_IDS or "realsense" in name


def _usb_device_path_from_physical_port(physical_port: str) -> Path | None:
    if not physical_port:
        return None
    path = Path(physical_port)
    for candidate in (path, *path.parents):
        if (candidate / "idVendor").is_file() and (candidate / "idProduct").is_file():
            return candidate.resolve()
    return None


def _query_realsense_sdk_devices() -> dict[Path, dict[str, str]]:
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        return {}

    devices_by_usb_path: dict[Path, dict[str, str]] = {}
    try:
        devices = rs.context().query_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] pyrealsense2 query_devices failed: {exc}", file=sys.stderr)
        return {}

    for device in devices:
        info: dict[str, str] = {}
        for key, camera_info in (
            ("camera_name", rs.camera_info.name),
            ("camera_serial", rs.camera_info.serial_number),
            ("camera_product_id", rs.camera_info.product_id),
            ("camera_physical_port", rs.camera_info.physical_port),
        ):
            try:
                info[key] = str(device.get_info(camera_info))
            except Exception:  # noqa: BLE001
                info[key] = ""
        usb_path = _usb_device_path_from_physical_port(info.get("camera_physical_port", ""))
        if usb_path is not None:
            devices_by_usb_path[usb_path] = info
    return devices_by_usb_path


def scan_realsense_usb_devices() -> list[UsbDevice]:
    devices: list[UsbDevice] = []
    sdk_devices = _query_realsense_sdk_devices()
    for path in sorted(Path("/sys/bus/usb/devices").glob("*")):
        if not (path / "idVendor").is_file() or not _is_realsense_usb_device(path):
            continue
        sdk_info = sdk_devices.get(path.resolve(), {})
        devices.append(
            UsbDevice(
                path=path,
                busnum=_read_text(path / "busnum"),
                devnum=_read_text(path / "devnum"),
                vendor=_read_text(path / "idVendor"),
                product=_read_text(path / "idProduct"),
                usb_serial=_read_text(path / "serial"),
                usb_name=_read_text(path / "product"),
                speed=_read_text(path / "speed"),
                camera_serial=sdk_info.get("camera_serial", ""),
                camera_name=sdk_info.get("camera_name", ""),
                camera_product_id=sdk_info.get("camera_product_id", ""),
                camera_physical_port=sdk_info.get("camera_physical_port", ""),
            )
        )
    return devices


def _load_config_serials(config_path: Path | None) -> set[str]:
    if config_path is None or not config_path.is_file():
        return set()
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return set()
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return set()
    cameras = loaded.get("cameras")
    if not isinstance(cameras, dict) and isinstance(loaded.get("record"), dict):
        cameras = loaded["record"].get("cameras")
    if not isinstance(cameras, dict):
        return set()
    return {
        str(value)
        for key, value in cameras.items()
        if key.endswith("_serial") and value not in (None, "")
    }


def _select_devices(devices: Iterable[UsbDevice], serials: set[str]) -> list[UsbDevice]:
    devices = list(devices)
    if not serials:
        return devices
    return [
        device
        for device in devices
        if device.camera_serial in serials or device.usb_serial in serials
    ]


def _print_devices(devices: Iterable[UsbDevice], *, prefix: str = "") -> None:
    for device in devices:
        print(
            f"{prefix}camera_serial={device.camera_serial or '<sdk-not-visible>'} "
            f"usb_serial={device.usb_serial or '<no-usb-serial>'} "
            f"bus={device.busnum} dev={device.devnum} "
            f"speed={device.speed}M path={device.path.name} "
            f"usb_product={device.product} "
            f"camera_name={device.camera_name or '<sdk-not-visible>'} "
            f"usb_name={device.usb_name}"
        )


def hardware_reset(serials: set[str]) -> set[str]:
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError as exc:
        print(f"[WARN] pyrealsense2 import failed: {exc}", file=sys.stderr)
        return set()

    reset_serials: set[str] = set()
    try:
        devices = rs.context().query_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] pyrealsense2 query_devices failed: {exc}", file=sys.stderr)
        return set()

    for device in devices:
        try:
            serial = device.get_info(rs.camera_info.serial_number)
        except Exception:  # noqa: BLE001
            serial = ""
        if serials and serial not in serials:
            continue
        try:
            print(f"[INFO] hardware_reset RealSense serial={serial or '<unknown>'}")
            device.hardware_reset()
            if serial:
                reset_serials.add(serial)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] hardware_reset failed for {serial or '<unknown>'}: {exc}", file=sys.stderr)
    return reset_serials


def sysfs_reset(devices: list[UsbDevice], reset_delay_sec: float) -> tuple[set[str], set[str]]:
    reset_camera_serials: set[str] = set()
    reset_usb_serials: set[str] = set()
    for device in devices:
        authorized = device.path / "authorized"
        if not authorized.is_file():
            print(
                f"[WARN] skip {device.camera_serial or device.usb_serial}: "
                f"no authorized file at {authorized}",
                file=sys.stderr,
            )
            continue
        try:
            print(
                "[INFO] sysfs reset RealSense "
                f"camera_serial={device.camera_serial or '<sdk-not-visible>'} "
                f"usb_serial={device.usb_serial} path={device.path.name}"
            )
            _write_text(authorized, "0")
            time.sleep(reset_delay_sec)
            _write_text(authorized, "1")
            if device.camera_serial:
                reset_camera_serials.add(device.camera_serial)
            if device.usb_serial:
                reset_usb_serials.add(device.usb_serial)
        except PermissionError:
            raise PermissionError(
                "sysfs reset needs root. Re-run with sudo, for example: "
                "sudo -E python scripts/tools/reset_realsense_usb.py --mode sysfs --all"
            )
        except OSError as exc:
            print(
                f"[WARN] sysfs reset failed for "
                f"{device.camera_serial or device.usb_serial}: {exc}",
                file=sys.stderr,
            )
    return reset_camera_serials, reset_usb_serials


def wait_for_serials(
    expected_camera_serials: set[str],
    expected_usb_serials: set[str],
    timeout_sec: float,
) -> bool:
    if not expected_camera_serials and not expected_usb_serials:
        return True
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        current_devices = scan_realsense_usb_devices()
        current_camera_serials = {
            device.camera_serial for device in current_devices if device.camera_serial
        }
        current_usb_serials = {device.usb_serial for device in current_devices if device.usb_serial}
        camera_ok = expected_camera_serials.issubset(current_camera_serials)
        usb_ok = expected_usb_serials.issubset(current_usb_serials)
        if camera_ok and usb_ok:
            return True
        time.sleep(0.5)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "hardware", "sysfs"),
        default="auto",
        help="Reset method. auto tries pyrealsense2 first and falls back to sysfs for visible devices.",
    )
    parser.add_argument(
        "--serial",
        action="append",
        default=[],
        help="Only reset this RealSense camera serial or USB serial. Can be provided multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Reset all RealSense devices found in /sys/bus/usb/devices.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/config/robots/flexiv_config.yaml"),
        help="Optional robot camera config used only for mismatch diagnostics.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only list detected devices.")
    parser.add_argument("--reset-delay-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    args = parser.parse_args(argv)

    before = scan_realsense_usb_devices()
    config_serials = _load_config_serials(args.config)
    requested_serials = {str(serial) for serial in args.serial}
    if not args.all and requested_serials:
        selected = _select_devices(before, requested_serials)
    else:
        selected = before

    print(f"[INFO] detected {len(before)} RealSense USB device(s)")
    _print_devices(before, prefix="  ")
    if config_serials:
        detected_camera_serials = {device.camera_serial for device in before if device.camera_serial}
        missing = sorted(config_serials - detected_camera_serials)
        extra = sorted(detected_camera_serials - config_serials)
        if missing:
            print(f"[WARN] configured camera serial(s) not visible to pyrealsense2: {missing}")
        if extra:
            print(f"[WARN] pyrealsense2 sees unconfigured camera serial(s): {extra}")
        sdk_invisible = [device.usb_serial for device in before if not device.camera_serial]
        if sdk_invisible:
            print(f"[WARN] USB RealSense device(s) not visible to pyrealsense2: {sdk_invisible}")

    if args.list_only:
        return 0
    if not selected:
        print("[ERROR] no matching RealSense USB devices found", file=sys.stderr)
        return 2

    target_camera_serials = {device.camera_serial for device in selected if device.camera_serial}
    target_usb_serials = {device.usb_serial for device in selected if device.usb_serial}
    reset_camera_serials: set[str] = set()
    reset_usb_serials: set[str] = set()
    if args.mode in ("auto", "hardware"):
        reset_camera_serials.update(hardware_reset(requested_serials if requested_serials else set()))
        time.sleep(args.reset_delay_sec)

    if args.mode == "sysfs" or (args.mode == "auto" and not reset_camera_serials):
        camera_serials, usb_serials = sysfs_reset(selected, args.reset_delay_sec)
        reset_camera_serials.update(camera_serials)
        reset_usb_serials.update(usb_serials)

    expected_camera_serials = reset_camera_serials or target_camera_serials
    expected_usb_serials = reset_usb_serials or (set() if expected_camera_serials else target_usb_serials)
    if not wait_for_serials(expected_camera_serials, expected_usb_serials, args.timeout_sec):
        current = scan_realsense_usb_devices()
        print("[ERROR] RealSense devices did not re-enumerate before timeout", file=sys.stderr)
        _print_devices(current, prefix="  ")
        return 1

    print("[INFO] RealSense reset complete")
    _print_devices(scan_realsense_usb_devices(), prefix="  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
