import pyrealsense2 as rs

from scripts.tools.reset_realsense_usb import _read_text, _usb_device_path_from_physical_port


def _get_info(dev, camera_info) -> str:
    try:
        return str(dev.get_info(camera_info))
    except Exception:  # noqa: BLE001
        return ""


# List the connected Intel RealSense cameras and print their serial numbers.
def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.devices
    num_devices = len(devices)
    print(f"------------Detected {num_devices} RealSense device------------")

    if num_devices == 0:
        return

    for i, dev in enumerate(devices):
        serial = _get_info(dev, rs.camera_info.serial_number)
        name = _get_info(dev, rs.camera_info.name)
        product_id = _get_info(dev, rs.camera_info.product_id)
        firmware = _get_info(dev, rs.camera_info.firmware_version)
        usb_type = _get_info(dev, rs.camera_info.usb_type_descriptor)
        physical_port = _get_info(dev, rs.camera_info.physical_port)
        usb_path = _usb_device_path_from_physical_port(physical_port)
        usb_serial = _read_text(usb_path / "serial") if usb_path is not None else ""
        usb_path_name = usb_path.name if usb_path is not None else ""
        print(
            f"Device {i}: Name={name}, Serial={serial}, ProductID={product_id}, "
            f"Firmware={firmware}, USB={usb_type}, USBSerial={usb_serial}, USBPath={usb_path_name}"
        )

def main():
    list_realsense_devices()


if __name__ == "__main__":
    main()
