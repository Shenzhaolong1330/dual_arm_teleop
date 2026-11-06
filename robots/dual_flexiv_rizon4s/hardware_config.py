"""Load the single Flexiv hardware YAML; derive diagnostic restrictions in memory."""
from pathlib import Path

import yaml


HARDWARE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "scripts/config/robots/flexiv_config.yaml"
READ_ONLY_FLAGS = (
    "enable_on_connect", "clear_fault_on_connect", "go_home_on_connect",
    "reset_go_home", "initialize_gripper_on_connect", "zero_ft_sensor_on_connect",
    "switch_cartesian_mode_on_connect", "use_cartesian_servo_thread",
    "switch_tool_on_connect", "open_grippers_on_connect", "reset_opens_grippers",
    "stop_grippers_on_disconnect",
)


def load_hardware_config(*, readonly: bool = False) -> dict:
    """Return an independent config; this function never connects to hardware.

    Read-only flags suppress automatic setup/reset operations. Diagnostic callers
    must still avoid send_action and use their no-actuator cleanup path.
    """
    config = yaml.safe_load(HARDWARE_CONFIG_PATH.read_text())
    if readonly:
        config["robot"].update(dict.fromkeys(READ_ONLY_FLAGS, False))
    return config
