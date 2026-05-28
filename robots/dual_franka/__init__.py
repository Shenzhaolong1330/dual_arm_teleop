"""Franka dual-arm robot module."""

from .config_franka import FrankaDualArmConfig
from .franka_dual_arm import FrankaDualArm
from .dual_franka_robotiq_rpc_client import FrankaDualArmClient


def __getattr__(name: str):
    if name == "FrankaDualArmServer":
        try:
            from .dual_franka_robotiq_rpc_server import DualFrankaRobotiqRpcApi
        except ImportError as exc:
            raise ImportError(
                "FrankaDualArmServer is not available in this package. "
                "Use robots.dual_franka.dual_franka_robotiq_rpc_server as the "
                "ZeroRPC server entrypoint, or install the ROS2 server dependencies."
            ) from exc

        return DualFrankaRobotiqRpcApi
    raise AttributeError(name)

__all__ = [
    "FrankaDualArmConfig",
    "FrankaDualArm",
    "FrankaDualArmClient",
]
