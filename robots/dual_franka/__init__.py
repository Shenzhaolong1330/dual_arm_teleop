"""Franka dual-arm robot module."""

from .config_franka import FrankaDualArmConfig
from .franka_dual_arm import FrankaDualArm
from .dual_franka_robotiq_rpc_client import FrankaDualArmClient


def __getattr__(name: str):
    if name == "FrankaDualArmServer":
        from .franka_interface_server import FrankaDualArmServer

        return FrankaDualArmServer
    raise AttributeError(name)

__all__ = [
    "FrankaDualArmConfig",
    "FrankaDualArm",
    "FrankaDualArmClient",
    "FrankaDualArmServer",
]
