"""
Example robot module template.
This module provides a template for implementing a new robot interface.

To implement a new robot:
1. Copy this example_robot folder and rename it to your robot name
2. Rename all files and classes accordingly
3. Implement the stub methods in each file
4. Register your robot in robots/__init__.py
"""

from .config_example import ExampleRobotConfig
from .example_robot import ExampleRobot
from .example_robot_interface_client import ExampleRobotInterfaceClient

__all__ = [
    "ExampleRobotConfig",
    "ExampleRobot",
    "ExampleRobotInterfaceClient",
]
