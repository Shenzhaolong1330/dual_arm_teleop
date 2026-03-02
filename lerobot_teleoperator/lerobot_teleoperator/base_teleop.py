#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base teleoperation class defining the common interface for all teleoperation modes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_teleop import BaseTeleopConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseTeleop(Teleoperator, ABC):
    """
    Abstract base class for all teleoperation modes.
    
    Subclasses must implement:
    - _connect_impl(): Connect to the teleoperation device
    - _disconnect_impl(): Disconnect from the teleoperation device
    - _get_action_impl(): Get action from the teleoperation device
    - action_features: Property returning the action feature dict
    """
    
    config_class = BaseTeleopConfig
    name = "BaseTeleop"  # Default name, will be overridden by subclasses
    
    def __init__(self, config: BaseTeleopConfig):
        super().__init__(config)
        self.cfg = config
        self._is_connected = False
        # Set the name from subclass method
        self.name = self._get_teleop_name()
    
    @abstractmethod
    def _get_teleop_name(self) -> str:
        """Return the name of this teleoperation mode."""
        pass
    
    @abstractmethod
    def _connect_impl(self) -> None:
        """Implementation of device connection. Subclasses must override."""
        pass
    
    @abstractmethod
    def _disconnect_impl(self) -> None:
        """Implementation of device disconnection. Subclasses must override."""
        pass
    
    @abstractmethod
    def _get_action_impl(self) -> Dict[str, Any]:
        """Implementation of getting action. Subclasses must override."""
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_connected
    
    @property
    def feedback_features(self) -> dict:
        return {}
    
    def connect(self) -> None:
        """Connect to the teleoperation device."""
        if self._is_connected:
            logger.warning(f"{self.name} is already connected.")
            return
        
        logger.info(f"\n===== [TELEOP] Connecting to {self.name} =====")
        self._connect_impl()
        self._is_connected = True
        logger.info(f"===== [TELEOP] {self.name} connected successfully =====\n")
    
    def disconnect(self) -> None:
        """Disconnect from the teleoperation device."""
        if not self._is_connected:
            return
        
        self._disconnect_impl()
        self._is_connected = False
        logger.info(f"[INFO] ===== {self.name} disconnected =====")
    
    def get_action(self) -> Dict[str, Any]:
        """Get the current action from the teleoperation device."""
        if not self._is_connected:
            raise RuntimeError(f"{self.name} is not connected.")
        return self._get_action_impl()
    
    def calibrate(self) -> None:
        """Calibrate the teleoperation device. Default: no-op."""
        pass
    
    def configure(self) -> None:
        """Configure the teleoperation device. Default: no-op."""
        pass
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperation device. Default: no-op."""
        pass
