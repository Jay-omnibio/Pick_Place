"""
Actuator backend abstraction: swap sim controller for real robot API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ActuatorBackend(ABC):
    """Abstract actuator backend. Real impl would send to robot API."""

    @abstractmethod
    def apply_action(self, action: Dict) -> None:
        """Apply action (move/ee_target_pos, grip) to robot."""
        pass


class SimActuatorBackend(ActuatorBackend):
    """Simulation backend: uses EEController + SafetyChecker."""

    def __init__(self, controller: Any, safety_checker: Any | None = None):
        self.controller = controller
        self.safety_checker = safety_checker

    def apply_action(self, action: Dict) -> None:
        if action is None:
            return
        if self.safety_checker is not None:
            current_ee = self.controller.simulator.get_ee_position()
            action = self.safety_checker.check(action, current_ee_pos=current_ee)
            if action is None:
                return
        self.controller.apply_action(action)
