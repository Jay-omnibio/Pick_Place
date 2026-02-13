"""
Sensor backend abstraction: swap sim sensors for real camera/encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import yaml

from env.sensors import SensorSuite
from perception.observation_filter import ObservationFilter


class SensorBackend(ABC):
    """Abstract sensor backend. Real impl would read from camera, encoders, etc."""

    @abstractmethod
    def get_observation(self, state: Any) -> Dict[str, np.ndarray]:
        """
        Return observation dict: o_ee, o_obj, o_target, o_grip, o_contact.
        state: backend-specific (sim_state for sim, frame/cache for real).
        """
        pass


class SimSensorBackend(SensorBackend):
    """
    Simulation backend: SensorSuite + ObservationFilter.
    state = sim_state from MujocoSimulator.get_state().
    """

    def __init__(self, config_path: str, config: Dict | None = None):
        if config is None:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        self.sensors = SensorSuite(config)
        self.filter = ObservationFilter(config)
        self.use_filter = config.get("observation_filter", {}).get("enabled", True)

    def get_observation(self, state: Dict) -> Dict[str, np.ndarray]:
        raw = self.sensors.get_observation(state)
        if self.use_filter:
            return {k: np.asarray(v) for k, v in self.filter.filter(raw).items()}
        return {k: np.asarray(v) for k, v in raw.items()}

    def reset(self) -> None:
        self.filter.reset()
