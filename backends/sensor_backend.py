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
from state.state_estimator import StateEstimator


class SensorBackend(ABC):
    """Abstract sensor backend. Real impl would read from camera, encoders, etc."""

    @abstractmethod
    def get_observation(self, state: Any) -> Dict[str, Any]:
        """
        Return observation dict: o_ee, o_obj, o_target, o_grip, o_contact, o_timestamp.
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
        estimator_cfg = config.get("state_estimator", {})
        self.state_estimator = StateEstimator(estimator_cfg)
        self.use_state_estimator = bool(estimator_cfg.get("enabled", True))

    @staticmethod
    def _to_obs_value(v):
        # Keep scalar values as native floats/ints so freshness/timing fields remain easy to use.
        if isinstance(v, (float, int, np.floating, np.integer)):
            if isinstance(v, (np.integer, int)):
                return int(v)
            return float(v)
        arr = np.asarray(v)
        if arr.shape == ():
            scalar = arr.item()
            if isinstance(scalar, (int, np.integer)):
                return int(scalar)
            return float(scalar)
        return arr

    def get_observation(self, state: Dict) -> Dict[str, Any]:
        raw = self.sensors.get_observation(state)
        src = self.filter.filter(raw) if self.use_filter else raw
        obs = {k: self._to_obs_value(v) for k, v in src.items()}
        if self.use_state_estimator:
            est = self.state_estimator.update(obs)
            for k, v in est.items():
                obs[k] = self._to_obs_value(v)
        return obs

    def reset(self) -> None:
        self.filter.reset()
        self.state_estimator.reset()
