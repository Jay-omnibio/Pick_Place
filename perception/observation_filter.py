"""
Observation filter: Kalman smoothing, outlier rejection, optional latency handling.

For real hardware you'd typically use:
- Kalman/EKF for state estimation
- Outlier rejection for sensor glitches
- Latency buffer for camera/network delay
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np


class ObservationFilter:
    """
    Filters raw observations: Kalman smoothing for positions, outlier rejection.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        filt = config.get("observation_filter", {})

        # Kalman: process/measurement noise (tune for your sensors)
        self.kalman_process_std = float(filt.get("kalman_process_std", 0.015))
        self.kalman_measure_std_obj = float(filt.get("kalman_measure_std_obj", 0.03))
        self.kalman_measure_std_target = float(filt.get("kalman_measure_std_target", 0.03))
        self.kalman_measure_std_ee = float(filt.get("kalman_measure_std_ee", 0.005))

        # Outlier rejection: max allowed change per step (meters)
        self.outlier_max_jump_obj = float(filt.get("outlier_max_jump_obj", 0.15))
        self.outlier_max_jump_target = float(filt.get("outlier_max_jump_target", 0.15))
        self.outlier_max_jump_ee = float(filt.get("outlier_max_jump_ee", 0.08))

        # Latency: number of steps to delay observations (0 = no delay, for real camera later)
        self.latency_steps = int(filt.get("latency_steps", 0))
        self._latency_buffer: Deque[Dict] = deque(maxlen=max(1, self.latency_steps + 1))

        # Kalman state: mean and variance for each 3D signal (position only, simple model)
        self._x_obj: Optional[np.ndarray] = None
        self._P_obj: float = 1.0
        self._x_target: Optional[np.ndarray] = None
        self._P_target: float = 1.0
        self._x_ee: Optional[np.ndarray] = None
        self._P_ee: float = 1.0

    def filter(self, raw_obs: Dict) -> Dict:
        """
        Apply Kalman smoothing + outlier rejection to raw observation.
        Returns filtered observation dict (same keys as input).
        """
        obs = dict(raw_obs)

        # Latency: delay observations if configured (for sim, typically 0)
        if self.latency_steps > 0:
            self._latency_buffer.append(obs)
            if len(self._latency_buffer) <= self.latency_steps:
                return obs  # Not enough history yet
            obs = self._latency_buffer[-1 - self.latency_steps]

        o_ee = np.asarray(obs["o_ee"], dtype=float)
        o_obj = np.asarray(obs["o_obj"], dtype=float)
        o_target = np.asarray(obs["o_target"], dtype=float)

        # Outlier rejection: clip large jumps
        o_obj = self._reject_outlier_3d(
            o_obj,
            self._x_obj,
            self.outlier_max_jump_obj,
        )
        o_target = self._reject_outlier_3d(
            o_target,
            self._x_target,
            self.outlier_max_jump_target,
        )
        o_ee = self._reject_outlier_3d(
            o_ee,
            self._x_ee,
            self.outlier_max_jump_ee,
        )

        # Kalman update (simple 1D per component, independent)
        self._x_obj, self._P_obj = self._kalman_update_3d(
            o_obj,
            self._x_obj,
            self._P_obj,
            self.kalman_process_std,
            self.kalman_measure_std_obj,
        )
        self._x_target, self._P_target = self._kalman_update_3d(
            o_target,
            self._x_target,
            self._P_target,
            self.kalman_process_std,
            self.kalman_measure_std_target,
        )
        self._x_ee, self._P_ee = self._kalman_update_3d(
            o_ee,
            self._x_ee,
            self._P_ee,
            self.kalman_process_std * 0.5,  # EE moves more predictably
            self.kalman_measure_std_ee,
        )

        filtered = {
            "o_ee": self._x_ee.tolist(),
            "o_obj": self._x_obj.tolist(),
            "o_target": self._x_target.tolist(),
            "o_grip": obs["o_grip"],
            "o_contact": obs["o_contact"],
        }
        # Pass through extra scalar/vector observations that do not use this filter,
        # such as object yaw used for dynamic gripper alignment.
        for k, v in obs.items():
            if k not in filtered:
                filtered[k] = v
        return filtered

    def _reject_outlier_3d(
        self,
        observed: np.ndarray,
        prev: Optional[np.ndarray],
        max_jump: float,
    ) -> np.ndarray:
        if prev is None:
            return observed
        delta = np.linalg.norm(observed - prev)
        if delta <= max_jump:
            return observed
        # Clamp to max_jump in direction of observed
        direction = observed - prev
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        return prev + direction * max_jump

    def _kalman_update_3d(
        self,
        z: np.ndarray,
        x: Optional[np.ndarray],
        P: float,
        process_std: float,
        measure_std: float,
    ) -> tuple[np.ndarray, float]:
        Q = process_std ** 2
        R = measure_std ** 2
        if x is None:
            return z.copy(), R
        # Predict
        x_pred = x  # constant position model (no velocity in state)
        P_pred = P + Q
        # Update
        K = P_pred / (P_pred + R)
        x_new = x_pred + K * (z - x_pred)
        P_new = (1 - K) * P_pred
        return x_new, float(P_new)

    def reset(self) -> None:
        """Reset filter state (e.g. new episode)."""
        self._x_obj = None
        self._P_obj = 1.0
        self._x_target = None
        self._P_target = 1.0
        self._x_ee = None
        self._P_ee = 1.0
        self._latency_buffer.clear()
