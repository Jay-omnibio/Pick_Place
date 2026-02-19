from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class _Track3D:
    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    last_t: float | None = None


class _AlphaBeta3D:
    """
    Simple alpha-beta estimator for 3D position/velocity.
    This is lightweight and robust for online robotics telemetry.
    """

    def __init__(self, alpha: float, beta: float, min_dt: float, max_dt: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.beta = float(max(0.0, beta))
        self.min_dt = float(max(1e-6, min_dt))
        self.max_dt = float(max(self.min_dt, max_dt))
        self.track = _Track3D()

    def reset(self):
        self.track = _Track3D()

    def update(self, meas_pos: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray, float]:
        z = np.asarray(meas_pos, dtype=float).reshape(3)
        t = float(timestamp)

        if self.track.pos is None or self.track.vel is None or self.track.last_t is None:
            self.track.pos = z.copy()
            self.track.vel = np.zeros(3, dtype=float)
            self.track.last_t = t
            return self.track.pos.copy(), self.track.vel.copy(), 0.0

        dt_raw = t - float(self.track.last_t)
        if not np.isfinite(dt_raw):
            dt_raw = self.min_dt
        dt = float(np.clip(dt_raw, self.min_dt, self.max_dt))

        # Predict
        x_pred = self.track.pos + self.track.vel * dt
        v_pred = self.track.vel

        # Innovation
        r = z - x_pred

        # Update
        x_new = x_pred + self.alpha * r
        v_new = v_pred + (self.beta / dt) * r

        self.track.pos = x_new
        self.track.vel = v_new
        self.track.last_t = t
        return x_new.copy(), v_new.copy(), dt


class StateEstimator:
    """
    Runtime estimator for ee/object/target signals.
    Adds estimated position/velocity to observation stream:
      - o_ee_est, o_obj_est, o_target_est
      - o_ee_vel, o_obj_vel, o_target_vel
      - o_dt
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        alpha = float(cfg.get("alpha", 0.65))
        beta = float(cfg.get("beta", 0.08))
        min_dt = float(cfg.get("min_dt", 0.001))
        max_dt = float(cfg.get("max_dt", 0.2))

        self.ee = _AlphaBeta3D(alpha=alpha, beta=beta, min_dt=min_dt, max_dt=max_dt)
        self.obj = _AlphaBeta3D(alpha=alpha, beta=beta, min_dt=min_dt, max_dt=max_dt)
        self.target = _AlphaBeta3D(alpha=alpha, beta=beta, min_dt=min_dt, max_dt=max_dt)

    @staticmethod
    def _safe_ts(obs: Dict[str, Any]) -> float:
        ts = obs.get("o_timestamp", 0.0)
        arr = np.asarray(ts, dtype=float).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            return float(arr[0])
        return 0.0

    def reset(self):
        self.ee.reset()
        self.obj.reset()
        self.target.reset()

    def update(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray | float]:
        if not self.enabled:
            return {}

        ts = self._safe_ts(observation)
        ee_m = np.asarray(observation.get("o_ee", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        obj_m = np.asarray(observation.get("o_obj", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        tgt_m = np.asarray(observation.get("o_target", [0.0, 0.0, 0.0]), dtype=float).reshape(3)

        ee_pos, ee_vel, dt_ee = self.ee.update(ee_m, ts)
        obj_pos, obj_vel, dt_obj = self.obj.update(obj_m, ts)
        tgt_pos, tgt_vel, dt_tgt = self.target.update(tgt_m, ts)
        dt = float(max(dt_ee, dt_obj, dt_tgt))

        return {
            "o_ee_est": ee_pos,
            "o_obj_est": obj_pos,
            "o_target_est": tgt_pos,
            "o_ee_vel": ee_vel,
            "o_obj_vel": obj_vel,
            "o_target_vel": tgt_vel,
            "o_dt": dt,
        }

