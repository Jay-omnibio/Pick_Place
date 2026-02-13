from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from tasks.pick_place_fsm import Phase, TaskConfig, TaskState


@dataclass(frozen=True)
class PolicyConfig:
    # Guarded motion: scale applied to controller max_step during descend (0.2 = 20% speed).
    descend_max_step_scale: float = 0.75 # 2 mean
    place_descend_max_step_scale: float = 0.75
    descend_xy_correction_gain: float = 1.15
    descend_xy_correction_max: float = 0.055 
    descend_xy_priority_threshold: float = 0.010
    # Lock XY once centered to prevent late-stage sideways drift while descending.
    descend_xy_lock_threshold: float = 0.004  # 
    descend_xy_unlock_threshold: float = 0.08  #

    # Reach: planar-first approach - move XY until above object, then descend.
    reach_xy_first_threshold: float = 0.03   # only allow Z movement when XY error below this
    reach_slowdown_threshold: float = 0.08 # start slowing down when within this distance to target
    reach_slowdown_scale: float = 0.75 # scale for max_step when within slowdown_threshold (0=no movement, 1=no slowdown)

    # EMA smoothing for o_obj / o_target to reduce noise-induced target jitter (alpha: 0=no filter, 1=no change).
    obs_smooth_alpha: float = 0.40

    # Lift: world Z offset per step (controller will clamp by max_step).
    lift_z_offset: float = 0.04 

    # Retreat: world offset from current EE (one-shot target).
    retreat_offset: np.ndarray = field(default_factory=lambda: np.array([-0.012, 0.0, 0.10], dtype=float))


def _ee_target_from_relative(
    o_ee: np.ndarray,
    o_obj: np.ndarray,
    desired_obj_rel: np.ndarray,
) -> np.ndarray:
    """Desired EE world position so that (obj_world - ee) = desired_obj_rel. obj_world = o_ee + o_obj."""
    obj_world = np.asarray(o_ee, dtype=float) + np.asarray(o_obj, dtype=float)
    return obj_world - np.asarray(desired_obj_rel, dtype=float)


def _ee_target_place_from_relative(
    o_ee: np.ndarray,
    o_target: np.ndarray,
    desired_target_rel: np.ndarray,
) -> np.ndarray:
    """Desired EE so that (place_target_world - ee) = desired_target_rel. target_world = o_ee + o_target."""
    target_world = np.asarray(o_ee, dtype=float) + np.asarray(o_target, dtype=float)
    return target_world - np.asarray(desired_target_rel, dtype=float)


class ScriptedPickPlacePolicy:
    """
    Pose-servo policy: outputs ee_target_pos (world) and per-step orientation flags.

    Action format for controller:
      ee_target_pos (optional): [x,y,z] world; if present, controller servos toward it with max_step_scale.
      move (fallback): [dx,dy,dz] if no ee_target_pos.
      grip: -1|0|1
      enable_yaw_objective: bool (False during approach/descend to avoid bad angles).
      enable_topdown_objective: bool (True to keep gripper top-down).
      max_step_scale: float (0..1, for guarded descend).
    """

    def __init__(
        self,
        task_cfg: TaskConfig = TaskConfig(),
        policy_cfg: PolicyConfig = PolicyConfig(),
    ):
        self.task_cfg = task_cfg
        self.policy_cfg = policy_cfg
        self._prev_o_obj: Optional[np.ndarray] = None
        self._prev_o_target: Optional[np.ndarray] = None
        self._prev_phase: Optional[Phase] = None
        self._descend_xy_locked: bool = False
        self._descend_xy_lock_target: Optional[np.ndarray] = None

    def act(self, state: TaskState, obs: Dict) -> Dict:
        phase = Phase(state.phase)
        o_ee = np.asarray(obs["o_ee"], dtype=float)
        o_obj_raw = np.asarray(obs["o_obj"], dtype=float)
        o_target_raw = np.asarray(obs["o_target"], dtype=float)

        def _finalize(out: Dict) -> Dict:
            self._prev_phase = phase
            return out

        # EMA smoothing to reduce noise-induced target jitter.
        alpha = self.policy_cfg.obs_smooth_alpha
        if self._prev_o_obj is not None:
            o_obj = alpha * self._prev_o_obj + (1.0 - alpha) * o_obj_raw
            o_target = alpha * self._prev_o_target + (1.0 - alpha) * o_target_raw
        else:
            o_obj = o_obj_raw.copy()
            o_target = o_target_raw.copy()
        self._prev_o_obj = o_obj.copy()
        self._prev_o_target = o_target.copy()

        # Reset descend XY lock when leaving Descend.
        if self._prev_phase == Phase.Descend and phase != Phase.Descend:
            self._descend_xy_locked = False
            self._descend_xy_lock_target = None

        # Orientation: no yaw during approach/descend (reduces bad angles); top-down always.
        topdown = True
        yaw = phase not in (
            Phase.ReachAbove,
            Phase.Descend,
            Phase.Transit,
            Phase.DescendToPlace,
        )

        if phase == Phase.ReachAbove:
            full_target = _ee_target_from_relative(o_ee, o_obj, self.task_cfg.pregrasp_obj_rel)
            # Planar-first: get above object in XY before descending.
            # Avoid "directly going down" with bad gripper posture - move laterally first.
            xy_err = float(np.linalg.norm((o_obj - self.task_cfg.pregrasp_obj_rel)[:2]))
            if xy_err > self.policy_cfg.reach_xy_first_threshold:
                ee_target = np.array([full_target[0], full_target[1], o_ee[2]], dtype=float)
            else:
                ee_target = full_target
            reach_err = float(np.linalg.norm(o_obj - self.task_cfg.pregrasp_obj_rel))
            max_step_scale = (
                self.policy_cfg.reach_slowdown_scale
                if reach_err < self.policy_cfg.reach_slowdown_threshold
                else 1.0
            )
            out = {
                "ee_target_pos": ee_target.tolist(),
                "grip": -1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            }
            if max_step_scale < 1.0:
                out["max_step_scale"] = max_step_scale
            return _finalize(out)

        if phase == Phase.Descend:
            full_target = _ee_target_from_relative(o_ee, o_obj, self.task_cfg.grasp_obj_rel)
            ee_target = full_target.copy()
            # Prioritize XY centering in descend: if XY is off, hold Z and correct XY first.
            xy_err = np.asarray((o_obj - self.task_cfg.grasp_obj_rel)[:2], dtype=float)
            xy_err_norm = float(np.linalg.norm(xy_err))

            # Hysteresis lock: once XY is good, lock it to avoid noise-induced drift.
            if not self._descend_xy_locked and xy_err_norm <= self.policy_cfg.descend_xy_lock_threshold:
                self._descend_xy_locked = True
                self._descend_xy_lock_target = o_ee[:2].copy()
            elif self._descend_xy_locked and xy_err_norm >= self.policy_cfg.descend_xy_unlock_threshold:
                self._descend_xy_locked = False
                self._descend_xy_lock_target = None

            corr = self.policy_cfg.descend_xy_correction_gain * xy_err
            corr_norm = float(np.linalg.norm(corr))
            if corr_norm > self.policy_cfg.descend_xy_correction_max and corr_norm > 0:
                corr = (corr / corr_norm) * self.policy_cfg.descend_xy_correction_max
            if self._descend_xy_locked and self._descend_xy_lock_target is not None:
                target_xy = self._descend_xy_lock_target.copy()
            else:
                target_xy = o_ee[:2] + corr
            target_z = o_ee[2] if xy_err_norm > self.policy_cfg.descend_xy_priority_threshold else full_target[2]
            ee_target = np.array([target_xy[0], target_xy[1], target_z], dtype=float)
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": -1,
                "max_step_scale": self.policy_cfg.descend_max_step_scale,
                "enable_yaw_objective": False,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.Close:
            return _finalize({
                "ee_target_pos": o_ee.tolist(),
                "grip": 1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.LiftTest:
            ee_target = o_ee + np.array([0.0, 0.0, self.policy_cfg.lift_z_offset], dtype=float)
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": 1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.Transit:
            # Collimator-style waypoint: lift to safe height before moving to place.
            ee_target = np.array(
                [o_ee[0], o_ee[1], self.task_cfg.transit_height],
                dtype=float,
            )
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": 1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.MoveToPlaceAbove:
            full_target = _ee_target_place_from_relative(o_ee, o_target, self.task_cfg.preplace_target_rel)
            xy_err = float(np.linalg.norm((o_target - self.task_cfg.preplace_target_rel)[:2]))
            if xy_err > self.policy_cfg.reach_xy_first_threshold:
                ee_target = np.array([full_target[0], full_target[1], o_ee[2]], dtype=float)
            else:
                ee_target = full_target
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": 1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.DescendToPlace:
            ee_target = _ee_target_place_from_relative(o_ee, o_target, self.task_cfg.place_target_rel)
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": 1,
                "max_step_scale": self.policy_cfg.place_descend_max_step_scale,
                "enable_yaw_objective": False,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.Open:
            return _finalize({
                "ee_target_pos": o_ee.tolist(),
                "grip": -1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            })

        if phase == Phase.Retreat:
            ee_target = o_ee + self.policy_cfg.retreat_offset
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": -1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            })

        # Done / Failure: hold, open gripper
        return _finalize({
            "ee_target_pos": o_ee.tolist(),
            "grip": -1,
            "enable_yaw_objective": yaw,
            "enable_topdown_objective": topdown,
        })
