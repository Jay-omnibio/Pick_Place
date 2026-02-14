from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from tasks.pick_place_fsm import Phase, TaskConfig, TaskState


@dataclass(frozen=True)
class PolicyConfig:
    # Guarded motion: scale applied to controller max_step during descend (0.2 = 20% speed).
    descend_max_step_scale: float
    place_descend_max_step_scale: float
    descend_xy_correction_gain: float
    descend_xy_correction_max: float
    descend_xy_priority_threshold: float
    # Lock XY once centered to prevent late-stage sideways drift while descending.
    descend_xy_lock_threshold: float
    descend_xy_unlock_threshold: float

    # Reach: planar-first approach - move XY until above object, then descend.
    reach_xy_first_threshold: float
    reach_slowdown_threshold: float
    reach_slowdown_scale: float
    # Dynamic grasp yaw alignment: yaw_target = object_yaw + offset (deg),
    # while keeping top-down objective active.
    enable_object_yaw_align: bool
    object_yaw_offset_deg: float

    # Place reach-above: mirror ReachAbove behavior with independent tuning knobs.
    preplace_xy_first_threshold: float
    preplace_slowdown_threshold: float
    preplace_slowdown_scale: float

    # Place descend: mirror Descend behavior with independent tuning knobs.
    place_descend_xy_correction_gain: float
    place_descend_xy_correction_max: float
    place_descend_xy_priority_threshold: float
    place_descend_xy_lock_threshold: float
    place_descend_xy_unlock_threshold: float

    # EMA smoothing for o_obj / o_target to reduce noise-induced target jitter (alpha: 0=no filter, 1=no change).
    obs_smooth_alpha: float

    # Lift: world Z offset per step (controller will clamp by max_step).
    lift_z_offset: float

    # Retreat: world offset from current EE (one-shot target).
    retreat_offset: np.ndarray


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


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


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
        task_cfg: TaskConfig,
        policy_cfg: PolicyConfig,
    ):
        self.task_cfg = task_cfg
        self.policy_cfg = policy_cfg
        self._prev_o_obj: Optional[np.ndarray] = None
        self._prev_o_target: Optional[np.ndarray] = None
        self._prev_phase: Optional[Phase] = None
        self._descend_xy_locked: bool = False
        self._descend_xy_lock_target: Optional[np.ndarray] = None
        self._place_descend_xy_locked: bool = False
        self._place_descend_xy_lock_target: Optional[np.ndarray] = None

    @staticmethod
    def _compute_guarded_descend_target(
        o_ee: np.ndarray,
        full_target: np.ndarray,
        rel_error_xy: np.ndarray,
        *,
        xy_locked: bool,
        xy_lock_target: Optional[np.ndarray],
        xy_correction_gain: float,
        xy_correction_max: float,
        xy_priority_threshold: float,
        xy_lock_threshold: float,
        xy_unlock_threshold: float,
    ):
        xy_err = np.asarray(rel_error_xy, dtype=float)
        xy_err_norm = float(np.linalg.norm(xy_err))

        # Hysteresis lock: once XY is good, keep XY fixed to avoid late-stage drift.
        if not xy_locked and xy_err_norm <= xy_lock_threshold:
            xy_locked = True
            xy_lock_target = np.asarray(o_ee[:2], dtype=float).copy()
        elif xy_locked and xy_err_norm >= xy_unlock_threshold:
            xy_locked = False
            xy_lock_target = None

        corr = float(xy_correction_gain) * xy_err
        corr_norm = float(np.linalg.norm(corr))
        if corr_norm > float(xy_correction_max) and corr_norm > 0.0:
            corr = (corr / corr_norm) * float(xy_correction_max)

        if xy_locked and xy_lock_target is not None:
            target_xy = np.asarray(xy_lock_target, dtype=float).copy()
        else:
            target_xy = np.asarray(o_ee[:2], dtype=float) + corr

        target_z = float(o_ee[2]) if xy_err_norm > float(xy_priority_threshold) else float(full_target[2])
        ee_target = np.array([float(target_xy[0]), float(target_xy[1]), target_z], dtype=float)
        return ee_target, xy_locked, xy_lock_target, xy_err_norm

    def act(self, state: TaskState, obs: Dict) -> Dict:
        phase = Phase(state.phase)
        o_ee = np.asarray(obs["o_ee"], dtype=float)
        o_obj_raw = np.asarray(obs["o_obj"], dtype=float)
        o_target_raw = np.asarray(obs["o_target"], dtype=float)
        o_obj_yaw_raw = obs.get("o_obj_yaw", None)

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

        dynamic_yaw_target = None
        if self.policy_cfg.enable_object_yaw_align and o_obj_yaw_raw is not None:
            try:
                obj_yaw = float(np.asarray(o_obj_yaw_raw, dtype=float).reshape(-1)[0])
                if np.isfinite(obj_yaw):
                    dynamic_yaw_target = _wrap_to_pi(
                        obj_yaw + np.deg2rad(float(self.policy_cfg.object_yaw_offset_deg))
                    )
            except (TypeError, ValueError, IndexError):
                dynamic_yaw_target = None

        # Reset descend XY lock when leaving Descend.
        if self._prev_phase == Phase.Descend and phase != Phase.Descend:
            self._descend_xy_locked = False
            self._descend_xy_lock_target = None
        if self._prev_phase == Phase.DescendToPlace and phase != Phase.DescendToPlace:
            self._place_descend_xy_locked = False
            self._place_descend_xy_lock_target = None

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
                "enable_yaw_objective": dynamic_yaw_target is not None,
                "enable_topdown_objective": topdown,
            }
            if dynamic_yaw_target is not None:
                out["yaw_target"] = float(dynamic_yaw_target)
                # Parallel gripper is 180-deg symmetric; choose nearest equivalent
                # yaw solution to avoid long 270-deg rotations.
                out["yaw_pi_symmetric"] = True
            if max_step_scale < 1.0:
                out["max_step_scale"] = max_step_scale
            return _finalize(out)

        if phase == Phase.Descend:
            full_target = _ee_target_from_relative(o_ee, o_obj, self.task_cfg.grasp_obj_rel)
            ee_target, self._descend_xy_locked, self._descend_xy_lock_target, _ = self._compute_guarded_descend_target(
                o_ee=o_ee,
                full_target=full_target,
                rel_error_xy=(o_obj - self.task_cfg.grasp_obj_rel)[:2],
                xy_locked=self._descend_xy_locked,
                xy_lock_target=self._descend_xy_lock_target,
                xy_correction_gain=self.policy_cfg.descend_xy_correction_gain,
                xy_correction_max=self.policy_cfg.descend_xy_correction_max,
                xy_priority_threshold=self.policy_cfg.descend_xy_priority_threshold,
                xy_lock_threshold=self.policy_cfg.descend_xy_lock_threshold,
                xy_unlock_threshold=self.policy_cfg.descend_xy_unlock_threshold,
            )
            return _finalize({
                "ee_target_pos": ee_target.tolist(),
                "grip": -1,
                "max_step_scale": self.policy_cfg.descend_max_step_scale,
                "enable_yaw_objective": dynamic_yaw_target is not None,
                "enable_topdown_objective": topdown,
                **({"yaw_target": float(dynamic_yaw_target)} if dynamic_yaw_target is not None else {}),
                **({"yaw_pi_symmetric": True} if dynamic_yaw_target is not None else {}),
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
            if xy_err > self.policy_cfg.preplace_xy_first_threshold:
                ee_target = np.array([full_target[0], full_target[1], o_ee[2]], dtype=float)
            else:
                ee_target = full_target
            preplace_err = float(np.linalg.norm(o_target - self.task_cfg.preplace_target_rel))
            max_step_scale = (
                self.policy_cfg.preplace_slowdown_scale
                if preplace_err < self.policy_cfg.preplace_slowdown_threshold
                else 1.0
            )
            out = {
                "ee_target_pos": ee_target.tolist(),
                "grip": 1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
            }
            if max_step_scale < 1.0:
                out["max_step_scale"] = max_step_scale
            return _finalize(out)

        if phase == Phase.DescendToPlace:
            full_target = _ee_target_place_from_relative(o_ee, o_target, self.task_cfg.place_target_rel)
            ee_target, self._place_descend_xy_locked, self._place_descend_xy_lock_target, _ = self._compute_guarded_descend_target(
                o_ee=o_ee,
                full_target=full_target,
                rel_error_xy=(o_target - self.task_cfg.place_target_rel)[:2],
                xy_locked=self._place_descend_xy_locked,
                xy_lock_target=self._place_descend_xy_lock_target,
                xy_correction_gain=self.policy_cfg.place_descend_xy_correction_gain,
                xy_correction_max=self.policy_cfg.place_descend_xy_correction_max,
                xy_priority_threshold=self.policy_cfg.place_descend_xy_priority_threshold,
                xy_lock_threshold=self.policy_cfg.place_descend_xy_lock_threshold,
                xy_unlock_threshold=self.policy_cfg.place_descend_xy_unlock_threshold,
            )
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
