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
    # Descend control mode:
    # - axis-priority: prioritize larger normalized axis error (x/y) instead of XY norm.
    # - xy-then-z: freeze Z motion until x and y satisfy thresholds (strict planar-first).
    descend_axis_priority_enabled: bool
    descend_xy_then_z: bool
    # Lock XY once centered to prevent late-stage sideways drift while descending.
    descend_xy_lock_threshold: float
    descend_xy_unlock_threshold: float

    # Reach: planar-first approach - move XY until above object, then descend.
    reach_xy_first_threshold: float
    reach_slowdown_threshold: float
    reach_slowdown_scale: float
    # Optional curved XY approach when target is largely around the robot.
    reach_arc_enabled: bool
    reach_arc_angle_trigger_deg: float
    reach_arc_max_theta_step_deg: float
    reach_arc_radial_step_max: float
    reach_arc_min_radius: float
    reach_arc_max_radius: float
    # Dynamic grasp yaw alignment: yaw_target = object_yaw + offset (deg),
    # while keeping top-down objective active.
    enable_object_yaw_align: bool
    object_yaw_offset_deg: float
    # Dynamic grip close target (FSM path): when contact is detected in Close,
    # set close width to (current_width - margin) instead of always forcing hard close.
    dynamic_grip_enabled: bool
    dynamic_grip_contact_margin: float
    dynamic_grip_min_close_width: float
    # Dynamic open target (FSM path): start from estimated object width + margin,
    # then boost wider while contact persists in Open.
    dynamic_open_enabled: bool
    dynamic_open_object_width: float
    dynamic_open_margin: float
    dynamic_open_contact_boost: float

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


def _object_rel_local_to_world(desired_obj_rel_local: np.ndarray, obj_yaw: float) -> np.ndarray:
    """
    Convert desired object-relative offset from object-local XY axes to world XY axes.
    Z is unchanged.
    """
    rel = np.asarray(desired_obj_rel_local, dtype=float).copy()
    c = float(np.cos(float(obj_yaw)))
    s = float(np.sin(float(obj_yaw)))
    x_l, y_l = float(rel[0]), float(rel[1])
    # world = Rz(yaw) * local
    rel[0] = c * x_l - s * y_l
    rel[1] = s * x_l + c * y_l
    return rel


def _desired_rel_for_frame(desired_rel_cfg: np.ndarray, obj_yaw: float, use_object_local_xy_errors: bool) -> np.ndarray:
    """
    Convert configured Reach/Descend desired relation into world frame that policy tracks.
    - object-local mode: rotate local XY by object yaw
    - legacy mode: interpret config directly in world XY
    """
    if bool(use_object_local_xy_errors):
        return _object_rel_local_to_world(desired_rel_cfg, obj_yaw)
    return np.asarray(desired_rel_cfg, dtype=float).copy()


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
        # Reach arc mode: lock CW/CCW turn direction during one ReachAbove approach.
        self._reach_arc_turn_sign: int = 0  # +1 CCW, -1 CW, 0 unlocked
        self._descend_xy_locked: bool = False
        self._descend_xy_lock_target: Optional[np.ndarray] = None
        self._place_descend_xy_locked: bool = False
        self._place_descend_xy_lock_target: Optional[np.ndarray] = None
        self._dynamic_close_target_width: Optional[float] = None
        self._dynamic_open_target_width: Optional[float] = None

    @staticmethod
    def _signed_angle_to_goal(theta_now: float, theta_goal: float, turn_sign: int) -> float:
        """
        Signed remaining angular distance to goal if we force a specific turn direction.
        +turn_sign means CCW-only, -turn_sign means CW-only.
        """
        if int(turn_sign) >= 0:
            return float((theta_goal - theta_now) % (2.0 * np.pi))
        return -float((theta_now - theta_goal) % (2.0 * np.pi))

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

    @staticmethod
    def _compute_guarded_descend_target_axis_priority(
        o_ee: np.ndarray,
        full_target: np.ndarray,
        rel_error_xy: np.ndarray,
        *,
        x_threshold: float,
        y_threshold: float,
        xy_locked: bool,
        xy_lock_target: Optional[np.ndarray],
        xy_correction_gain: float,
        xy_correction_max: float,
        xy_priority_threshold: float,
        xy_then_z: bool,
        xy_lock_threshold: float,
        xy_unlock_threshold: float,
    ):
        err_xy = np.asarray(rel_error_xy, dtype=float)
        err_x = float(err_xy[0])
        err_y = float(err_xy[1])
        abs_x = abs(err_x)
        abs_y = abs(err_y)

        x_thr = max(1e-6, float(x_threshold))
        y_thr = max(1e-6, float(y_threshold))
        x_ratio = abs_x / x_thr
        y_ratio = abs_y / y_thr

        # Hysteresis lock using per-axis gates (no XY norm use).
        lock_x = min(x_thr, float(xy_lock_threshold))
        lock_y = min(y_thr, float(xy_lock_threshold))
        if not xy_locked and abs_x <= lock_x and abs_y <= lock_y:
            xy_locked = True
            xy_lock_target = np.asarray(o_ee[:2], dtype=float).copy()
        elif xy_locked:
            unlock_x = max(float(xy_unlock_threshold), 2.0 * x_thr)
            unlock_y = max(float(xy_unlock_threshold), 2.0 * y_thr)
            if abs_x >= unlock_x or abs_y >= unlock_y:
                xy_locked = False
                xy_lock_target = None

        # Axis-priority XY correction:
        # prioritize whichever axis is farther from its own threshold.
        if x_ratio >= y_ratio:
            scale_x = 1.0
            scale_y = max(0.35, y_ratio / (x_ratio + 1e-9))
        else:
            scale_y = 1.0
            scale_x = max(0.35, x_ratio / (y_ratio + 1e-9))
        corr = np.array(
            [
                float(xy_correction_gain) * err_x * scale_x,
                float(xy_correction_gain) * err_y * scale_y,
            ],
            dtype=float,
        )
        corr_norm = float(np.linalg.norm(corr))
        if corr_norm > float(xy_correction_max) and corr_norm > 0.0:
            corr = (corr / corr_norm) * float(xy_correction_max)

        if xy_locked and xy_lock_target is not None:
            target_xy = np.asarray(xy_lock_target, dtype=float).copy()
        else:
            target_xy = np.asarray(o_ee[:2], dtype=float) + corr

        # Z gating by per-axis threshold proximity (no XY norm).
        if bool(xy_then_z):
            xy_ready = (abs_x <= x_thr) and (abs_y <= y_thr)
        else:
            # Loose gate: allow Z once both axes are near threshold + buffer.
            buf = max(0.0, float(xy_priority_threshold))
            xy_ready = (abs_x <= (x_thr + buf)) and (abs_y <= (y_thr + buf))
        target_z = float(full_target[2]) if xy_ready else float(o_ee[2])

        ee_target = np.array([float(target_xy[0]), float(target_xy[1]), target_z], dtype=float)
        return ee_target, xy_locked, xy_lock_target, float(max(x_ratio, y_ratio))

    def act(self, state: TaskState, obs: Dict) -> Dict:
        phase = Phase(state.phase)
        o_ee = np.asarray(obs["o_ee"], dtype=float)
        o_obj_raw = np.asarray(obs["o_obj"], dtype=float)
        o_target_raw = np.asarray(obs["o_target"], dtype=float)
        o_obj_yaw_raw = obs.get("o_obj_yaw", None)
        obj_yaw = 0.0
        if o_obj_yaw_raw is not None:
            try:
                obj_yaw = float(np.asarray(o_obj_yaw_raw, dtype=float).reshape(-1)[0])
                if not np.isfinite(obj_yaw):
                    obj_yaw = 0.0
            except (TypeError, ValueError, IndexError):
                obj_yaw = 0.0

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
        if self._prev_phase == Phase.ReachAbove and phase != Phase.ReachAbove:
            self._reach_arc_turn_sign = 0
        if self._prev_phase == Phase.DescendToPlace and phase != Phase.DescendToPlace:
            self._place_descend_xy_locked = False
            self._place_descend_xy_lock_target = None
        if self._prev_phase == Phase.Close and phase != Phase.Close:
            self._dynamic_close_target_width = None
        if self._prev_phase == Phase.Open and phase != Phase.Open:
            self._dynamic_open_target_width = None

        # Orientation: no yaw during approach/descend (reduces bad angles); top-down always.
        topdown = True
        yaw = phase not in (
            Phase.ReachAbove,
            Phase.Descend,
            Phase.Transit,
            Phase.DescendToPlace,
        )

        if phase == Phase.ReachAbove:
            desired_rel_world = _desired_rel_for_frame(
                self.task_cfg.pregrasp_obj_rel,
                obj_yaw,
                self.task_cfg.use_object_local_xy_errors,
            )
            full_target = _ee_target_from_relative(o_ee, o_obj, desired_rel_world)
            xy_err = float(np.linalg.norm((o_obj - desired_rel_world)[:2]))
            if xy_err > self.policy_cfg.reach_xy_first_threshold:
                # Default: linear XY approach at current Z.
                ee_target = np.array([full_target[0], full_target[1], o_ee[2]], dtype=float)
                # Optional arc waypoint for behind/side approaches to avoid
                # pure straight-line XY pulls across the workspace.
                if bool(self.policy_cfg.reach_arc_enabled):
                    ee_xy = np.asarray(o_ee[:2], dtype=float)
                    goal_xy = np.asarray(full_target[:2], dtype=float)
                    r_ee = float(np.linalg.norm(ee_xy))
                    r_goal = float(np.linalg.norm(goal_xy))
                    if r_ee > 1e-5 and r_goal > 1e-5:
                        theta_ee = float(np.arctan2(ee_xy[1], ee_xy[0]))
                        theta_goal = float(np.arctan2(goal_xy[1], goal_xy[0]))
                        dtheta_short = _wrap_to_pi(theta_goal - theta_ee)
                        arc_angle_trigger = float(np.deg2rad(float(self.policy_cfg.reach_arc_angle_trigger_deg)))
                        max_theta_step = max(0.0, float(np.deg2rad(float(self.policy_cfg.reach_arc_max_theta_step_deg))))
                        radial_step_max = max(0.0, float(self.policy_cfg.reach_arc_radial_step_max))
                        r_min = max(1e-3, float(self.policy_cfg.reach_arc_min_radius))
                        r_max = max(r_min, float(self.policy_cfg.reach_arc_max_radius))

                        # Rule 1: choose CW/CCW once and lock for this approach.
                        if self._reach_arc_turn_sign == 0 and abs(dtheta_short) >= arc_angle_trigger:
                            self._reach_arc_turn_sign = 1 if dtheta_short >= 0.0 else -1

                        # Release lock near alignment, then default to linear target.
                        if self._reach_arc_turn_sign != 0 and abs(dtheta_short) < (0.5 * arc_angle_trigger):
                            self._reach_arc_turn_sign = 0

                        if max_theta_step > 0.0 and radial_step_max > 0.0 and self._reach_arc_turn_sign != 0:
                            # Rule 2: angle and radius updated together every step.
                            dtheta_signed = self._signed_angle_to_goal(
                                theta_now=theta_ee,
                                theta_goal=theta_goal,
                                turn_sign=self._reach_arc_turn_sign,
                            )
                            theta_step = float(np.sign(self._reach_arc_turn_sign)) * min(
                                max_theta_step,
                                abs(float(dtheta_signed)),
                            )
                            theta_next = theta_ee + theta_step
                            r_next = r_ee + float(np.clip(r_goal - r_ee, -radial_step_max, radial_step_max))
                            # Rule 3: keep radius in safe band to avoid center collapse / huge swing.
                            r_next = float(np.clip(r_next, r_min, r_max))
                            arc_xy = np.array(
                                [r_next * np.cos(theta_next), r_next * np.sin(theta_next)],
                                dtype=float,
                            )
                            ee_target = np.array([arc_xy[0], arc_xy[1], o_ee[2]], dtype=float)
            else:
                ee_target = full_target
            reach_err = float(np.linalg.norm(o_obj - desired_rel_world))
            max_step_scale = (
                self.policy_cfg.reach_slowdown_scale
                if reach_err < self.policy_cfg.reach_slowdown_threshold
                else 1.0
            )
            out = {
                "ee_target_pos": ee_target.tolist(),
                "grip": -1,
                # Keep ReachAbove primarily translational (XY/Z). Enabling yaw
                # here can fight IK tracking and cause lateral drift loops.
                # Yaw alignment remains active in Descend.
                "enable_yaw_objective": False,
                "enable_topdown_objective": topdown,
            }
            if max_step_scale < 1.0:
                out["max_step_scale"] = max_step_scale
            return _finalize(out)

        if phase == Phase.Descend:
            desired_rel_world = _desired_rel_for_frame(
                self.task_cfg.grasp_obj_rel,
                obj_yaw,
                self.task_cfg.use_object_local_xy_errors,
            )
            full_target = _ee_target_from_relative(o_ee, o_obj, desired_rel_world)
            if bool(self.policy_cfg.descend_axis_priority_enabled):
                ee_target, self._descend_xy_locked, self._descend_xy_lock_target, _ = self._compute_guarded_descend_target_axis_priority(
                    o_ee=o_ee,
                    full_target=full_target,
                    rel_error_xy=(o_obj - desired_rel_world)[:2],
                    x_threshold=self.task_cfg.descend_x_threshold,
                    y_threshold=self.task_cfg.descend_y_threshold,
                    xy_locked=self._descend_xy_locked,
                    xy_lock_target=self._descend_xy_lock_target,
                    xy_correction_gain=self.policy_cfg.descend_xy_correction_gain,
                    xy_correction_max=self.policy_cfg.descend_xy_correction_max,
                    xy_priority_threshold=self.policy_cfg.descend_xy_priority_threshold,
                    xy_then_z=self.policy_cfg.descend_xy_then_z,
                    xy_lock_threshold=self.policy_cfg.descend_xy_lock_threshold,
                    xy_unlock_threshold=self.policy_cfg.descend_xy_unlock_threshold,
                )
            else:
                ee_target, self._descend_xy_locked, self._descend_xy_lock_target, _ = self._compute_guarded_descend_target(
                    o_ee=o_ee,
                    full_target=full_target,
                    rel_error_xy=(o_obj - desired_rel_world)[:2],
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
            if bool(self.policy_cfg.dynamic_grip_enabled):
                contact = int(obs.get("o_contact", 0))
                o_grip_raw = obs.get("o_grip", None)
                if o_grip_raw is not None:
                    try:
                        grip_width = float(np.asarray(o_grip_raw, dtype=float).reshape(-1)[0])
                    except (TypeError, ValueError, IndexError):
                        grip_width = float("nan")
                    if np.isfinite(grip_width) and contact == 1:
                        candidate = max(
                            float(self.policy_cfg.dynamic_grip_min_close_width),
                            grip_width - float(self.policy_cfg.dynamic_grip_contact_margin),
                        )
                        if self._dynamic_close_target_width is None:
                            self._dynamic_close_target_width = float(candidate)
                        else:
                            # Never reopen during close hold; only tighten or keep.
                            self._dynamic_close_target_width = min(
                                float(self._dynamic_close_target_width),
                                float(candidate),
                            )
            return _finalize({
                "ee_target_pos": o_ee.tolist(),
                "grip": 1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
                **(
                    {"grip_close_target_width": float(self._dynamic_close_target_width)}
                    if self._dynamic_close_target_width is not None
                    else {}
                ),
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
            if bool(self.policy_cfg.dynamic_open_enabled):
                contact = int(obs.get("o_contact", 0))
                # Base release width from object size estimate.
                base_release_width = (
                    float(self.policy_cfg.dynamic_open_object_width)
                    + float(self.policy_cfg.dynamic_open_margin)
                )
                if self._dynamic_open_target_width is None:
                    self._dynamic_open_target_width = float(base_release_width)
                # If still touching object while opening, widen target further.
                if contact == 1:
                    self._dynamic_open_target_width = float(self._dynamic_open_target_width) + float(
                        self.policy_cfg.dynamic_open_contact_boost
                    )
            return _finalize({
                "ee_target_pos": o_ee.tolist(),
                "grip": -1,
                "enable_yaw_objective": yaw,
                "enable_topdown_objective": topdown,
                **(
                    {"grip_open_target_width": float(self._dynamic_open_target_width)}
                    if self._dynamic_open_target_width is not None
                    else {}
                ),
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
