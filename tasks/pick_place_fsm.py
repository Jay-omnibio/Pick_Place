from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Optional

import numpy as np


class Phase(str, Enum):
    ReachAbove = "ReachAbove"
    Descend = "Descend"
    Close = "Close"
    LiftTest = "LiftTest"
    Transit = "Transit"  # Collimator-style waypoint: lift to safe height before moving to place
    MoveToPlaceAbove = "MoveToPlaceAbove"
    DescendToPlace = "DescendToPlace"
    Open = "Open" # Hold open for a while to ensure object release before retreating; also gives more time for contact to register if place was slightly off and object is still in gripper.
    Retreat = "Retreat"
    Done = "Done"
    Failure = "Failure"


@dataclass(frozen=True)
class TaskConfig:
    # Geometry targets in EE-relative coordinates.
    # Note: o_obj = obj_world - ee_world, so a less-negative Z means EE is closer to the object.
    # For stable top-down descend, keep X/Y identical between pregrasp and grasp;
    # only Z should change across ReachAbove -> Descend.
    pregrasp_obj_rel: np.ndarray
    grasp_obj_rel: np.ndarray
    preplace_target_rel: np.ndarray
    place_target_rel: np.ndarray

    # Thresholds (wide enough to transition when "close enough" before drift).
    # Tight XY centering before starting Descend.
    reach_xy_threshold: float
    reach_z_threshold: float
    descend_threshold: float
    descend_xy_threshold: float
    # Tight XY centering required to finish Descend -> Close.
    descend_x_threshold: float
    descend_y_threshold: float
    descend_z_threshold: float
    descend_contact_z_threshold: float
    descend_timeout_xy_threshold: float
    descend_timeout_x_threshold: float
    descend_timeout_y_threshold: float
    descend_timeout_z_threshold: float
    descend_max_steps: int
    preplace_threshold: float
    place_threshold: float
    # Axis-wise place gates (kept alongside legacy norm thresholds for robustness).
    preplace_xy_threshold: float
    preplace_z_threshold: float
    place_xy_threshold: float
    place_z_threshold: float

    # Timing / hysteresis.
    stable_contact_steps: int
    close_hold_steps: int
    lift_test_steps: int
    open_hold_steps: int
    retreat_steps: int

    # Lift-test pass condition (object should remain in contact / not “drift away” too much).
    lift_test_obj_rel_drift_max: float

    # Transit (Collimator waypoint): safe height before moving to place.
    transit_height: float
    transit_z_threshold: float

    # Retry policy.
    max_retries: int
    retry_reach_cooldown_steps: int

    # Guarded descend: stop on contact after this many consecutive contact steps (hysteresis).
    descend_stop_contact_steps: int
    descend_ready_steps: int


@dataclass(frozen=True)
class TaskState:
    phase: Phase = Phase.ReachAbove
    step_in_phase: int = 0
    retry_count: int = 0
    reach_cooldown: int = 0

    # Counters/flags derived from observations.
    stable_contact_counter: int = 0
    descend_contact_hold: int = 0  # consecutive steps with contact during Descend (stop-on-contact)
    descend_ready_counter: int = 0  # consecutive steps where descend XY/Z are both within threshold

    # Lift-test tracking.
    lift_test_timer: int = 0
    lift_test_ref_obj_rel: Optional[np.ndarray] = None


def _norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _xy_norm(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.linalg.norm(v[:2]))


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _update_contact_counter(prev: int, contact: int) -> int:
    if contact == 1:
        return _clamp_int(prev + 1, 0, 1000)
    return 0


def step_fsm(state: Optional[TaskState], obs: Dict, cfg: TaskConfig) -> TaskState:
    """
    Pure task-level state machine.

    obs must include:
      - o_obj: object position relative to EE (3,)
      - o_target: target position relative to EE (3,)
      - o_ee: EE world position (3,) [needed for Transit]
      - o_contact: {0,1}
      - o_grip: float (width)
    """
    if state is None:
        state = TaskState()

    phase = Phase(state.phase)
    step_in_phase = int(state.step_in_phase) + 1
    retry_count = int(state.retry_count)
    reach_cooldown = max(0, int(state.reach_cooldown) - 1)

    o_obj = np.asarray(obs["o_obj"], dtype=float)
    o_target = np.asarray(obs["o_target"], dtype=float)
    contact = int(obs.get("o_contact", 0))

    stable_contact_counter = _update_contact_counter(state.stable_contact_counter, contact)

    # Default: carry lift-test references.
    lift_test_timer = int(state.lift_test_timer)
    lift_test_ref_obj_rel = state.lift_test_ref_obj_rel

    # ------------------------------------------------------------
    # Phase transition guards
    # ------------------------------------------------------------
    if phase == Phase.ReachAbove:
        # Drive to pregrasp pose above object.
        err = o_obj - cfg.pregrasp_obj_rel
        xy_ok = _xy_norm(err) <= cfg.reach_xy_threshold
        z_ok = abs(float(err[2])) <= cfg.reach_z_threshold
        if reach_cooldown == 0 and xy_ok and z_ok:
            return replace(
                state,
                phase=Phase.Descend,
                step_in_phase=0,
                stable_contact_counter=0,
                descend_contact_hold=0,
                descend_ready_counter=0,
                lift_test_timer=0,
                lift_test_ref_obj_rel=None,
                reach_cooldown=reach_cooldown,
            )

    elif phase == Phase.Descend:
        # Guarded descend: stop on contact (with hysteresis) or when position reached.
        descend_contact_hold = (state.descend_contact_hold + 1) if contact == 1 else 0
        err = o_obj - cfg.grasp_obj_rel
        x_err = abs(float(err[0]))
        y_err = abs(float(err[1]))
        xy_err = _xy_norm(err)
        z_err = abs(float(err[2]))
        xy_ok = (x_err <= cfg.descend_x_threshold and y_err <= cfg.descend_y_threshold)
        z_ok = z_err <= cfg.descend_z_threshold
        descend_ready_counter = (state.descend_ready_counter + 1) if (xy_ok and z_ok) else 0
        ready_ok = descend_ready_counter >= cfg.descend_ready_steps
        # Close only when truly near grasp pose (or very close by Euclidean check).
        position_ok = ((_norm(err) <= cfg.descend_threshold) or (xy_ok and z_ok)) and ready_ok
        # Contact can end descend only when Z is also reasonably near grasp level.
        contact_stop = (
            descend_contact_hold >= cfg.descend_stop_contact_steps
            and z_err <= cfg.descend_contact_z_threshold
            and ready_ok
        )
        # Timeout is a fallback only when near enough, to avoid closing while still high above object.
        timeout_near = (
            x_err <= cfg.descend_timeout_x_threshold
            and y_err <= cfg.descend_timeout_y_threshold
            and z_err <= cfg.descend_timeout_z_threshold
        )
        timeout_stop = step_in_phase >= cfg.descend_max_steps and timeout_near and ready_ok
        if position_ok or contact_stop or timeout_stop:
            return replace(
                state,
                phase=Phase.Close,
                step_in_phase=0,
                stable_contact_counter=stable_contact_counter,
                descend_contact_hold=0,
                descend_ready_counter=0,
                lift_test_timer=0,
                lift_test_ref_obj_rel=None,
                reach_cooldown=reach_cooldown,
            )
        # Stay in Descend; carry contact hold counter.
        return replace(
            state,
            phase=Phase.Descend,
            step_in_phase=step_in_phase,
            reach_cooldown=reach_cooldown,
            descend_contact_hold=descend_contact_hold,
            descend_ready_counter=descend_ready_counter,
        )

    elif phase == Phase.Close:
        # Hold close for N steps and require stable contact.
        if step_in_phase >= cfg.close_hold_steps and stable_contact_counter >= cfg.stable_contact_steps:
            return replace(
                state,
                phase=Phase.LiftTest,
                step_in_phase=0,
                stable_contact_counter=stable_contact_counter,
                lift_test_timer=0,
                lift_test_ref_obj_rel=o_obj.copy(),
                reach_cooldown=reach_cooldown,
            )

        # Timeout: retry from above.
        if step_in_phase >= cfg.close_hold_steps * 3:
            retry_count += 1
            if retry_count > cfg.max_retries:
                return replace(state, phase=Phase.Failure, step_in_phase=0, retry_count=retry_count)
            return replace(
                state,
                phase=Phase.ReachAbove,
                step_in_phase=0,
                retry_count=retry_count,
                stable_contact_counter=0,
                reach_cooldown=cfg.retry_reach_cooldown_steps,
                lift_test_timer=0,
                lift_test_ref_obj_rel=None,
            )

    elif phase == Phase.LiftTest:
        lift_test_timer += 1
        if lift_test_ref_obj_rel is None:
            lift_test_ref_obj_rel = o_obj.copy()
        drift = _norm(o_obj - lift_test_ref_obj_rel)

        # If contact is lost, fail grasp immediately and retry.
        if contact == 0:
            retry_count += 1
            if retry_count > cfg.max_retries:
                return replace(state, phase=Phase.Failure, step_in_phase=0, retry_count=retry_count)
            return replace(
                state,
                phase=Phase.ReachAbove,
                step_in_phase=0,
                retry_count=retry_count,
                stable_contact_counter=0,
                reach_cooldown=cfg.retry_reach_cooldown_steps,
                lift_test_timer=0,
                lift_test_ref_obj_rel=None,
            )

        # If object-relative belief drifts too much during lift-test, treat as failure.
        if drift > cfg.lift_test_obj_rel_drift_max:
            retry_count += 1
            if retry_count > cfg.max_retries:
                return replace(state, phase=Phase.Failure, step_in_phase=0, retry_count=retry_count)
            return replace(
                state,
                phase=Phase.ReachAbove,
                step_in_phase=0,
                retry_count=retry_count,
                stable_contact_counter=0,
                reach_cooldown=cfg.retry_reach_cooldown_steps,
                lift_test_timer=0,
                lift_test_ref_obj_rel=None,
            )

        if lift_test_timer >= cfg.lift_test_steps:
            return replace(
                state,
                phase=Phase.Transit,
                step_in_phase=0,
                stable_contact_counter=stable_contact_counter,
                lift_test_timer=lift_test_timer,
                lift_test_ref_obj_rel=lift_test_ref_obj_rel,
                reach_cooldown=reach_cooldown,
            )

    elif phase == Phase.Transit:
        # Collimator-style waypoint: lift to safe height before moving to place.
        o_ee = np.asarray(obs.get("o_ee", [0.0, 0.0, 0.0]), dtype=float)
        ee_z = float(o_ee[2])
        if ee_z >= cfg.transit_height - cfg.transit_z_threshold:
            return replace(state, phase=Phase.MoveToPlaceAbove, step_in_phase=0, reach_cooldown=reach_cooldown)
        if contact == 0:
            return replace(state, phase=Phase.ReachAbove, step_in_phase=0, stable_contact_counter=0)

    elif phase == Phase.MoveToPlaceAbove:
        # Navigate to preplace above target.
        err = o_target - cfg.preplace_target_rel
        preplace_xy_ok = _xy_norm(err) <= cfg.preplace_xy_threshold
        preplace_z_ok = abs(float(err[2])) <= cfg.preplace_z_threshold
        if (_norm(err) <= cfg.preplace_threshold) or (preplace_xy_ok and preplace_z_ok):
            return replace(state, phase=Phase.DescendToPlace, step_in_phase=0, reach_cooldown=reach_cooldown)
        # Drop detection -> restart.
        if contact == 0:
            return replace(state, phase=Phase.ReachAbove, step_in_phase=0, stable_contact_counter=0)

    elif phase == Phase.DescendToPlace:
        err = o_target - cfg.place_target_rel
        place_xy_ok = _xy_norm(err) <= cfg.place_xy_threshold
        place_z_ok = abs(float(err[2])) <= cfg.place_z_threshold
        if (_norm(err) <= cfg.place_threshold) or (place_xy_ok and place_z_ok):
            return replace(state, phase=Phase.Open, step_in_phase=0, reach_cooldown=reach_cooldown)
        if contact == 0:
            return replace(state, phase=Phase.ReachAbove, step_in_phase=0, stable_contact_counter=0)

    elif phase == Phase.Open:
        if step_in_phase >= cfg.open_hold_steps:
            return replace(state, phase=Phase.Retreat, step_in_phase=0, reach_cooldown=reach_cooldown)

    elif phase == Phase.Retreat:
        if step_in_phase >= cfg.retreat_steps:
            return replace(state, phase=Phase.Done, step_in_phase=0, reach_cooldown=reach_cooldown)

    elif phase in (Phase.Done, Phase.Failure):
        # Terminal.
        return replace(state, step_in_phase=step_in_phase, reach_cooldown=reach_cooldown)

    # Default: remain in same phase.
    return replace(
        state,
        phase=phase,
        step_in_phase=step_in_phase,
        retry_count=retry_count,
        reach_cooldown=reach_cooldown,
        stable_contact_counter=stable_contact_counter,
        lift_test_timer=lift_test_timer,
        lift_test_ref_obj_rel=lift_test_ref_obj_rel,
    )
