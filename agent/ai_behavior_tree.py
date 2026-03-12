from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np

from agent.phase_managers import (
    PickPhaseManager,
    PlacePhaseManager,
    TERMINAL_FAILURE_PHASES as TERMINAL_FAILURE_PHASE_SET,
    TERMINAL_SUCCESS_PHASES as TERMINAL_SUCCESS_PHASE_SET,
)


class BTStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class BTContext:
    phase: str
    belief: Dict


class BTNode:
    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        raise NotImplementedError


class SequenceNode(BTNode):
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        for child in self.children:
            status = child.tick(ctx, bt)
            if status == BTStatus.FAILURE:
                return BTStatus.FAILURE
            if status == BTStatus.RUNNING:
                return BTStatus.RUNNING
        return BTStatus.SUCCESS


class SelectorNode(BTNode):
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        for child in self.children:
            status = child.tick(ctx, bt)
            if status in (BTStatus.SUCCESS, BTStatus.RUNNING):
                return status
        return BTStatus.FAILURE


class AcquireNode(BTNode):
    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        phase = ctx.phase
        if phase in bt.TERMINAL_SUCCESS_PHASES:
            return BTStatus.SUCCESS
        if phase in bt.PLACE_PHASES:
            return BTStatus.SUCCESS
        if phase in bt.PICK_PHASES:
            # Regression after place started should be treated as tree failure.
            if bt.place_started:
                bt.last_reason = "regressed_to_pick_after_place"
                return BTStatus.FAILURE
            return BTStatus.RUNNING
        bt.last_reason = f"unknown_phase:{phase}"
        return BTStatus.FAILURE


class PlaceNode(BTNode):
    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        phase = ctx.phase
        if phase in bt.TERMINAL_SUCCESS_PHASES:
            return BTStatus.SUCCESS
        if phase in bt.PLACE_PHASES:
            return BTStatus.RUNNING
        if phase in bt.PICK_PHASES:
            bt.last_reason = "place_not_started_or_lost"
            return BTStatus.FAILURE
        bt.last_reason = f"unknown_phase:{phase}"
        return BTStatus.FAILURE


class RecoverNode(BTNode):
    def tick(self, _ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        bt.recover_requested = True
        if not bt.last_reason:
            bt.last_reason = "tree_failure"
        return BTStatus.RUNNING


class SetPickPriorsNode(BTNode):
    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        bt.apply_pick_priors(ctx.phase, ctx.belief)
        return BTStatus.SUCCESS


class SetPlacePriorsNode(BTNode):
    def tick(self, ctx: BTContext, bt: "AIPickPlaceBehaviorTree") -> BTStatus:
        bt.apply_place_priors(ctx.phase, ctx.belief)
        return BTStatus.SUCCESS


class AIPickPlaceBehaviorTree:
    PICK_PHASES = PickPhaseManager.INFO.phases
    PLACE_PHASES = PlacePhaseManager.INFO.phases
    TERMINAL_SUCCESS_PHASES = TERMINAL_SUCCESS_PHASE_SET
    TERMINAL_FAILURE_PHASES = TERMINAL_FAILURE_PHASE_SET
    RECOVERY_BRANCHES = ("ReScanTable", "ReApproachOffset", "SafeBackoff")
    TASK_INTENTS = ("PICK", "PLACE", "RECOVER", "DONE", "FAILURE")
    BT_DECISIONS = ("CONTINUE", "RECOVER", "TASK_SWITCH", "SUCCESS", "TERMINAL_FAILURE")

    def __init__(
        self,
        max_retries: int = 3,
        reach_reentry_cooldown_steps: int = 20,
        stall_limit: int = 1200,
        progress_eps: float = 1e-3,
        no_progress_limit: int = 240,
        set_priors_enabled: bool = True,
        retry_reach_z_step: float = 0.005,
        retry_reach_z_max: float = 0.02,
        vfe_recover_enabled: bool = False,
        vfe_recover_threshold: float = 2.0,
        vfe_recover_steps: int = 50,
        branch_retry_cap: int = 2,
        global_recovery_cap: int = 6,
        rescan_hold_steps: int = 40,
        reapproach_offset_xy: float = 0.01,
        safe_backoff_hold_steps: int = 30,
        safe_backoff_z_boost: float = 0.02,
    ):
        self.max_retries = int(max_retries)
        self.reach_reentry_cooldown_steps = int(reach_reentry_cooldown_steps)
        self.stall_limit = int(stall_limit)
        self.progress_eps = float(progress_eps)
        self.no_progress_limit = int(no_progress_limit)
        self.set_priors_enabled = bool(set_priors_enabled)
        self.retry_reach_z_step = float(max(0.0, retry_reach_z_step))
        self.retry_reach_z_max = float(max(0.0, retry_reach_z_max))
        self.vfe_recover_enabled = bool(vfe_recover_enabled)
        self.vfe_recover_threshold = float(vfe_recover_threshold)
        self.vfe_recover_steps = int(vfe_recover_steps)
        self.branch_retry_cap = int(max(1, branch_retry_cap))
        self.global_recovery_cap = int(max(self.branch_retry_cap, global_recovery_cap))
        self.rescan_hold_steps = int(max(0, rescan_hold_steps))
        self.reapproach_offset_xy = float(max(0.0, reapproach_offset_xy))
        self.safe_backoff_hold_steps = int(max(0, safe_backoff_hold_steps))
        self.safe_backoff_z_boost = float(max(0.0, safe_backoff_z_boost))

        self.place_started = False
        self.last_phase = ""
        self.phase_steps = 0
        self.phase_best_error = float("inf")
        self.phase_no_progress_steps = 0
        self.last_reason = ""
        self.status: BTStatus = BTStatus.RUNNING
        self.recover_requested = False
        self.vfe_high_steps = 0
        self.base_priors: Dict[str, np.ndarray] = {}
        self.global_recovery_count = 0
        self.branch_retry_counts = {name: 0 for name in self.RECOVERY_BRANCHES}
        self.active_recovery_branch = ""
        self.active_branch_steps_remaining = 0
        self.active_reapproach_offset_local = np.zeros(2, dtype=float)
        self.active_z_boost = 0.0
        self.reapproach_cycle_index = 0

        # Root: try normal pick->place sequence; if it fails, run recovery.
        self.root: BTNode = SelectorNode(
            [
                SequenceNode([SetPickPriorsNode(), AcquireNode(), SetPlacePriorsNode(), PlaceNode()]),
                RecoverNode(),
            ]
        )

    @staticmethod
    def _safe_vec3(value) -> np.ndarray | None:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
            return None
        return arr.copy()

    def task_intent_for_phase(self, phase: str) -> str:
        phase = str(phase or "Reach")
        if phase in self.PICK_PHASES:
            return "PICK"
        if phase in self.PLACE_PHASES:
            return "PLACE"
        if phase in self.TERMINAL_SUCCESS_PHASES:
            return "DONE"
        if phase in self.TERMINAL_FAILURE_PHASES:
            return "FAILURE"
        return "PICK"

    @staticmethod
    def _make_tick_result(
        *,
        recover: bool,
        terminal_failure: bool,
        task_intent: str,
        bt_decision: str,
    ) -> Dict:
        return {
            "recover": bool(recover),
            "terminal_failure": bool(terminal_failure),
            "task_intent": str(task_intent),
            "bt_decision": str(bt_decision),
        }

    def _ensure_base_priors(self, belief: Dict) -> None:
        if self.base_priors:
            return
        key_aliases = (
            ("reach_obj_rel_local", ("reach_obj_rel_local", "reach_obj_rel")),
            ("align_obj_rel_local", ("align_obj_rel_local", "align_obj_rel")),
            ("descend_obj_rel_local", ("descend_obj_rel_local", "descend_obj_rel")),
            ("preplace_target_rel", ("preplace_target_rel",)),
            ("place_target_rel", ("place_target_rel",)),
            ("retreat_move", ("retreat_move",)),
        )
        for out_key, aliases in key_aliases:
            vec = None
            for key in aliases:
                vec = self._safe_vec3(belief.get(key, [0.0, 0.0, 0.0]))
                if vec is not None:
                    break
            if vec is not None:
                self.base_priors[out_key] = vec

    def _clear_active_recovery_branch(self) -> None:
        self.active_recovery_branch = ""
        self.active_branch_steps_remaining = 0
        self.active_reapproach_offset_local = np.zeros(2, dtype=float)
        self.active_z_boost = 0.0

    def _pick_reapproach_offset_local(self) -> np.ndarray:
        if self.reapproach_offset_xy <= 0.0:
            return np.zeros(2, dtype=float)
        # Cycle deterministic local offsets so retries do not repeat same line.
        dirs = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=float,
        )
        idx = int(self.reapproach_cycle_index % len(dirs))
        self.reapproach_cycle_index += 1
        return self.reapproach_offset_xy * dirs[idx]

    def _activate_recovery_branch(self, branch: str, belief: Dict) -> None:
        self._clear_active_recovery_branch()
        self.active_recovery_branch = branch

        if branch == "ReScanTable":
            self.active_branch_steps_remaining = self.rescan_hold_steps
            self.active_z_boost = max(self.retry_reach_z_step * 2.0, 0.01)
            self.active_reapproach_offset_local = np.zeros(2, dtype=float)
            return

        if branch == "ReApproachOffset":
            self.active_branch_steps_remaining = max(
                self.reach_reentry_cooldown_steps, self.rescan_hold_steps // 2
            )
            self.active_z_boost = self.retry_reach_z_step
            self.active_reapproach_offset_local = self._pick_reapproach_offset_local()
            return

        # SafeBackoff: add upward clearance and shift sideways.
        side_sign = 1.0 if float(belief.get("approach_side_sign", 1.0)) >= 0.0 else -1.0
        self.active_branch_steps_remaining = self.safe_backoff_hold_steps
        self.active_z_boost = self.safe_backoff_z_boost
        self.active_reapproach_offset_local = np.asarray(
            [0.0, -side_sign * self.reapproach_offset_xy], dtype=float
        )

    def _select_recovery_branch(self, reason: str) -> str:
        reason = str(reason or "tree_failure")
        if reason == "stale_observation":
            ordered = ("ReScanTable", "ReApproachOffset", "SafeBackoff")
        elif reason == "reach_stall":
            ordered = ("ReApproachOffset", "ReScanTable", "SafeBackoff")
        elif reason in ("singularity_risk", "unexpected_contact"):
            ordered = ("SafeBackoff", "ReScanTable", "ReApproachOffset")
        elif reason == "grasp_failed":
            # First refresh belief, then retry with offset line if still failing.
            ordered = ("ReScanTable", "ReApproachOffset", "SafeBackoff")
        elif reason in ("place_alignment_failed", "release_failed"):
            ordered = ("ReApproachOffset", "ReScanTable", "SafeBackoff")
        elif reason == "high_vfe":
            ordered = ("ReApproachOffset", "ReScanTable", "SafeBackoff")
        else:
            ordered = ("ReApproachOffset", "ReScanTable", "SafeBackoff")

        for branch in ordered:
            if self.branch_retry_counts.get(branch, 0) < self.branch_retry_cap:
                return branch
        return ""

    def apply_pick_priors(self, phase: str, belief: Dict) -> None:
        if (not self.set_priors_enabled) or phase not in self.PICK_PHASES:
            return
        self._ensure_base_priors(belief)
        if not self.base_priors:
            return
        retry_count = int(belief.get("retry_count", 0))
        retry_z = min(float(retry_count) * self.retry_reach_z_step, self.retry_reach_z_max)

        reach_ref = self.base_priors.get("reach_obj_rel_local")
        align_ref = self.base_priors.get("align_obj_rel_local")
        descend_ref = self.base_priors.get("descend_obj_rel_local")
        xy_offset = self.active_reapproach_offset_local.copy()
        z_boost = float(self.active_z_boost)
        if reach_ref is not None:
            updated = reach_ref.copy()
            updated[2] -= retry_z
            updated[0] += float(xy_offset[0])
            updated[1] += float(xy_offset[1])
            updated[2] -= z_boost
            belief["reach_obj_rel_local"] = updated
        if align_ref is not None:
            updated = align_ref.copy()
            updated[2] -= retry_z
            updated[0] += float(xy_offset[0])
            updated[1] += float(xy_offset[1])
            updated[2] -= z_boost
            belief["align_obj_rel_local"] = updated
        if descend_ref is not None:
            belief["descend_obj_rel_local"] = descend_ref.copy()

    def apply_place_priors(self, phase: str, belief: Dict) -> None:
        if (not self.set_priors_enabled) or phase not in self.PLACE_PHASES + self.TERMINAL_SUCCESS_PHASES:
            return
        self._ensure_base_priors(belief)
        if not self.base_priors:
            return
        preplace_ref = self.base_priors.get("preplace_target_rel")
        place_ref = self.base_priors.get("place_target_rel")
        retreat_ref = self.base_priors.get("retreat_move")
        xy_offset = self.active_reapproach_offset_local.copy()
        z_boost = float(self.active_z_boost)
        if preplace_ref is not None:
            updated = preplace_ref.copy()
            updated[0] += float(xy_offset[0])
            updated[1] += float(xy_offset[1])
            updated[2] -= z_boost
            belief["preplace_target_rel"] = updated
        if place_ref is not None:
            updated = place_ref.copy()
            updated[0] += float(xy_offset[0])
            updated[1] += float(xy_offset[1])
            belief["place_target_rel"] = updated
        if retreat_ref is not None:
            belief["retreat_move"] = retreat_ref.copy()

    def _update_phase_progress(self, phase: str) -> None:
        if phase == self.last_phase:
            self.phase_steps += 1
        else:
            self.phase_steps = 0
            self.last_phase = phase
            self.phase_best_error = float("inf")
            self.phase_no_progress_steps = 0

        if phase in self.PLACE_PHASES:
            self.place_started = True
        elif phase in self.TERMINAL_SUCCESS_PHASES:
            self.place_started = False

    @staticmethod
    def _safe_norm(v: np.ndarray) -> float:
        n = float(np.linalg.norm(v))
        if not np.isfinite(n):
            return float("inf")
        return n

    def _phase_error(self, phase: str, belief: Dict) -> float | None:
        s_obj = np.asarray(belief.get("s_obj_mean", [0.0, 0.0, 0.0]), dtype=float)
        s_target = np.asarray(belief.get("s_target_mean", [0.0, 0.0, 0.0]), dtype=float)
        s_ee = np.asarray(belief.get("s_ee_mean", [0.0, 0.0, 0.0]), dtype=float)
        if phase == "Reach":
            ref = np.asarray(belief.get("reach_obj_rel", [0.0, 0.0, 0.0]), dtype=float)
            return self._safe_norm(s_obj - ref)
        if phase == "Align":
            ref = np.asarray(belief.get("align_obj_rel", [0.0, 0.0, 0.0]), dtype=float)
            return self._safe_norm(s_obj - ref)
        if phase in ("Descend", "CloseHold", "LiftTest"):
            ref = np.asarray(belief.get("descend_obj_rel", [0.0, 0.0, 0.0]), dtype=float)
            return self._safe_norm(s_obj - ref)
        if phase == "MoveToPlaceAbove":
            ref = np.asarray(belief.get("preplace_target_rel", [0.0, 0.0, 0.0]), dtype=float)
            return self._safe_norm(s_target - ref)
        if phase == "DescendToPlace":
            ref = np.asarray(belief.get("place_target_rel", [0.0, 0.0, 0.0]), dtype=float)
            return self._safe_norm(s_target - ref)
        if phase == "Transit":
            # No explicit transit target in belief; use Z-height ascent progress proxy.
            return float(-s_ee[2])
        return None

    def _update_progress_watchdog(self, phase: str, belief: Dict) -> bool:
        """
        Returns True if current phase appears stalled.
        """
        err = self._phase_error(phase, belief)
        # For timer phases (Open/Retreat), avoid false stall triggers.
        if err is None:
            return self.phase_steps >= self.stall_limit

        if not np.isfinite(err):
            self.phase_no_progress_steps += 1
            return self.phase_no_progress_steps >= self.no_progress_limit

        if err < (self.phase_best_error - self.progress_eps):
            self.phase_best_error = err
            self.phase_no_progress_steps = 0
        else:
            self.phase_no_progress_steps += 1

        return self.phase_no_progress_steps >= self.no_progress_limit

    @staticmethod
    def _stall_reason(phase: str) -> str:
        if phase in ("Reach", "Align", "Descend"):
            return "reach_stall"
        if phase in ("CloseHold", "LiftTest"):
            return "grasp_failed"
        if phase in ("MoveToPlaceAbove", "DescendToPlace"):
            return "place_alignment_failed"
        if phase in ("Open", "Retreat"):
            return "release_failed"
        return f"phase_stall:{phase}"

    def tick(self, belief: Dict) -> Dict:
        phase = str(belief.get("phase", "Reach"))
        phase_task_intent = self.task_intent_for_phase(phase)
        self.recover_requested = False
        self.last_reason = ""

        self._update_phase_progress(phase)
        self._ensure_base_priors(belief)
        belief["recovery_branch"] = self.active_recovery_branch
        belief["recovery_global_count"] = int(self.global_recovery_count)
        belief["recovery_branch_retry"] = int(
            self.branch_retry_counts.get(self.active_recovery_branch, 0)
        )
        belief["task_intent"] = str(phase_task_intent)
        belief["bt_decision"] = "CONTINUE"

        if self.active_branch_steps_remaining > 0:
            self.active_branch_steps_remaining -= 1
            if self.active_branch_steps_remaining <= 0:
                self._clear_active_recovery_branch()

        if phase in self.TERMINAL_SUCCESS_PHASES:
            self.status = BTStatus.SUCCESS
            self._clear_active_recovery_branch()
            return self._make_tick_result(
                recover=False,
                terminal_failure=False,
                task_intent="DONE",
                bt_decision="SUCCESS",
            )

        if phase in self.TERMINAL_FAILURE_PHASES:
            self.status = BTStatus.FAILURE
            self.last_reason = str(belief.get("failure_reason", "phase_failure") or "phase_failure")
            self._clear_active_recovery_branch()
            return self._make_tick_result(
                recover=False,
                terminal_failure=True,
                task_intent="FAILURE",
                bt_decision="TERMINAL_FAILURE",
            )

        if str(belief.get("task_switch_event", "")).strip() == "object_dropped":
            self.last_reason = "object_dropped"
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return self._make_tick_result(
                recover=True,
                terminal_failure=False,
                task_intent="PICK",
                bt_decision="TASK_SWITCH",
            )

        if int(belief.get("obs_stale_warn", 0)) == 1:
            self.last_reason = "stale_observation"
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return self._make_tick_result(
                recover=True,
                terminal_failure=False,
                task_intent="RECOVER",
                bt_decision="RECOVER",
            )

        if int(belief.get("risk_singularity_warn", 0)) == 1:
            self.last_reason = "singularity_risk"
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return self._make_tick_result(
                recover=True,
                terminal_failure=False,
                task_intent="RECOVER",
                bt_decision="RECOVER",
            )

        if int(belief.get("risk_unexpected_contact_warn", 0)) == 1:
            self.last_reason = "unexpected_contact"
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return self._make_tick_result(
                recover=True,
                terminal_failure=False,
                task_intent="RECOVER",
                bt_decision="RECOVER",
            )

        if self.vfe_recover_enabled:
            vfe = float(belief.get("vfe_total", 0.0))
            if np.isfinite(vfe) and vfe > self.vfe_recover_threshold:
                self.vfe_high_steps += 1
            else:
                self.vfe_high_steps = 0
            if self.vfe_high_steps >= self.vfe_recover_steps:
                self.last_reason = "high_vfe"
                self.recover_requested = True
                self.status = BTStatus.RUNNING
                return self._make_tick_result(
                    recover=True,
                    terminal_failure=False,
                    task_intent="RECOVER",
                    bt_decision="RECOVER",
                )
        else:
            self.vfe_high_steps = 0

        if self._update_progress_watchdog(phase, belief):
            self.last_reason = self._stall_reason(phase)
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return self._make_tick_result(
                recover=True,
                terminal_failure=False,
                task_intent="RECOVER",
                bt_decision="RECOVER",
            )

        result = self.root.tick(BTContext(phase=phase, belief=belief), self)
        if result == BTStatus.SUCCESS:
            self.status = BTStatus.SUCCESS
            self._clear_active_recovery_branch()
            return self._make_tick_result(
                recover=False,
                terminal_failure=False,
                task_intent="DONE",
                bt_decision="SUCCESS",
            )

        self.status = BTStatus.RUNNING
        if self.recover_requested:
            return self._make_tick_result(
                recover=True,
                terminal_failure=False,
                task_intent="RECOVER",
                bt_decision="RECOVER",
            )
        return self._make_tick_result(
            recover=False,
            terminal_failure=False,
            task_intent=phase_task_intent,
            bt_decision="CONTINUE",
        )

    def recover_belief(self, belief: Dict) -> Dict:
        next_belief = dict(belief)
        reason = str(self.last_reason or "tree_failure")
        next_belief["bt_decision"] = "RECOVER"
        next_belief["last_retry_reason"] = reason
        next_belief["failure_reason"] = ""

        if reason == "object_dropped":
            pick_entry = str(PickPhaseManager.retry_entry_phase())
            next_belief["bt_decision"] = "TASK_SWITCH"
            next_belief["phase"] = pick_entry
            next_belief["task_intent"] = "PICK"
            next_belief["requested_task_intent"] = "PICK"
            next_belief["task_switch_event"] = "object_dropped"
            next_belief["retry_scope"] = "BT_TASK_SWITCH"
            # Start fresh retry budget after explicit task switch PLACE->PICK.
            next_belief["retry_count"] = 0
            next_belief["reach_cooldown"] = self.reach_reentry_cooldown_steps
            next_belief["reach_best_error"] = float("inf")
            next_belief["reach_no_progress_steps"] = 0
            next_belief["reach_watchdog_active"] = 0
            next_belief["reach_yaw_align_active"] = 0
            next_belief["reach_yaw_align_timer"] = 0
            next_belief["reach_yaw_align_done"] = 0
            next_belief["align_timer"] = 0
            next_belief["align_settle_counter"] = 0
            next_belief["descend_timer"] = 0
            next_belief["descend_best_error"] = float("inf")
            next_belief["descend_no_progress_steps"] = 0
            next_belief["descend_timeout_extensions"] = 0
            next_belief["close_hold_timer"] = 0
            next_belief["lift_test_timer"] = 0
            next_belief["transit_timer"] = 0
            next_belief["open_timer"] = 0
            next_belief["retreat_timer"] = 0
            next_belief["release_detach_counter"] = 0
            next_belief["release_stable_counter"] = 0
            next_belief["place_descend_timer"] = 0
            next_belief["place_descend_best_error"] = float("inf")
            next_belief["place_descend_no_progress_steps"] = 0
            next_belief["place_descend_timeout_extensions"] = 0
            next_belief["place_reapproach_count"] = 0
            next_belief["release_reapproach_count"] = 0
            self.place_started = False
            self.phase_steps = 0
            self.phase_best_error = float("inf")
            self.phase_no_progress_steps = 0
            self.last_phase = pick_entry
            self.vfe_high_steps = 0
            self._clear_active_recovery_branch()
            return next_belief

        retries = int(next_belief.get("retry_count", 0)) + 1
        self.global_recovery_count += 1
        next_belief["retry_count"] = retries
        next_belief["recovery_global_count"] = int(self.global_recovery_count)
        next_belief["retry_scope"] = "BT_RECOVERY"

        if retries > self.max_retries:
            next_belief["phase"] = "Failure"
            next_belief["failure_reason"] = reason
            next_belief["recovery_branch"] = "Failure"
            next_belief["recovery_branch_retry"] = 0
            next_belief["task_intent"] = "FAILURE"
            self.status = BTStatus.FAILURE
            self.place_started = False
            self.last_reason = f"max_retries_exceeded:{retries}>{self.max_retries}"
            self._clear_active_recovery_branch()
            return next_belief

        if self.global_recovery_count > self.global_recovery_cap:
            next_belief["phase"] = "Failure"
            next_belief["failure_reason"] = "retry_budget_exceeded"
            next_belief["recovery_branch"] = "Failure"
            next_belief["recovery_branch_retry"] = 0
            next_belief["task_intent"] = "FAILURE"
            self.status = BTStatus.FAILURE
            self.place_started = False
            self.last_reason = (
                f"global_recovery_cap_exceeded:{self.global_recovery_count}>{self.global_recovery_cap}"
            )
            self._clear_active_recovery_branch()
            return next_belief

        branch = self._select_recovery_branch(reason)
        if not branch:
            next_belief["phase"] = "Failure"
            next_belief["failure_reason"] = "branch_retry_budget_exceeded"
            next_belief["recovery_branch"] = "Failure"
            next_belief["recovery_branch_retry"] = 0
            next_belief["task_intent"] = "FAILURE"
            self.status = BTStatus.FAILURE
            self.place_started = False
            self.last_reason = "branch_retry_budget_exceeded"
            self._clear_active_recovery_branch()
            return next_belief

        self.branch_retry_counts[branch] = int(self.branch_retry_counts.get(branch, 0)) + 1
        next_belief["recovery_branch"] = branch
        next_belief["recovery_branch_retry"] = int(self.branch_retry_counts[branch])
        self._activate_recovery_branch(branch, next_belief)

        cur_phase = str(next_belief.get("phase", PickPhaseManager.entry_phase()))
        has_grasp = int(next_belief.get("s_grasp", 0)) == 1
        place_side_failure = reason in ("place_alignment_failed", "release_failed")
        place_retry_phase = str(PlacePhaseManager.retry_entry_phase())
        pick_retry_phase = str(PickPhaseManager.retry_entry_phase())
        if place_side_failure and cur_phase in self.PLACE_PHASES and has_grasp:
            # Keep object grasped and retry from place-approach, not pick-reach.
            next_belief["phase"] = place_retry_phase
            next_belief["task_intent"] = "PLACE"
            next_belief["transit_timer"] = 0
            next_belief["open_timer"] = 0
            next_belief["retreat_timer"] = 0
            next_belief["place_descend_timer"] = 0
            next_belief["place_descend_best_error"] = float("inf")
            next_belief["place_descend_no_progress_steps"] = 0
            next_belief["place_descend_timeout_extensions"] = 0
            next_belief["release_detach_counter"] = 0
            next_belief["release_stable_counter"] = 0
            self.phase_steps = 0
            self.phase_best_error = float("inf")
            self.phase_no_progress_steps = 0
            self.last_phase = place_retry_phase
            self.vfe_high_steps = 0
            return next_belief

        next_belief["phase"] = pick_retry_phase
        next_belief["task_intent"] = "PICK"
        if branch == "ReScanTable":
            next_belief["reach_cooldown"] = max(
                self.reach_reentry_cooldown_steps, self.rescan_hold_steps
            )
        elif branch == "SafeBackoff":
            next_belief["reach_cooldown"] = max(
                self.reach_reentry_cooldown_steps, self.safe_backoff_hold_steps
            )
        else:
            next_belief["reach_cooldown"] = self.reach_reentry_cooldown_steps
        next_belief["reach_best_error"] = float("inf")
        next_belief["reach_no_progress_steps"] = 0
        next_belief["reach_watchdog_active"] = 0
        next_belief["reach_yaw_align_active"] = 0
        next_belief["reach_yaw_align_timer"] = 0
        next_belief["reach_yaw_align_done"] = 0
        next_belief["align_timer"] = 0
        next_belief["align_settle_counter"] = 0
        next_belief["descend_timer"] = 0
        next_belief["descend_best_error"] = float("inf")
        next_belief["descend_no_progress_steps"] = 0
        next_belief["descend_timeout_extensions"] = 0
        next_belief["close_hold_timer"] = 0
        next_belief["lift_test_timer"] = 0
        next_belief["transit_timer"] = 0
        next_belief["open_timer"] = 0
        next_belief["retreat_timer"] = 0
        next_belief["release_detach_counter"] = 0
        next_belief["release_stable_counter"] = 0
        next_belief["place_descend_timer"] = 0
        next_belief["place_descend_best_error"] = float("inf")
        next_belief["place_descend_no_progress_steps"] = 0
        next_belief["place_descend_timeout_extensions"] = 0
        next_belief["place_reapproach_count"] = 0
        next_belief["release_reapproach_count"] = 0

        self.place_started = False
        self.phase_steps = 0
        self.phase_best_error = float("inf")
        self.phase_no_progress_steps = 0
        self.last_phase = pick_retry_phase
        self.vfe_high_steps = 0
        return next_belief
