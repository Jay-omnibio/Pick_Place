from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np


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
    PICK_PHASES = ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest")
    PLACE_PHASES = ("Transit", "MoveToPlaceAbove", "DescendToPlace", "Open", "Retreat")
    TERMINAL_SUCCESS_PHASES = ("Done",)
    TERMINAL_FAILURE_PHASES = ("Failure",)

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

    def _ensure_base_priors(self, belief: Dict) -> None:
        if self.base_priors:
            return
        keys = (
            "reach_obj_rel",
            "align_obj_rel",
            "descend_obj_rel",
            "preplace_target_rel",
            "place_target_rel",
            "retreat_move",
        )
        for key in keys:
            vec = self._safe_vec3(belief.get(key, [0.0, 0.0, 0.0]))
            if vec is not None:
                self.base_priors[key] = vec

    def apply_pick_priors(self, phase: str, belief: Dict) -> None:
        if (not self.set_priors_enabled) or phase not in self.PICK_PHASES:
            return
        self._ensure_base_priors(belief)
        if not self.base_priors:
            return
        retry_count = int(belief.get("retry_count", 0))
        retry_z = min(float(retry_count) * self.retry_reach_z_step, self.retry_reach_z_max)

        reach_ref = self.base_priors.get("reach_obj_rel")
        align_ref = self.base_priors.get("align_obj_rel")
        descend_ref = self.base_priors.get("descend_obj_rel")
        if reach_ref is not None:
            updated = reach_ref.copy()
            updated[2] -= retry_z
            belief["reach_obj_rel"] = updated
        if align_ref is not None:
            updated = align_ref.copy()
            updated[2] -= retry_z
            belief["align_obj_rel"] = updated
        if descend_ref is not None:
            belief["descend_obj_rel"] = descend_ref.copy()

    def apply_place_priors(self, phase: str, belief: Dict) -> None:
        if (not self.set_priors_enabled) or phase not in self.PLACE_PHASES + self.TERMINAL_SUCCESS_PHASES:
            return
        self._ensure_base_priors(belief)
        if not self.base_priors:
            return
        preplace_ref = self.base_priors.get("preplace_target_rel")
        place_ref = self.base_priors.get("place_target_rel")
        retreat_ref = self.base_priors.get("retreat_move")
        if preplace_ref is not None:
            belief["preplace_target_rel"] = preplace_ref.copy()
        if place_ref is not None:
            belief["place_target_rel"] = place_ref.copy()
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
        if phase in ("Align", "PreGraspHold"):
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

    def tick(self, belief: Dict) -> Dict:
        phase = str(belief.get("phase", "Reach"))
        self.recover_requested = False
        self.last_reason = ""

        self._update_phase_progress(phase)
        self._ensure_base_priors(belief)

        if phase in self.TERMINAL_SUCCESS_PHASES:
            self.status = BTStatus.SUCCESS
            return {"recover": False, "terminal_failure": False}

        if phase in self.TERMINAL_FAILURE_PHASES:
            self.status = BTStatus.FAILURE
            self.last_reason = "phase_failure"
            return {"recover": False, "terminal_failure": True}

        if self.vfe_recover_enabled:
            vfe = float(belief.get("vfe_total", 0.0))
            if np.isfinite(vfe) and vfe > self.vfe_recover_threshold:
                self.vfe_high_steps += 1
            else:
                self.vfe_high_steps = 0
            if self.vfe_high_steps >= self.vfe_recover_steps:
                self.last_reason = f"high_vfe:{vfe:.3f}"
                self.recover_requested = True
                self.status = BTStatus.RUNNING
                return {"recover": True, "terminal_failure": False}
        else:
            self.vfe_high_steps = 0

        if self._update_progress_watchdog(phase, belief):
            self.last_reason = f"phase_stall:{phase}"
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return {"recover": True, "terminal_failure": False}

        result = self.root.tick(BTContext(phase=phase, belief=belief), self)
        if result == BTStatus.SUCCESS:
            self.status = BTStatus.SUCCESS
            return {"recover": False, "terminal_failure": False}

        self.status = BTStatus.RUNNING
        if self.recover_requested:
            return {"recover": True, "terminal_failure": False}
        return {"recover": False, "terminal_failure": False}

    def recover_belief(self, belief: Dict) -> Dict:
        next_belief = dict(belief)
        retries = int(next_belief.get("retry_count", 0)) + 1
        next_belief["retry_count"] = retries

        if retries > self.max_retries:
            next_belief["phase"] = "Failure"
            self.status = BTStatus.FAILURE
            self.place_started = False
            self.last_reason = f"max_retries_exceeded:{retries}>{self.max_retries}"
            return next_belief

        next_belief["phase"] = "Reach"
        next_belief["reach_cooldown"] = self.reach_reentry_cooldown_steps
        next_belief["reach_best_error"] = float("inf")
        next_belief["reach_no_progress_steps"] = 0
        next_belief["reach_watchdog_active"] = 0
        next_belief["reach_yaw_align_active"] = 0
        next_belief["reach_yaw_align_timer"] = 0
        next_belief["reach_yaw_align_done"] = 0
        next_belief["align_timer"] = 0
        next_belief["pregrasp_hold_timer"] = 0
        next_belief["descend_timer"] = 0
        next_belief["descend_best_error"] = float("inf")
        next_belief["descend_no_progress_steps"] = 0
        next_belief["descend_timeout_extensions"] = 0
        next_belief["close_hold_timer"] = 0
        next_belief["lift_test_timer"] = 0
        next_belief["transit_timer"] = 0
        next_belief["open_timer"] = 0
        next_belief["retreat_timer"] = 0

        self.place_started = False
        self.phase_steps = 0
        self.phase_best_error = float("inf")
        self.phase_no_progress_steps = 0
        self.last_phase = "Reach"
        self.vfe_high_steps = 0
        return next_belief
