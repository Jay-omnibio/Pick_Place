from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class BTStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class BTContext:
    phase: str


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


class AIPickPlaceBehaviorTree:
    PICK_PHASES = ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest")
    PLACE_PHASES = ("Transit", "MoveToPlaceAbove", "DescendToPlace", "Open", "Retreat")
    TERMINAL_SUCCESS_PHASES = ("Done",)
    TERMINAL_FAILURE_PHASES = ("Failure",)

    def __init__(self, max_retries: int = 3, reach_reentry_cooldown_steps: int = 20, stall_limit: int = 1200):
        self.max_retries = int(max_retries)
        self.reach_reentry_cooldown_steps = int(reach_reentry_cooldown_steps)
        self.stall_limit = int(stall_limit)

        self.place_started = False
        self.last_phase = ""
        self.phase_steps = 0
        self.last_reason = ""
        self.status: BTStatus = BTStatus.RUNNING
        self.recover_requested = False

        # Root: try normal pick->place sequence; if it fails, run recovery.
        self.root: BTNode = SelectorNode(
            [
                SequenceNode([AcquireNode(), PlaceNode()]),
                RecoverNode(),
            ]
        )

    def _update_phase_progress(self, phase: str) -> None:
        if phase == self.last_phase:
            self.phase_steps += 1
        else:
            self.phase_steps = 0
            self.last_phase = phase

        if phase in self.PLACE_PHASES:
            self.place_started = True
        elif phase in self.TERMINAL_SUCCESS_PHASES:
            self.place_started = False

    def tick(self, belief: Dict) -> Dict:
        phase = str(belief.get("phase", "Reach"))
        self.recover_requested = False
        self.last_reason = ""

        self._update_phase_progress(phase)

        if phase in self.TERMINAL_SUCCESS_PHASES:
            self.status = BTStatus.SUCCESS
            return {"recover": False, "terminal_failure": False}

        if phase in self.TERMINAL_FAILURE_PHASES:
            self.status = BTStatus.FAILURE
            self.last_reason = "phase_failure"
            return {"recover": False, "terminal_failure": True}

        if self.phase_steps >= self.stall_limit:
            self.last_reason = f"phase_stall:{phase}"
            self.recover_requested = True
            self.status = BTStatus.RUNNING
            return {"recover": True, "terminal_failure": False}

        result = self.root.tick(BTContext(phase=phase), self)
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
        self.last_phase = "Reach"
        return next_belief
