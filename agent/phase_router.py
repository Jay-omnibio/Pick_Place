from __future__ import annotations

from dataclasses import dataclass

from agent.phase_managers import (
    PickPhaseManager,
    PlacePhaseManager,
    task_intent_from_phase,
)


@dataclass(frozen=True)
class TaskRouteDecision:
    task_intent: str
    route_source: str
    entry_phase: str


class PhaseRouter:
    """
    Non-breaking task router.
    Current runtime still keeps phase transitions in inference/BT.
    Router is used to resolve authoritative task intent and entry mapping.
    """

    TASK_ENTRY_PHASE = {
        "PICK": PickPhaseManager.entry_phase(),
        "PLACE": PlacePhaseManager.entry_phase(),
    }

    @staticmethod
    def entry_phase_for_task(task_intent: str) -> str:
        return str(PhaseRouter.TASK_ENTRY_PHASE.get(str(task_intent), "Reach"))

    @staticmethod
    def task_intent_for_phase(phase: str) -> str:
        return str(task_intent_from_phase(phase))

    def resolve_task_intent(self, bt_task_intent: str, belief: dict | None) -> TaskRouteDecision:
        if not isinstance(belief, dict):
            intent = str(bt_task_intent or "PICK")
            return TaskRouteDecision(
                task_intent=intent,
                route_source="bt_default",
                entry_phase=self.entry_phase_for_task(intent),
            )

        switch_event = str(belief.get("task_switch_event", "")).strip()
        if switch_event == "object_dropped":
            intent = "PICK"
            return TaskRouteDecision(
                task_intent=intent,
                route_source="event:object_dropped",
                entry_phase=self.entry_phase_for_task(intent),
            )

        bt_intent = str(bt_task_intent or "").strip()
        if bt_intent in ("PICK", "PLACE", "RECOVER", "DONE", "FAILURE"):
            if bt_intent == "RECOVER":
                # During RECOVER, keep task-aligned entry based on current phase.
                phase_intent = self.task_intent_for_phase(str(belief.get("phase", "Reach")))
                intent = "PICK" if phase_intent == "PICK" else "PLACE" if phase_intent == "PLACE" else "PICK"
                return TaskRouteDecision(
                    task_intent=intent,
                    route_source="bt_recover_phase_infer",
                    entry_phase=self.entry_phase_for_task(intent),
                )
            if bt_intent in ("DONE", "FAILURE"):
                return TaskRouteDecision(
                    task_intent=bt_intent,
                    route_source="bt_terminal",
                    entry_phase=self.entry_phase_for_task("PICK"),
                )
            return TaskRouteDecision(
                task_intent=bt_intent,
                route_source="bt_intent",
                entry_phase=self.entry_phase_for_task(bt_intent),
            )

        phase_intent = self.task_intent_for_phase(str(belief.get("phase", "Reach")))
        return TaskRouteDecision(
            task_intent=phase_intent,
            route_source="phase_fallback",
            entry_phase=self.entry_phase_for_task(phase_intent),
        )
