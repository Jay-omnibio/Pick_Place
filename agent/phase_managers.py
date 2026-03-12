from __future__ import annotations

from dataclasses import dataclass


PICK_PHASES = ("Reach", "Align", "Descend", "CloseHold", "LiftTest")
PLACE_PHASES = ("Transit", "MoveToPlaceAbove", "DescendToPlace", "Open", "Retreat")
TERMINAL_SUCCESS_PHASES = ("Done",)
TERMINAL_FAILURE_PHASES = ("Failure",)


@dataclass(frozen=True)
class PhaseManagerInfo:
    name: str
    task_intent: str
    entry_phase: str
    phases: tuple[str, ...]


class PickPhaseManager:
    INFO = PhaseManagerInfo(
        name="PickPhaseManager",
        task_intent="PICK",
        entry_phase="Reach",
        phases=PICK_PHASES,
    )

    @classmethod
    def owns_phase(cls, phase: str) -> bool:
        return str(phase) in cls.INFO.phases

    @classmethod
    def entry_phase(cls) -> str:
        return str(cls.INFO.entry_phase)

    @classmethod
    def retry_entry_phase(cls) -> str:
        return str(cls.INFO.entry_phase)


class PlacePhaseManager:
    INFO = PhaseManagerInfo(
        name="PlacePhaseManager",
        task_intent="PLACE",
        entry_phase="Transit",
        phases=PLACE_PHASES,
    )

    @classmethod
    def owns_phase(cls, phase: str) -> bool:
        return str(phase) in cls.INFO.phases

    @classmethod
    def entry_phase(cls) -> str:
        return str(cls.INFO.entry_phase)

    @classmethod
    def retry_entry_phase(cls) -> str:
        # Keep place retries restarting from the approach-above phase.
        return "MoveToPlaceAbove"


def task_intent_from_phase(phase: str) -> str:
    p = str(phase or "Reach")
    if p in PICK_PHASES:
        return "PICK"
    if p in PLACE_PHASES:
        return "PLACE"
    if p in TERMINAL_SUCCESS_PHASES:
        return "DONE"
    if p in TERMINAL_FAILURE_PHASES:
        return "FAILURE"
    return "PICK"


def manager_name_from_phase(phase: str) -> str:
    p = str(phase or "Reach")
    if p in PICK_PHASES:
        return PickPhaseManager.INFO.name
    if p in PLACE_PHASES:
        return PlacePhaseManager.INFO.name
    if p in TERMINAL_SUCCESS_PHASES or p in TERMINAL_FAILURE_PHASES:
        return "TerminalPhase"
    return "UnknownPhaseOwner"
