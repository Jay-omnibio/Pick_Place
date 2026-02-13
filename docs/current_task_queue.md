# Current Task Queue (FSM Stability)

Status: discussion/planning only. No code changes in this step.

Config/Constant Audit:
- Completed and documented in `docs/config_audit_notes.md`.

## A) Grip Improvement Checklist (No Code Change Yet)

Planned behavior changes for later implementation:
- Use only `position_ok` for starting `Close` (disable `contact_stop` / `timeout_stop` as grip-start triggers).
- Before lift, require gripper truly finished closing (`controller READY` + near close width), not only contact hold.
- Do not leave `Close` while gripper is still `MOVING`.
- Keep one fixed grip-start Z offset and stable XY first, then close.

## B) Log Analysis Tasks (From run_20260212_182953)

These 5 points are tracked as investigation tasks:

1. ReachAbove quality check
- Confirm ReachAbove remains stable and accurate at transition in repeated runs.

2. First Descend success check
- Confirm first Descend can reach near-object Z and transition to Close normally.

3. Retry-cycle failure point
- Verify that instability starts after `Close -> ReachAbove -> Descend` retry cycle.

4. Freeze cause hypothesis (non-threshold-first)
- Investigate physical interaction/approach geometry in retry Descend (object/hand contact, approach pose), not only threshold tuning.

5. Descend Story blocker confirmation
- Use Descend Story to confirm repeated blockers (`z>threshold`, `ready=0/6`) and correlate with stalled EE Z in retry cycle.

## C) Next Discussion Phase

Pending:
- Your next questions for phase-2 investigation.
- We will use this task list as base before any further code changes.
