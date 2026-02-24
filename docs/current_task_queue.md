# Current Task Queue (Active-Inference Focus)

Status: active queue contains only remaining work.

## A) Robotics Remaining Work
See: `docs/robotics_upgrade_plan.md`

Current priorities:
1. Release verification robustness after `Open`.
2. Episode-level robotics health summary counters.
3. (Later) optional compliant control mode.
4. (Later) BT node modularization.

## B) Recovery and Failure Work (After Robotics)
See: `docs/recovery_failure_upgrade_plan.md`

Current sequence:
1. Done: failure taxonomy + reason codes.
2. Done: semantic BT recovery branches (`ReScanTable`, `ReApproachOffset`, `SafeBackoff`).
3. Done: reason-to-branch gating policy.
4. In progress: release verification robustness and place-side retry behavior tuning.
5. Done: retry budget policy (branch-local + global cap).

## C) Deferred / Not in Active Queue
1. Old FSM-only tuning queue from early runs (archived in history).
2. Constant-audit baseline is already documented in `docs/config_audit_notes.md`.

## D) End-Of-Day Update (2026-02-24)
Ready-to-push status for today: core active-inference pipeline updated and compile-verified.

Implemented today:
1. Place goal source centralized to `common_robot.yaml -> task_shared` and injected at runtime for AI/FSM consistency.
2. `MoveToPlaceAbove` changed to XY-first (no early Z pull while XY is out), with local constrained EFE candidate selection.
3. XY-first guard aligned with phase-gate semantics by using XY-norm threshold in local EFE for `MoveToPlaceAbove`/`DescendToPlace` (prevents Z sneaking in while XY norm is still out).
4. `MoveToPlaceAbove -> DescendToPlace` transition tightened to strict axis gate (`preplace_xy_ok && preplace_z_ok`), removing norm shortcut.
5. `Descend`/`DescendToPlace` local constrained EFE selection active in Python path.
6. `Retreat` tuned for clearance behavior (upward retreat vector + translation-priority action settings).
7. Crash fix: `preplace_z_threshold` binding/defensive fallback in `inference/action_selection.py`.
8. Consistency guard: if Julia is available, phases `MoveToPlaceAbove`, `Descend`, `DescendToPlace`, and `Retreat` now still use Python policy path (latest logic).

Validation completed:
1. `python -m compileall inference/action_selection.py inference_interface.py agent/agent_loop.py run_pick_place.py`
2. Direct `_select_action_python(..., phase='MoveToPlaceAbove')` call succeeds with current params.

Tomorrow first checks:
1. Run full episode and confirm no repeated `place_alignment_failed` loops post-lift.
2. Measure `DescendToPlace` segment count and steps-to-`Open`.
3. Verify retreat clears upward quickly (`Retreat` phase EE Z monotonic increase).
