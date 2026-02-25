# Implementation Backlog (AI-Only)

This file is the single source for near-term engineering work.
It replaces:
- `docs/current_task_queue.md`
- `docs/robotics_upgrade_plan.md`
- `docs/recovery_failure_upgrade_plan.md`

## Scope
- Runtime: active-inference + BT only.
- Focus: stability, recovery quality, and measurable progress.

## Baseline Already Completed
1. FSM mode removed from runtime path.
2. BT semantic recovery branches implemented (`ReScanTable`, `ReApproachOffset`, `SafeBackoff`).
3. Reason-to-branch mapping and retry budgets implemented.
4. Place goal source centralized in `config/common_robot.yaml` (`task_shared`) and injected into AI runtime.
5. Place pipeline guards tightened (`MoveToPlaceAbove` / `DescendToPlace` strict axis gates).
6. Release verification infrastructure exists in `Open` (detach/stability checks, timeout/retry hooks).
7. Runtime timing/freshness and risk warning signals available in logs.

## Active Priorities (Now)

### P0) Reach/Descend Reliability
1. Reduce Reach stalls and late XY correction behavior.
2. Ensure Reach -> Descend transition is robust under moderate observation noise.
3. Keep phase progression monotonic where possible (avoid repeated recovery loops for same reason).

Done criteria:
1. Reach error trends down in most runs.
2. Fewer repeated `reach_stall` recoveries.
3. Stable transition into `CloseHold` in successful pick episodes.

### P0) Release Verification Robustness
1. Tune detach/stability thresholds across object variants.
2. Confirm `Open -> Retreat` only occurs after verification passes or explicit failure/recovery branch.

Done criteria:
1. Fewer false `release_failed` outcomes.
2. Reduced object drop/slip during place completion.

### P1) Episode Health Summary
1. Add aggregated end-of-episode counters:
 - stale observation warnings
 - control saturation pressure (`dq_raw` vs `dq_applied`)
 - singularity-like warnings
 - unintended-contact warning streaks
2. Emit one concise summary block at terminal phase.

Done criteria:
1. Each run has one comparable monitor summary.
2. Health metrics can be trended across batch runs.

### P1) Place-Side Retry Tuning
1. Tune `MoveToPlaceAbove`/`DescendToPlace` thresholds and watchdog budgets.
2. Reduce repeated place re-approach loops while keeping safety.

Done criteria:
1. Lower `place_alignment_failed` recurrence.
2. Improved steps-to-`Open` consistency after `Transit`.

## Later (Not Blocking Current Work)
1. Optional compliant/admittance control mode behind config flag (default remains current IK path).
2. BT node modularization into `agent/bt_nodes/` without behavior change.
3. RxInfer-backed belief updates in `infer_beliefs` after current stability goals.

## Validation Standard
For each change set, compare at least 10 episodes:
1. Lift success rate
2. Done success rate
3. Mean steps to `LiftTest` and `Done`
4. Recovery count by reason
5. `release_failed` count
6. Health warning counters
