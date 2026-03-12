# Implementation Backlog (AI-Only)

This file is the single source for near-term engineering work.
Consolidated note:
1. Earlier planning docs were merged here and removed from the repo.
2. Keep this file as the active execution backlog.

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
8. Optional RxInfer-backed belief update path integrated in `infer_beliefs` with Python fallback.
9. `MoveToPlaceAbove` watchdog/progress timeout extension logic is active in baseline.
10. Gripper open-ready gating is synchronized with controller dynamic open target each step.
11. Place-above keeps topdown objective enabled while yaw objective is conditionally gated.
12. Pick-side hold-stage cleanup completed:
 - separate `PreGraspHold` phase removed
 - merged `Align` settle gate in place (`align_settle_steps`, `align_settle_step`)
13. Align yaw gate before descend implemented (`align_pick_yaw_*` keys).
14. Place-side `object_dropped` task-switch routing implemented with explicit BT observability fields.
15. Retry budget reset on `object_dropped` task switch (`retry_count=0`).
16. EE yaw observation (`o_ee_yaw`) aligned to controller yaw axis semantics.
17. Place keepout no-reentry projection is active (post-selection correction included).
18. Reach orientation policy now translation-first when far, with final yaw-align hold preserved near transition.

## Reverted Experiments (For Future Re-Test)
These were tested and then reverted. Keep as reference so they can be re-applied later in a controlled A/B run.

| ID | Experiment | Main files | Intended effect | Behavior impact observed during test |
|---|---|---|---|---|
| R1 | IK translation-priority scaling in controller | `control/controller.py` | Reduce XY wrong-direction motion by attenuating orientation objectives during large XY demand and increasing nullspace pull. | Successful scenarios became slower and less consistent (more step count variance). |
| R2 | Reach watchdog XY-first variant | `inference/action_selection.py`, `inference/action_selection.jl` | Keep Reach watchdog from adding Z pull while XY is still outside threshold. | Reduced forced Z coupling, but far-pose robustness did not improve enough and stall patterns remained. |

## Adopted From Earlier Experiments
These were previously tested as experiments and are now part of baseline because they improved behavior.

| ID | Change | Main files | Baseline impact |
|---|---|---|---|
| A1 | `MoveToPlaceAbove` watchdog/progress timer | `inference_interface.py`, `config/active_inference_config.yaml`, `config/runtime_loader.py` | Reduced hard stalls in place-above and improved eventual progression to place descend. |
| A2 | Dynamic gripper-open target sync into AI gating | `agent/agent_loop.py`, `inference_interface.py` | Eliminated align-open gate mismatch when size-based open target differs from max-open width. |

Re-test rule:
1. Re-enable one experiment at a time.
2. Run the same fixed scenario matrix and episode count.
3. Keep only changes that improve both success rate and median steps without introducing new dominant failure modes.

## Active Priorities (Now)

### P0) Reach/Descend Reliability
1. Reduce Reach stalls and late XY correction behavior.
2. Ensure Reach -> Descend transition is robust under moderate observation noise.
3. Keep phase progression monotonic where possible (avoid repeated recovery loops for same reason).

Done criteria:
1. Reach error trends down in most runs.
2. Fewer repeated `reach_stall` recoveries.
3. Stable transition into `CloseHold` in successful pick episodes.

### P0 Exit Targets (Current Cycle, 10 Episodes)
Use these as hard pass/fail gates before moving priority to release tuning.

| Metric | Target | Pass/Fail Rule |
|---|---|---|
| Reach -> Descend transition success | >= 9/10 episodes | Fail if < 9 |
| Reach -> CloseHold success | >= 8/10 episodes | Fail if < 8 |
| `reach_stall` recoveries | <= 8 total across 10 episodes | Fail if > 8 |
| Median steps to `LiftTest` | <= 2600 | Fail if > 2600 |
| P90 steps to `LiftTest` | <= 3400 | Fail if > 3400 |
| Hard stuck in Reach (`Reach` for >1200 consecutive steps) | 0 episodes | Fail if any |

Measurement source:
1. Run CSV logs in `logs/run_*.csv`
2. Diagnostics from `tools/analyze_run_diagnostics.py` (`--save-report`)
3. Scenario coverage matrix and tier gates: `docs/scenario_matrix.md`

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
3. Tune and benchmark RxInfer noise parameters across scenario matrix.

## Future Track: Real-Robot Motion Hardening
This is intentionally deferred until current P0/P1 stability gates are passing.

1. Add planner-backed motion layer for collision-aware approach/re-approach trajectories.
2. Add compliant control mode (admittance/impedance) with safe contact behavior.
3. Add deterministic real-time execution/watchdog path for hardware control loop.
4. Add stronger calibration and frame/latency compensation pipeline.
5. Add hardware fault/degraded-mode handling (comm drop, actuator saturation, emergency stop path).

### Future Track Detail: Collision/Singularity Safety Stack
Target: avoid self-contact and unstable postures while preserving task completion.

1. Add explicit self-collision distance checks (link-link and tool-link) with a configured safety margin.
2. Add environment collision checks (table/object/fixtures) using inflated collision geometry for conservative clearance.
3. Add command-time velocity scaling by minimum collision distance (slow down near constraints, not only hard-stop).
4. Add singularity proximity monitor from Jacobian condition/velocity amplification metrics, with fallback posture behavior.
5. Add joint-limit proximity penalties in nullspace to keep away from hard-limit corners.
6. Add phase-aware safe waypoint policy (`lift -> translate -> descend`) as a hard preference for risky transitions.
7. Add runtime guard metrics to batch reports:
 - self-collision near-miss count
 - environment near-miss count
 - singularity warning count
 - command scaling ratio distribution
8. Define hardware-readiness gate:
 - no collision events in validation matrix
 - bounded singularity warnings
 - stable cycle-time under guard load

## Validation Standard
For each change set, compare at least 10 episodes:
1. Lift success rate
2. Done success rate
3. Mean steps to `LiftTest` and `Done`
4. Recovery count by reason
5. `release_failed` count
6. Health warning counters
