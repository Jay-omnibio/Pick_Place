# AI+BT Partwise Findings (Deep Review)

Last updated: 2026-03-04

Purpose:
1. Keep one deep, code-grounded status review of the AI+BT runtime.
2. Separate what is solved, what is still risky, and what to prioritize next.
3. Use one shared language (`P0`, `P1`, `P2`) for future fixes.

Evidence baseline used for this revision:
1. Latest multi-scenario sweep: `logs/reports/position_sweep_20260301_134501.md`.
2. Recent failed-run diagnostics for bottleneck analysis:
 - `logs/reports/run_20260301_135118_diagnostics.md`
 - `logs/reports/run_20260301_140218_diagnostics.md`
 - `logs/reports/run_20260301_140453_diagnostics.md`
 - `logs/reports/run_20260301_140952_diagnostics.md`
3. Current runtime code in `agent/`, `inference/`, `control/`, `env/`, `config/`.

---

## Part 1: Belief and Inference Core
Scope:
- `inference_interface.py`
- `inference/rxinfer_beliefs.jl`

What is solid now:
1. Full belief loop with phase state, timers, confidence, VFE, and retry metadata.
2. Optional RxInfer-backed update path with safe Python fallback.
3. Place-side watchdogs are active (preplace and place-descend progress/timeout handling).
4. Place-goal world pose integration is consistent with runtime injection.

Findings:
1. `P0`: Phase progression still depends on hard thresholds; near-threshold oscillation can still produce retry loops in hard geometries.
2. `P1`: Dual smoothing (RxInfer posterior + alpha blending) can trade robustness for lag; needs scenario-based tuning.
3. `P2`: The belief object is large and mixes state, watchdog, and diagnostics fields; maintainability cost is rising.

Recommended next action:
1. Add a compact "phase gate decision trace" per step (which sub-condition blocked transition).
2. Tune RxInfer noise and alpha jointly using scenario matrix, not one-run tuning.

Completion estimate:
- 84%

---

## Part 2: BT Supervision and Recovery
Scope:
- `agent/ai_behavior_tree.py`

What is solid now:
1. Structured recovery branches (`ReScanTable`, `ReApproachOffset`, `SafeBackoff`) exist.
2. Branch and global retry budgets are enforced.
3. Place-side failures can recover back to `MoveToPlaceAbove` while retaining grasp.

Findings:
1. `P0`: Recovery ownership is still split between BT and phase-local fallbacks in inference, which complicates root-cause attribution.
2. `P1`: Branch success metrics are not summarized per run, so branch efficacy is hard to compare.
3. `P2`: Prior injection is effective but still tightly coupled to belief key naming.

Recommended next action:
1. Add per-run branch summary counters (triggers, successes, terminal failures by branch).
2. Define one authoritative recovery path policy (BT-first or phase-first) and document it.

Completion estimate:
- 82%

---

## Part 3: Action Selection and Phase Policy
Scope:
- `inference/action_selection.py`
- `inference/action_selection.jl`

What is solid now:
1. Phase-specific policy is complete for pick and place pipelines.
2. Reach arc logic is available for hard angular approaches.
3. `MoveToPlaceAbove` and `DescendToPlace` are XY-first.
4. `MoveToPlaceAbove` now keeps topdown objective enabled while gating yaw when far in XY.

Findings:
1. `P0`: `MoveToPlaceAbove` is still the dominant time sink in difficult runs.
2. `P1`: Policy has grown complex (EFE, watchdog interaction, per-phase scaling), making single-parameter tuning fragile.
3. `P2`: Python and Julia action paths are not always kept behavior-identical for every new tweak.

Recommended next action:
1. Add a "policy efficiency" metric: commanded error-reduction vs realized error-reduction per phase.
2. Keep one canonical policy path for production tuning; keep the other clearly marked experimental.

Completion estimate:
- 80%

---

## Part 4: Controller, IK, and Gripper Execution
Scope:
- `control/controller.py`
- `backends/actuator_backend.py`

What is solid now:
1. Damped IK with stacked objectives and nullspace posture bias is stable.
2. Stale-direction smoothing guard exists (per-axis sign-flip memory reset).
3. Gripper open mode supports size-based dynamic target.
4. AI runtime now synchronizes open-ready gating with controller dynamic open target each step.

Findings:
1. `P0`: Requested Cartesian move vs realized EE move can still diverge in constrained postures.
2. `P1`: Objective weight interactions (position/yaw/topdown/nullspace) remain the main source of "unexpected path shape."
3. `P2`: No explicit collision-aware motion layer yet; safety is bounds/limits driven, not path-planning driven.

Recommended next action:
1. Add phase-level realized-motion efficiency summary (`||dEE|| / ||command||`) to diagnostics output.
2. Add a simple controller health table at end-of-run (saturation ratio, fallback count, guard activation count).

Completion estimate:
- 78%

---

## Part 5: Perception, Sensors, and State Estimation
Scope:
- `env/sensors.py`
- `backends/sensor_backend.py`
- `state/state_estimator.py`

What is solid now:
1. Sensor model is clean: EE, object-relative, target-relative, contact, gripper width.
2. Estimator and velocity support are integrated into inference.
3. Confidence and VFE already use observation dynamics to gate transitions/recovery.

Findings:
1. `P0`: Noise sensitivity is still scenario-dependent; some layouts remain fragile.
2. `P1`: Runtime lacks explicit frame-contract assertions for all critical vectors during difficult episodes.
3. `P2`: Observation and belief drift diagnostics are available, but not yet summarized into a single "sensor quality score."

Recommended next action:
1. Add per-run frame sanity checks (`target-ee`, `obj-ee`) with fail-fast warnings.
2. Track confidence distribution by phase in diagnostics report.

Completion estimate:
- 76%

---

## Part 6: Runtime Orchestration, Logging, and Tooling
Scope:
- `agent/agent_loop.py`
- `run_pick_place.py`
- `tools/analyze_run_diagnostics.py`
- `tools/run_batch_eval.py`
- `tools/run_position_sweep.py`

What is solid now:
1. Runtime loop is clear: `sense -> infer -> BT -> action -> act`.
2. Batch and sweep tooling exist and are usable.
3. Diagnostics reports already capture transitions, drift, and phase durations.
4. Gripper readiness (`ai_gripper_open_ready`, `ai_gripper_close_ready`) is now log-visible.

Findings:
1. `P0`: No single run-summary KPI block yet for quick comparison across runs.
2. `P1`: Some root-cause checks still require manual CSV inspection.
3. `P2`: Log schema is broad; pruning/structuring can reduce analysis overhead.

Recommended next action:
1. Add auto-generated end-of-run KPI summary (success, retries by reason, bottleneck phase).
2. Add one command for "compare last N runs" on key metrics.

Completion estimate:
- 83%

---

## Part 7: Config, Governance, and Documentation
Scope:
- `config/*.yaml`
- `config/runtime_loader.py`
- `docs/*.md`

What is solid now:
1. Config loader is strict and catches missing/unknown keys.
2. Shared target/world goal config model is clear.
3. Runtime is AI-only, removing old FSM split complexity.

Findings:
1. `P0`: Parameter count is high; local tuning can have hidden global effects.
2. `P1`: Some docs drift can happen quickly after behavior updates unless explicitly maintained.
3. `P2`: More config keys now deserve grouping by "core vs experimental."

Recommended next action:
1. Add a "config key ownership map" (key -> file/function where used).
2. Freeze a baseline profile and use scenario matrix for all changes.

Completion estimate:
- 79%

---

## Current Priority Order
1. `P0`: Reduce `MoveToPlaceAbove` duration variance while preserving successful opposite-side placement.
2. `P0`: Reduce recurring `reach_stall` loops across multi-scenario sweeps.
3. `P1`: Add end-of-run KPI summary and branch-efficacy reporting.
4. `P1`: Tune RxInfer + alpha as a coupled set across scenario tiers.

## Snapshot Conclusion
1. RxInfer integration is a major step forward and has improved difficult reach cases.
2. End-to-end `Done` is stable in easier workspace slices but still variable in hard geometry/noise slices.
3. Current dominant bottleneck in latest sweep is pick-side progression robustness (`Reach/Align/Descend`) rather than place-side completion once post-lift phases are reached.
