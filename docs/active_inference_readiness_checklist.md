# Active Inference Readiness Checklist

Use this checklist before replacing FSM behavior with Active Inference control.

## 1) Baseline FSM Must Be Reliable

- Phase flow should consistently complete:
  `ReachAbove -> Descend -> Close -> LiftTest -> Transit -> MoveToPlaceAbove -> DescendToPlace -> Open -> Retreat -> Done`
- No random freezes, endless loops, or unsafe jumps.

## 2) Define Success Metrics

- Pick success rate.
- Place success rate.
- Release success rate (object leaves gripper during `Open`).
- Max step budget per episode.
- Number of retries/failures.

## 3) Validate Phase Gate Correctness

- `ReachAbove -> Descend` only when XY/Z are truly aligned for pregrasp.
- `Descend -> Close` should be intentional (position/gate-based), not accidental.
- `Close -> LiftTest` should wait for stable grasp condition.
- `Open -> Retreat` should happen after enough release time.

## 4) Stabilize Grasp + Release Mechanics

- Gripper close/open behavior should be deterministic.
- Do not lift before close is actually completed.
- Confirm object is physically released in `Open` (not stuck by contact/friction side effects).

## 5) Freeze and Audit Config Constants

- Thresholds, hold steps, and gains should be reviewed and cleaned.
- Keep a known-good baseline config and avoid ad-hoc changes between runs.
- Track every config change with reason and outcome.

## 6) Keep Logs Explainable

- Heartbeat log for periodic health.
- Transition log with trigger reason.
- Descend gate/blockers log for failed transitions.
- CSV should stay consistent for plotting/regression.

## 7) Run Multi-Run Validation

- Test many episodes, not one.
- Include varied object/target initial positions.
- Report success/failure distribution, not single anecdotal result.

## 8) Recovery and Failure Paths

- Retry behavior should be bounded and deterministic.
- Timeouts should be safe and interpretable.
- Failure terminal state should be explicit and logged.

## 9) Regression Baseline

- Keep a locked baseline run profile (steps, phase durations, success rates).
- Compare every tuning change against this baseline.

## 10) Add Active Inference Incrementally

- Start with one phase (usually Reach/Descend).
- Compare against FSM baseline metrics.
- Expand phase-by-phase only after stable improvement.

