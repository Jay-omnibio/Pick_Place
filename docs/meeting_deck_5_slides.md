# 5-Slide Meeting Deck: Active-Inference Pick-and-Place

Last updated: 2026-03-04

---

## Slide 1: Problem, Goal, and Business Value

### Problem
Robotic pick-and-place fails in real settings when noise, geometry shifts, and partial observability cause stalls or wrong moves.

### Goal
Build a robust adaptive runtime that can:
1. Reach, grasp, lift, place, and release reliably.
2. Recover automatically from common failures.
3. Provide measurable diagnostics for engineering decisions.

### Value
1. Faster iteration to real-robot readiness.
2. Better explainability than black-box-only control.
3. Lower manual intervention during runs.

---

## Slide 2: System Architecture (BT + Active Inference)

### Core idea
1. Active Inference does low-level adaptive action selection.
2. Behavior Tree (BT) supervises progression and recovery.

### Runtime loop
1. Sense (`o_ee`, `o_obj`, `o_target`, `o_grip`, `o_contact`)
2. Infer beliefs (`infer_beliefs`)
3. BT tick (normal or recovery branch)
4. Select action (`select_action`)
5. Control + IK execution

### Why this combination
1. Active Inference handles uncertainty online.
2. BT provides deterministic task structure and failure routing.

---

## Slide 3: Execution Flow and Recovery

### Phase pipeline
1. `Reach -> Align -> Descend -> CloseHold -> LiftTest -> Transit -> MoveToPlaceAbove -> DescendToPlace -> Open -> Retreat -> Done`

### Key transition controls
1. Confidence gating (`phase_gate_ok`).
2. VFE-aware supervision.
3. Hysteresis/debounce around thresholds.
4. Align settle gate (`align_settle_steps`, `align_settle_step`).

### Recovery branches
1. `ReScanTable`
2. `ReApproachOffset`
3. `SafeBackoff`

### Retry policy
1. Branch caps + global cap to prevent infinite loops.
2. Place-side failures with grasp kept can re-enter place subtree.

---

## Slide 4: Current Status (Evidence)

### Latest sweep reference
`logs/reports/position_sweep_20260301_134501.md`

### Snapshot
1. 12 episodes total (4 scenarios x 3 episodes).
2. 8/12 reached `Done`.
3. Nominal scenario strong; harder geometry remains recovery-heavy.

### Bottleneck today
1. Pick-side robustness (`Reach/Align/Descend`) under hard poses/noise.
2. Place side is generally reliable once post-lift phases are reached.

### Engineering maturity
1. Structured diagnostics and batch tooling are in place.
2. Strict runtime config and reproducible scenario matrix are in place.

---

## Slide 5: Next 2-Week Plan and Ask

### Plan
1. Improve first-try pass rates for `Reach/Align/Descend`.
2. Reduce `reach_stall` loops with targeted tuning and A/B validation.
3. Add concise end-of-run KPI summary for leadership review.
4. Preserve place-side stability while improving pick-side robustness.

### Success criteria
1. Higher `Done` rate across scenario tiers.
2. Lower recovery count per run.
3. Better median/P90 steps to `LiftTest` and `Done`.

### Ask for leadership support
1. Keep architecture direction (BT + Active Inference).
2. Prioritize robustness gates before new feature expansion.
3. Continue scenario-matrix-based acceptance for release decisions.

