# Movement Baseline (Do-Not-Regress)

This document captures the current robot movement behavior that fixed the old major motion issue (bad reach path / unstable descend behavior).

Use this as the reference before making future controller/FSM changes.

## Current Goal
- Keep robust motion to pre-grasp and descend.
- Prevent old regressions where robot moved badly or closed too early.
- Grasp success tuning is later. First priority is correct approach geometry.

## Movement Design (Current)
- `ReachAbove`: approach object from above with XY-first behavior.
- `Descend`: mostly vertical descend with limited XY correction.
- `Close`: only after descend is consistently near grasp pose for multiple steps.

## Key Parameters To Preserve

### FSM thresholds (`tasks/pick_place_fsm.py`)
- `reach_xy_threshold = 0.06`
- `reach_z_threshold = 0.10`
- `descend_x_threshold = 0.055`
- `descend_y_threshold = 0.055`
- `descend_z_threshold = 0.035`
- `descend_timeout_x_threshold = 0.05`
- `descend_timeout_y_threshold = 0.05`
- `descend_timeout_z_threshold = 0.03`
- `descend_ready_steps = 6`
- `descend_max_steps = 220`

### Descend policy behavior (`policies/scripted_pick_place.py`)
- Descend uses XY anchor from Descend entry.
- Small XY correction during descend:
  - `descend_xy_correction_gain = 0.40`
  - `descend_xy_correction_max = 0.012`
- Descend speed:
  - `descend_max_step_scale = 0.40`

## Why These Matter
- Tight reach/descend thresholds prevent closing while too far from object.
- Consecutive-step readiness (`descend_ready_steps`) prevents noise-triggered early close.
- XY anchor + limited correction keeps descend stable but still allows center correction.

## Required Debug Logs (Keep)
From `agent/agent_loop.py` periodic logs:
- `Pregrasp error`
- `Descend error`
- `Descend X error`
- `Descend Y error`
- `Descend Z error`
- `True Descend X error`
- `True Descend Y error`
- `True Descend Z error`
- `Contact`

From phase transitions:
- `descend_x_error`, `descend_y_error`, `descend_z_error`
- `true_descend_x_error`, `true_descend_y_error`, `true_descend_z_error`

These are the minimum logs needed to diagnose current XY/Z alignment behavior.

## Regression Signals
If any of these appear, movement likely regressed:
- Reach path becomes erratic or drifts away from object.
- `Descend Z error` is low but `Descend X/Y error` stay high and never improve.
- Frequent `Descend -> Close` with high XY error.
- Robot appears stuck near table height with no XY convergence.

## Note
- Remaining issue: object is sometimes not centered between fingers at close.
- Likely causes to tune later:
  - residual XY/orientation mismatch at close,
  - gripper-frame alignment vs object frame,
  - final wrist orientation handling.
