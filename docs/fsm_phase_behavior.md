# FSM Phase Behavior Guide

This document explains each FSM phase in plain terms:
- what the phase tries to do
- when the phase should end
- what robot pose/state should be true at phase end

All thresholds and targets are from current code in `tasks/pick_place_fsm.py` and `policies/scripted_pick_place.py`.

## Important Coordinate Meaning

- `o_obj = obj_world - ee_world` (object relative to end-effector)
- `o_target = target_world - ee_world`

So phase targets like `[0.004, 0.004, -0.13]` are relative offsets, not fixed world points.

## 1) ReachAbove

- Goal:
  - Move EE above object at pre-grasp relation.
  - Keep gripper open.
- Main target:
  - `pregrasp_obj_rel = [0.001, 0.001, -0.08]`
- Transition to next phase:
  - `xy_error <= reach_xy_threshold` and `abs(z_error) <= reach_z_threshold`
  - Current values: `0.015` and `0.020`
- What should be true at end:
  - EE is above object (not yet at grasp depth).
  - XY is aligned enough for descend.
  - Contact should normally still be `0`.

## 2) Descend

- Goal:
  - Move from above-object to grasp-height while keeping XY aligned.
  - Policy prioritizes XY correction and limits descend speed.
- Main target:
  - `grasp_obj_rel = [0.004, 0.004, -0.010]`
- Transition to next phase (Close):
  - X and Y errors below per-axis thresholds.
  - Z error below descend Z threshold.
  - Ready condition stable for `descend_ready_steps`.
  - Or guarded stop on contact/timeout fallback.
- What should be true at end:
  - EE is at grasp height near object.
  - XY should be centered within configured thresholds.
  - Still open gripper.

## 3) Close

- Goal:
  - Close gripper and verify stable contact.
- Transition to next phase (LiftTest):
  - `step_in_phase >= close_hold_steps`
  - `stable_contact_counter >= stable_contact_steps`
- What should be true at end:
  - Object should be between fingers.
  - Contact stable, not just a one-step touch.

## 4) LiftTest

- Goal:
  - Validate grasp quality before moving to place.
- Transition to next phase (Transit):
  - Hold grasp through lift test window.
  - Contact stays on.
  - Object-relative drift does not exceed `lift_test_obj_rel_drift_max`.
- What should be true at end:
  - Object follows gripper while lifting.
  - No major slip.

## 5) Transit

- Goal:
  - Lift to safe travel height.
- Transition to next phase (MoveToPlaceAbove):
  - EE `z >= transit_height - transit_z_threshold`
- What should be true at end:
  - Object safely above table/obstacles.

## 6) MoveToPlaceAbove

- Goal:
  - Move to pre-place point above target.
- Main target:
  - `preplace_target_rel = [0.0, 0.0, -0.13]`
- Transition to next phase (DescendToPlace):
  - `norm(o_target - preplace_target_rel) <= preplace_threshold`
- What should be true at end:
  - EE is above place target, still holding object.

## 7) DescendToPlace

- Goal:
  - Lower to placement depth.
- Main target:
  - `place_target_rel = [0.0, 0.0, -0.05]`
- Transition to next phase (Open):
  - `norm(o_target - place_target_rel) <= place_threshold`
- What should be true at end:
  - Object is at place height near target.

## 8) Open

- Goal:
  - Release object.
- Transition to next phase (Retreat):
  - `step_in_phase >= open_hold_steps`
- What should be true at end:
  - Gripper open, object released.

## 9) Retreat

- Goal:
  - Move EE away to safe post-place position.
- Transition to terminal phase (Done):
  - `step_in_phase >= retreat_steps`
- What should be true at end:
  - EE separated from object/target area.

## Terminal Phases

- `Done`: normal completion.
- `Failure`: retry limit exceeded or grasp validation failed repeatedly.

## Practical Debug Checks

For approach quality, watch these first:
- `true_descend_x_error`
- `true_descend_y_error`
- `true_descend_z_error`
- `phase transition logs`

If `Descend` keeps looping before `Close`, usually one of:
- XY never reaches threshold,
- Z not within threshold,
- stable-ready counter not sustained.

## Critical Descend Z Clamp

If `true_descend_z_error` stalls around a fixed value (for example `~0.05` to `~0.06`) and never reaches the threshold, check workspace Z floors first:

- `env/simulator.py`: `workspace_min` (mocap clamp)
- `config/safety_config.yaml`: `workspace_min` (safety clamp)
- `safety/safety_checker.py`: fallback `workspace_min`

Current defaults are aligned to:

- `workspace_min z = 0.02`
- `descend_z_threshold = 0.010`

If workspace floor is higher than your desired grasp approach, Descend will keep moving in XY while Z remains blocked.

## Quick Tuning Cheat Sheet

### Core Relative Targets

| Name | Current Value | Used In |
|---|---|---|
| `pregrasp_obj_rel` | `[0.001, 0.001, -0.08]` | ReachAbove target |
| `grasp_obj_rel` | `[0.004, 0.004, -0.010]` | Descend / Close readiness |
| `preplace_target_rel` | `[0.0, 0.0, -0.13]` | MoveToPlaceAbove target |
| `place_target_rel` | `[0.0, 0.0, -0.05]` | DescendToPlace target |

### Reach/Descend Thresholds

| Parameter | Current Value | Meaning |
|---|---|---|
| `reach_xy_threshold` | `0.015` | ReachAbove XY completion |
| `reach_z_threshold` | `0.020` | ReachAbove Z completion |
| `descend_threshold` | `0.02` | Descend norm fallback completion |
| `descend_x_threshold` | `0.055` | Descend X completion |
| `descend_y_threshold` | `0.055` | Descend Y completion |
| `descend_z_threshold` | `0.010` | Descend Z completion |
| `descend_contact_z_threshold` | `0.06` | Allow contact-based descend stop only if near Z |
| `descend_ready_steps` | `6` | Consecutive ready steps before Close |
| `descend_max_steps` | `220` | Descend timeout limit |
| `descend_timeout_x_threshold` | `0.05` | Timeout near-enough X guard |
| `descend_timeout_y_threshold` | `0.05` | Timeout near-enough Y guard |
| `descend_timeout_z_threshold` | `0.05` | Timeout near-enough Z guard |

### Phase Timing / Hysteresis

| Parameter | Current Value | Meaning |
|---|---|---|
| `stable_contact_steps` | `8` | Close -> LiftTest contact stability |
| `close_hold_steps` | `25` | Minimum close hold before lift-test |
| `lift_test_steps` | `18` | Lift validation window |
| `open_hold_steps` | `12` | Open hold before retreat |
| `retreat_steps` | `24` | Retreat duration |
| `descend_stop_contact_steps` | `2` | Consecutive contact steps for descend contact stop |

### Place / Transit Thresholds

| Parameter | Current Value | Meaning |
|---|---|---|
| `transit_height` | `0.35` | Safe travel height |
| `transit_z_threshold` | `0.02` | Transit completion margin |
| `preplace_threshold` | `0.04` | MoveToPlaceAbove completion |
| `place_threshold` | `0.03` | DescendToPlace completion |

### Policy Motion Controls (Scripted Policy)

| Parameter | Current Value | Meaning |
|---|---|---|
| `reach_xy_first_threshold` | `0.03` | In place-approach phases, hold Z until XY is this good |
| `reach_slowdown_threshold` | `0.12` | Slow when near pregrasp |
| `reach_slowdown_scale` | `0.35` | Reach slowdown factor |
| `descend_max_step_scale` | `0.40` | Descend speed scale |
| `place_descend_max_step_scale` | `0.35` | Place descend speed scale |
| `descend_xy_priority_threshold` | `0.035` | In Descend, hold Z if XY error larger than this |
| `descend_xy_lock_threshold` | `0.0` | XY lock disabled (diagnostic mode) |
| `descend_xy_unlock_threshold` | `0.0` | XY lock disabled (diagnostic mode) |
| `descend_xy_correction_gain` | `1.15` | XY correction gain in Descend |
| `descend_xy_correction_max` | `0.055` | Max per-step XY correction in Descend |
| `obs_smooth_alpha` | `0.55` | Observation EMA smoothing |

## Descend XY Anti-Drift A/B Test Matrix

Use this table to compare profiles in a controlled way (same object start and same run length).

| Profile | `descend_xy_lock_threshold` | `descend_xy_unlock_threshold` | `descend_xy_correction_gain` | `descend_xy_correction_max` | `descend_xy_priority_threshold` | `obs_smooth_alpha` | Expected |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | `0.045` | `0.070` | `0.65` | `0.020` | `0.050` | `0.75` | Current behavior reference |
| A (Balanced) | `0.030` | `0.050` | `0.85` | `0.028` | `0.045` | `0.68` | Better XY hold, still stable |
| B (Stronger) | `0.025` | `0.045` | `1.00` | `0.035` | `0.040` | `0.62` | Stronger anti-drift |
| C (Aggressive) | `0.020` | `0.040` | `1.15` | `0.040` | `0.035` | `0.55` | Maximum correction, may oscillate |

### Run Result Sheet

Fill this after each run for objective comparison.

| Profile | Run ID | ReachAbove->Descend step | `true_descend_x_error` at transition | `true_descend_y_error` at transition | `true_descend_z_error` at transition | Descend trend (X/Y up or down) | Lowest true Z in Descend | Descend->Close happened (Y/N) | Notes |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| Baseline |  |  |  |  |  |  |  |  |  |
| A |  |  |  |  |  |  |  |  |  |
| B |  |  |  |  |  |  |  |  |  |
| C |  |  |  |  |  |  |  |  |  |
