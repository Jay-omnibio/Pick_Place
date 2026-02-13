# Config/Constant Audit Notes (Completed)

Date: 2026-02-12
Scope: constants/thresholds in FSM + policy + controller + safety.
Goal: identify values that can destabilize behavior and mark keep/revisit candidates.

## 1) Audit Outcome

- Reach/Descend logic is mostly consistent with current behavior goals.
- Biggest practical issue is not missing constants; it is gating interaction in Descend/Close and retry-cycle behavior.
- A few constants are legacy/unused and can confuse tuning.

## 2) Constants Snapshot and Recommendations

### A. `tasks/pick_place_fsm.py`

| Constant | Current | Role | Recommendation |
|---|---:|---|---|
| `pregrasp_obj_rel` | `[0.001, 0.001, -0.08]` | ReachAbove target relative to object | Keep (current runs show good handoff) |
| `grasp_obj_rel` | `[0.001, 0.001, 0.0]` | Descend target relative to object | Keep for now |
| `reach_xy_threshold` | `0.010` | ReachAbove XY gate | Keep |
| `reach_z_threshold` | `0.025` | ReachAbove Z gate | Keep |
| `descend_threshold` | `0.02` | Euclidean fallback gate in Descend | Keep |
| `descend_x_threshold` | `0.04` | Descend X gate | Keep |
| `descend_y_threshold` | `0.04` | Descend Y gate | Keep |
| `descend_z_threshold` | `0.005` | Descend Z gate | Keep |
| `descend_contact_z_threshold` | `0.00` | Contact-triggered close requires very near Z | Keep if contact path remains enabled |
| `descend_timeout_x/y/z_threshold` | `0.05` each | Timeout-near fallback gate | Keep for now |
| `descend_max_steps` | `220` | Timeout step gate | Keep |
| `descend_stop_contact_steps` | `2` | Contact hysteresis | Keep |
| `descend_ready_steps` | `6` | Consecutive in-threshold requirement | Keep |
| `stable_contact_steps` | `8` | Close->LiftTest contact hold | Keep |
| `close_hold_steps` | `25` | Close dwell | Keep |
| `lift_test_steps` | `18` | Lift test dwell | Keep |
| `open_hold_steps` | `12` | Open dwell | Keep |
| `retreat_steps` | `24` | Retreat dwell | Keep |
| `lift_test_obj_rel_drift_max` | `0.06` | Grasp drift fail condition | Keep |
| `transit_height` | `0.35` | Safe transit Z | Keep |
| `transit_z_threshold` | `0.02` | Transit reach gate | Keep |
| `max_retries` | `3` | Retry budget | Keep |
| `retry_reach_cooldown_steps` | `20` | Retry cooldown | Keep |
| `descend_xy_threshold` | `0.50` | Legacy field | Revisit/Remove later (unused) |
| `descend_timeout_xy_threshold` | `0.05` | Legacy field | Revisit/Remove later (unused) |

### B. `policies/scripted_pick_place.py`

| Constant | Current | Role | Recommendation |
|---|---:|---|---|
| `descend_max_step_scale` | `2` | Descend speed multiplier | Revisit: effectively clipped to `1.0` in controller |
| `place_descend_max_step_scale` | `0.35` | Place descend speed | Keep |
| `descend_xy_correction_gain` | `1.15` | XY correction gain in Descend | Keep |
| `descend_xy_correction_max` | `0.055` | XY correction clamp | Keep |
| `descend_xy_priority_threshold` | `0.015` | Hold Z until XY close | Keep (important behavior switch) |
| `descend_xy_lock_threshold` | `0.04` | XY lock-on threshold | Keep |
| `descend_xy_unlock_threshold` | `0.10` | XY unlock threshold | Keep |
| `reach_xy_first_threshold` | `0.03` | Reach planar-first gate | Keep |
| `reach_slowdown_threshold` | `0.08` | Reach slowdown near target | Keep |
| `reach_slowdown_scale` | `0.75` | Slowdown intensity | Keep |
| `obs_smooth_alpha` | `0.40` | EMA smoothing | Keep |
| `lift_z_offset` | `0.04` | Lift target increment | Keep |

### C. `control/controller.py`

| Constant | Current | Role | Recommendation |
|---|---:|---|---|
| `max_step` | `0.03` | EE step clamp per control step | Keep |
| `max_joint_step` | `0.20` | Joint delta clamp | Keep |
| `move_smoothing` | `0.55` | Command smoothing | Keep |
| `min_height` | `0.02` | EE floor in controller | Keep |
| `max_target_radius` | `0.78` | XY radius clamp | Keep |
| `max_step_scale clip` | `[0.01, 1.0]` | Clamps policy scale | Keep; note this clips `descend_max_step_scale=2` |

### D. Safety / Workspace

| Constant | Current | Role | Recommendation |
|---|---:|---|---|
| `workspace_min` | `[0.05, -0.50, 0.005]` | Global lower workspace bound | Keep |
| `workspace_max` | `[0.85, 0.50, 0.95]` | Global upper workspace bound | Keep |
| `max_move_norm` | `0.05` | Safety move clamp | Keep |
| `max_ee_target_radius` | `0.85` | Safety radial clamp | Keep |
| `simulator.workspace_min/max` | same defaults | Mocap clamp in sim | Keep |

## 3) High-Impact Notes

1. `descend_max_step_scale=2` does not produce 2x speed because controller clamps `max_step_scale` to `1.0`.
2. Two FSM constants are currently unused (`descend_xy_threshold`, `descend_timeout_xy_threshold`) and can mislead tuning.
3. `descend_xy_priority_threshold=0.015` is active and materially changes descend behavior.
4. Current thresholds are internally consistent for ongoing discussion; no immediate constant reset required before next grip-behavior design pass.

## 4) Next-Step Reminder (No Code Change Yet)

For better grip later, use this behavior plan:
- Start `Close` only from `position_ok`.
- Require gripper truly closed/ready before lift progression.
- Do not leave `Close` while gripper is still moving.
- Keep fixed grip-start Z offset after stable XY.
