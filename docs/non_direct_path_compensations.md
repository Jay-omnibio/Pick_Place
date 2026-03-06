# Non-Direct Path Logic and Compensations

This file lists behavior components that intentionally make motion differ from a pure straight-line "go directly to object" policy.

Status note:
1. Place-side behavior (`Transit -> MoveToPlaceAbove -> DescendToPlace`) is currently under active tuning.
2. Place-side entries in this file describe design intent and active mechanisms, not final performance guarantees.

## Direct Path Baseline (Reference)

Direct path means:
1. Move end-effector along shortest Cartesian error vector to the phase reference.
2. No watchdog offsets, no arc/side-lock, no timeout extensions, no recovery priors.
3. Minimal gating (only strict success thresholds).

Everything below is an intentional compensation beyond that baseline.

## Compensation Components

| Component | Where Implemented | Why It Exists | Main Config Keys |
|---|---|---|---|
| Object-local reference conversion + side-sign selection | `inference_interface.py` | Keep pick references tied to object yaw and choose safer approach side relative to base. | `reach_obj_rel`, `align_obj_rel`, `descend_obj_rel` |
| Reach turn-lock / arc guidance | `inference_interface.py`, `inference/action_selection.py` | Avoid unstable direct pushes for large angular gaps by using curved approach behavior. | `reach_arc_angle_trigger_deg`, `reach_arc_release_deg`, `reach_arc_enabled`, `reach_arc_max_theta_step_deg`, `reach_arc_min_radius`, `reach_arc_max_radius` |
| Reach yaw-align gate/hold | `inference_interface.py` | Prevent phase flapping near reach completion and stabilize orientation before descend. | `reach_yaw_align_xy_enter`, `reach_yaw_align_steps` |
| Align settle gate (merged from old hold stage) | `inference_interface.py`, `inference/action_selection.py` | Keep `Align -> Descend` robust by requiring consecutive stable-ready checks and small near-threshold correction before descent. | `align_settle_steps`, `align_settle_step` |
| Hysteresis/debounce phase gates | `inference_interface.py` | Stop threshold chatter from noisy observations. | `approach_threshold`, `align_threshold`, `descend_x_threshold`, `descend_y_threshold`, `descend_z_threshold` |
| Observation confidence gate | `inference_interface.py` | Block transitions when data quality is weak. | `confidence_enabled`, `confidence_min_for_phase_change`, `confidence_alpha`, `obj_jump_confidence_scale`, `target_jump_confidence_scale`, `contact_flip_confidence_penalty` |
| VFE gate/recovery signals | `inference_interface.py`, `agent/ai_behavior_tree.py` | Penalize high uncertainty/error and allow BT recovery when persistent. | `vfe_enabled`, `vfe_epistemic_weight`, `vfe_phase_change_enabled`, `vfe_phase_change_max`, `vfe_recover_enabled`, `vfe_recover_threshold`, `vfe_recover_steps` |
| Descend/place timeout extensions and stall checks | `inference_interface.py` | Avoid premature failure when still making progress; retry when truly stalled. | `descend_progress_eps`, `descend_stall_steps`, `descend_timeout_extension_steps`, `descend_max_timeout_extensions`, `place_descend_progress_eps`, `place_descend_stall_steps`, `place_descend_timeout_extension_steps`, `place_descend_max_timeout_extensions` |
| Preplace watchdog/timeout extensions | `inference_interface.py` | Prevent `MoveToPlaceAbove` from hanging forever when near-goal or still improving. | `preplace_max_steps`, `preplace_progress_eps`, `preplace_stall_steps`, `preplace_timeout_extension_steps`, `preplace_max_timeout_extensions`, `preplace_timeout_xy_threshold`, `preplace_timeout_z_threshold` |
| BT recovery priors (Rescan/Reapproach/SafeBackoff) | `agent/ai_behavior_tree.py` | Structured retries using semantic branches instead of random retries. | `bt_set_priors_enabled`, `bt_branch_retry_cap`, `bt_global_recovery_cap`, `bt_rescan_hold_steps`, `bt_reapproach_offset_xy`, `bt_safe_backoff_hold_steps`, `bt_safe_backoff_z_boost` |
| Controller command smoothing + safety correction | `control/controller.py` | Reduce jitter and prevent stale-direction IK output under conflicting objectives. | `move_smoothing`, `max_step`, `max_joint_step`, orientation/top-down gains |
| Contact hysteresis + gripper-ready checks | `inference_interface.py` | Avoid contact flicker and unreliable close/open transitions. | `contact_on_count`, `contact_off_count`, `grip_close_ready_max_width`, `grip_ready_width_tol`, `grip_ready_speed_tol` |
| Dynamic gripper-open gate sync | `agent/agent_loop.py`, `control/controller.py`, `inference_interface.py` | Keep AI open-ready gate aligned with controller size-based open target (avoid Align timeout mismatch). | `gripper_size_based_open`, `gripper_open_width`, `gripper_open_min_width`, `gripper_open_clearance`, `grip_open_target` (runtime) |
| Place yaw gate + release verification | `inference_interface.py` | Ensure stable placement orientation and object detach/stability before retreat. | `place_goal_yaw_enabled`, `place_goal_yaw_threshold_deg`, `release_verify_enabled`, `release_detach_hold_steps`, `release_stable_hold_steps`, `release_obj_xy_threshold`, `release_obj_z_threshold`, `release_obj_speed_threshold` |
| Optional RxInfer belief backend | `inference/rxinfer_beliefs.jl`, `inference_interface.py` | Replace simple EMA-only belief update with Bayesian one-step posterior update while keeping Python fallback. | `rxinfer_enabled`, `rxinfer_process_noise_ee`, `rxinfer_process_noise_obj`, `rxinfer_process_noise_target`, `rxinfer_obs_noise_ee`, `rxinfer_obs_noise_obj`, `rxinfer_obs_noise_target`, `rxinfer_min_variance` |

## Notes for Tuning

1. Tune only one component group at a time (for example, Reach arc or BT recovery offsets), then run the same scenario matrix.
2. Keep direct-path baseline metrics (reach success, steps to lift, retries) as reference before enabling extra compensation.
3. If two compensations are active on the same phase, prefer reducing one first instead of increasing both.
4. For post-transit placement, keep topdown objective enabled and only gate yaw when far in XY to avoid sideways tilt-first behavior.
