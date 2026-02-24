# Motion Wave Isolation Checklist

Goal: identify why EE path looks like right-left sine wave before converging.

Use this when motion is oscillatory in both `fsm` and `active_inference`.

## Keep Constant Across Tests
- Same object start pose.
- Same place target.
- Same random seed/setup (if applicable).
- Same `log_every_steps`.
- Run each test 3 times.

Record for every run:
- CSV path.
- Final phase.
- Whether path is smooth or wave-like.
- Peak `true_reach_xy_error`.
- Number of sign flips in `action_move_x` and `action_move_y` during Reach.

## Baseline Run
1. Keep current configs unchanged.
2. Run once in `active_inference`.
3. Run once in `fsm`.

Commands:
```bash
python3 run_pick_place.py --no-pause
```

Generate plots:
```bash
python tools/plot_run_metrics.py --csv logs/run_YYYYMMDD_HHMMSS.csv
python tools/analyze_run_diagnostics.py --csv logs/run_YYYYMMDD_HHMMSS.csv --save-report --plot
```

## Test A: Yaw Coupling Isolation
Hypothesis: yaw objective is injecting lateral oscillation.

Edit `config/common_robot.yaml`:
- `controller.yaw_weight: 0.0`
- `controller.yaw_weight_grasp: 0.0`

Run AI and FSM again.

Interpretation:
- If wave strongly reduces: yaw coupling is a major cause.
- If no meaningful change: yaw is not primary.

## Test B: Smoothing Lag Isolation
Hypothesis: command smoothing causes phase lag and snake-like path.

Edit `config/common_robot.yaml`:
- `controller.move_smoothing: 0.0` (or very low like `0.05`)

Keep yaw weights as baseline (or keep from Test A if you want stacked tests, but note it clearly).

Interpretation:
- If wave reduces: smoothing lag is a major cause.
- If no change: smoothing is secondary.

## Test C: Arc/Turn Strategy Isolation
Hypothesis: arc-turn logic produces side-to-side convergence.

For AI mode, edit `config/active_inference_config.yaml`:
- `action_selection.reach_arc_enabled: false`

For FSM mode, edit `config/fsm_config.yaml`:
- `policy.reach_arc_enabled: false`

Interpretation:
- If wave reduces: arc strategy is a major cause.
- If no change: look more at controller coupling.

## Test D: Target Frame Rotation Isolation
Hypothesis: per-step object yaw changes rotate the world target and create zig-zag.

Use fixed object yaw scene (no yaw randomization) and rerun baseline.

Interpretation:
- If wave reduces with fixed yaw: target-frame rotation is contributing.

## What To Look At In Logs
- `[CtrlDbg] req_move` vs `applied_move`
  - Large persistent mismatch means controller limits/coupling are shaping path.
- `[CtrlDbg] dq_raw` vs `dq_applied`
  - Frequent saturation indicates IK pressure and possible lateral artifacts.
- `[HB] x/y/xy` trend
  - Smooth convergence should show mostly monotonic reduction (small noise okay).

## Pass Criteria (Practical)
- Reach path visually smooth (no repeated right-left swings).
- `true_reach_xy_error` trends down with few reversals.
- Reach completes without repeated retry loops.

## Revert Checklist
After experiments, restore:
- `config/common_robot.yaml` (`yaw_weight`, `yaw_weight_grasp`, `move_smoothing`)
- `config/active_inference_config.yaml` (`action_selection.reach_arc_enabled`)
- `config/fsm_config.yaml` (`policy.reach_arc_enabled`)

Keep one short summary table:
- Test name
- Changed keys
- AI result
- FSM result
- Conclusion
