# Scenario Matrix (Real-Robot Style Validation)

Last updated: 2026-03-04

Use this to verify that tuning works across workspace and not only one easy pose.

## Goal
Validate robust behavior for:
1. object position changes
2. object yaw changes
3. sensor noise changes

## How To Run
For each scenario:
1. Set object pose in `assets/pick_and_place.xml` (`body name="obj"` `pos` and `quat`).
2. Keep target pose fixed unless scenario says otherwise.
3. Run:

```bash
python3 tools/run_batch_eval.py --episodes 10 --timeout-sec 240 --save-per-run-report
```

4. Save batch summary path in your notes.

Optional automation (run multiple scenarios without editing XML each time):

```bash
python3 tools/run_position_sweep.py ^
  --scenarios "A1:0.40,0.00,0.20,0;A2:0.50,0.00,0.20,0;A3:0.60,0.00,0.20,0" ^
  --episodes 10 --timeout-sec 240 --save-per-run-report
```

This uses env overrides (`OBJ_WORLD_XYZ`, `OBJ_WORLD_QUAT_WXYZ`) per scenario and writes a sweep summary in `logs/reports/`.

Latest reference sweep:
1. `logs/reports/position_sweep_20260301_134501.md`
2. Use it as baseline when comparing new tuning changes.

## Pose Conventions
- MuJoCo object quaternion in this file is `[w x y z]`.
- Pure yaw `theta` around Z:
  - `w = cos(theta/2)`
  - `z = sin(theta/2)`
  - `x = y = 0`

Common yaw quaternions:
1. `0 deg`: `[1.0000, 0, 0, 0]`
2. `+45 deg`: `[0.9239, 0, 0, 0.3827]`
3. `+90 deg`: `[0.7071, 0, 0, 0.7071]`
4. `-45 deg`: `[0.9239, 0, 0, -0.3827]`

## Scenario Set

### Tier A (Nominal Workspace)
| ID | obj pos (x,y,z) | obj yaw | noise profile |
|---|---|---|---|
| A1 | (0.40, 0.00, 0.20) | 0 deg | baseline |
| A2 | (0.50, 0.00, 0.20) | 0 deg | baseline |
| A3 | (0.60, 0.00, 0.20) | 0 deg | baseline |

### Tier B (Lateral Offset)
| ID | obj pos (x,y,z) | obj yaw | noise profile |
|---|---|---|---|
| B1 | (0.50, +0.12, 0.20) | 0 deg | baseline |
| B2 | (0.50, -0.12, 0.20) | 0 deg | baseline |
| B3 | (0.60, +0.12, 0.20) | 0 deg | baseline |

### Tier C (Yaw Variation)
| ID | obj pos (x,y,z) | obj yaw | noise profile |
|---|---|---|---|
| C1 | (0.40, 0.00, 0.20) | +45 deg | baseline |
| C2 | (0.50, 0.00, 0.20) | +90 deg | baseline |
| C3 | (0.60, 0.00, 0.20) | -45 deg | baseline |

### Tier D (Noise Stress)
Use center pose `(0.50, 0.00, 0.20)`, yaw `0 deg`.

| ID | sensor config changes in `config/sensor_config.yaml` |
|---|---|
| D1 | baseline (current file as-is) |
| D2 | `object_sensor.noise_std=0.03`, `target_sensor.noise_std=0.03` |
| D3 | same as D2 + `observation_filter.enabled=false` |

## Pass/Fail Gates

### Per Scenario (10 episodes)
1. `Done` rate:
 - Tier A: `>= 9/10`
 - Tier B: `>= 8/10`
 - Tier C: `>= 7/10`
 - Tier D: `>= 6/10`
2. Hard stuck Reach (`Reach` > 1200 consecutive rows): `0` episodes for Tier A/B, `<= 1` for Tier C/D.
3. Total `reach_stall` retries per 10 episodes: `<= 8`.

### Overall Exit Rule
1. All Tier A scenarios pass.
2. At least 2/3 pass in each of Tier B/C.
3. At least 2/3 pass in Tier D.

## Results Template
Fill this after each scenario run.

| Scenario | Done rate | hard-stuck count | reach_stall total | Median LiftTest step | P90 LiftTest step | Pass/Fail | Batch report path |
|---|---:|---:|---:|---:|---:|---|---|
| A1 |  |  |  |  |  |  |  |
| A2 |  |  |  |  |  |  |  |
| A3 |  |  |  |  |  |  |  |
| B1 |  |  |  |  |  |  |  |
| B2 |  |  |  |  |  |  |  |
| B3 |  |  |  |  |  |  |  |
| C1 |  |  |  |  |  |  |  |
| C2 |  |  |  |  |  |  |  |
| C3 |  |  |  |  |  |  |  |
| D1 |  |  |  |  |  |  |  |
| D2 |  |  |  |  |  |  |  |
| D3 |  |  |  |  |  |  |  |
