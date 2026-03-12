# Active Inference Pick-and-Place (MuJoCo)

This repo runs a pick -> lift -> place pipeline in MuJoCo with an active-inference + behavior-tree runtime.

Core modules:

- **Simulation**: `env/`
- **Sensors**: `env/sensors.py` (noisy observations)
- **Perception**: `perception/` (Kalman filter, outlier rejection, latency handling)
- **Backends**: `backends/` (SensorBackend, ActuatorBackend for sim/real swap)
- **Safety**: `safety/` (workspace bounds, velocity limits)
- **Controller**: `control/controller.py`
- **Agent**: `agent/agent_loop.py`
- **Inference**: `inference_interface.py`, `inference/action_selection.py`, `agent/ai_behavior_tree.py`

The runtime is **active-inference only** (FSM mode removed).

## Documentation Map

- Architecture: `docs/ai_bt_architecture.md`
- Current vs hybrid architecture graphs: `docs/architecture_graphs_current_vs_hybrid.md`
- BT/phase design-lock draft: `docs/bt_phase_architecture_draft.md`
- Project roadmap: `docs/project_roadmap.md`
- Backlog and gates: `docs/implementation_backlog.md`
- Scenario validation matrix: `docs/scenario_matrix.md`
- Recovery/failure policy: `docs/recovery_failure_agreed_solutions.md`
- Run commands: `docs/python3_run_commands.md`
- Executive/meeting brief: `docs/project_brief.md`
- 5-slide meeting deck: `docs/meeting_deck_5_slides.md`
- Full concepts and layers explainer: `docs/system_concepts_and_layers.md`

## Release Notes (2026-03-12)

- BT task-switch path for place-side drop is active:
  - emits `object_dropped`
  - switches task intent to `PICK`
  - routes back to pick entry (`Reach`)
  - resets `retry_count` on this task switch
- Align yaw gate added before `Align -> Descend`:
  - uses `align_pick_yaw_gate_enabled`
  - `align_pick_yaw_threshold_deg`
  - `align_pick_yaw_hold_steps`
- EE yaw observation (`o_ee_yaw`) now aligns with controller yaw definition (EE-site axis via `yaw_axis`).
- Reach uses translation-first orientation policy:
  - yaw objective is reduced/disabled while far
  - final yaw-align hold near transition remains active
  - topdown objective stays enabled with distance-based `topdown_weight_scale`
- Place keepout logic hardened:
  - no-reentry projection in keepout zone
  - keepout applied after local move search too
- `MoveToPlaceAbove` topdown emphasis increased for this release.
- CSV runtime logs now include:
  - `ai_belief_ee_yaw`
  - `ai_align_pick_yaw_error`

## Assets and simulation setup

- **panda_mocap.xml**: Put your Panda robot MuJoCo model in `assets/` as `assets/panda_mocap.xml`.
- **Place target**: configured in `config/common_robot.yaml` under `task_shared`.
- **Loop rate**: default control loop is ~50 Hz in `run_pick_place.py`.

1. **Clone Dependencies**: 
   The robot models are sourced from [Google DeepMind's MuJoCo Menagerie](https://github.com).
   ```bash
   git clone https://github.com/google-deepmind/mujoco_menagerie.git
   ```

### ⚙️ Configure Asset Paths

MuJoCo requires correct relative paths to load meshes and textures. You must ensure `panda_mocap.xml` can find its assets:

*   **Option A**: In `assets/panda_mocap.xml`, update the `<compiler meshdir="..." />` or individual asset paths to point to your local `mujoco_menagerie/franka_emika_panda/assets/` directory.
*   **Option B**: Move/copy the `panda_mocap.xml` file directly into the `mujoco_menagerie/franka_emika_panda/` folder so it sits alongside its native assets.

### ✅ Verify Main Import

1.  Open `pick_and_place.xml`.
2.  Check the `<include file="..." />` tag to ensure it correctly points to the location of your `panda_mocap.xml`.

## Run the simulation

```bash
python3 run_pick_place.py
```

Command reference with `python3` examples and flags:
- `docs/python3_run_commands.md`

Environment variables:

- `LOG_EVERY_STEPS` (default `100`): console log frequency.
- `LOG_CONTACT_EVENTS=1`: print contact transitions.
- `CTRL_DEBUG_EVERY_STEPS` (default `100`): controller debug prints.

## Plot the latest run

The agent writes `logs/run_YYYYMMDD_HHMMSS.csv`. To generate plots:

```bash
python tools/plot_run_metrics.py
```

Plots are saved under `logs/plots/run_.../`.

## Batch Evaluation

Run repeated episodes and get one summary report with P0 gate checks:

```bash
python tools/run_batch_eval.py --episodes 10 --save-per-run-report
```

Outputs:
- Batch summary: `logs/reports/batch_eval_YYYYMMDD_HHMMSS.md`
- Per-run diagnostics (optional): `logs/reports/run_..._diagnostics.md`

Run multi-scenario sweep (different object positions/yaws) without editing XML each time:

```bash
python tools/run_position_sweep.py --scenarios "A1:0.40,0.00,0.20,0;A2:0.50,0.00,0.20,0;A3:0.60,0.00,0.20,0" --episodes 10 --timeout-sec 240 --save-per-run-report
```

## Dependencies

Install Python deps:

```bash
pip install -r requirements.txt
```

Notes:

- You need a working MuJoCo install for the `mujoco` Python package.
- Julia/PyJulia is optional and not required for the baseline.
- If enabled, RxInfer-backed beliefs use Julia through `inference.rxinfer_enabled`.

## Config

- `config/common_robot.yaml`: shared runtime config (`run`, `controller`, `task_shared`).
  - Gripper hybrid-open tuning is in `controller`:
    - `gripper_open_width` (max open)
    - `gripper_open_min_width` (minimum pre-open)
    - `gripper_open_clearance` (object width margin)
    - `gripper_size_based_open` / `gripper_open_unknown_full_open`
- `config/active_inference_config.yaml`: active inference + BT + phase/action tuning.
  - Optional RxInfer belief path knobs: `rxinfer_*` keys in `inference`.
- `config/sensor_config.yaml`: sensor noise and observation filtering.
- `config/safety_config.yaml`: workspace bounds and velocity limits.
