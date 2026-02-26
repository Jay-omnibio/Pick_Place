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

## Assets and simulation setup

- **panda_mocap.xml**: Put your Panda robot MuJoCo model in `assets/` as `assets/panda_mocap.xml`.
- **Place target**: configured in `config/common_robot.yaml` under `task_shared`.
- **Loop rate**: default control loop is ~50 Hz in `run_pick_place.py`.

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
