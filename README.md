# Active Inference Pick-and-Place (MuJoCo)

This repo runs a pick → lift → place baseline in MuJoCo using a clean separation of:

- **Simulation**: `env/`
- **Sensors**: `env/sensors.py` (noisy observations)
- **Perception**: `perception/` (Kalman filter, outlier rejection, latency handling)
- **Backends**: `backends/` (SensorBackend, ActuatorBackend for sim/real swap)
- **Safety**: `safety/` (workspace bounds, velocity limits)
- **Task FSM**: `tasks/pick_place_fsm.py`
- **Policy**: `policies/scripted_pick_place.py`
- **Controller**: `control/controller.py`
- **Agent**: `agent/agent_loop.py`

The **active inference / RxInfer** pieces are kept in `inference/` for later integration.

## Assets and simulation setup

- **panda_mocap.xml**: Put your Panda robot MuJoCo model in `assets/` (same dir as `pick_and_place.xml`). The scene expects `assets/panda_mocap.xml`. If it lives elsewhere, create a symlink: `ln -s /path/to/panda_mocap.xml assets/panda_mocap.xml`
- **Scene**: Object at (0.6, 0, 0.2), place target at (0.4, 0, 0.2), 50 Hz control loop.

## Run the simulation

```bash
python run_pick_place.py
```

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

## Dependencies

Install Python deps:

```bash
pip install -r requirements.txt
```

Notes:

- You need a working MuJoCo install for the `mujoco` Python package.
- Julia/PyJulia is optional and not required for the baseline.

## Config

- `config/sensor_config.yaml`: sensor noise, observation filter (Kalman, outlier rejection, latency).
- `config/safety_config.yaml`: workspace bounds, max velocity.
- Set `observation_filter.enabled: false` in sensor config to disable the filter.

