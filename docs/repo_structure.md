# Repository Structure (AI-Only Runtime)

## Runtime Entry
- `run_pick_place.py`: main simulation runner and wiring for config/simulator/backends/agent.

## Agent and Decision
- `agent/agent_loop.py`: closed-loop runtime (`sense -> infer -> BT -> action -> act`).
- `agent/ai_behavior_tree.py`: phase orchestration and recovery branching.
- `inference_interface.py`: belief update integration point.
- `inference/action_selection.py`: action selection logic.
- `inference/action_selection.jl`: optional Julia path.
- `inference/rxinfer_beliefs.jl`: optional RxInfer belief update path.

## Robot Interface and Control
- `backends/sensor_backend.py`: sensor abstraction.
- `backends/actuator_backend.py`: actuator abstraction.
- `control/controller.py`: IK + motion command application.
- `state/state_estimator.py`: motion/state estimation helpers.

## Simulation and Safety
- `env/simulator.py`: MuJoCo simulator wrapper.
- `env/sensors.py`: observation extraction/noise model.
- `safety/safety_checker.py`: runtime safety checks.

## Config
- `config/common_robot.yaml`: shared runtime/controller/target config.
- `config/active_inference_config.yaml`: AI + BT + phase/action tuning.
- `config/runtime_loader.py`: strict runtime config loading and validation.
- `config/sensor_config.yaml`: observation/noise filtering config.
- `config/safety_config.yaml`: safety limit config.

## Diagnostics
- `tools/plot_run_metrics.py`: run plotting utility.
- `tools/analyze_run_diagnostics.py`: run diagnostics.
- `tools/run_batch_eval.py`: batch episode evaluator.
- `tools/run_position_sweep.py`: multi-scenario object-pose sweep runner.
- `logs/`: generated CSVs and plots.

## Design Notes
- `docs/ai_bt_partwise_findings.md`: deep part-by-part status review and priorities.
- `docs/non_direct_path_compensations.md`: map of non-direct-path behavior logic and tuning knobs.
