# Project File Map

This is a quick reference for what each important file is used for.

## Entry Point

| File | Use |
|---|---|
| `run_pick_place.py` | Main runner. Loads config, builds simulator/agent/controller stack, runs loop, stops on terminal FSM phase. |

## Agent / Task / Policy / Control (Main Runtime)

| File | Use |
|---|---|
| `agent/agent_loop.py` | Core loop logic: sense -> estimate -> FSM -> policy -> actuator, logging, heartbeat/phase prints, CSV writing. |
| `tasks/pick_place_fsm.py` | Finite state machine and transitions (`ReachAbove`, `Descend`, `Close`, `LiftTest`, `Transit`, `MoveToPlaceAbove`, `DescendToPlace`, `Open`, `Retreat`, `Done`, `Failure`). |
| `policies/scripted_pick_place.py` | Phase-wise motion targets and grip commands. Contains reach/descend behavior and pose objectives. |
| `control/controller.py` | EE and gripper controller (IK, smoothing, motion clamps, orientation objectives, gripper mode/state handling). |
| `safety/safety_checker.py` | Safety gates on commands (workspace bounds, speed/limit checks). |

## Environment / Simulation / Observation

| File | Use |
|---|---|
| `env/simulator.py` | MuJoCo simulator wrapper, stepping, state access. |
| `env/sensors.py` | Sensor observation generation/noise interface from simulator state. |
| `backends/sensor_backend.py` | Sensor backend abstraction (sim path currently). |
| `backends/actuator_backend.py` | Actuator backend abstraction (sim path currently). |
| `perception/observation_filter.py` | Optional observation filtering (smoothing/robustness). |

## Config (Split Runtime Configs)

| File | Use |
|---|---|
| `config/common_robot.yaml` | Common runtime config: run mode/log cadence and controller/gripper parameters. |
| `config/fsm_config.yaml` | FSM task + scripted policy parameters (thresholds, holds, retry logic, policy gains). |
| `config/active_inference_config.yaml` | Active-inference parameters for inference and action-selection behavior. |
| `config/runtime_loader.py` | Strict split-config loader/validator for common + fsm + active-inference configs. |
| `config/sensor_config.yaml` | Sensor/perception settings. |
| `config/safety_config.yaml` | Safety constraints. |

## Scene / Robot Assets

| File | Use |
|---|---|
| `assets/pick_and_place.xml` | Main MuJoCo scene (robot, object, target, world setup). |
| `assets/panda_mocap.xml` | Panda robot model used by scene. |

## Inference (For Later Integration)

| File | Use |
|---|---|
| `inference_interface.py` | Python interface boundary for inference integration. |
| `inference/action_selection.py` | Python action selection utilities. |
| `inference/action_selection.jl` | Julia action selection implementation candidate. |
| `inference/generative_model.jl` | Julia generative model for active inference work. |

## Tools / Logs

| File | Use |
|---|---|
| `tools/plot_run_metrics.py` | Converts run CSV logs into plots for analysis. |
| `logs/run_*.csv` | Runtime outputs (phase, errors, action, true states). |
| `logs/plots/run_*/` | Generated figures from log plotting tool. |

## Docs (Planning + Behavior Notes)

| File | Use |
|---|---|
| `docs/current_task_queue.md` | Current work queue and priorities. |
| `docs/fsm_phase_behavior.md` | FSM phase behavior and transition notes. |
| `docs/movement_baseline.md` | Baseline movement behavior reference. |
| `docs/config_audit_notes.md` | Notes on config constants and cleanup/audit findings. |
| `docs/fsm_state_flow_diagram.md` | Diagram/version of FSM flow. |
| `docs/active_inference_readiness_checklist.md` | Checklist before switching to active inference mode. |
| `docs/not_needed_inprovment.md` | Deferred improvements (non-blocking now). |
| `docs/project_roadmap.md` | Roadmap level planning. |
| `docs/real_world_grasp_transition_plan.md` | Transition plan from sim behavior to real-world grasping constraints. |
| `docs/robot_mind_spec.md` | Higher-level design/spec notes. |
| `docs/plots_guide.md` | Plot interpretation and usage guide. |

## Quick "Where To Edit" Guide

| If you want to change... | Edit this file first |
|---|---|
| FSM phase transition condition | `tasks/pick_place_fsm.py` |
| Reach/Descend motion style | `policies/scripted_pick_place.py` |
| IK or gripper behavior | `control/controller.py` |
| Numeric thresholds and behavior values | `config/fsm_config.yaml`, `config/active_inference_config.yaml`, `config/common_robot.yaml` |
| Scene object/target/robot placement | `assets/pick_and_place.xml` |
| Safety limits | `config/safety_config.yaml` + `safety/safety_checker.py` |
| Logging format/CSV fields | `agent/agent_loop.py` |

## Detailed Roles And Descriptions

### Runtime Control Path (what runs each step)

`run_pick_place.py`  
Role: Runtime bootstrap.  
Description: Starts the whole app, loads strict split runtime configs, creates simulator/controller/agent objects, and runs the simulation loop until terminal FSM phase.

`agent/agent_loop.py`  
Role: Coordinator/orchestrator.  
Description: Collects observations, computes errors/metrics, advances FSM state, asks policy for action, sends action to actuator backend, and writes logs/heartbeats/phase transitions.

`tasks/pick_place_fsm.py`  
Role: Task logic/state machine.  
Description: Defines all phases and transition rules (thresholds, counters, holds, retries, contact gating). This is the source of truth for "when phase changes."

`policies/scripted_pick_place.py`  
Role: Motion decision layer for each phase.  
Description: Converts current phase + observations into end-effector target commands and grip commands (ReachAbove, Descend, Close, LiftTest, place-side phases).

`control/controller.py`  
Role: Low-level motion executor.  
Description: Takes high-level EE targets and produces robot joint commands using IK, orientation objectives (yaw/top-down), smoothing, limits, and gripper mode logic.

### Environment And IO

`env/simulator.py`  
Role: MuJoCo integration wrapper.  
Description: Owns model/data stepping, rendering, and exposes robot/object/target state used by sensors and agent.

`env/sensors.py`  
Role: Observation generator.  
Description: Builds observation dictionary from simulator state (including noisy/processed signals depending on config).

`backends/sensor_backend.py`  
Role: Sensor abstraction.  
Description: Provides a stable interface so agent can read observations from sim now and real hardware later with minimal agent changes.

`backends/actuator_backend.py`  
Role: Actuator abstraction.  
Description: Applies policy/controller outputs with safety checks, allowing clean sim/real swapping at the boundary.

`safety/safety_checker.py`  
Role: Command guardrail.  
Description: Validates/clamps unsafe commands (workspace/radius/height/speed constraints) before command reaches controller/simulator.

### Configuration

`config/common_robot.yaml`  
Role: Common runtime config source.  
Description: Contains run settings and controller/gripper parameters shared by control modes.

`config/fsm_config.yaml`  
Role: FSM config source.  
Description: Contains task thresholds, phase timers, retry behavior, and scripted policy parameters.

`config/active_inference_config.yaml`  
Role: Active-inference config source.  
Description: Contains active-inference phase gates, smoothing/contact settings, and action-selection parameters.

`config/runtime_loader.py`  
Role: Strict split-config validator/loader.  
Description: Enforces required keys and rejects unknown keys across common/fsm/active-inference sections, then returns typed runtime config objects.

`config/sensor_config.yaml`  
Role: Sensor/perception tuning.  
Description: Controls observation filtering/noise parameters and related sensor behavior.

`config/safety_config.yaml`  
Role: Safety tuning.  
Description: Defines safety thresholds and limits used by `safety/safety_checker.py`.

### Scene And Robot Model

`assets/pick_and_place.xml`  
Role: Task scene definition.  
Description: Defines world, object, target, table/environment setup, and references robot model for simulation.

`assets/panda_mocap.xml`  
Role: Robot model asset.  
Description: Panda robot structure/joints/actuation used by MuJoCo scene.

### Inference Layer (future integration path)

`inference_interface.py`  
Role: Integration boundary.  
Description: Bridge point where active inference output can plug into current agent/policy flow.

`inference/action_selection.py`  
Role: Python action-selection helpers.  
Description: Contains action selection utilities in Python form for inference experiments.

`inference/action_selection.jl`  
Role: Julia action-selection implementation.  
Description: Julia-side counterpart for active inference experiments.

`inference/generative_model.jl`  
Role: Julia generative model.  
Description: Model definition for active inference work planned after baseline robustness.

### Observability / Analysis

`tools/plot_run_metrics.py`  
Role: Offline analysis tool.  
Description: Reads run CSVs and generates plots to inspect phase behavior, errors, commands, and contact events.

`logs/run_*.csv`  
Role: Ground-truth run record.  
Description: Per-step data for debugging, regression comparison, and tuning decisions.

### Documentation

`docs/current_task_queue.md`  
Role: Working backlog.  
Description: Current agreed tasks, active priorities, and hold/defer items.

`docs/fsm_phase_behavior.md`  
Role: FSM behavior reference.  
Description: Human-readable explanation of what each phase is intended to do and what gates/transitions are expected.

`docs/config_audit_notes.md`  
Role: Config sanity notes.  
Description: Tracks suspicious constants, duplicates, and cleanup decisions from tuning audits.
