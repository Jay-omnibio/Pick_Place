# Runtime System Flow

This doc shows how the current code runs end-to-end, which files call which files, and how data moves in both `fsm` and `active_inference` modes.

## 1) Top-Level Startup Flow

```mermaid
flowchart TD
    A[run_pick_place.py] --> B[config/runtime_loader.py]
    B --> B1[config/common_robot.yaml]
    B --> B2[config/fsm_config.yaml]
    B --> B3[config/active_inference_config.yaml]

    A --> C[env/simulator.py]
    C --> C1[assets/pick_and_place.xml]
    C1 --> C2[assets/panda_mocap.xml]

    A --> D[control/controller.py]
    A --> E[backends/sensor_backend.py]
    E --> E1[env/sensors.py]
    E --> E2[perception/observation_filter.py]
    E --> E3[config/sensor_config.yaml]

    A --> F[safety/safety_checker.py]
    F --> F1[config/safety_config.yaml]

    A --> G[backends/actuator_backend.py]
    G --> D
    G --> F

    A --> H[agent/agent_loop.py]
    H --> I[logs/run_*.csv]
```

## 2) Main Runtime Loop (Per Step)

`run_pick_place.py` loop:
1. `agent.step()`
2. `agent.is_terminal()` check
3. `simulator.step()`
4. sleep `0.02` (about 50 Hz)

```mermaid
sequenceDiagram
    participant R as run_pick_place.py
    participant A as agent/agent_loop.py
    participant S as backends/sensor_backend.py
    participant M as env/simulator.py
    participant X as backends/actuator_backend.py
    participant C as control/controller.py
    participant K as safety/safety_checker.py

    R->>A: step()
    A->>M: get_state()
    A->>S: get_observation(sim_state)
    A->>A: mode branch (FSM or AI)
    A->>A: logging/events/heartbeat/CSV
    A->>X: apply_action(action)
    X->>K: check(action, current_ee)
    K-->>X: safe action (or None)
    X->>C: apply_action(action)
    C->>M: set_ee_position / IK + gripper command
    R->>M: step()
```

## 3) Mode Branch Inside `agent.step()`

```mermaid
flowchart TD
    A0[Observation ready] --> A1{control_mode}
    A1 -->|fsm| F0[step_fsm in tasks/pick_place_fsm.py]
    F0 --> F1[policy.act in policies/scripted_pick_place.py]
    F1 --> O0[Common post-processing]

    A1 -->|active_inference| I0[infer_beliefs in inference_interface.py]
    I0 --> I1[BT tick in agent/ai_behavior_tree.py]
    I1 --> I2{BT recover?}
    I2 -->|yes| I3[recover_belief reset to Reach]
    I2 -->|no| I4[continue]
    I3 --> I5[select_action in inference/action_selection.py]
    I4 --> I5
    I5 --> I6[detect-only risk update in agent_loop]
    I6 --> O0

    O0 --> O1[_maybe_escape_stall]
    O1 --> O2[_log_step + _log_events + heartbeat]
    O2 --> O3[actuator_backend.apply_action]
```

## 4) Active-Inference Phase Flow

Main phase path currently used by AI mode:

```mermaid
flowchart LR
    R[Reach] --> A[Align]
    A --> P[PreGraspHold]
    P --> D[Descend]
    D --> C[CloseHold]
    C --> L[LiftTest]
    L --> T[Transit]
    T --> M[MoveToPlaceAbove]
    M --> DP[DescendToPlace]
    DP --> O[Open]
    O --> RT[Retreat]
    RT --> DN[Done]

    D -.timeout/fail.-> R
    C -.grasp fail.-> R
    L -.drop/fail.-> R
    T -.lost grasp.-> R
    M -.lost grasp.-> R
    DP -.lost grasp.-> R
```

Notes:
- BT supervises this flow and can request reset/retry when progress stalls.
- Risk detection is detect-only now (warning logs, no forced branch).

## 5) FSM Phase Flow (Reference)

FSM mode uses `tasks/pick_place_fsm.py` and `policies/scripted_pick_place.py`.

For full FSM diagram, see:
- `docs/fsm_state_flow_diagram.md`
- `docs/fsm_phase_behavior.md`

## 6) Data Contracts

### Observation dict (sensor -> agent)
- `o_ee`: end-effector world position (with sensor noise)
- `o_obj`: object relative vector (EE frame)
- `o_target`: place target relative vector (EE frame)
- `o_grip`: gripper opening width
- `o_contact`: object-gripper contact flag
- `o_obj_yaw`: object yaw

### Belief dict (AI mode)
Built in `inference_interface.py`, includes:
- state means: `s_ee_mean`, `s_obj_mean`, `s_target_mean`
- phase state/timers
- confidence state: `obs_confidence`
- release verification state: `release_contact_counter`, `release_warning`

### Action dict (agent -> actuator)
- motion: `move` or `ee_target_pos`
- grip: `grip` (`1` close, `-1` open, `0` hold)
- optional orientation flags:
  - `enable_yaw_objective`
  - `yaw_target`
  - `yaw_pi_symmetric`
  - `enable_topdown_objective`

## 7) File-to-File Call Path

| Caller | Calls | Why |
|---|---|---|
| `run_pick_place.py` | `load_runtime_sections` | Strict config load/validation |
| `run_pick_place.py` | `MujocoSimulator` | Sim world/model lifecycle |
| `run_pick_place.py` | `EEController` | Low-level command execution |
| `run_pick_place.py` | `ActiveInferenceAgent` | Main decision loop owner |
| `ActiveInferenceAgent.step` | `SimSensorBackend.get_observation` | Get filtered observation |
| `ActiveInferenceAgent.step` (FSM) | `step_fsm` + `ScriptedPickPlacePolicy.act` | FSM decision path |
| `ActiveInferenceAgent.step` (AI) | `infer_beliefs` + `AIPickPlaceBehaviorTree.tick` + `select_action` | AI decision path |
| `SimActuatorBackend.apply_action` | `SafetyChecker.check` | Enforce motion safety before control |
| `SimActuatorBackend.apply_action` | `EEController.apply_action` | Execute final safe action |
| `EEController.apply_action` | `MujocoSimulator` methods | Drive EE and gripper in sim |

## 8) Where "Model" Logic Lives

There are multiple model layers:

1. **Physics model**: MuJoCo XML scene and robot (`assets/*.xml`)  
2. **Belief model**: phase/belief update logic (`inference_interface.py`)  
3. **Action-selection model**: EFE-style chooser (`inference/action_selection.py`, optional Julia bridge in `inference/action_selection.jl`)  
4. **Execution model**: IK + constraints (`control/controller.py`)  
5. **Safety model**: bounds and action validation (`safety/safety_checker.py`)  


