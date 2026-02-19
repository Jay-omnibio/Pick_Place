# Robotics Upgrade Plan (Execution-Focused)

## Goal
Improve the robotics stack quality (timing, control robustness, hardware readiness) without redesigning the AI model first.

## Scope
- In scope: sensing, state estimation, control, safety, runtime monitoring, BT/AI integration boundaries.
- Out of scope: replacing active-inference math/model internals.

## Quick Verdict On Suggested Ideas
| Idea | Value for this project now | Decision |
|---|---|---|
| Add `world_model/state_estimator` | High | Do now |
| Add admittance/impedance layer | Medium now, High for hardware | Do phased version now (gain scheduling), full admittance later |
| Split BT nodes into separate folder files | Medium (maintainability) | Do later after behavior stabilizes |
| Add latency monitor in backend | High | Do now |
| Dynamic gain scaling by phase | High | Do now |

## Recommended Hybrid Structure (for this repo)
Most practical structure is:
1. High-level supervisor: BT (`agent/ai_behavior_tree.py`)
2. Mid-level belief + policy: active inference (`inference_interface.py`, `inference/action_selection.py`)
3. Low-level deterministic control: IK + safety (`control/controller.py`, `safety/safety_checker.py`)

This is the best near-term structure for your project because:
- BT gives explicit failure recovery and mission flow.
- Active inference handles uncertainty and target-seeking behavior.
- Controller/safety remain deterministic and hardware-safe.

## Layer Contracts (important)
1. BT layer outputs:
- phase supervision intent
- prior overrides (optional)
- recovery triggers

2. Active inference layer outputs:
- `move`, `grip`
- optional confidence/risk metadata
- optional per-phase gain hints

3. Controller layer consumes:
- command target (`move` or `ee_target_pos`)
- gain/compliance scalars (new)
- hard limits always enforced by safety layer

## Priority Plan

## P0 (Do First)
1. Timing and latency observability
- Add observation timestamp and age tracking.
- Add control loop cycle time and jitter tracking.
- Log stale observation warnings.

Files:
- `env/simulator.py`
- `env/sensors.py`
- `backends/sensor_backend.py`
- `agent/agent_loop.py`

Expected result:
- You can answer: "Was this decision based on fresh data?"

2. State estimator (position + velocity belief)
- Add estimator for object/target/ee with velocity estimate.
- Keep current observation filter; estimator sits after filter.
- Feed estimated state to AI and logs.

Files:
- `state/state_estimator.py` (new)
- `agent/agent_loop.py`
- `config/sensor_config.yaml`

Expected result:
- Better prediction stability during motion and contact transitions.

3. Phase-based control gain scheduling
- Add action fields for gain scaling:
  - `position_gain_scale`
  - `yaw_weight_scale`
  - `topdown_weight_scale`
  - `nullspace_gain_scale`
- Apply these in controller per step.
- Use lower stiffness in grasp/close/open windows.

Files:
- `control/controller.py`
- `inference/action_selection.py`
- `inference/action_selection.jl` (if kept in sync)
- `config/common_robot.yaml`

Expected result:
- Less overshoot/knock in delicate phases, faster reach when safe.

## P1 (After P0)
1. Robotics health monitor
- Add a monitor struct with counters:
  - stale frame count
  - control saturation count
  - high `dq_raw/dq_applied` count
  - unintended contact streak
- Print concise heartbeat summary and write CSV fields.

Files:
- `agent/agent_loop.py`
- `docs/` (metrics definitions)

2. Release verification robustness
- Keep current release check, extend with:
  - "object separated from gripper" condition
  - "object stable near place target" condition

Files:
- `inference_interface.py`
- `agent/agent_loop.py`

## P2 (Later, hardware-focused)
1. Full admittance/impedance control mode
- Add optional force-aware mode in controller.
- Use it only when force/torque sensing is available.

Files:
- `control/controller.py`
- hardware backend adapters

2. BT node modularization
- Split node classes into `agent/bt_nodes/`.

Files:
- `agent/bt_nodes/*.py` (new)
- `agent/ai_behavior_tree.py`

## Test Strategy For Robotics Layer
For each change, run 10 episodes and compare:
1. Lift success rate
2. Done success rate
3. Mean steps to LiftTest and Done
4. Recover count
5. Stale-frame warnings
6. Control saturation rate

## "Hybrid vs Full AI" Practical Guidance
Use hybrid now.

Why:
- You already have BT and deterministic controller safety.
- Full AI phase replacement needs larger redesign of transition semantics.
- Hybrid is standard and production-friendly for manipulation systems.

Move toward "more AI" only after:
1. P0 metrics are stable
2. Lift/place success is consistent
3. Timing/latency behavior is understood

