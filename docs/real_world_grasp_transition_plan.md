# Real-World Grasp Transition Plan

## Goal
Build a robust pick-and-place pipeline in current MuJoCo setup, then transition the same logic to a real robot with minimal rewrites.

## Phase A: Improve Current Setup (No Major Rewrite)

### 1. Sensor Realism
- Extend `env/sensors.py` to output camera-like object pose observations (noise, occasional dropouts).
- Keep current simulator ground truth internally, but agent uses only noisy observations.

### 2. Explicit Task State Machine
- Implement/complete staged FSM in `inference_interface.py`:
  - `ReachHigh`
  - `PreGraspHold`
  - `Descend`
  - `CloseHold`
  - `LiftTest`
  - `Place`
- Keep phase transitions guarded by clear conditions (position, contact, gripper readiness, timers).

### 3. Grasp Success Criteria
- Do not treat "close command sent" as success.
- Success should require:
  - stable contact for N steps,
  - gripper motion slowing/stopping,
  - short lift-test where object follows EE.

### 4. Retry Policy
- On failed lift-test:
  - open gripper,
  - return above object,
  - retry with small XY/yaw offset.
- Add max retry count and reset/re-detect when exceeded.

### 5. Keep Controller Role Clean
- `control/controller.py` should execute commands (IK + orientation + gripper), not decide task phase.
- Task logic remains in `inference_interface.py`.

## Phase B: Real Robot Transition (Same Architecture)

### 1. Replace Observation Sources
- Replace simulated object pose with real RGB-D / vision pose stream.
- Replace simulated contact with gripper force/current + finger encoder signals.

### 2. Keep Existing FSM and Interfaces
- Reuse same state machine and action interface.
- Only swap sensor backends and actuator backend.

### 3. Add Safety Layer
- Joint limits, workspace limits, velocity limits.
- Force limits and timeout aborts.
- Emergency stop behavior.

## File-Level Action Map
- `env/sensors.py`: camera-like noisy observations, dropout simulation.
- `inference_interface.py`: robust phase gates, timers, retry logic.
- `inference/action_selection.py`: phase-specific motion policies.
- `control/controller.py`: motion/gripper execution with orientation constraints.
- `agent/agent_loop.py`: concise event logs + run metrics.

## Suggested Work Order (Next Session)
1. Finalize FSM transitions (`PreGraspHold`, `Descend`, `CloseHold`, `LiftTest`).
2. Add strict grasp confirmation + retry loop.
3. Run 3 benchmark trials and compare success rate + step count.
4. Tune orientation + descend-only final approach.
5. Freeze MuJoCo baseline, then start real sensor adapter.

## Baseline KPIs
- Reach success rate: percent of runs entering grasp-preparation zone within max steps.
- Grasp success rate: percent of runs where grasp is confirmed (not just close command).
- Lift-test pass rate: percent of runs where object stays with EE during short lift.
- Median steps to reach: median step count to first `Align`/`PreGraspHold` entry.
- Median steps to grasp: median step count to first confirmed grasp.
- Retry burden: average retries per successful pick.

## Acceptance Gates
- Gate A (Reach baseline):
  Reach success >= 90% across 20 runs, with no unstable simulation events.
- Gate B (Grasp baseline):
  Grasp success >= 60% across 20 runs.
- Gate C (Lift reliability):
  Lift-test pass >= 50% across 20 runs.
- Move to real-sensor integration only after Gate A + Gate B pass.

## Experiment Protocol
- Use fixed random seeds for reproducibility.
- Test minimum 3 object placements: center, +Y offset, -Y offset.
- Keep same max step budget and logging config per comparison batch.
- Save CSV + plots for every run in `logs/` and `logs/plots/`.
- Compare against previous baseline by run date/time and KPI table.

## Real-Robot Safety Checklist
- Hard workspace bounds and minimum table clearance.
- Joint velocity/acceleration caps enabled.
- Gripper force/current limit configured.
- Per-phase timeout and global episode timeout.
- Emergency stop path tested before autonomous run.
- Recovery behavior defined for camera loss / pose jump.
