# System Concepts and Layers: Full Explanation

Last updated: 2026-03-04

## 1) System Objective
The project implements a full pick-and-place runtime in simulation with:
1. Adaptive low-level control under uncertainty.
2. Structured high-level task orchestration and recovery.
3. Measurable diagnostics for reliability improvement.

Primary objective:
1. Move from scripted behavior to robust policy-driven behavior that can be hardened for real robotics.

## 2) Core Concepts Used

### Active Inference (low-level decision)
1. Maintains beliefs about robot/object/target states.
2. Chooses actions that reduce expected future "badness" (Expected Free Energy, EFE).
3. Balances:
 - Pragmatic term: reduce task error.
 - Epistemic term: reduce uncertainty.

Practical effect:
1. Not only "go to target," but "go to target while staying confident enough to trust transitions."

### Behavior Tree (high-level supervision)
1. Defines mission structure (`Acquire`, `Place`, `Recover`).
2. Monitors progress and risk.
3. Routes to semantic recovery branches when stalled or unsafe.

Practical effect:
1. Keeps runtime explainable and debuggable.
2. Prevents uncontrolled infinite loops.

### Phase Machine (task staging)
1. Breaks task into explicit phases:
 - `Reach`, `Align`, `Descend`, `CloseHold`, `LiftTest`, `Transit`, `MoveToPlaceAbove`, `DescendToPlace`, `Open`, `Retreat`, `Done`.
2. Each phase has phase-specific objectives, thresholds, and watchdog logic.

Practical effect:
1. Easier debugging and targeted tuning by phase.

### Confidence and VFE gating
1. Confidence gate blocks phase transitions when observations are weak.
2. VFE metrics provide a quality signal for supervision and possible recovery triggers.

Practical effect:
1. Fewer brittle transitions caused by noisy flicker around thresholds.

### Structured recovery model
1. Recovery branches:
 - `ReScanTable`
 - `ReApproachOffset`
 - `SafeBackoff`
2. Branch retry caps and global caps are enforced.

Practical effect:
1. Controlled retries with bounded failure behavior.

## 3) Layered Model (What Each Layer Does)

### Layer A: Task and Mission Layer
Files:
1. `agent/ai_behavior_tree.py`

Responsibility:
1. Decide whether to continue normal sequence or recover.
2. Maintain retry/recovery budgets.
3. Manage pick-side vs place-side recovery routing.

### Layer B: Belief and Phase Layer
Files:
1. `inference_interface.py`
2. `inference/rxinfer_beliefs.jl` (optional backend)

Responsibility:
1. Convert observations into belief state (`s_ee_mean`, `s_obj_mean`, `s_target_mean`, covariances).
2. Evaluate phase transition conditions and gates.
3. Track phase timers, progress counters, and failure reasons.

### Layer C: Action Policy Layer
Files:
1. `inference/action_selection.py`
2. `inference/action_selection.jl` (optional path)

Responsibility:
1. Produce phase-aware action commands (`move`, `grip`).
2. Apply EFE-driven local selection and axis-priority behavior.
3. Apply phase-specific orientation/topdown/step-scaling behavior.

### Layer D: Control and Kinematics Layer
Files:
1. `control/controller.py`

Responsibility:
1. Convert task-space command into joint-space command (IK).
2. Apply smoothing, gains, limits, and safety constraints.
3. Manage gripper command tracking.

### Layer E: Sensing and Simulation Layer
Files:
1. `env/simulator.py`
2. `env/sensors.py`
3. `backends/sensor_backend.py`
4. `backends/actuator_backend.py`

Responsibility:
1. Generate and expose observations.
2. Provide swappable interfaces for sim/real.
3. Keep runtime loop independent from concrete hardware transport.

### Layer F: Safety and Estimation Layer
Files:
1. `safety/safety_checker.py`
2. `state/state_estimator.py`

Responsibility:
1. Enforce workspace and motion safety checks.
2. Provide velocity/state estimation support for diagnostics and gating.

### Layer G: Config Governance Layer
Files:
1. `config/common_robot.yaml`
2. `config/active_inference_config.yaml`
3. `config/runtime_loader.py`

Responsibility:
1. Separate shared robot/task settings from AI tuning settings.
2. Enforce strict key validation to avoid silent config drift.
3. Keep experiments reproducible.

### Layer H: Observability and Evaluation Layer
Files:
1. `tools/analyze_run_diagnostics.py`
2. `tools/plot_run_metrics.py`
3. `tools/run_batch_eval.py`
4. `tools/run_position_sweep.py`

Responsibility:
1. Create run-level and batch-level evidence.
2. Report pass/fail gates, retries, timings, and bottlenecks.
3. Support controlled A/B tuning.

## 4) End-to-End Data and Control Path
1. Sensors provide observations.
2. Belief layer updates state estimates and phase context.
3. BT inspects phase/progress/risk and may request recovery.
4. Action layer generates command for current phase.
5. Controller executes command under constraints.
6. New state produces next observation.

## 5) Why This Design (Compared to Common Alternatives)

### Versus pure scripted FSM
1. Better adaptation under noise and geometry changes.
2. Still retains phase explainability through BT + phase machine.

### Versus pure black-box policy
1. Easier to debug because each layer is explicit.
2. Recovery logic is inspectable and bounded.

### Versus planner-only stack
1. Keeps reactive adaptation loop tight.
2. Easier early-stage iteration before full planning stack integration.

## 6) Current Performance Story
Evidence reference:
1. `logs/reports/position_sweep_20260301_134501.md`

Current summary:
1. Nominal scenario can complete reliably.
2. Hard geometry/noise cases still show pick-side retries.
3. Place-side generally performs well once lift is achieved.

Main open issue:
1. Improve first-try reliability in `Reach/Align/Descend`.

## 7) Risks and Controls

### Key risks
1. Threshold-sensitive oscillation near gates.
2. Recovery loops in difficult object poses.
3. Policy/controller mismatch under constrained postures.

### Controls already implemented
1. Hysteresis/debounce gates.
2. Align settle gate and near-threshold correction.
3. Branch-capped BT recovery.
4. Confidence + VFE-informed supervision.
5. Strict config validation and batch diagnostics.

## 8) Meeting Explanation Script (Quick)
1. "Our system has a layered architecture: sensing, belief inference, BT supervision, action policy, controller execution, and diagnostics."
2. "Active Inference handles adaptive local behavior; BT handles structured progression and recovery."
3. "The phase machine makes behavior inspectable phase-by-phase and tunable with explicit gates."
4. "Current bottleneck is pick-side robustness under hard geometry, and we already have measurable gates to track progress."

## 9) Glossary
1. Belief: current estimated state of robot/object/target.
2. EFE: Expected Free Energy used for action scoring.
3. BT: Behavior Tree supervising sequence/recovery.
4. Phase gate: condition required to transition phase.
5. VFE: free-energy-derived quality signal for supervision.
6. Recovery branch: semantic retry strategy (`ReScanTable`, `ReApproachOffset`, `SafeBackoff`).
