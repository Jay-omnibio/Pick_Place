# AI-Only RxInfer + BT Redesign Plan

## Goal
Build a new project variant that is:
- AI-only (no FSM runtime path)
- BT-supervised
- RxInfer-based for belief updates
- EFE-driven for motion decisions
- Safe for later real-robot transition

This plan is for the new copied folder first. The original project remains unchanged for ongoing experiments.

---

## 1) Core Direction

### What we want
1. One runtime mode: `active_inference` only.
2. One decision stack: `BT -> Belief (RxInfer) -> Planner (EFE) -> Controller`.
3. No scripted policy path and no runtime FSM path.
4. Start with smooth normal motion-to-object first, then tighten reliability gates later.

### What we do not want (initially)
1. Large hard XY/Z phase thresholds as primary logic.
2. Mixed dual behavior owners (FSM + AI + local escape all deciding).
3. Multiple competing action-selection backends by default.

---

## 2) Important Constraint

Do not remove all thresholds on day 1.

Keep hard bounds for:
1. Safety limits: workspace, velocity, z-floor, joint limits.
2. Episode timeout/watchdog.
3. Contact validation sanity bounds.

Reduce or postpone task-specific hard XY/Z thresholds, but keep safety thresholds.

---

## 3) Target Runtime Architecture

## 3.1 Control Pipeline
1. Sensors -> observation stream.
2. State estimator -> filtered position/velocity.
3. RxInfer update -> belief posterior.
4. BT tick -> task intent + recovery branch.
5. EFE planner -> best action sequence / action.
6. Controller -> IK + gripper execution.
7. Safety checker -> final clamp/guard.

## 3.2 Ownership Rules
1. BT owns high-level phase and recovery policy.
2. RxInfer owns latent belief/state estimation.
3. Planner owns action choice.
4. Controller only executes commands; no phase logic.
5. Safety checker only enforces constraints.

---

## 4) Phase Redesign (BT Nodes)

Replace legacy runtime phase logic with BT-intent nodes:
1. `PerceiveObject`
2. `ApproachAbove`
3. `Descend`
4. `CloseAndVerify`
5. `LiftVerify`
6. `MoveToPlaceAbove`
7. `PlaceDescend`
8. `Release`
9. `Retreat`
10. `Done` / `Recover`

### Transition Style (initial)
Use confidence/evidence transitions instead of strict XY threshold transitions:
1. Posterior confidence above minimum for N consecutive steps.
2. Error trend improving (negative slope) for N steps.
3. Contact/gripper consistency checks.

---

## 5) Belief and Inference Design

## 5.1 RxInfer Responsibility
Infer posterior over:
1. `ee_state` (pos, vel)
2. `obj_rel_state` (pos, vel)
3. `target_rel_state` (pos, vel)
4. `grasp_state` (latent discrete)
5. Optional orientation latent terms (yaw first, roll/pitch later)

## 5.2 Why this helps
1. Reduces hard gate chatter near boundaries.
2. Handles sensor noise with explicit uncertainty.
3. Enables probabilistic completion checks and better recovery triggers.

## 5.3 Current reality to acknowledge
Current `inference_interface.py` is heuristic/placeholder-oriented and should become a thin adapter around RxInfer updates.

---

## 6) Planner Design (Action Selection)

## 6.1 Initial Planner
1. Keep a bounded candidate action set.
2. Score candidates with EFE.
3. Add short horizon lookahead (multi-step rollout).
4. Penalize unsafe or unstable motions.

## 6.2 Later upgrades
1. CEM/MPPI over continuous actions.
2. Learned proposal distribution.
3. Adaptive horizon by BT intent.

---

## 7) Threshold Strategy

## 7.1 Initial rule
1. Keep safety thresholds.
2. Keep only minimal completion guards.
3. Use probabilistic confidence + trend for transitions.

## 7.2 Later reintroduction
Add task thresholds only where data proves needed:
1. Place precision constraints.
2. Release stability constraints.
3. Recovery trigger boundaries.

---

## 8) File-Level Keep / Deprecate / Remove Map

## 8.1 Keep (core runtime)
1. `run_pick_place.py` (AI-only entry after cleanup)
2. `agent/agent_loop.py` (AI path only after cleanup)
3. `agent/ai_behavior_tree.py` (BT supervisor)
4. `inference_interface.py` (to be simplified into RxInfer adapter)
5. `inference/action_selection.py` (single planner path)
6. `control/controller.py`
7. `backends/sensor_backend.py`
8. `backends/actuator_backend.py`
9. `state/state_estimator.py`
10. `safety/safety_checker.py`
11. `config/common_robot.yaml`
12. `config/active_inference_config.yaml`
13. `config/sensor_config.yaml`

## 8.2 Deprecate (move to legacy folder or disable import path)
1. `tasks/pick_place_fsm.py`
2. `policies/scripted_pick_place.py`
3. `config/fsm_config.yaml`

## 8.3 Remove later (after AI-only stable)
1. Runtime mode switching branches (`fsm` path in agent/runner).
2. Legacy action branches (`LegacyGrasp`, `Lift` compatibility paths).
3. Unused belief fields that do not affect planner/BT decisions.

---

## 9) Migration Sequence (Recommended)

## M0: Project copy and freeze
1. Copy whole project to new folder.
2. Tag baseline commit.
3. Keep original project untouched.

## M1: AI-only skeleton
1. Force `control_mode=active_inference`.
2. Remove runtime FSM execution branches.
3. Disable scripted policy usage in active path.
4. Keep current controller/safety unchanged.

## M2: Single action-selection backend
1. Pick Python path first for determinism.
2. Disable automatic Julia fallback switching for now.

## M3: Inference refactor
1. Replace monolithic heuristic belief transition with RxInfer adapter.
2. Keep BT state and planner contract stable.

## M4: Transition criteria shift
1. Replace hard task transitions with confidence/trend checks.
2. Keep fallback timeout and failure routing.

## M5: Multi-step EFE
1. Add short horizon candidate rollouts.
2. Compare against one-step baseline on same seeds.

## M6: Cleanup pass
1. Remove dead legacy branches and unused fields.
2. Trim config to AI-only keys.

---

## 10) Logging and Metrics (must keep from day 1)

Track at least:
1. Success/failure phase.
2. Steps-to-reach and steps-to-descend.
3. Recovery count and reason.
4. Mean/peak command norm.
5. Belief confidence and uncertainty trend.
6. Direction consistency (command vs actual EE delta cosine).

---

## 11) Definition of Done (AI-only baseline)

1. No FSM runtime dependency in active path.
2. Smooth approach and descend without frequent reversal jitter.
3. Stable success across multi-run seeded tests.
4. Recovery bounded and explainable.
5. Safety limits never violated.
6. Config and code are understandable without legacy branches.

---

## 12) Practical Notes for Start

1. Start with normal motion quality first, not perfect grasp logic first.
2. Do not overfit thresholds early; use trends and confidence.
3. Keep one owner per concern (BT, inference, planner, controller, safety).
4. Remove complexity only after replacement path is validated.

---

## 13) Immediate Next Actions (when you start new folder)

1. Create new repo/folder from full copy.
2. Apply M1 and M2 only.
3. Run 10 seeded trials and collect baseline metrics.
4. Then begin M3 (RxInfer adapter integration).

