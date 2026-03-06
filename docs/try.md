# Robotic Manipulation System Architecture

## Adaptive Pick-and-Place with Active Inference and Behavior Tree

Author: Jay  
Purpose: explain the current runtime architecture used in this repository.

---

## 1. System Overview

This project controls a robot arm for autonomous pick-and-place in simulation (MuJoCo), with a path toward real-robot deployment.

Core design goals:
- robust behavior under noise and uncertainty
- explainable high-level task progression
- bounded, structured recovery when phases fail
- modular separation between decision, control, and hardware/sim interfaces

Main task loop:
1. approach and align to object
2. descend and grasp
3. lift and move to place area
4. place and retreat
5. recover on failure when possible

---

## 2. High-Level Runtime Architecture

Current runtime is AI-only (legacy FSM mode removed).

Flow:

Behavior Tree Supervisor
-> Phase Pipeline
-> Active Inference Action Selection
-> Controller (IK + limits + smoothing)
-> Simulator or Robot Backend

---

## 3. Layer Responsibilities

## 3.1 Behavior Tree Supervisor

The BT manages mission-level control and recovery routing.

Implemented structure:
- Root: `Selector`
- Primary branch: `Sequence(Acquire, Place)`
- Fallback branch: `Recover`

Responsibilities:
- keep task running when progress is valid
- detect stalled progression and trigger recovery
- cap retries and route to terminal `Failure` when budget is exceeded

## 3.2 Phase Pipeline

The active phase machine stages the task into explicit operational phases:
1. `Reach`
2. `Align`
3. `Descend`
4. `CloseHold`
5. `LiftTest`
6. `Transit`
7. `MoveToPlaceAbove`
8. `DescendToPlace`
9. `Open`
10. `Retreat`
11. `Done`

Responsibilities:
- provide phase-specific objectives and transition gates
- separate pick-side and place-side behavior for easier debugging
- expose phase timing and progression for diagnostics

## 3.3 Belief and Active Inference Layer

This layer updates beliefs from observations and selects actions by minimizing expected free energy (EFE).

Responsibilities:
- maintain robot/object/target belief state
- combine pragmatic objective (reduce task error) with epistemic objective (reduce uncertainty)
- apply phase-aware action shaping (position, yaw, topdown behavior, step scaling)

Notes:
- Python belief path is available
- optional RxInfer-backed belief path is supported

## 3.4 Controller Layer

Converts task-space commands into bounded robot motion commands.

Responsibilities:
- inverse kinematics solve
- motion smoothing and rate limiting
- joint step limits and command clipping
- gripper command tracking

## 3.5 Interface and Execution Layer

Provides sensor/actuator abstraction and simulation execution.

Responsibilities:
- collect observations (`ee`, object, target, contact, gripper state)
- execute commanded motion through backend
- allow sim/real backend swap with minimal changes in decision code

---

## 4. Failure Handling and Recovery

Recovery is structured, not ad hoc.

Common recovery branches:
- `ReScanTable`
- `ReApproachOffset`
- `SafeBackoff`

Behavior:
- pick-side failures generally reset to `Reach`
- place-side alignment failures can re-enter place-above flow when grasp is retained
- retry counters and cooldowns prevent unbounded loops
- terminal `Failure` is reached when retry budget is exhausted

---

## 5. What Is Important for Stakeholders

- The system is not a black box: BT, phases, and recovery are explicit and inspectable.
- Active inference gives adaptive motion decisions under uncertainty.
- Reliability is measured through run logs, diagnostics, batch evaluation, and position sweeps.
- Current known bottleneck remains first-try robustness in hard pick geometries; architecture is in place to improve this systematically.
