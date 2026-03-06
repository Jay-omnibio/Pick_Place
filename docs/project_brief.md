# CEO Meeting Brief: Active-Inference Pick-and-Place

Last updated: 2026-03-04

## 1) 60-Second Summary
This project is a robotic pick-and-place system in simulation (MuJoCo) using:
1. Active Inference for low-level action selection.
2. Behavior Tree (BT) for high-level supervision and recovery.

Why this matters:
1. It is not a fixed scripted path; it adapts online to noisy observations.
2. It has explicit recovery logic when a phase stalls or fails.
3. It is structured for future sim-to-real transfer (backends, safety, strict config).

## 2) What Problem It Solves
Goal:
1. Reach object.
2. Grasp and verify.
3. Move to place target.
4. Release and retreat.

Main challenge:
1. Noise, imperfect sensing, and geometric edge cases cause phase stalls.
2. System must recover without manual reset.

## 3) How It Works (High Level)
Architecture:
1. `infer_beliefs(...)` estimates current world/phase state from sensor observations.
2. BT checks progress/risk and decides normal progression vs recovery.
3. `select_action(...)` computes phase-appropriate robot action (move + gripper command).
4. Controller executes with IK/safety constraints.

Core runtime loop:
1. Sense -> 2. Infer -> 3. BT tick -> 4. Action select -> 5. Control/actuate -> repeat.

## 4) Why BT + Active Inference Together
Active Inference strengths:
1. Online adaptation to uncertainty.
2. Goal-directed behavior with confidence/VFE-aware gating.

BT strengths:
1. Clear phase orchestration (`Reach -> Align -> Descend -> ... -> Done`).
2. Explicit recovery branches and retry budgets.

Combined result:
1. Better robustness than pure scripted FSM.
2. Better explainability than monolithic black-box control.

## 5) Current Pipeline
Implemented phase pipeline:
1. `Reach`
2. `Align` (includes merged settle gate; no separate pregrasp phase)
3. `Descend`
4. `CloseHold`
5. `LiftTest`
6. `Transit`
7. `MoveToPlaceAbove`
8. `DescendToPlace`
9. `Open`
10. `Retreat`
11. `Done`

## 6) Current Status (Evidence)
Latest multi-scenario sweep:
1. Report: `logs/reports/position_sweep_20260301_134501.md`
2. Episodes: 12 total (4 scenarios x 3 episodes)
3. Completed (`Done`): 8/12

Interpretation:
1. Nominal scenario is strong.
2. Hard geometry still causes pick-side retries/failures.
3. Place side is generally reliable once post-lift phases are reached.

## 7) Main Technical Strengths to Highlight
1. Active-inference-only runtime (old FSM path removed for clarity).
2. Optional RxInfer-backed belief update with safe Python fallback.
3. Structured BT recovery branches (`ReScanTable`, `ReApproachOffset`, `SafeBackoff`).
4. Confidence and VFE-informed transition/recovery support.
5. Strict config loader to prevent hidden parameter drift.
6. Full diagnostics toolchain: single-run analysis, batch eval, position sweep.

## 8) Known Gaps (Open Work)
1. Pick-side robustness under hard geometry/noise is still the main bottleneck.
2. Need to reduce repeated `reach_stall` loops.
3. Need tighter first-try pass rates for `Reach/Align/Descend`.

## 9) Next 2-Week Plan (Practical)
1. Target pick-side first-try reliability (`Reach -> Descend`, `Reach -> CloseHold`).
2. Run scenario matrix with fixed pass/fail gates.
3. Add concise end-of-run KPI summary for quick decision meetings.
4. Keep place-side stability while improving pick-side robustness.

## 10) Suggested Talking Track for Meeting
Use this sequence:
1. "We built a robust robotic pick-place stack that combines adaptive inference with explicit recovery logic."
2. "It already completes full tasks across multiple scenarios, and we have measurable diagnostics for every run."
3. "Current bottleneck is not architecture; it is specific hard-pose pick robustness, which we are addressing with targeted gates and recovery tuning."
4. "This structure is ready for progressive hardening toward real robot constraints."

## 11) Quick Q&A Prep
Q: Is this production-ready for real robot?
A: Not yet; it is simulation-mature with strong observability and recovery scaffolding.

Q: Why not use only one approach (only planner / only RL / only BT)?
A: BT gives deterministic supervision and recovery; active inference gives adaptive low-level behavior under uncertainty.

Q: How do we prove progress?
A: Fixed scenario matrix + batch reports + phase-level first-try metrics + failure taxonomy.
