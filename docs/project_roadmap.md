# Active Inference Pick-and-Place Roadmap (Simple Version)

## Main Goal
Build a project that can run and show a full pick-and-place simulation using an active inference model.
The model can be imperfect at first. Plots and logs should show what is happening.

## Current Status
- Reach behavior works better than before, but is not fully reliable yet.
- Plot and CSV logging are already working.
- System runs in Python fallback mode when PyJulia is not available.

## Work Order (What we do first)
### Step 1 - Make current behavior stable
Focus only on making the current Reach/Grasp flow reliable.

Tasks:
1. Improve Reach convergence (already in progress).
2. Make Reach -> Grasp transition reliable.
3. Keep plotting every run and compare trends.

Done criteria:
- Reach error goes down consistently.
- Phase does not stay stuck in Reach for long runs.

### Step 2 - Connect to active inference core objective
After Step 1 is stable, connect proper inference.

Tasks:
1. Replace heuristic belief update with RxInfer inference in `infer_beliefs`.
2. Use posterior mean/covariance from the generative model.
3. Keep current control logic initially to reduce risk.

Done criteria:
- Beliefs come from RxInfer output.
- Simulation still runs end-to-end with same logging pipeline.

### Step 3 - Full demo target (MVP)
Goal: one command that runs simulation and shows active inference pick-and-place behavior.

Requirements:
1. Arm performs reach, grasp, lift, and place in simulation.
2. Logs and plots are saved every run.
3. Behavior is understandable from plots, even if not optimal.

### Step 4 - Refine active inference model
Once MVP works, improve model quality:
1. Better generative model state structure.
2. Better EFE action selection (multi-step horizon).
3. Better safety preferences.

### Step 5 - Failure and recovery
After model refinement:
1. Detect grasp failure.
2. Retry strategy (re-approach, re-grasp).
3. Track success rate across runs.

### Step 6 - Advanced improvements (Later)
Only after above is stable:
1. Pseudo vision/depth sensor fusion.
2. Hierarchical policy (Reach/Align/Grasp/Lift/Place).
3. Multi-object and research-level extensions.

## Backlog Board (Now / Next / Later)
| ID | Task | Bucket | Status | Notes |
|---|---|---|---|---|
| T1 | Reach convergence tuning | Now | In Progress | Reduce stalls and bad descent behavior |
| T2 | Reliable Reach -> Grasp transition | Now | Pending | Use robust transition conditions |
| T3 | RxInfer integration in `infer_beliefs` | Next | Pending | Keep controller unchanged first |
| T4 | End-to-end MVP pick-and-place demo | Next | Pending | Must run with logs + plots |
| T5 | Multi-step EFE and model refinement | Later | Pending | Improve decision quality |
| T6 | Failure detection + retry | Later | Pending | Add recovery behavior |
| T7 | Sensor fusion / pseudo-vision | Later | Pending | Add realistic perception |

## Rule for each completed task
For every completed task, record:
1. Run CSV path (for example: `logs/run_YYYYMMDD_HHMMSS.csv`)
2. Plot folder path (for example: `logs/plots/run_.../`)
3. One-line pass/fail note
