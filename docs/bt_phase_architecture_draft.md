# BT + Phase Architecture Draft (Design Lock Before Refactor)

Last updated: 2026-03-12
Status: Draft with partial implementation (active runtime)

## Implemented Slice (Current Runtime)
1. Task-intent/BT decision observability is present in runtime logs.
2. Place-side grasp loss emits `object_dropped` and routes through BT task switch to pick entry (`Reach`).
3. Retry budget is reset on `object_dropped` task switch (`retry_count=0`).
4. Align yaw gate before descend is implemented with configurable threshold/hold.
5. EE yaw observation and controller yaw frame are aligned via runtime wiring (`yaw_axis`).
6. Place keepout no-reentry behavior is implemented in `MoveToPlaceAbove`/`DescendToPlace`.
7. Router/phase-manager scaffolding files exist, while primary phase transition logic remains in `inference_interface.py`.

## 1) Goal
Define a clear, debuggable runtime architecture where:
1. BT owns mission/task routing.
2. Pick/Place phase managers own phase transitions.
3. Active inference owns belief update and action selection.
4. Recovery and failure handling are explicit and observable.

This is a design-lock document before code refactor.

---

## 2) Current Runtime (As Implemented)

```mermaid
flowchart LR
  S[Sensor Observation] --> B[infer_beliefs]
  B --> T[BT tick]
  T -->|recover?| R[recover_belief]
  T -->|normal| A[select_action]
  R --> A
  A --> C[Controller + Safety]
  C --> P[Plant/Simulator]
  P --> S
```

Notes:
1. This loop already exists and works.
2. Main pain point is ownership overlap: BT recovery and phase-local retry logic both influence flow.

---

## 3) Target Ownership Model (Recommended)

```mermaid
flowchart TD
  BT[BT Supervisor] -->|task_intent| PM[Phase Manager Router]
  PM -->|active_phase + references| AI[Active Inference Engine]
  AI -->|action| CTRL[Controller + Safety]
  CTRL --> PLANT[Robot/Sim]
  PLANT --> OBS[Observations]
  OBS --> AI
  PM --> MON[Run Monitor/Logs]
  AI --> MON
  BT --> MON
```

Ownership contract:
1. BT decides only `task_intent`: `PICK`, `PLACE`, `RECOVER`, `FAILURE`, `SAFE_STOP`.
2. Phase manager decides only phase progression within chosen task.
3. Active inference computes beliefs + action for current phase.
4. Controller/safety executes constrained motion.

---

## 4) BT Top-Level Graph (Task-Level Only)

```mermaid
flowchart TD
  START([Start]) --> PICK[PICK Node]
  PICK -->|pick_success| PLACE[PLACE Node]
  PICK -->|retry_allowed| RECOVER[RECOVER Node]
  PICK -->|catastrophic/safety| SAFE[SAFE_STOP]

  PLACE -->|place_success| DONE([DONE])
  PLACE -->|retry_allowed| RECOVER
  PLACE -->|catastrophic/safety| SAFE

  RECOVER -->|success_to_pick| PICK
  RECOVER -->|success_to_place_with_grasp| PLACE
  RECOVER -->|retry_exhausted| FAIL([FAILURE])
  RECOVER -->|safety| SAFE
```

Key rule:
1. BT never micromanages phase names like `Align` or `DescendToPlace`.
2. BT works at mission level only.

---

## 5) PICK Internal Phase Graph

```mermaid
flowchart TD
  PR[Reach] -->|gate_ok| PA[Align]
  PA -->|settle_ok| PD[Descend]
  PD -->|contact+axis_ok| PC[CloseHold]
  PC -->|grasp_stable| PL[LiftTest]
  PL -->|lift_pass| PS([PICK_SUCCESS])

  PA -->|timeout/stall| PRT([RETRY_PICK])
  PD -->|timeout/stall| PRT
  PC -->|grasp_search_fail| PRT
  PL -->|grasp_lost/drift| PRT
```

---

## 6) PLACE Internal Phase Graph

```mermaid
flowchart TD
  PT[Transit] --> PMA[MoveToPlaceAbove]
  PMA -->|pose_gate_ok| PDD[DescendToPlace]
  PDD -->|xy+z+yaw_ok| PO[Open]
  PO -->|release_verified| PR[Retreat]
  PR -->|timer_done| PSS([PLACE_SUCCESS])

  PMA -->|stall/timeout| PLR([RETRY_PLACE])
  PDD -->|stall/timeout| PLR
  PO -->|release_failed| PLR
```

---

## 7) Recovery/Fault Routing (Explicit Matrix)

```mermaid
flowchart LR
  E1[reach_stall] --> R1[ReApproachOffset]
  E2[stale_observation] --> R2[ReScanTable]
  E3[unexpected_contact/singularity] --> R3[SafeBackoff]
  E4[place_alignment_failed] --> R1
  E5[release_failed] --> R1
  E6[grasp_lost] --> R4[RePick Path]

  R1 --> DEC{retry budget left?}
  R2 --> DEC
  R3 --> DEC
  R4 --> DEC

  DEC -->|yes + grasp=1 + place-side event| G1[Resume PLACE at MoveToPlaceAbove]
  DEC -->|yes otherwise| G2[Resume PICK at Reach]
  DEC -->|no| F[FAILURE]
```

---

## 8) Failure Severity Model

```mermaid
flowchart TD
  WARN[Warning]
  RETRY[Recoverable Retry]
  FAIL[Terminal Failure]
  SAFE[Safe Stop]

  WARN --> RETRY
  RETRY -->|budget exceeded| FAIL
  WARN -->|safety breach| SAFE
  RETRY -->|hard safety breach| SAFE
```

Severity guidance:
1. `Warning`: detect-only signals, continue.
2. `Recoverable Retry`: branch-based recovery.
3. `Terminal Failure`: retry budgets exhausted.
4. `Safe Stop`: safety-critical condition.

---

## 9) Interface Contracts (Needed Before Refactor)

BT output contract:
1. `task_intent` in `{PICK, PLACE, RECOVER, FAILURE, SAFE_STOP}`
2. `recovery_reason`
3. `recovery_branch`
4. `global_retry_count`

Phase manager output contract:
1. `active_phase`
2. `phase_status` in `{RUNNING, SUCCESS, RETRY, FAILURE}`
3. `retry_reason`
4. `has_object`

AI output contract:
1. `belief_state` (filtered state, confidence, VFE)
2. `action` (`move`, `grip`, optional objective flags)
3. `risk_flags`

---

## 10) Observability Graph (What To See Per Run)

```mermaid
flowchart LR
  STEP[Step Log] --> KPI[Episode KPI Summary]
  BTLOG[BT Events] --> KPI
  PHLOG[Phase Events] --> KPI
  RLOG[Recovery Events] --> KPI
  SLOG[Safety/Risk Events] --> KPI
```

Minimum end-of-run KPI block:
1. Final status (`Done`, `Failure`, `SafeStop`).
2. Retry counts by reason.
3. Recovery branch usage and success rate.
4. Hard-stuck counts by phase.
5. First-try pass rates for key transitions.

---

## 11) Incremental Migration Plan (No Big-Bang Rewrite)

1. **Step A: Design Lock**
Define and approve contracts in this doc.

2. **Step B: Add Task Intent Layer**
Keep behavior same; add explicit `task_intent` plumbing.

3. **Step C: Split Internal Managers**
Create `pick_phase_manager` and `place_phase_manager` wrappers that call existing logic.

4. **Step D: Move Transition Ownership**
Gradually move phase-transition rules from mixed locations into managers.

5. **Step E: Unify Recovery Routing**
Keep one authoritative retry routing path with clear budgets.

6. **Step F: Freeze and Benchmark**
Run scenario matrix and compare to baseline before further tuning.

---

## 12) Decision Items To Finalize Before Coding

1. Should `SafeStop` be BT terminal only, or also callable directly by phase manager?
2. Which layer is authoritative for retry budget checks (BT only recommended)?
3. Should release verification be mandatory for `Open -> Retreat` in this cycle?
4. Do we expose both `phase` and `task_intent` in every CSV row (recommended: yes)?
5. Should place-side grasp loss be emitted as semantic event `object_dropped` and switched by BT to `PICK` intent (recommended: yes)?
