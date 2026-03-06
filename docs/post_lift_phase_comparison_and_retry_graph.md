# Full High-Level Flow: BT + Phase Machine + Retry Paths

Last updated: 2026-03-04

Source files used:
- `agent/ai_behavior_tree.py`
- `inference_interface.py`
- `inference/action_selection.py`

This diagram set shows:
1. How BT supervises phase execution and recovery.
2. Full phase-state machine transitions.
3. Exactly where failures go (retry, fallback, or terminal failure).

---

## 1) Runtime Control Stack (High Level)

```mermaid
flowchart LR
  OBS[Sensor Observation]
  INF[infer_beliefs]
  BT[Behavior Tree Tick]
  RB[recover_belief]
  ACT[select_action]
  CTRL[Controller / IK]
  SIM[Simulator/Robot State]

  OBS --> INF --> BT
  BT -->|normal| ACT
  BT -->|recover requested| RB --> ACT
  ACT --> CTRL --> SIM --> OBS
```

---

## 2) BT Logic and Recovery Routing

### 2.1 BT decision path

```mermaid
flowchart TD
  A[Tick BT with current phase/belief]
  B{Terminal phase?}
  C{Risk/quality trigger?}
  D{Progress watchdog stalled?}
  E[Root Selector]
  F[Sequence: SetPickPriors -> Acquire -> SetPlacePriors -> Place]
  G[RecoverNode]
  H[recover_requested = true]
  I[No recovery this tick]

  A --> B
  B -->|Done| I
  B -->|Failure| H
  B -->|No| C
  C -->|stale/risk/high_vfe| H
  C -->|No| D
  D -->|Yes| H
  D -->|No| E
  E --> F
  E --> G
  G --> H
  F -->|running/success| I
  F -->|failure| H
```

### 2.2 Recovery reason -> preferred branch order

```mermaid
flowchart LR
  RS[reach_stall] --> RA[ReApproachOffset]
  ST[stale_observation] --> RT[ReScanTable]
  SR[singularity_risk / unexpected_contact] --> SB[SafeBackoff]
  GF[grasp_failed] --> RT
  PF[place_alignment_failed / release_failed] --> RA
  HV[high_vfe] --> RA
```

Notes:
- If preferred branch cap is reached, next branch is tried.
- If all branches are exhausted, BT returns terminal failure path.
- Retry caps apply in `recover_belief`:
  - `max_retries`
  - `global_recovery_cap`

---

## 3) Full Phase-State Machine with Retry/Fallback Paths

```mermaid
flowchart TD
  R[Reach]
  A[Align]
  D[Descend]
  CH[CloseHold]
  LT[LiftTest]
  T[Transit]
  MPA[MoveToPlaceAbove]
  DTP[DescendToPlace]
  O[Open]
  RT[Retreat]
  DN[Done]
  FL[Failure]

  %% Nominal success path
  R -->|reach gate stable + phase gate| A
  A -->|ready/near + settle gate| D
  D -->|position ok OR timeout-near + phase gate| CH
  CH -->|stable grasp + close ready + phase gate| LT
  LT -->|lift test pass + phase gate| T
  T -->|ee_z reaches transit target| MPA
  MPA -->|preplace ok OR timeout-near + phase gate| DTP
  DTP -->|place xy+z+yaw ok + phase gate| O
  O -->|open hold + release ok| RT
  RT -->|retreat timer done| DN

  %% Pick-side retries/fallbacks
  A -->|timeout while far| R
  D -->|timeout while far| R
  CH -->|grasp search timeout| R
  LT -->|grasp lost OR drift too high| R

  %% Post-lift fallbacks on grasp loss
  T -->|grasp lost| R
  MPA -->|grasp lost| R
  DTP -->|grasp lost| R

  %% Place-side retries/failures
  MPA -->|stall/timeout -> place_alignment_failed + retry budget left| MPA
  MPA -->|place_alignment_failed + retry budget exceeded| FL

  DTP -->|timeout + grasp held + retry budget left| MPA
  DTP -->|timeout + grasp held + retry budget exceeded| FL

  O -->|open_max timeout + release_failed + retry budget left| MPA
  O -->|open_max timeout + release_failed + retry budget exceeded| FL
```

---

## 4) What Happens When BT Recovery Triggers

When BT triggers recovery (`recover_belief`):

1. If reason is place-side (`place_alignment_failed` or `release_failed`) and:
- current phase is place-side, and
- grasp is still held

Then BT forces:
- phase -> `MoveToPlaceAbove`
- resets place-side timers/counters

2. Otherwise BT forces:
- phase -> `Reach`
- resets pick/place timers and watchdog counters
- applies selected recovery branch prior offsets/boost

3. Hard failure conditions in BT:
- `retry_count > max_retries`
- `global_recovery_count > global_recovery_cap`
- no recovery branch available under branch caps

All of these go to phase `Failure`.

---

## 5) Quick Edge Summary (fail -> where)

- `Align` fail -> `Reach` (`reach_stall`)
- `Descend` fail -> `Reach` (`grasp_failed`)
- `CloseHold` fail -> `Reach` (`grasp_failed`)
- `LiftTest` fail -> `Reach` (`grasp_failed`)
- `Transit` grasp loss -> `Reach` (`grasp_failed`)
- `MoveToPlaceAbove` fail -> self-retry or `Failure` (`place_alignment_failed`)
- `DescendToPlace` fail -> `MoveToPlaceAbove` or `Reach` or `Failure`
- `Open` fail -> `MoveToPlaceAbove` or `Failure` (`release_failed`)
- Any BT hard-limit breach -> `Failure`
