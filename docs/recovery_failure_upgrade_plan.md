# Recovery and Failure Improvement Plan (In Progress)

## Goal
Add semantic failure handling for active-inference + BT so recovery is not only "retry same thing."

## Scope
- Control mode: `active_inference`
- Focus: detection, branch selection, and safe recovery actions
- Not in scope: full model redesign

## Why this is needed
Current system detects many risks/warnings, but most recoveries still reset to Reach and retry.  
Real robot behavior needs branch-specific recovery policies.

## Priority (when this work starts)

## R1 - Failure taxonomy and reason codes (Completed)
Define standard failure reasons and log them consistently.

Examples:
- `reach_stall`
- `high_vfe`
- `stale_observation`
- `singularity_risk`
- `unexpected_contact`
- `grasp_failed`
- `release_failed`
- `place_alignment_failed`

Files:
- `agent/ai_behavior_tree.py`
- `agent/agent_loop.py`
- `inference_interface.py`

## R2 - Semantic BT recovery branches (Completed)
Add dedicated BT branches instead of one generic retry.

Minimum branches:
1. `ReScanTable`
- Move EE to safe high vantage pose.
- Hold briefly, reacquire object belief.

2. `ReApproachOffset`
- Re-enter Reach with XY offset and/or higher Z.
- Avoid repeating same failed approach line.

3. `SafeBackoff`
- Small retreat when unexpected contact/singularity risk persists.
- Then restart from Reach.

Implemented:
- branch selection and activation in `agent/ai_behavior_tree.py`
- branch-aware prior modulation (pick/place targets) in `agent/ai_behavior_tree.py`
- branch + retry counters surfaced in `agent/agent_loop.py` logs/CSV

Files:
- `agent/ai_behavior_tree.py`
- `agent/agent_loop.py`

## R3 - Recovery gating rules (Completed)
Recovery branch selection should depend on reason:
- `stale_observation` -> `ReScanTable`
- `reach_stall` -> `ReApproachOffset`
- `singularity_risk` -> `SafeBackoff`
- `grasp_failed` -> `ReScanTable` then `ReApproachOffset`

Status:
- implemented in `agent/ai_behavior_tree.py` via reason-to-branch mapping.

## R4 - Release/placement failure verification (Partially Completed)
Add explicit post-open checks:
- object detached from gripper
- object stable near place target for a short hold window

If fail:
- branch to `ReScanTable` or `RePlaceAttempt`

Critical rule:
- If place-side alignment fails (`DescendToPlace` not converged) but object is still grasped,
  do **not** fall back to `Reach`.
- Instead, go to `MoveToPlaceAbove` (or `RePlaceAttempt`) while keeping grip closed.
- Reason: switching to `Reach` typically commands open-grip behavior and can drop the object.

Status:
- place-side failure with grasp retained already routes to `MoveToPlaceAbove` (not `Reach`).
- explicit release verification gating added in `Open` (`detach + stable` checks, timeout, re-approach limit).
- remaining: optional dedicated `RePlaceAttempt` branch naming (behavior is currently handled through `ReApproachOffset` + place re-entry).

## R5 - Retry budget policy (Completed)
Use branch-local retry counters and global mission cap:
- branch retry cap (e.g., 2 each)
- global cap (e.g., 6 total recoveries)
- terminal `Failure` when exceeded

Status:
- implemented in `agent/ai_behavior_tree.py` with configurable caps.

## Dependencies on robotics foundation
Prerequisite robotics foundation is now available in codebase:
1. timing/freshness monitoring
2. state estimator
3. phase-based gain scheduling

Recovery work can proceed directly using these signals.

## Suggested order once started
1. R1 (reason codes)
2. R2 (three semantic branches)
3. R3 (reason-to-branch mapping)
4. R4 (release verification)
5. R5 (retry budget policy)

## Success criteria
1. Fewer repeated "same failure" loops.
2. Higher lift/place completion under perturbations.
3. Logs show clear reason -> branch -> outcome chain.
