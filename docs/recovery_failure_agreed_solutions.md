# Recovery/Failure Agreed Solutions (Finalization Draft)

Date: 2026-03-01  
Scope: BT + phase-machine recovery policy only (not full motion tuning).
Status update: 2026-03-04 (phase pipeline uses merged `Align` settle, no separate `PreGraspHold` phase).

Reference flow/graph (current state): `docs/post_lift_phase_comparison_and_retry_graph.md`

---

## 1) Agreed High-Level Policy

1. Treat pipeline as two logical tasks:
   - `PickTask`: `Reach -> Align -> Descend -> CloseHold -> LiftTest`
   - `PlaceTask`: `Transit -> MoveToPlaceAbove -> DescendToPlace -> Open -> Retreat`
2. Do not route every place-side failure back to `Reach`.
3. Re-enter `Reach` only when object possession is actually lost (`s_grasp == 0`) before successful release.

---

## 2) Failure Taxonomy We Agreed

1. Place motion failure, object still held:
   - Examples: `place_alignment_failed`, place-side timeout/stall.
   - Recovery target: stay in place-side subtree (`MoveToPlaceAbove` first).
2. Object slip/drop failure:
   - Condition: `s_grasp == 0` before successful `Open` completion.
   - Recovery target: return to pick-side (`Reach` / re-pick).
3. Release failure (still attached after open attempts):
   - Recovery target: place-side recovery first (re-open/reposition), not immediate full re-pick.

---

## 3) Agreed BT Root Behavior

1. Root routing should be possession-aware:
   - If `has_object == 0`: run `PickTask`.
   - If `has_object == 1`: run `PlaceTask`.
2. Recovery branch selection remains BT-managed (branch caps + global cap).
3. `Failure` should represent exhausted budgets, not first place-side difficulty.

---

## 4) Current Behavior vs Target

1. Current behavior already supports place-side re-entry to `MoveToPlaceAbove` in several place failures.
2. Current behavior also returns to `Reach` when grasp is lost (correct).
3. Remaining target: make this routing fully possession-first and reason-consistent across all place-side failure exits.

---

## 5) Implementation Actions (Next)

Priority order:

1. `P0` Normalize recovery routing table:
   - Explicit mapping: `(phase, reason, s_grasp)` -> next phase.
   - Ensure all place-side reasons with `s_grasp==1` stay place-side.
2. `P0` Add possession-first BT guard:
   - Before fallback-to-`Reach`, confirm `s_grasp==0` or explicit pick-side reason.
3. `P1` Standardize release-failure handling:
   - Retry `Open` and/or reposition in place subtree before re-pick.
4. `P1` Add reason consistency checks:
   - Prevent same event being tagged differently across phase-machine vs BT.
5. `P1` Add metrics:
   - `place_failure_with_grasp_count`
   - `place_failure_without_grasp_count`
   - `repick_trigger_count`
   - `place_recovery_success_count`

---

## 6) Acceptance Criteria

1. Place-side failures with `s_grasp==1` should mostly avoid `Reach` fallback.
2. Re-pick (`Reach` re-entry) should correlate with true grasp loss events.
3. Batch reports should show lower unnecessary repick loops and better place recovery conversion.

---

## 7) Out of Scope for This Doc

1. Reach/Descend path-shape tuning.
2. Controller gain redesign.
3. RxInfer model improvements.

These stay tracked in existing roadmap/backlog docs.
