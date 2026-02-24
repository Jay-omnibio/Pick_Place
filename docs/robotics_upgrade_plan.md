# Robotics Upgrade Plan (Remaining Work)

## Goal
Track only robotics-layer tasks that are still pending.

## Completed (Removed from active queue)
1. Timing and latency observability.
2. State estimator (position + velocity) in sensor pipeline.
3. Phase-based gain scheduling from policy to controller.
4. Detect-only robotics risk monitoring in heartbeat/events.

These are now baseline features, so they are not active plan items anymore.

## Active Remaining Tasks

### R1) Release Verification Robustness (Highest Priority)
Current behavior in `Open` includes timer + contact warning + detach/stability checks.
Still needed:
1. Tune detach/stability thresholds across object variants.
2. Validate branch outcomes in batch runs (release fail -> recovery branch -> outcome).

Primary files:
- `inference_interface.py`
- `agent/agent_loop.py`

Done criteria:
- `Open -> Retreat` happens only after detach/stability checks pass, or explicit recovery path is triggered.

### R2) Health Monitor Completion
Current monitor already logs risk warnings and timing.
Still needed:
1. Add aggregated episode counters (stale frames, saturation events, singularity-like warnings, unintended-contact streaks).
2. Emit one concise end-of-episode monitor summary.

Primary files:
- `agent/agent_loop.py`
- `docs/` metrics description

Done criteria:
- One per-episode health summary can be compared across runs.

### R3) Hardware-Focused Control Mode (Later)
1. Add optional admittance/impedance mode behind config flag.
2. Keep current deterministic IK path as default.

Primary files:
- `control/controller.py`
- backend adapters

Done criteria:
- Optional compliant mode available without changing default sim behavior.

### R4) BT Node Modularization (Later)
1. Split BT leaf/condition nodes into `agent/bt_nodes/`.
2. Keep behavior identical; refactor for maintainability only.

Primary files:
- `agent/bt_nodes/*.py` (new)
- `agent/ai_behavior_tree.py`

Done criteria:
- No behavior change, cleaner testable BT node boundaries.

## Test Strategy
For each remaining change, compare at least 10 episodes:
1. Lift success rate
2. Done success rate
3. Mean steps to LiftTest and Done
4. Recovery count
5. Stale-frame warnings
6. Control saturation / singularity-like warning count
