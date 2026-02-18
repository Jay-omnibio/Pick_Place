# Not Needed Inprovment (For Later)

This file tracks improvements that are useful in the future, but not required right now for your current goal.

## Current Status (Good Enough Now)
- ReachAbove and Descend are now yaw-aware for object rotation around Z.
- Gripper yaw is aligned dynamically to object yaw with offset, while keeping top-down constraint.
- Yaw shortest-turn logic with 180-degree symmetry is active, so large unnecessary flips are reduced.
- Reach/Descend X-Y gating now uses object-local axes, which is better for rotated objects.

## Not Needed Right Now (Add Later If Required)
- Full 3D object-frame control (roll/pitch + yaw), not only yaw around Z.
- Tilt-aware approach direction (change wrist tilt for non-flat objects or cluttered scenes).
- Yaw smoothing/hysteresis (reduce jitter when object yaw estimate is noisy).
- Multi-grasp candidate selection (choose best side based on reachability/collision, not fixed offset only).
- Dynamic grasp offset from object size/shape (instead of one fixed pregrasp/grasp offset).
- Contact-based micro-adjust before Close (small local search if near but not stable).
- Placement orientation goal (currently place is mostly position-driven; add orientation goal if needed).
- Failure-adaptive retries (change offset/yaw strategy across retries, not same retry every time).
- Confidence-aware policy (if object orientation estimate is uncertain, reduce yaw weight temporarily).
- Real-world timing model (phase timers by seconds instead of only step counts).

## When To Add These
- Only after current two-phase baseline (ReachAbove + Descend + stable grasp) is consistently successful.
- Add one improvement at a time, then re-test success rate.
