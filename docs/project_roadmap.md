# Project Roadmap (High-Level)

## Mission
Deliver a reliable active-inference pick-and-place runtime that is measurable, debuggable, and ready for incremental real-robot hardening.

## Runtime Scope
- Active-inference + BT only (FSM removed).
- Shared robot/target settings from `config/common_robot.yaml`.
- Behavior tuning from `config/active_inference_config.yaml`.

## Milestones
1. Stabilize core motion (Reach/Descend/Place reliability).
2. Improve recovery quality and reduce repeated failure loops.
3. Deliver consistent end-to-end simulation success (pick -> place -> done).
4. Integrate stronger inference model path (RxInfer-backed beliefs).
5. Prepare hardware-oriented robustness features.

## Execution Source
All actionable tasks, priorities, and done criteria are tracked only in:
- `docs/implementation_backlog.md`

## Completion Evidence Rule
For each completed change set, capture:
1. Run CSV path (`logs/run_YYYYMMDD_HHMMSS.csv`)
2. Plot/report outputs (`logs/plots/...`, `logs/reports/...`)
3. One-line outcome note (pass/fail + main reason)
