# AI Failure Detection Flow

This is the current detect-only failure monitoring path for active-inference mode.

## 1) Detection Pipeline (Per Step)

```mermaid
flowchart TD
    A[agent.step in active_inference] --> B[infer_beliefs]
    B --> C[update obs_confidence]
    C --> D[phase transition gates use confidence]
    D --> E[select_action]
    E --> F[_update_ai_risk_detection]
    F --> G[risk state fields updated]
    G --> H[_log_heartbeat]
    H --> I[_log_events]
```

## 2) Confidence + Release Verification

```mermaid
flowchart TD
    A[New observation] --> B[obj jump, target jump, contact flip]
    B --> C[raw confidence score]
    C --> D[EMA smoothing -> obs_confidence]
    D --> E{obs_confidence >= confidence_min_for_phase_change}
    E -->|yes| F[allow phase transition gates]
    E -->|no| G[hold current phase]

    O[phase == Open] --> O1{contact == 1}
    O1 -->|yes| O2[release_contact_counter++]
    O1 -->|no| O3[counter reset to 0]
    O2 --> O4{counter >= release_contact_warn_steps}
    O4 -->|yes| O5[release_warning = 1]
```

## 3) Risk Detection (Detect-Only)

```mermaid
flowchart TD
    A[controller debug norms] --> B[dq_ratio = dq_raw / dq_applied]
    C[phase error progress] --> D[phase_no_progress_steps]
    B --> E{dq_ratio high AND no progress}
    D --> E
    E -->|yes| F[singularity_counter++]
    E -->|no| G[counter reset]
    F --> H{counter >= singularity_no_progress_steps}
    H -->|yes| I[singularity_warn=1]

    J[current phase + contact] --> K[unexpected_contact?]
    K -->|yes| L[unintended_contact_counter++]
    K -->|no| M[counter reset]
    L --> N{counter >= unintended_contact_warn_steps}
    N -->|yes| O[unintended_contact_warn=1]
```

## 4) Important Note

Current behavior is detect-only:
- No forced emergency transition is triggered from these warnings.
- Warnings are surfaced in heartbeat/event logs for analysis and later recovery policy design.

