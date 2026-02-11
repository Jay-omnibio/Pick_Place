using LinearAlgebra
using Distributions

# ------------------------------------------------
# Configuration
# ------------------------------------------------

const LAMBDA_EPISTEMIC = 0.1   # curiosity weight
const DELTA = 0.05            # step size for EE movement

# ------------------------------------------------
# Action definitions
# ------------------------------------------------

"""
Generate a small discrete set of candidate actions.

Each action is a NamedTuple:
- move :: Vector{Float64} (ΔEE)
- grip :: Int (0 = no-op, 1 = close, -1 = open)
"""
function generate_candidate_actions()

    moves = [
        [ DELTA, 0.0, 0.0],
        [-DELTA, 0.0, 0.0],
        [0.0,  DELTA, 0.0],
        [0.0, -DELTA, 0.0],
        [0.0, 0.0,  DELTA],
        [0.0, 0.0, -DELTA],
        [0.0, 0.0,  0.0],   # no movement
    ]

    actions = []

    for m in moves
        push!(actions, (move = m, grip = 0))
    end

    # gripper actions
    push!(actions, (move = [0.0, 0.0, 0.0], grip = 1))   # close
    push!(actions, (move = [0.0, 0.0, 0.0], grip = -1))  # open

    return actions
end

# ------------------------------------------------
# Expected Free Energy computation
# ------------------------------------------------

"""
Compute Expected Free Energy for a predicted belief.

belief is a Dict containing:
- :s_obj_mean
- :s_obj_cov
- :s_target_mean
"""
function compute_efe(belief, phase)

    # -------------------------
    # Pragmatic term
    # -------------------------

    pragmatic_cost = 0.0

    if phase == :Reach
        # want object close to EE → relative position near zero
        pragmatic_cost = norm(belief[:s_obj_mean])^2

    elseif phase == :Place
        # want object close to target
        pragmatic_cost = norm(belief[:s_target_mean])^2
    end

    # -------------------------
    # Epistemic term
    # -------------------------

    epistemic_cost =
        tr(belief[:s_obj_cov]) +
        tr(belief[:s_target_cov])

    # -------------------------
    # Expected Free Energy
    # -------------------------

    G = pragmatic_cost + LAMBDA_EPISTEMIC * epistemic_cost
    return G
end

# ------------------------------------------------
# Belief prediction (1-step)
# ------------------------------------------------

"""
Predict next belief given current belief and action.

This is a *belief-level* dynamics model, not physics.
"""
function predict_next_belief(current_belief, action)

    s_ee_mean = current_belief[:s_ee_mean]
    s_obj_mean = current_belief[:s_obj_mean]
    s_target_mean = current_belief[:s_target_mean]

    # EE update
    next_s_ee = s_ee_mean + action.move

    # Object relative update (free object assumption)
    next_s_obj = s_obj_mean - (next_s_ee - s_ee_mean)

    # Target relative update
    next_s_target = s_target_mean - (next_s_ee - s_ee_mean)

    # Covariances grow slightly (uncertainty propagation)
    base_obj_cov = current_belief[:s_obj_cov]
    base_target_cov = current_belief[:s_target_cov]
    next_obj_cov = base_obj_cov + 0.01I
    next_target_cov = base_target_cov + 0.01I

    return Dict(
        :s_ee_mean => next_s_ee,
        :s_obj_mean => next_s_obj,
        :s_target_mean => next_s_target,
        :s_obj_cov => next_obj_cov,
        :s_target_cov => next_target_cov
    )
end

# ------------------------------------------------
# Main action selection function
# ------------------------------------------------

"""
Select the action that minimizes Expected Free Energy.
"""
function select_action(current_belief)

    candidate_actions = generate_candidate_actions()

    best_action = nothing
    best_G = Inf

    phase = current_belief[:phase]

    for action in candidate_actions
        predicted_belief = predict_next_belief(current_belief, action)
        G = compute_efe(predicted_belief, phase)

        if G < best_G
            best_G = G
            best_action = action
        end
    end

    return best_action
end
