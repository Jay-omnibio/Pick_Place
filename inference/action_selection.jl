using LinearAlgebra
using Distributions

# ------------------------------------------------
# Configuration
# ------------------------------------------------

const LAMBDA_EPISTEMIC = 0.1   # curiosity weight
const DELTA = 0.018            # step size for EE movement
const REACH_OBJ_REL = [0.0, 0.0, -0.08]
const ACTION_EFFECTIVENESS = 0.4
const XY_ALIGN_THRESHOLD = 0.04
const XY_BLEND_FLOOR = 0.25
const GRASP_STEP = 0.008
const ALIGN_STEP = 0.010
const GRASP_SIDE_OFFSET_Y = 0.020
const ALIGN_REL_Z = -0.09

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

    effective_move = ACTION_EFFECTIVENESS * action.move

    # EE update
    next_s_ee = s_ee_mean + effective_move

    # Object relative update (free object assumption)
    next_s_obj = s_obj_mean - effective_move

    # Target relative update
    next_s_target = s_target_mean - effective_move

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

    phase = current_belief[:phase]

    # Phase-specific control to mirror stable scripted behavior.
    if phase == :Reach
        s_obj = current_belief[:s_obj_mean]
        err = s_obj - REACH_OBJ_REL
        xy_err_norm = norm(err[1:2])
        z_weight = min(1.0, max(XY_BLEND_FLOOR, 1.0 - (xy_err_norm / XY_ALIGN_THRESHOLD)))
        desired_move = [err[1], err[2], z_weight * err[3]]
        n = norm(desired_move)
        if n > DELTA && n > 0.0
            desired_move = (desired_move / n) * DELTA
        end
        return (move = desired_move, grip = -1)
    elseif phase == :Align
        s_obj = current_belief[:s_obj_mean]
        side_sign = get(current_belief, :grasp_side_sign, s_obj[2] >= 0.0 ? -1.0 : 1.0)
        align_rel = [0.0, side_sign * GRASP_SIDE_OFFSET_Y, ALIGN_REL_Z]
        err = s_obj - align_rel
        desired = [0.7 * err[1], 1.0 * err[2], 0.8 * err[3]]
        n = norm(desired)
        if n > ALIGN_STEP && n > 0.0
            desired = (desired / n) * ALIGN_STEP
        end
        return (move = desired, grip = -1)
    elseif phase == :Grasp
        if get(current_belief, :s_grasp, 0) == 0
            s_obj = current_belief[:s_obj_mean]
            grasp_rel = [0.0, 0.0, -0.08]
            err = s_obj - grasp_rel
            desired = [0.9 * err[1], 0.9 * err[2], 0.9 * err[3]]
            n = norm(desired)
            if n > GRASP_STEP && n > 0.0
                desired = (desired / n) * GRASP_STEP
            end
            return (move = desired, grip = 1)
        end
        return (move = [0.0, 0.0, 0.0], grip = 1)
    elseif phase == :Lift
        return (move = [0.0, 0.0, DELTA], grip = 1)
    end

    candidate_actions = generate_candidate_actions()

    best_action = nothing
    best_G = Inf

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
