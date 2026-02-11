using RxInfer
using Distributions
using LinearAlgebra

# ------------------------------------------------
# Constants and enums
# ------------------------------------------------

const DIM = 3  # 3D space

@enum Phase begin
    Reach = 1
    Grasp = 2
    Lift  = 3
    Place = 4
end

@enum GraspState begin
    Open    = 1
    Holding = 2
end

# ------------------------------------------------
# Generative Model
# ------------------------------------------------
# This model defines:
# - latent beliefs (continuous + discrete)
# - observation likelihoods
# - state transition dynamics
# - preferences as priors
#
# Actions are GIVEN (conditioning inputs), not inferred
# ------------------------------------------------

@model function pick_and_place_model(
    T,                      # number of time steps

    # Observations (given)
    o_ee,                   # Vector{Vector{Float64}}
    o_obj,                  # Vector{Vector{Float64}}
    o_grip,                 # Vector{Float64}
    o_contact,              # Vector{Int}

    # Actions (given)
    a_move,                 # Vector{Vector{Float64}}
    a_grip                  # Vector{Int}
)

    # --------------------------------------------
    # Initial beliefs (t = 1)
    # --------------------------------------------
    s_ee_prev     ~ MvNormal(zeros(DIM), 0.1I)
    s_obj_prev    ~ MvNormal(zeros(DIM), 0.1I)
    s_target_prev ~ MvNormal(zeros(DIM), 0.1I)

    s_grasp_prev ~ Categorical([0.9, 0.1])   # mostly open
    s_phase_prev ~ Categorical([1.0, 0.0, 0.0, 0.0])  # start in Reach

    # --------------------------------------------
    # Time loop
    # --------------------------------------------
    for t in 1:T

        # ----------------------------------------
        # Continuous state transitions
        # ----------------------------------------

        # EE motion model
        s_ee ~ MvNormal(
            s_ee_prev + a_move[t],
            0.01I
        )

        # Object relative position (assume free object here)
        s_obj ~ MvNormal(
            s_obj_prev - (s_ee - s_ee_prev),
            0.05I
        )

        # Target relative position (static in world)
        s_target ~ MvNormal(
            s_target_prev - (s_ee - s_ee_prev),
            0.01I
        )

        # ----------------------------------------
        # Discrete state transitions
        # ----------------------------------------

        # Grasp state transition (placeholder probabilities)
        s_grasp ~ Categorical([0.8, 0.2])

        # Task phase transition (simple HMM placeholder)
        s_phase ~ Categorical([0.7, 0.1, 0.1, 0.1])

        # ----------------------------------------
        # Observation likelihoods
        # ----------------------------------------

        # EE proprioception
        o_ee[t] ~ MvNormal(s_ee, 0.01I)

        # Object-relative vision
        o_obj[t] ~ MvNormal(s_obj, 0.05I)

        # Gripper encoder (depends on grasp belief)
        o_grip[t] ~ Normal(
            s_grasp == Holding ? 0.0 : 0.04,
            0.002
        )

        # Contact sensor (binary, placeholder)
        o_contact[t] ~ Bernoulli(0.5)

        # ----------------------------------------
        # Preferences as priors (Active Inference)
        # ----------------------------------------

        if s_phase == Reach
            # Prefer object close to EE
            s_obj ~ MvNormal(zeros(DIM), 0.01I)

        elseif s_phase == Place
            # Prefer object close to target
            s_target ~ MvNormal(zeros(DIM), 0.01I)
        end

        # ----------------------------------------
        # Carry state forward
        # ----------------------------------------
        s_ee_prev     = s_ee
        s_obj_prev    = s_obj
        s_target_prev = s_target
        s_grasp_prev  = s_grasp
        s_phase_prev  = s_phase
    end
end
