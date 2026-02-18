using LinearAlgebra

"""
Julia action-selection path for active inference.

All tuning comes from `params` (strict, no defaults).
"""

function _require(params, key::Symbol)
    if !haskey(params, key)
        error("Missing active-inference config key: $(key)")
    end
    return params[key]
end

function _require_vec3(params, key::Symbol)
    v = Float64.(_require(params, key))
    if length(v) != 3
        error("Active-inference config key '$(key)' must be a length-3 vector.")
    end
    return v
end

_wrap_to_pi(angle::Float64) = mod(angle + pi, 2.0 * pi) - pi

function _signed_angle_to_goal(theta_now::Float64, theta_goal::Float64, turn_sign::Int)
    if turn_sign >= 0
        return (theta_goal - theta_now) % (2.0 * pi)
    end
    return -((theta_now - theta_goal) % (2.0 * pi))
end

function _object_yaw_target(current_belief)
    obj_yaw = Float64(get(current_belief, :s_obj_yaw, 0.0))
    if !isfinite(obj_yaw)
        obj_yaw = 0.0
    end
    return _wrap_to_pi(obj_yaw + 0.5 * pi)
end

function generate_candidate_actions(delta::Float64)
    moves = [
        [ delta, 0.0, 0.0],
        [-delta, 0.0, 0.0],
        [0.0,  delta, 0.0],
        [0.0, -delta, 0.0],
        [0.0, 0.0,  delta],
        [0.0, 0.0, -delta],
        [0.0, 0.0,  0.0],
    ]

    actions = []
    for m in moves
        push!(actions, (move = m, grip = 0))
    end
    push!(actions, (move = [0.0, 0.0, 0.0], grip = 1))
    push!(actions, (move = [0.0, 0.0, 0.0], grip = -1))
    return actions
end

function compute_efe(belief, phase, lambda_epistemic::Float64)
    pragmatic_cost = 0.0
    if phase == :Reach
        pragmatic_cost = norm(belief[:s_obj_mean])^2
    elseif phase == :Place
        pragmatic_cost = norm(belief[:s_target_mean])^2
    end

    epistemic_cost = tr(belief[:s_obj_cov]) + tr(belief[:s_target_cov])
    return pragmatic_cost + lambda_epistemic * epistemic_cost
end

function predict_next_belief(current_belief, action, action_effectiveness::Float64)
    s_ee_mean = current_belief[:s_ee_mean]
    s_obj_mean = current_belief[:s_obj_mean]
    s_target_mean = current_belief[:s_target_mean]

    effective_move = action_effectiveness .* action.move

    next_s_ee = s_ee_mean + effective_move
    next_s_obj = s_obj_mean - effective_move
    next_s_target = s_target_mean - effective_move

    base_obj_cov = current_belief[:s_obj_cov]
    base_target_cov = current_belief[:s_target_cov]
    next_obj_cov = base_obj_cov + 0.01I
    next_target_cov = base_target_cov + 0.01I

    return Dict(
        :s_ee_mean => next_s_ee,
        :s_obj_mean => next_s_obj,
        :s_target_mean => next_s_target,
        :s_obj_cov => next_obj_cov,
        :s_target_cov => next_target_cov,
    )
end

function select_action(current_belief, params)
    lambda_epistemic = Float64(_require(params, :lambda_epistemic))
    delta = Float64(_require(params, :delta))
    reach_delta = Float64(_require(params, :reach_delta))
    action_effectiveness = Float64(_require(params, :action_effectiveness))

    reach_axis_threshold_x = Float64(_require(params, :reach_axis_threshold_x))
    reach_axis_threshold_y = Float64(_require(params, :reach_axis_threshold_y))
    reach_z_blend_start = Float64(_require(params, :reach_z_blend_start))
    reach_z_blend_full = Float64(_require(params, :reach_z_blend_full))
    reach_step_min = Float64(_require(params, :reach_step_min))

    reach_arc_enabled = Bool(_require(params, :reach_arc_enabled))
    reach_arc_max_theta_step_deg = Float64(_require(params, :reach_arc_max_theta_step_deg))
    reach_arc_radial_step_max = Float64(_require(params, :reach_arc_radial_step_max))
    reach_arc_min_radius = Float64(_require(params, :reach_arc_min_radius))
    reach_arc_max_radius = Float64(_require(params, :reach_arc_max_radius))

    grasp_step = Float64(_require(params, :grasp_step))
    align_step = Float64(_require(params, :align_step))
    pregrasp_hold_step = Float64(_require(params, :pregrasp_hold_step))
    descend_x_threshold = Float64(_require(params, :descend_x_threshold))
    descend_y_threshold = Float64(_require(params, :descend_y_threshold))
    descend_z_threshold = Float64(_require(params, :descend_z_threshold))
    preplace_xy_threshold = Float64(_require(params, :preplace_xy_threshold))
    place_xy_threshold = Float64(_require(params, :place_xy_threshold))
    place_z_threshold = Float64(_require(params, :place_z_threshold))

    phase = current_belief[:phase]
    reach_obj_rel = haskey(current_belief, :reach_obj_rel) ? Float64.(current_belief[:reach_obj_rel]) : _require_vec3(params, :reach_obj_rel)
    align_obj_rel = haskey(current_belief, :align_obj_rel) ? Float64.(current_belief[:align_obj_rel]) : _require_vec3(params, :align_obj_rel)
    descend_obj_rel = haskey(current_belief, :descend_obj_rel) ? Float64.(current_belief[:descend_obj_rel]) : _require_vec3(params, :descend_obj_rel)
    preplace_target_rel = haskey(current_belief, :preplace_target_rel) ? Float64.(current_belief[:preplace_target_rel]) : _require_vec3(params, :preplace_target_rel)
    place_target_rel = haskey(current_belief, :place_target_rel) ? Float64.(current_belief[:place_target_rel]) : _require_vec3(params, :place_target_rel)
    retreat_move = haskey(current_belief, :retreat_move) ? Float64.(current_belief[:retreat_move]) : _require_vec3(params, :retreat_move)

    if phase == :Reach
        s_obj = Float64.(current_belief[:s_obj_mean])
        s_ee = Float64.(current_belief[:s_ee_mean])
        reach_turn_sign = Int(get(current_belief, :reach_turn_sign, 0))
        reach_watchdog_active = Int(get(current_belief, :reach_watchdog_active, 0)) != 0
        reach_yaw_align_active = Int(get(current_belief, :reach_yaw_align_active, 0)) != 0

        if reach_yaw_align_active
            return (
                move = [0.0, 0.0, 0.0],
                grip = -1,
                enable_yaw_objective = true,
                yaw_target = _object_yaw_target(current_belief),
                yaw_pi_symmetric = true,
                enable_topdown_objective = true,
            )
        end

        err = s_obj - reach_obj_rel
        abs_x = abs(err[1])
        abs_y = abs(err[2])
        x_ratio = abs_x / max(1e-6, reach_axis_threshold_x)
        y_ratio = abs_y / max(1e-6, reach_axis_threshold_y)

        if x_ratio >= y_ratio
            w_x = 1.0
            w_y = max(0.35, y_ratio / (x_ratio + 1e-9))
        else
            w_y = 1.0
            w_x = max(0.35, x_ratio / (y_ratio + 1e-9))
        end

        xy_max = max(abs_x, abs_y)
        if xy_max >= reach_z_blend_start
            z_weight = 0.0
        elseif xy_max <= reach_z_blend_full
            z_weight = 1.0
        else
            z_weight = (reach_z_blend_start - xy_max) / max(1e-6, (reach_z_blend_start - reach_z_blend_full))
        end

        desired_xy = [w_x * err[1], w_y * err[2]]

        if reach_arc_enabled && !reach_watchdog_active && reach_turn_sign != 0
            ee_xy = [s_ee[1], s_ee[2]]
            goal_xy = [s_ee[1] + err[1], s_ee[2] + err[2]]
            r_ee = norm(ee_xy)
            r_goal = norm(goal_xy)
            if r_ee > 1e-6 && r_goal > 1e-6
                theta_ee = atan(ee_xy[2], ee_xy[1])
                theta_goal = atan(goal_xy[2], goal_xy[1])
                dtheta_signed = _signed_angle_to_goal(theta_ee, theta_goal, reach_turn_sign)
                max_theta_step = deg2rad(reach_arc_max_theta_step_deg)
                theta_step = sign(reach_turn_sign) * min(max_theta_step, abs(dtheta_signed))
                theta_next = theta_ee + theta_step
                r_next = r_ee + clamp(r_goal - r_ee, -reach_arc_radial_step_max, reach_arc_radial_step_max)
                r_next = clamp(r_next, reach_arc_min_radius, reach_arc_max_radius)
                target_xy = [r_next * cos(theta_next), r_next * sin(theta_next)]
                desired_xy = target_xy - ee_xy
            end
        end

        step_floor = reach_step_min
        if reach_watchdog_active
            desired_xy = [err[1], err[2]]
            z_weight = max(z_weight, 0.35)
            step_floor = max(step_floor, 0.012)
        end

        desired_move = [desired_xy[1], desired_xy[2], z_weight * err[3]]
        desired_norm = norm(desired_move)
        if desired_norm > 0.0
            err_norm = norm(err)
            step_limit = clamp(0.35 * err_norm, step_floor, reach_delta)
            if desired_norm > step_limit
                desired_move = (desired_move / desired_norm) * step_limit
            end
        end
        return (
            move = desired_move,
            grip = -1,
            enable_yaw_objective = false,
            enable_topdown_objective = true,
        )
    elseif phase == :Align
        s_obj = Float64.(current_belief[:s_obj_mean])
        err = s_obj - align_obj_rel
        desired = [0.9 * err[1], 0.9 * err[2], 0.9 * err[3]]
        n = norm(desired)
        if n > align_step && n > 0.0
            desired = (desired / n) * align_step
        end
        return (
            move = desired,
            grip = -1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :PreGraspHold
        s_obj = Float64.(current_belief[:s_obj_mean])
        err = s_obj - align_obj_rel
        desired = [0.35 * err[1], 0.35 * err[2], 0.35 * err[3]]
        n = norm(desired)
        if n > pregrasp_hold_step && n > 0.0
            desired = (desired / n) * pregrasp_hold_step
        end
        return (
            move = desired,
            grip = -1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :Descend
        s_obj = Float64.(current_belief[:s_obj_mean])
        err = s_obj - descend_obj_rel

        abs_x = abs(err[1])
        abs_y = abs(err[2])
        abs_z = abs(err[3])

        # "Threshold-left" control:
        # prioritize axes still outside threshold.
        left_x = max(0.0, abs_x - descend_x_threshold)
        left_y = max(0.0, abs_y - descend_y_threshold)
        left_z = max(0.0, abs_z - descend_z_threshold)

        x_ratio = left_x / max(descend_x_threshold, 1e-6)
        y_ratio = left_y / max(descend_y_threshold, 1e-6)

        x_out = left_x > 0.0
        y_out = left_y > 0.0

        if x_out && y_out
            # Both axes out: priority by normalized threshold-left amount.
            w_x = max(0.35, x_ratio / (x_ratio + y_ratio + 1e-9))
            w_y = max(0.35, y_ratio / (x_ratio + y_ratio + 1e-9))
        elseif x_out
            # Keep solved axis nearly frozen while fixing the out-of-threshold axis.
            w_x = 1.0
            w_y = 0.05
        elseif y_out
            w_x = 0.05
            w_y = 1.0
        else
            # XY already within threshold: tiny XY correction only.
            w_x = 0.10
            w_y = 0.10
        end

        # Strict XY-first descent:
        # do not descend in Z until BOTH X and Y are inside threshold.
        if x_out || y_out
            z_weight = 0.0
        elseif left_z > 0.0
            z_weight = 1.0
        else
            z_weight = 0.0
        end

        desired = [w_x * err[1], w_y * err[2], z_weight * err[3]]
        n = norm(desired)
        if n > grasp_step && n > 0.0
            desired = (desired / n) * grasp_step
        end
        yaw_enable = !(x_out || y_out)
        return (
            move = desired,
            grip = -1,
            # Yaw alignment can perturb XY while descending near the object.
            # Enable yaw only after XY is settled.
            enable_yaw_objective = yaw_enable,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :CloseHold || phase == :Grasp
        return (
            move = [0.0, 0.0, 0.0],
            grip = 1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :LiftTest
        return (
            move = [0.0, 0.0, delta],
            grip = 1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :Transit
        return (
            move = [0.0, 0.0, delta],
            grip = 1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :MoveToPlaceAbove
        s_target = Float64.(current_belief[:s_target_mean])
        err = s_target - preplace_target_rel
        desired = [err[1], err[2], err[3]]
        n = norm(desired)
        if n > reach_delta && n > 0.0
            desired = (desired / n) * reach_delta
        end
        return (
            move = desired,
            grip = 1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :DescendToPlace
        s_target = Float64.(current_belief[:s_target_mean])
        err = s_target - place_target_rel
        xy_err = norm(err[1:2])
        z_err = abs(err[3])
        z_weight = (xy_err > place_xy_threshold) ? 0.0 : ((z_err > place_z_threshold) ? 1.0 : 0.0)
        desired = [err[1], err[2], z_weight * err[3]]
        n = norm(desired)
        if n > grasp_step && n > 0.0
            desired = (desired / n) * grasp_step
        end
        return (
            move = desired,
            grip = 1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :Open
        return (
            move = [0.0, 0.0, 0.0],
            grip = -1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :Retreat
        rv_norm = norm(retreat_move)
        desired = rv_norm > 1e-9 ? (retreat_move / rv_norm) * delta : [0.0, 0.0, delta]
        return (
            move = desired,
            grip = -1,
            enable_yaw_objective = true,
            yaw_target = _object_yaw_target(current_belief),
            yaw_pi_symmetric = true,
            enable_topdown_objective = true,
        )
    elseif phase == :Done
        return (
            move = [0.0, 0.0, 0.0],
            grip = -1,
            enable_yaw_objective = false,
            enable_topdown_objective = true,
        )
    elseif phase == :Lift
        return (move = [0.0, 0.0, delta], grip = 1)
    end

    candidate_actions = generate_candidate_actions(delta)

    best_action = nothing
    best_G = Inf
    for action in candidate_actions
        predicted_belief = predict_next_belief(current_belief, action, action_effectiveness)
        G = compute_efe(predicted_belief, phase, lambda_epistemic)
        if G < best_G
            best_G = G
            best_action = action
        end
    end

    return best_action
end
