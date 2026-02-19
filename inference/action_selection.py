"""
Python wrapper for Julia-based action selection (Expected Free Energy).

Uses PyJulia to call Julia functions from Python.
"""

import numpy as np
from pathlib import Path
import os

# ================================================
# Julia interface setup
# ================================================

try:
    from julia import Main as jl
    
    # Add inference directory to Julia's load path using Julia syntax
    inference_dir = str(Path(__file__).parent)
    jl.eval(f'pushfirst!(LOAD_PATH, "{inference_dir}")')
    
    # Load the Julia module
    jl.include(str(Path(__file__).parent / "action_selection.jl"))
    
    JULIA_AVAILABLE = True
    
except Exception as e:
    # Silence PyJulia setup errors by default; pure Python path is supported.
    if os.getenv("ACTIVE_INFERENCE_VERBOSE_JULIA", "0") == "1":
        print(f"Warning: Julia/PyJulia initialization failed: {type(e).__name__}: {e}")
        print("Falling back to pure Python implementation.")
    JULIA_AVAILABLE = False


# ================================================
# Python wrapper function
# ================================================

def _require_param(params, key):
    if key not in params:
        raise KeyError(f"Missing active-inference config key: {key}")
    return params[key]


def _require_vec3(params, key):
    raw = np.asarray(_require_param(params, key), dtype=float).reshape(-1)
    if raw.shape[0] != 3:
        raise ValueError(f"Active-inference config key '{key}' must be a length-3 vector.")
    return raw.copy()


def select_action(current_belief, params=None):
    """
    Select the action that minimizes Expected Free Energy.
    
    Parameters
    ----------
    current_belief : dict
        Belief state containing:
        - s_ee_mean : np.ndarray (3,)
        - s_obj_mean : np.ndarray (3,)
        - s_target_mean : np.ndarray (3,)
        - s_ee_cov : np.ndarray (3, 3)
        - s_obj_cov : np.ndarray (3, 3)
        - s_target_cov : np.ndarray (3, 3)
        - s_grasp : int
        - phase : str or int
    
    Returns
    -------
    action : dict
        Action with keys:
        - "move" : list of 3 floats (ΔEE position)
        - "grip" : int (0=no-op, 1=close, -1=open)
    """
    
    params = params or {}

    if not JULIA_AVAILABLE:
        return _select_action_python(current_belief, params=params)
    
    try:
        # Convert Python belief dict to Julia dict
        belief_jl = _python_dict_to_julia(current_belief)
        params_jl = _python_dict_to_julia(params)
        
        # Call Julia function
        action_jl = jl.select_action(belief_jl, params_jl)
        
        # Convert Julia action back to Python dict
        action_py = _julia_action_to_python(action_jl)
        
        return action_py
        
    except Exception as e:
        print(f"Error calling Julia function: {e}")
        print("Falling back to pure Python implementation.")
        return _select_action_python(current_belief, params=params)


# ================================================
# Type conversions: Python ↔ Julia
# ================================================

def _python_dict_to_julia(data_py):
    """Convert Python dict to Julia-compatible format with symbol keys."""
    
    # Build dict with symbol keys using PyJulia's native approach
    out_jl = jl.Dict()
    
    for key, value in data_py.items():
        # Create symbol key (:key)
        sym_key = jl.Symbol(key)
        
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists (PyJulia auto-converts to Julia arrays)
            if value.ndim == 1:
                # 1D: list becomes Julia Vector
                out_jl[sym_key] = value.tolist()
            else:
                # 2D: nested list becomes Julia Matrix
                out_jl[sym_key] = value.tolist()
        elif isinstance(value, str):
            # Convert strings to Julia symbols (e.g., "Reach" -> :Reach)
            out_jl[sym_key] = jl.Symbol(value)
        else:
            # Numbers, ints, etc. pass through directly
            out_jl[sym_key] = value
    
    return out_jl


def _julia_action_to_python(action_jl):
    """Convert Julia action (NamedTuple) to Python dict."""
    
    # Julia NamedTuple can be accessed as attributes or indexed
    try:
        move = np.array(action_jl.move)
        grip = int(action_jl.grip)
    except:
        # Fallback if structure is different
        move = np.array([0.0, 0.0, 0.0])
        grip = 0

    action_py = {
        "move": move.tolist(),
        "grip": grip
    }

    optional_fields = (
        "enable_yaw_objective",
        "yaw_target",
        "yaw_pi_symmetric",
        "enable_topdown_objective",
    )
    for key in optional_fields:
        try:
            value = getattr(action_jl, key)
        except Exception:
            continue
        if key == "yaw_target":
            action_py[key] = float(value)
        else:
            action_py[key] = bool(value)

    return action_py


# ================================================
# Pure Python fallback implementation
# ================================================

def _select_action_python(current_belief, params=None):
    """
    Pure Python implementation of Expected Free Energy action selection.
    Mirrors the Julia version for when PyJulia is unavailable.
    """
    
    params = params or {}

    lambda_epistemic = float(_require_param(params, "lambda_epistemic"))
    delta = float(_require_param(params, "delta"))
    reach_delta = float(_require_param(params, "reach_delta"))
    action_effectiveness = float(_require_param(params, "action_effectiveness"))

    reach_axis_threshold_x = float(_require_param(params, "reach_axis_threshold_x"))
    reach_axis_threshold_y = float(_require_param(params, "reach_axis_threshold_y"))
    reach_z_blend_start = float(_require_param(params, "reach_z_blend_start"))
    reach_z_blend_full = float(_require_param(params, "reach_z_blend_full"))
    reach_step_min = float(_require_param(params, "reach_step_min"))

    reach_arc_enabled = bool(_require_param(params, "reach_arc_enabled"))
    reach_arc_max_theta_step_deg = float(_require_param(params, "reach_arc_max_theta_step_deg"))
    reach_arc_radial_step_max = float(_require_param(params, "reach_arc_radial_step_max"))
    reach_arc_min_radius = float(_require_param(params, "reach_arc_min_radius"))
    reach_arc_max_radius = float(_require_param(params, "reach_arc_max_radius"))

    grasp_step = float(_require_param(params, "grasp_step"))
    align_step = float(_require_param(params, "align_step"))
    pregrasp_hold_step = float(_require_param(params, "pregrasp_hold_step"))
    descend_x_threshold = float(_require_param(params, "descend_x_threshold"))
    descend_y_threshold = float(_require_param(params, "descend_y_threshold"))
    descend_z_threshold = float(_require_param(params, "descend_z_threshold"))
    preplace_xy_threshold = float(_require_param(params, "preplace_xy_threshold"))
    place_xy_threshold = float(_require_param(params, "place_xy_threshold"))
    place_z_threshold = float(_require_param(params, "place_z_threshold"))
    confidence_speed_scaling_enabled = bool(_require_param(params, "confidence_speed_scaling_enabled"))
    confidence_speed_min_scale = float(_require_param(params, "confidence_speed_min_scale"))
    reach_confidence_speed_power = float(_require_param(params, "reach_confidence_speed_power"))
    descend_confidence_speed_power = float(_require_param(params, "descend_confidence_speed_power"))

    cfg_reach_obj_rel = _require_vec3(params, "reach_obj_rel")
    cfg_align_obj_rel = _require_vec3(params, "align_obj_rel")
    cfg_descend_obj_rel = _require_vec3(params, "descend_obj_rel")
    cfg_preplace_target_rel = _require_vec3(params, "preplace_target_rel")
    cfg_place_target_rel = _require_vec3(params, "place_target_rel")
    cfg_retreat_move = _require_vec3(params, "retreat_move")

    def _wrap_to_pi(angle):
        return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)

    def _object_yaw_target():
        obj_yaw = float(current_belief.get("s_obj_yaw", 0.0))
        if not np.isfinite(obj_yaw):
            obj_yaw = 0.0
        # Top-down grasp orientation: gripper jaw axis orthogonal to object long axis.
        return _wrap_to_pi(obj_yaw + 0.5 * np.pi)

    # Generate candidate actions
    moves = [
        [delta, 0.0, 0.0],
        [-delta, 0.0, 0.0],
        [0.0, delta, 0.0],
        [0.0, -delta, 0.0],
        [0.0, 0.0, delta],
        [0.0, 0.0, -delta],
        [0.0, 0.0, 0.0],  # no movement
    ]
    
    candidate_actions = []
    for m in moves:
        candidate_actions.append({"move": m, "grip": 0})
    
    # gripper actions
    candidate_actions.append({"move": [0.0, 0.0, 0.0], "grip": 1})   # close
    candidate_actions.append({"move": [0.0, 0.0, 0.0], "grip": -1})  # open
    
    phase = current_belief.get("phase", "Reach")
    obs_conf = float(current_belief.get("obs_confidence", 1.0))
    if not np.isfinite(obs_conf):
        obs_conf = 1.0
    obs_conf = float(np.clip(obs_conf, 0.0, 1.0))

    def _phase_conf_scale(power):
        if not confidence_speed_scaling_enabled:
            return 1.0
        return float(max(confidence_speed_min_scale, obs_conf ** float(power)))

    reach_obj_rel = np.array(current_belief.get("reach_obj_rel", cfg_reach_obj_rel), dtype=float)
    align_obj_rel = np.array(
        current_belief.get("align_obj_rel", cfg_align_obj_rel),
        dtype=float,
    )
    descend_obj_rel = np.array(
        current_belief.get("descend_obj_rel", cfg_descend_obj_rel),
        dtype=float,
    )

    # Phase-specific control to match stable scripted behavior.
    if phase == "Reach" or phase == 1:
        def _signed_angle_to_goal(theta_now, theta_goal, turn_sign):
            if int(turn_sign) >= 0:
                return float((theta_goal - theta_now) % (2.0 * np.pi))
            return -float((theta_now - theta_goal) % (2.0 * np.pi))

        s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
        s_ee = np.array(current_belief.get("s_ee_mean", [0, 0, 0]), dtype=float)
        obj_yaw = float(current_belief.get("s_obj_yaw", 0.0))
        if not np.isfinite(obj_yaw):
            obj_yaw = 0.0
        reach_turn_sign = int(current_belief.get("reach_turn_sign", 0))
        reach_watchdog_active = bool(int(current_belief.get("reach_watchdog_active", 0)))
        reach_yaw_align_active = bool(int(current_belief.get("reach_yaw_align_active", 0)))

        if reach_yaw_align_active:
            yaw_target = _object_yaw_target()
            return {
                "move": [0.0, 0.0, 0.0],
                "grip": -1,
                "enable_yaw_objective": True,
                "yaw_target": float(yaw_target),
                "yaw_pi_symmetric": True,
                "enable_topdown_objective": True,
            }

        err = s_obj - reach_obj_rel
        abs_x = abs(float(err[0]))
        abs_y = abs(float(err[1]))
        x_ratio = abs_x / max(1e-6, reach_axis_threshold_x)
        y_ratio = abs_y / max(1e-6, reach_axis_threshold_y)

        # Axis-priority XY correction: whichever axis is farther from threshold gets more authority.
        if x_ratio >= y_ratio:
            w_x = 1.0
            w_y = max(0.35, y_ratio / (x_ratio + 1e-9))
        else:
            w_y = 1.0
            w_x = max(0.35, x_ratio / (y_ratio + 1e-9))

        xy_max = max(abs_x, abs_y)
        # Delay Z while XY is far; blend Z in smoothly as XY converges.
        if xy_max >= reach_z_blend_start:
            z_weight = 0.0
        elif xy_max <= reach_z_blend_full:
            z_weight = 1.0
        else:
            z_weight = (reach_z_blend_start - xy_max) / max(
                1e-6, (reach_z_blend_start - reach_z_blend_full)
            )

        desired_xy = np.array([w_x * err[0], w_y * err[1]], dtype=float)

        # Reach arc/spiral mode with locked CW/CCW direction.
        if reach_arc_enabled and (not reach_watchdog_active) and reach_turn_sign != 0:
            ee_xy = np.asarray(s_ee[:2], dtype=float)
            goal_xy = np.asarray(s_ee[:2] + err[:2], dtype=float)
            r_ee = float(np.linalg.norm(ee_xy))
            r_goal = float(np.linalg.norm(goal_xy))
            if r_ee > 1e-6 and r_goal > 1e-6:
                theta_ee = float(np.arctan2(ee_xy[1], ee_xy[0]))
                theta_goal = float(np.arctan2(goal_xy[1], goal_xy[0]))
                dtheta_signed = _signed_angle_to_goal(
                    theta_now=theta_ee,
                    theta_goal=theta_goal,
                    turn_sign=reach_turn_sign,
                )
                max_theta_step = float(np.deg2rad(reach_arc_max_theta_step_deg))
                theta_step = float(np.sign(reach_turn_sign)) * min(max_theta_step, abs(float(dtheta_signed)))
                theta_next = theta_ee + theta_step
                r_next = r_ee + float(
                    np.clip(r_goal - r_ee, -reach_arc_radial_step_max, reach_arc_radial_step_max)
                )
                r_next = float(np.clip(r_next, reach_arc_min_radius, reach_arc_max_radius))
                target_xy = np.array([r_next * np.cos(theta_next), r_next * np.sin(theta_next)], dtype=float)
                desired_xy = target_xy - ee_xy

        # Watchdog fallback: disable arc and push a little stronger linear correction.
        reach_conf_scale = _phase_conf_scale(reach_confidence_speed_power)
        step_floor = reach_step_min * reach_conf_scale
        if reach_watchdog_active:
            desired_xy = np.array([err[0], err[1]], dtype=float)
            # Critical: when arc stalls, force non-zero Z coupling so we can escape
            # XY-only deadlocks near kinematic limits.
            z_weight = max(z_weight, 0.35)
            step_floor = max(step_floor, 0.012 * reach_conf_scale)

        desired_move = np.array([desired_xy[0], desired_xy[1], z_weight * err[2]], dtype=float)
        desired_norm = float(np.linalg.norm(desired_move))
        if desired_norm > 0.0:
            # Dynamic step: larger when far, but never tiny-stuck near saturation.
            err_norm = float(np.linalg.norm(err))
            step_limit = float(np.clip(0.35 * err_norm, step_floor, reach_delta * reach_conf_scale))
            if desired_norm > step_limit:
                desired_move = (desired_move / desired_norm) * step_limit
        return {
            "move": desired_move.tolist(),
            "grip": -1,
            "enable_yaw_objective": False,
            "enable_topdown_objective": True,
        }

    if phase == "Align":
        s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
        err = s_obj - align_obj_rel
        desired = np.array([0.9 * err[0], 0.9 * err[1], 0.9 * err[2]], dtype=float)
        n = np.linalg.norm(desired)
        if n > align_step and n > 0:
            desired = (desired / n) * align_step
        # Keep gripper open while aligning.
        return {
            "move": desired.tolist(),
            "grip": -1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "PreGraspHold":
        # Hold above object to settle posture before final descend, but keep a
        # tiny corrective motion so drift does not accumulate into a stall.
        s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
        err = s_obj - align_obj_rel
        desired = np.array([0.35 * err[0], 0.35 * err[1], 0.35 * err[2]], dtype=float)
        n = np.linalg.norm(desired)
        if n > pregrasp_hold_step and n > 0:
            desired = (desired / n) * pregrasp_hold_step
        return {
            "move": desired.tolist(),
            "grip": -1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "Descend":
        s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
        err = s_obj - descend_obj_rel
        abs_x = abs(float(err[0]))
        abs_y = abs(float(err[1]))
        abs_z = abs(float(err[2]))

        # "Threshold-left" control:
        # prioritize axes still outside threshold.
        left_x = max(0.0, abs_x - descend_x_threshold)
        left_y = max(0.0, abs_y - descend_y_threshold)
        left_z = max(0.0, abs_z - descend_z_threshold)

        x_ratio = left_x / max(descend_x_threshold, 1e-6)
        y_ratio = left_y / max(descend_y_threshold, 1e-6)

        x_out = left_x > 0.0
        y_out = left_y > 0.0

        if x_out and y_out:
            # Both axes out: priority by normalized threshold-left amount.
            w_x = max(0.35, x_ratio / (x_ratio + y_ratio + 1e-9))
            w_y = max(0.35, y_ratio / (x_ratio + y_ratio + 1e-9))
        elif x_out:
            # Keep solved axis nearly frozen while fixing the out-of-threshold axis.
            w_x = 1.0
            w_y = 0.05
        elif y_out:
            w_x = 0.05
            w_y = 1.0
        else:
            # XY already within threshold: tiny XY correction only.
            w_x = 0.10
            w_y = 0.10

        # Strict XY-first descent:
        # do not descend in Z until BOTH X and Y are inside threshold.
        if x_out or y_out:
            z_weight = 0.0
        elif left_z > 0.0:
            z_weight = 1.0
        else:
            z_weight = 0.0

        desired = np.array([w_x * err[0], w_y * err[1], z_weight * err[2]], dtype=float)
        n = np.linalg.norm(desired)
        descend_conf_scale = _phase_conf_scale(descend_confidence_speed_power)
        descend_step_cap = max(1e-6, grasp_step * descend_conf_scale)
        if n > descend_step_cap and n > 0:
            desired = (desired / n) * descend_step_cap
        yaw_enable = not (x_out or y_out)
        return {
            "move": desired.tolist(),
            "grip": -1,
            # Yaw alignment can perturb XY while descending near the object.
            # Enable yaw only after XY is settled.
            "enable_yaw_objective": yaw_enable,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "CloseHold" or phase == "Grasp" or phase == 2:
        # Close in place: no Cartesian motion while fingers settle/contact.
        return {
            "move": [0.0, 0.0, 0.0],
            "grip": 1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "LiftTest":
        return {
            "move": [0.0, 0.0, delta],
            "grip": 1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "Transit":
        return {
            "move": [0.0, 0.0, delta],
            "grip": 1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "MoveToPlaceAbove":
        s_target = np.array(current_belief.get("s_target_mean", [0, 0, 0]), dtype=float)
        preplace_target_rel = np.array(
            current_belief.get("preplace_target_rel", cfg_preplace_target_rel), dtype=float
        )
        err = s_target - preplace_target_rel
        desired = np.array([err[0], err[1], err[2]], dtype=float)
        n = np.linalg.norm(desired)
        if n > reach_delta and n > 0:
            desired = (desired / n) * reach_delta
        return {
            "move": desired.tolist(),
            "grip": 1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "DescendToPlace":
        s_target = np.array(current_belief.get("s_target_mean", [0, 0, 0]), dtype=float)
        place_target_rel = np.array(
            current_belief.get("place_target_rel", cfg_place_target_rel), dtype=float
        )
        err = s_target - place_target_rel
        xy_err = float(np.linalg.norm(err[:2]))
        z_err = abs(float(err[2]))
        z_weight = 0.0 if (xy_err > place_xy_threshold) else (1.0 if z_err > place_z_threshold else 0.0)
        desired = np.array([err[0], err[1], z_weight * err[2]], dtype=float)
        n = np.linalg.norm(desired)
        if n > grasp_step and n > 0:
            desired = (desired / n) * grasp_step
        return {
            "move": desired.tolist(),
            "grip": 1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "Open":
        return {
            "move": [0.0, 0.0, 0.0],
            "grip": -1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "Retreat":
        retreat_move = np.array(current_belief.get("retreat_move", cfg_retreat_move), dtype=float)
        n = float(np.linalg.norm(retreat_move))
        if n > 1e-9:
            desired = (retreat_move / n) * delta
        else:
            desired = np.array([0.0, 0.0, delta], dtype=float)
        return {
            "move": desired.tolist(),
            "grip": -1,
            "enable_yaw_objective": True,
            "yaw_target": float(_object_yaw_target()),
            "yaw_pi_symmetric": True,
            "enable_topdown_objective": True,
        }

    if phase == "Done":
        return {
            "move": [0.0, 0.0, 0.0],
            "grip": -1,
            "enable_yaw_objective": False,
            "enable_topdown_objective": True,
        }

    if phase == "LegacyGrasp":
        # legacy fallback kept for compatibility
        if int(current_belief.get("s_grasp", 0)) == 0:
            s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
            err = s_obj - descend_obj_rel
            # Final descend from above: prioritize Z, keep XY drift very small.
            desired = np.array([0.15 * err[0], 0.15 * err[1], 1.0 * err[2]], dtype=float)
            n = np.linalg.norm(desired)
            if n > grasp_step and n > 0:
                desired = (desired / n) * grasp_step
            return {"move": desired.tolist(), "grip": 1}
        return {"move": [0.0, 0.0, 0.0], "grip": 1}

    if phase == "Lift" or phase == 3:
        return {"move": [0.0, 0.0, delta], "grip": 1}

    # Place (or unknown) falls back to EFE scan.
    best_action = candidate_actions[0]
    best_G = float('inf')
    
    for action in candidate_actions:
        # Predict next belief
        predicted_belief = _predict_next_belief_python(
            current_belief,
            action,
            action_effectiveness=action_effectiveness,
        )
        
        # Compute EFE
        G = _compute_efe_python(predicted_belief, phase, lambda_epistemic)
        
        if G < best_G:
            best_G = G
            best_action = action
    
    return best_action


def _predict_next_belief_python(current_belief, action, action_effectiveness):
    """Predict next belief given current belief and action (Python)."""
    
    s_ee_mean = np.array(current_belief.get("s_ee_mean", [0, 0, 0]))
    s_obj_mean = np.array(current_belief.get("s_obj_mean", [0, 0, 0]))
    s_target_mean = np.array(current_belief.get("s_target_mean", [0, 0, 0]))
    
    move = np.array(action["move"])
    effective_move = action_effectiveness * move

    # EE update
    next_s_ee = s_ee_mean + effective_move

    # Object relative update with damped closed-loop response.
    next_s_obj = s_obj_mean - effective_move

    # Target relative update
    next_s_target = s_target_mean - effective_move
    
    # Covariances grow slightly (uncertainty propagation)
    base_cov = np.array(current_belief.get("s_obj_cov", np.eye(3)))
    next_cov = base_cov + 0.01 * np.eye(3)
    
    return {
        "s_ee_mean": next_s_ee,
        "s_obj_mean": next_s_obj,
        "s_target_mean": next_s_target,
        "s_obj_cov": next_cov,
        "s_target_cov": next_cov,
        "phase": current_belief.get("phase", "Reach")
    }


def _compute_efe_python(belief, phase, lambda_epistemic):
    """Compute Expected Free Energy (Python)."""
    
    # Pragmatic term
    pragmatic_cost = 0.0
    
    if phase == "Reach" or phase == 1:
        # want object close to EE → relative position near zero
        obj_mean = np.array(belief.get("s_obj_mean", [0, 0, 0]))
        pragmatic_cost = np.linalg.norm(obj_mean) ** 2
        
    elif phase == "Place" or phase == 4:
        # want object close to target
        target_mean = np.array(belief.get("s_target_mean", [0, 0, 0]))
        pragmatic_cost = np.linalg.norm(target_mean) ** 2
    
    # Epistemic term
    obj_cov = np.array(belief.get("s_obj_cov", np.eye(3)))
    target_cov = np.array(belief.get("s_target_cov", np.eye(3)))
    
    epistemic_cost = np.trace(obj_cov) + np.trace(target_cov)
    
    # Expected Free Energy
    G = pragmatic_cost + lambda_epistemic * epistemic_cost
    
    return G
