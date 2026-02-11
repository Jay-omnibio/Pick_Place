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

def select_action(current_belief):
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
    
    if not JULIA_AVAILABLE:
        return _select_action_python(current_belief)
    
    try:
        # Convert Python belief dict to Julia dict
        belief_jl = _python_belief_to_julia(current_belief)
        
        # Call Julia function
        action_jl = jl.select_action(belief_jl)
        
        # Convert Julia action back to Python dict
        action_py = _julia_action_to_python(action_jl)
        
        return action_py
        
    except Exception as e:
        print(f"Error calling Julia function: {e}")
        print("Falling back to pure Python implementation.")
        return _select_action_python(current_belief)


# ================================================
# Type conversions: Python ↔ Julia
# ================================================

def _python_belief_to_julia(belief_py):
    """Convert Python belief dict to Julia-compatible format with symbol keys."""
    
    # Build dict with symbol keys using PyJulia's native approach
    belief_jl = jl.Dict()
    
    for key, value in belief_py.items():
        # Create symbol key (:key)
        sym_key = jl.Symbol(key)
        
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists (PyJulia auto-converts to Julia arrays)
            if value.ndim == 1:
                # 1D: list becomes Julia Vector
                belief_jl[sym_key] = value.tolist()
            else:
                # 2D: nested list becomes Julia Matrix
                belief_jl[sym_key] = value.tolist()
        elif isinstance(value, str):
            # Convert strings to Julia symbols (e.g., "Reach" -> :Reach)
            belief_jl[sym_key] = jl.Symbol(value)
        else:
            # Numbers, ints, etc. pass through directly
            belief_jl[sym_key] = value
    
    return belief_jl


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
    
    return {
        "move": move.tolist(),
        "grip": grip
    }


# ================================================
# Pure Python fallback implementation
# ================================================

def _select_action_python(current_belief):
    """
    Pure Python implementation of Expected Free Energy action selection.
    Mirrors the Julia version for when PyJulia is unavailable.
    """
    
    # Configuration
    LAMBDA_EPISTEMIC = 0.1
    DELTA = 0.018
    REACH_OBJ_REL = np.array([0.0, 0.0, -0.08])
    XY_ALIGN_THRESHOLD = 0.04
    XY_BLEND_FLOOR = 0.25
    GRASP_STEP = 0.008
    ALIGN_STEP = 0.010
    GRASP_SIDE_OFFSET_Y = 0.020
    ALIGN_REL_Z = -0.09

    # Generate candidate actions
    moves = [
        [DELTA, 0.0, 0.0],
        [-DELTA, 0.0, 0.0],
        [0.0, DELTA, 0.0],
        [0.0, -DELTA, 0.0],
        [0.0, 0.0, DELTA],
        [0.0, 0.0, -DELTA],
        [0.0, 0.0, 0.0],  # no movement
    ]
    
    candidate_actions = []
    for m in moves:
        candidate_actions.append({"move": m, "grip": 0})
    
    # gripper actions
    candidate_actions.append({"move": [0.0, 0.0, 0.0], "grip": 1})   # close
    candidate_actions.append({"move": [0.0, 0.0, 0.0], "grip": -1})  # open
    
    phase = current_belief.get("phase", "Reach")

    # Phase-specific control to match stable scripted behavior.
    if phase == "Reach" or phase == 1:
        s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
        err = s_obj - REACH_OBJ_REL
        xy_err_norm = float(np.linalg.norm(err[:2]))

        # Adaptive blend:
        # - large XY error => mostly XY correction
        # - small XY error => allow stronger Z correction
        # Keeps some Z authority in all cases to avoid hard-mode switching stalls.
        z_weight = min(1.0, max(XY_BLEND_FLOOR, 1.0 - (xy_err_norm / XY_ALIGN_THRESHOLD)))
        desired_move = np.array([err[0], err[1], z_weight * err[2]], dtype=float)

        norm = np.linalg.norm(desired_move)
        if norm > DELTA and norm > 0:
            desired_move = (desired_move / norm) * DELTA
        return {"move": desired_move.tolist(), "grip": -1}

    if phase == "Align":
        s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
        side_sign = float(current_belief.get("grasp_side_sign", -1.0 if s_obj[1] >= 0.0 else 1.0))
        align_rel = np.array([0.0, side_sign * GRASP_SIDE_OFFSET_Y, ALIGN_REL_Z], dtype=float)
        err = s_obj - align_rel
        desired = np.array([0.7 * err[0], 1.0 * err[1], 0.8 * err[2]], dtype=float)
        n = np.linalg.norm(desired)
        if n > ALIGN_STEP and n > 0:
            desired = (desired / n) * ALIGN_STEP
        # Keep gripper open while aligning.
        return {"move": desired.tolist(), "grip": -1}

    if phase == "Grasp" or phase == 2:
        # In grasp, center object between fingers and close.
        if int(current_belief.get("s_grasp", 0)) == 0:
            s_obj = np.array(current_belief.get("s_obj_mean", [0, 0, 0]), dtype=float)
            grasp_rel = np.array([0.0, 0.0, -0.08], dtype=float)
            err = s_obj - grasp_rel
            desired = np.array([0.9 * err[0], 0.9 * err[1], 0.9 * err[2]], dtype=float)
            n = np.linalg.norm(desired)
            if n > GRASP_STEP and n > 0:
                desired = (desired / n) * GRASP_STEP
            return {"move": desired.tolist(), "grip": 1}
        return {"move": [0.0, 0.0, 0.0], "grip": 1}

    if phase == "Lift" or phase == 3:
        return {"move": [0.0, 0.0, DELTA], "grip": 1}

    # Place (or unknown) falls back to EFE scan.
    best_action = candidate_actions[0]
    best_G = float('inf')
    
    for action in candidate_actions:
        # Predict next belief
        predicted_belief = _predict_next_belief_python(current_belief, action)
        
        # Compute EFE
        G = _compute_efe_python(predicted_belief, phase, LAMBDA_EPISTEMIC)
        
        if G < best_G:
            best_G = G
            best_action = action
    
    return best_action


def _predict_next_belief_python(current_belief, action):
    """Predict next belief given current belief and action (Python)."""
    
    s_ee_mean = np.array(current_belief.get("s_ee_mean", [0, 0, 0]))
    s_obj_mean = np.array(current_belief.get("s_obj_mean", [0, 0, 0]))
    s_target_mean = np.array(current_belief.get("s_target_mean", [0, 0, 0]))
    
    move = np.array(action["move"])
    action_effectiveness = 0.4
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
