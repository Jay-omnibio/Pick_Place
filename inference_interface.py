import numpy as np

"""
This module bridges Python to Julia (RxInfer).

Responsibilities:
- maintain belief state
- convert observations into inference inputs
- call RxInfer (later)
- return belief dictionary usable by action selection
"""

# Reach objective in object-relative coordinates: keep EE above the object center.
REACH_OBJ_REL = np.array([0.0, 0.0, -0.08])
APPROACH_THRESHOLD = 0.03
GRASP_HOLD_STEPS = 8
LIFT_STEPS = 20


def infer_beliefs(observation, previous_belief=None):
    """
    Perform one inference step.

    Parameters
    ----------
    observation : dict
        {
          "o_ee": np.array(3),
          "o_obj": np.array(3),
          "o_grip": float,
          "o_contact": int
        }
    previous_belief : dict or None
        belief state from previous timestep
    """

    # Initial belief (t = 0)
    if previous_belief is None:
        return {
            "s_ee_mean": observation["o_ee"].copy(),
            "s_obj_mean": observation["o_obj"].copy(),
            "s_target_mean": np.zeros(3),
            "s_ee_cov": np.eye(3) * 0.05,
            "s_obj_cov": np.eye(3) * 0.1,
            "s_target_cov": np.eye(3) * 0.1,
            "s_grasp": 0,   # Open
            "phase": "Reach",
            "grasp_timer": 0,
            "lift_timer": 0,
        }

    # Placeholder inference update (to be replaced by RxInfer call)
    alpha = 0.7

    s_ee_mean = alpha * observation["o_ee"] + (1 - alpha) * previous_belief["s_ee_mean"]
    s_obj_mean = alpha * observation["o_obj"] + (1 - alpha) * previous_belief["s_obj_mean"]

    # Simple grasp belief from contact signal.
    s_grasp = 1 if observation["o_contact"] == 1 else previous_belief["s_grasp"]

    # Phase logic aligned with manual baseline:
    # Reach (approach above object) -> Grasp (close) -> Lift -> Place.
    phase = previous_belief["phase"]
    grasp_timer = previous_belief.get("grasp_timer", 0)
    lift_timer = previous_belief.get("lift_timer", 0)

    if phase == "Reach":
        if np.linalg.norm(s_obj_mean - REACH_OBJ_REL) < APPROACH_THRESHOLD:
            phase = "Grasp"
            grasp_timer = 0
            lift_timer = 0
    elif phase == "Grasp":
        grasp_timer += 1
        if s_grasp == 1 and grasp_timer >= GRASP_HOLD_STEPS:
            phase = "Lift"
            lift_timer = 0
    elif phase == "Lift":
        # If grasp is lost while lifting, retry grasp.
        if s_grasp == 0:
            phase = "Grasp"
            grasp_timer = 0
        else:
            lift_timer += 1
            if lift_timer >= LIFT_STEPS:
                phase = "Place"

    return {
        "s_ee_mean": s_ee_mean,
        "s_obj_mean": s_obj_mean,
        "s_target_mean": previous_belief["s_target_mean"],
        "s_ee_cov": previous_belief["s_ee_cov"],
        "s_obj_cov": previous_belief["s_obj_cov"],
        "s_target_cov": previous_belief["s_target_cov"],
        "s_grasp": s_grasp,
        "phase": phase,
        "grasp_timer": grasp_timer,
        "lift_timer": lift_timer,
    }
