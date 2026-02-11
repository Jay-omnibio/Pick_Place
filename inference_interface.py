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
ALIGN_SIDE_OFFSET_Y = 0.02
ALIGN_REL_Z = -0.09
ALIGN_THRESHOLD = 0.025
ALIGN_MAX_STEPS = 40
GRASP_HOLD_STEPS = 12
GRASP_SEARCH_STEPS = 60
GRASP_STABLE_STEPS_FOR_LIFT = 10
LIFT_STEPS = 20
ALPHA_EE = 0.7
ALPHA_OBJ_DEFAULT = 0.7
ALPHA_OBJ_REACH = 0.9
OBJ_REACQUIRE_JUMP = 0.05
CONTACT_ON_COUNT = 2
CONTACT_OFF_COUNT = 2


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
            "align_timer": 0,
            "lift_timer": 0,
            "contact_counter": 0,
            "stable_grasp_counter": 0,
            "grasp_side_sign": 1.0,
        }

    # Placeholder inference update (to be replaced by RxInfer call)
    s_ee_mean = ALPHA_EE * observation["o_ee"] + (1 - ALPHA_EE) * previous_belief["s_ee_mean"]

    prev_phase = previous_belief["phase"]
    prev_obj = previous_belief["s_obj_mean"]
    obj_obs = observation["o_obj"]
    obj_jump = np.linalg.norm(obj_obs - prev_obj)
    alpha_obj = ALPHA_OBJ_REACH if prev_phase == "Reach" else ALPHA_OBJ_DEFAULT
    if obj_jump > OBJ_REACQUIRE_JUMP:
        # Fast object motion (fall/slide): trust current observation more.
        alpha_obj = max(alpha_obj, 0.95)
    s_obj_mean = alpha_obj * obj_obs + (1 - alpha_obj) * prev_obj

    # Contact hysteresis to avoid sticky/latched grasp state.
    contact_counter = previous_belief.get("contact_counter", 0)
    if observation["o_contact"] == 1:
        contact_counter = min(contact_counter + 1, 10)
    else:
        contact_counter = max(contact_counter - 1, -10)

    prev_grasp = previous_belief["s_grasp"]
    if prev_grasp == 0:
        s_grasp = 1 if contact_counter >= CONTACT_ON_COUNT else 0
    else:
        s_grasp = 0 if contact_counter <= -CONTACT_OFF_COUNT else 1

    # Phase logic aligned with manual baseline:
    # Reach (approach above object) -> Grasp (close) -> Lift -> Place.
    phase = prev_phase
    grasp_timer = previous_belief.get("grasp_timer", 0)
    align_timer = previous_belief.get("align_timer", 0)
    lift_timer = previous_belief.get("lift_timer", 0)
    stable_grasp_counter = previous_belief.get("stable_grasp_counter", 0)
    grasp_side_sign = float(previous_belief.get("grasp_side_sign", 1.0))
    reach_error = np.linalg.norm(s_obj_mean - REACH_OBJ_REL)

    if phase == "Reach":
        # Enter alignment stage when pregrasp geometry is good.
        if reach_error < APPROACH_THRESHOLD:
            phase = "Align"
            # Pick a side based on object lateral sign to approach between fingers.
            grasp_side_sign = -1.0 if s_obj_mean[1] >= 0.0 else 1.0
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
    elif phase == "Align":
        align_timer += 1
        align_rel = np.array([0.0, grasp_side_sign * ALIGN_SIDE_OFFSET_Y, ALIGN_REL_Z])
        align_error = np.linalg.norm(s_obj_mean - align_rel)
        if align_error < ALIGN_THRESHOLD or align_timer >= ALIGN_MAX_STEPS:
            phase = "Grasp"
            grasp_timer = 0
            stable_grasp_counter = 0
    elif phase == "Grasp":
        # Allow several grasp-attempt steps before giving up and reacquiring.
        grasp_timer += 1
        # Require stable consecutive contact+grasp before lifting.
        if s_grasp == 1 and observation["o_contact"] == 1:
            stable_grasp_counter += 1
        else:
            stable_grasp_counter = 0

        if s_grasp == 1:
            if grasp_timer >= GRASP_HOLD_STEPS and stable_grasp_counter >= GRASP_STABLE_STEPS_FOR_LIFT:
                phase = "Lift"
                lift_timer = 0
        else:
            if grasp_timer >= GRASP_SEARCH_STEPS:
                phase = "Reach"
                align_timer = 0
                grasp_timer = 0
                lift_timer = 0
                stable_grasp_counter = 0
    elif phase == "Lift":
        # If grasp is lost while lifting, reacquire from current observation.
        if s_grasp == 0:
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
        else:
            lift_timer += 1
            if lift_timer >= LIFT_STEPS:
                phase = "Place"
    elif phase == "Place":
        # If object is dropped at place stage, reacquire and retry.
        grasp_timer += 1
        if s_grasp == 0:
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0

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
        "align_timer": align_timer,
        "lift_timer": lift_timer,
        "contact_counter": contact_counter,
        "stable_grasp_counter": stable_grasp_counter,
        "grasp_side_sign": grasp_side_sign,
    }

