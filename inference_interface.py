import numpy as np

"""
This module bridges Python ↔ Julia (RxInfer).

Responsibilities:
- maintain belief state
- convert observations into inference inputs
- call RxInfer (later)
- return belief dictionary usable by EFE
"""


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

    Returns
    -------
    belief : dict
        belief over latent states
    """

    # ------------------------------------------------
    # INITIAL BELIEF (t = 0)
    # ------------------------------------------------
    if previous_belief is None:
        belief = {
            "s_ee_mean": observation["o_ee"].copy(),
            "s_obj_mean": observation["o_obj"].copy(),
            "s_target_mean": np.zeros(3),

            "s_ee_cov": np.eye(3) * 0.05,
            "s_obj_cov": np.eye(3) * 0.1,
            "s_target_cov": np.eye(3) * 0.1,

            "s_grasp": 0,   # Open
            "phase": "Reach",
        }
        return belief

    # ------------------------------------------------
    # PLACEHOLDER INFERENCE UPDATE
    # (This will be replaced by RxInfer call)
    # ------------------------------------------------

    # Simple Bayesian update intuition (mock)
    alpha = 0.7

    s_ee_mean = (
        alpha * observation["o_ee"]
        + (1 - alpha) * previous_belief["s_ee_mean"]
    )

    s_obj_mean = (
        alpha * observation["o_obj"]
        + (1 - alpha) * previous_belief["s_obj_mean"]
    )

    # Update grasp belief based on contact
    if observation["o_contact"] == 1:
        s_grasp = 1
    else:
        s_grasp = previous_belief["s_grasp"]

    # Phase logic (VERY simple placeholder)
    phase = previous_belief["phase"]
    if phase == "Reach" and np.linalg.norm(s_obj_mean) < 0.05:
        phase = "Grasp"
    elif phase == "Grasp" and s_grasp == 1:
        phase = "Lift"
    elif phase == "Lift":
        phase = "Place"

    belief = {
        "s_ee_mean": s_ee_mean,
        "s_obj_mean": s_obj_mean,
        "s_target_mean": previous_belief["s_target_mean"],

        "s_ee_cov": previous_belief["s_ee_cov"],
        "s_obj_cov": previous_belief["s_obj_cov"],
        "s_target_cov": previous_belief["s_target_cov"],

        "s_grasp": s_grasp,
        "phase": phase,
    }

    return belief
