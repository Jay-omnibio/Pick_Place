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
ALIGN_SIDE_OFFSET_Y = 0.00
ALIGN_REL_Z = -0.10
DESCEND_REL_Z = -0.10
ALIGN_THRESHOLD = 0.025
DESCEND_THRESHOLD = 0.018
ALIGN_MAX_STEPS = 40
ALIGN_MIN_STEPS = 6
PREGRASP_HOLD_STEPS = 20
DESCEND_MAX_STEPS = 80
CLOSE_HOLD_STEPS = 20
GRASP_SEARCH_STEPS = 60
GRASP_STABLE_STEPS_FOR_LIFT = 10
REACH_REENTRY_COOLDOWN_STEPS = 20
LIFT_TEST_STEPS = 16
LIFT_TEST_OBJ_REL_DRIFT_MAX = 0.06
MAX_RETRIES = 3
ALPHA_EE = 0.7
ALPHA_OBJ_DEFAULT = 0.7
ALPHA_OBJ_REACH = 0.9
OBJ_REACQUIRE_JUMP = 0.05
CONTACT_ON_COUNT = 2
CONTACT_OFF_COUNT = 2
GRIP_OPEN_TARGET = 0.04
GRIP_CLOSE_TARGET = 0.0
GRIP_READY_WIDTH_TOL = 0.003
GRIP_READY_SPEED_TOL = 0.0015
GRIP_CLOSE_READY_MAX_WIDTH = 0.035


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
            "reach_cooldown": 0,
            "prev_o_grip": float(observation["o_grip"]),
            "pregrasp_hold_timer": 0,
            "descend_timer": 0,
            "close_hold_timer": 0,
            "lift_test_timer": 0,
            "lift_test_ref_obj_rel": observation["o_obj"].copy(),
            "retry_count": 0,
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
    reach_cooldown = int(previous_belief.get("reach_cooldown", 0))
    pregrasp_hold_timer = int(previous_belief.get("pregrasp_hold_timer", 0))
    descend_timer = int(previous_belief.get("descend_timer", 0))
    close_hold_timer = int(previous_belief.get("close_hold_timer", 0))
    lift_test_timer = int(previous_belief.get("lift_test_timer", 0))
    lift_test_ref_obj_rel = np.array(
        previous_belief.get("lift_test_ref_obj_rel", s_obj_mean.copy()), dtype=float
    )
    retry_count = int(previous_belief.get("retry_count", 0))
    reach_error = np.linalg.norm(s_obj_mean - REACH_OBJ_REL)
    prev_o_grip = float(previous_belief.get("prev_o_grip", observation["o_grip"]))
    grip_obs = float(observation["o_grip"])
    grip_speed = abs(grip_obs - prev_o_grip)
    gripper_open_ready = (
        abs(grip_obs - GRIP_OPEN_TARGET) <= GRIP_READY_WIDTH_TOL and grip_speed <= GRIP_READY_SPEED_TOL
    )
    gripper_close_ready = (
        grip_speed <= GRIP_READY_SPEED_TOL
        and (
            abs(grip_obs - GRIP_CLOSE_TARGET) <= GRIP_READY_WIDTH_TOL
            or (grip_obs <= GRIP_CLOSE_READY_MAX_WIDTH and observation["o_contact"] == 1)
        )
    )
    if reach_cooldown > 0:
        reach_cooldown -= 1

    if phase == "Reach":
        # Enter alignment stage when pregrasp geometry is good.
        if reach_cooldown == 0 and reach_error < APPROACH_THRESHOLD:
            phase = "Align"
            # Top-down approach: keep centerline, no side offset.
            grasp_side_sign = 0.0
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            pregrasp_hold_timer = 0
            descend_timer = 0
            close_hold_timer = 0
            lift_test_timer = 0
    elif phase == "Align":
        align_timer += 1
        align_rel = np.array([0.0, 0.0, ALIGN_REL_Z])
        align_error = np.linalg.norm(s_obj_mean - align_rel)
        align_ready = align_timer >= ALIGN_MIN_STEPS and gripper_open_ready
        # Timeout should force progression to avoid getting stuck in Align.
        if (align_ready and align_error < ALIGN_THRESHOLD) or (align_timer >= ALIGN_MAX_STEPS):
            phase = "PreGraspHold"
            pregrasp_hold_timer = 0
            grasp_timer = 0
            stable_grasp_counter = 0
            descend_timer = 0
            close_hold_timer = 0
            lift_test_timer = 0
    elif phase == "PreGraspHold":
        pregrasp_hold_timer += 1
        if pregrasp_hold_timer >= PREGRASP_HOLD_STEPS and gripper_open_ready:
            phase = "Descend"
            descend_timer = 0
            stable_grasp_counter = 0
    elif phase == "Descend":
        descend_timer += 1
        descend_rel = np.array([0.0, 0.0, DESCEND_REL_Z])
        descend_error = np.linalg.norm(s_obj_mean - descend_rel)
        if descend_error < DESCEND_THRESHOLD or descend_timer >= DESCEND_MAX_STEPS:
            phase = "CloseHold"
            close_hold_timer = 0
            stable_grasp_counter = 0
    elif phase == "CloseHold":
        close_hold_timer += 1
        if s_grasp == 1 and observation["o_contact"] == 1:
            stable_grasp_counter += 1
        else:
            stable_grasp_counter = 0

        if (
            close_hold_timer >= CLOSE_HOLD_STEPS
            and stable_grasp_counter >= GRASP_STABLE_STEPS_FOR_LIFT
            and gripper_close_ready
        ):
            phase = "LiftTest"
            lift_test_timer = 0
            lift_test_ref_obj_rel = s_obj_mean.copy()
        elif close_hold_timer >= GRASP_SEARCH_STEPS:
            # Timed out without reliable grasp confirmation.
            retry_count += 1
            if retry_count > MAX_RETRIES:
                retry_count = 0
                phase = "Reach"
            else:
                phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = REACH_REENTRY_COOLDOWN_STEPS
            pregrasp_hold_timer = 0
            descend_timer = 0
            close_hold_timer = 0
            lift_test_timer = 0
    elif phase == "LiftTest":
        lift_test_timer += 1
        obj_rel_drift = np.linalg.norm(s_obj_mean - lift_test_ref_obj_rel)
        if s_grasp == 0:
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = REACH_REENTRY_COOLDOWN_STEPS
            pregrasp_hold_timer = 0
            descend_timer = 0
            close_hold_timer = 0
            lift_test_timer = 0
        elif obj_rel_drift > LIFT_TEST_OBJ_REL_DRIFT_MAX:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                retry_count = 0
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = REACH_REENTRY_COOLDOWN_STEPS
            pregrasp_hold_timer = 0
            descend_timer = 0
            close_hold_timer = 0
            lift_test_timer = 0
        else:
            if lift_test_timer >= LIFT_TEST_STEPS:
                retry_count = 0
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
            reach_cooldown = REACH_REENTRY_COOLDOWN_STEPS
            pregrasp_hold_timer = 0
            descend_timer = 0
            close_hold_timer = 0
            lift_test_timer = 0

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
        "reach_cooldown": reach_cooldown,
        "prev_o_grip": grip_obs,
        "pregrasp_hold_timer": pregrasp_hold_timer,
        "descend_timer": descend_timer,
        "close_hold_timer": close_hold_timer,
        "lift_test_timer": lift_test_timer,
        "lift_test_ref_obj_rel": lift_test_ref_obj_rel,
        "retry_count": retry_count,
    }
