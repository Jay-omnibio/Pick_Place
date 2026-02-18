import numpy as np

"""
This module bridges Python to Julia (RxInfer).

Responsibilities:
- maintain belief state
- convert observations into inference inputs
- call RxInfer (later)
- return belief dictionary usable by action selection
"""


def _require_param(params, key):
    if key not in params:
        raise KeyError(f"Missing active-inference config key: {key}")
    return params[key]


def _require_vec3(params, key):
    raw = np.asarray(_require_param(params, key), dtype=float).reshape(-1)
    if raw.shape[0] != 3:
        raise ValueError(f"Active-inference config key '{key}' must be a length-3 vector.")
    return raw.copy()


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def infer_beliefs(observation, previous_belief=None, params=None):
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

    params = params or {}
    reach_obj_rel = _require_vec3(params, "reach_obj_rel")
    align_obj_rel = _require_vec3(params, "align_obj_rel")
    descend_obj_rel = _require_vec3(params, "descend_obj_rel")

    approach_threshold = float(_require_param(params, "approach_threshold"))
    align_threshold = float(_require_param(params, "align_threshold"))
    descend_threshold = float(_require_param(params, "descend_threshold"))
    align_max_steps = int(_require_param(params, "align_max_steps"))
    align_min_steps = int(_require_param(params, "align_min_steps"))
    pregrasp_hold_steps = int(_require_param(params, "pregrasp_hold_steps"))
    pregrasp_hold_ready_grace_steps = int(_require_param(params, "pregrasp_hold_ready_grace_steps"))
    descend_max_steps = int(_require_param(params, "descend_max_steps"))
    descend_x_threshold = float(_require_param(params, "descend_x_threshold"))
    descend_y_threshold = float(_require_param(params, "descend_y_threshold"))
    descend_z_threshold = float(_require_param(params, "descend_z_threshold"))
    descend_timeout_x_threshold = float(_require_param(params, "descend_timeout_x_threshold"))
    descend_timeout_y_threshold = float(_require_param(params, "descend_timeout_y_threshold"))
    descend_timeout_z_threshold = float(_require_param(params, "descend_timeout_z_threshold"))
    descend_progress_eps = float(_require_param(params, "descend_progress_eps"))
    descend_stall_steps = int(_require_param(params, "descend_stall_steps"))
    descend_timeout_extension_steps = int(_require_param(params, "descend_timeout_extension_steps"))
    descend_max_timeout_extensions = int(_require_param(params, "descend_max_timeout_extensions"))

    close_hold_steps = int(_require_param(params, "close_hold_steps"))
    grasp_search_steps = int(_require_param(params, "grasp_search_steps"))
    grasp_stable_steps_for_lift = int(_require_param(params, "grasp_stable_steps_for_lift"))
    reach_reentry_cooldown_steps = int(_require_param(params, "reach_reentry_cooldown_steps"))
    lift_test_steps = int(_require_param(params, "lift_test_steps"))
    lift_test_obj_rel_drift_max = float(_require_param(params, "lift_test_obj_rel_drift_max"))
    max_retries = int(_require_param(params, "max_retries"))

    alpha_ee = float(_require_param(params, "alpha_ee"))
    alpha_obj_default = float(_require_param(params, "alpha_obj_default"))
    alpha_obj_reach = float(_require_param(params, "alpha_obj_reach"))
    alpha_target = float(_require_param(params, "alpha_target"))
    obj_reacquire_jump = float(_require_param(params, "obj_reacquire_jump"))
    contact_on_count = int(_require_param(params, "contact_on_count"))
    contact_off_count = int(_require_param(params, "contact_off_count"))

    reach_arc_angle_trigger_deg = float(_require_param(params, "reach_arc_angle_trigger_deg"))
    reach_arc_release_deg = float(_require_param(params, "reach_arc_release_deg"))
    reach_progress_eps = float(_require_param(params, "reach_progress_eps"))
    reach_stall_steps = int(_require_param(params, "reach_stall_steps"))
    reach_yaw_align_xy_enter = float(_require_param(params, "reach_yaw_align_xy_enter"))
    reach_yaw_align_steps = int(_require_param(params, "reach_yaw_align_steps"))
    grip_close_ready_max_width = float(_require_param(params, "grip_close_ready_max_width"))
    preplace_target_rel = _require_vec3(params, "preplace_target_rel")
    place_target_rel = _require_vec3(params, "place_target_rel")
    preplace_threshold = float(_require_param(params, "preplace_threshold"))
    place_threshold = float(_require_param(params, "place_threshold"))
    preplace_xy_threshold = float(_require_param(params, "preplace_xy_threshold"))
    preplace_z_threshold = float(_require_param(params, "preplace_z_threshold"))
    place_xy_threshold = float(_require_param(params, "place_xy_threshold"))
    place_z_threshold = float(_require_param(params, "place_z_threshold"))
    transit_height = float(_require_param(params, "transit_height"))
    transit_z_threshold = float(_require_param(params, "transit_z_threshold"))
    open_hold_steps = int(_require_param(params, "open_hold_steps"))
    retreat_steps = int(_require_param(params, "retreat_steps"))
    retreat_move = _require_vec3(params, "retreat_move")

    grip_open_target = float(_require_param(params, "grip_open_target"))
    grip_close_target = float(_require_param(params, "grip_close_target"))
    grip_ready_width_tol = float(_require_param(params, "grip_ready_width_tol"))
    grip_ready_speed_tol = float(_require_param(params, "grip_ready_speed_tol"))

    # Initial belief (t = 0)
    if previous_belief is None:
        return {
            "s_ee_mean": observation["o_ee"].copy(),
            "s_obj_mean": observation["o_obj"].copy(),
            "s_target_mean": observation["o_target"].copy(),
            "s_ee_cov": np.eye(3) * 0.05,
            "s_obj_cov": np.eye(3) * 0.1,
            "s_target_cov": np.eye(3) * 0.1,
            "s_obj_yaw": float(observation.get("o_obj_yaw", 0.0)),
            "reach_obj_rel": reach_obj_rel.copy(),
            "align_obj_rel": align_obj_rel.copy(),
            "descend_obj_rel": descend_obj_rel.copy(),
            "preplace_target_rel": preplace_target_rel.copy(),
            "place_target_rel": place_target_rel.copy(),
            "retreat_move": retreat_move.copy(),
            "s_grasp": 0,   # Open
            "phase": "Reach",
            "grasp_timer": 0,
            "align_timer": 0,
            "lift_timer": 0,
            "contact_counter": 0,
            "stable_grasp_counter": 0,
            "grasp_side_sign": 1.0,
            "reach_cooldown": 0,
            "reach_turn_sign": 0,
            "reach_best_error": float("inf"),
            "reach_no_progress_steps": 0,
            "reach_watchdog_active": 0,
            "reach_yaw_align_active": 0,
            "reach_yaw_align_timer": 0,
            "reach_yaw_align_done": 0,
            "prev_o_grip": float(observation["o_grip"]),
            "pregrasp_hold_timer": 0,
            "descend_timer": 0,
            "descend_best_error": float("inf"),
            "descend_no_progress_steps": 0,
            "descend_timeout_extensions": 0,
            "close_hold_timer": 0,
            "lift_test_timer": 0,
            "lift_test_ref_obj_rel": observation["o_obj"].copy(),
            "transit_timer": 0,
            "open_timer": 0,
            "retreat_timer": 0,
            "retry_count": 0,
        }

    # Placeholder inference update (to be replaced by RxInfer call)
    s_ee_mean = alpha_ee * observation["o_ee"] + (1 - alpha_ee) * previous_belief["s_ee_mean"]

    prev_phase = previous_belief["phase"]
    prev_obj = previous_belief["s_obj_mean"]
    obj_obs = observation["o_obj"]
    obj_yaw_obs = float(observation.get("o_obj_yaw", 0.0))
    if not np.isfinite(obj_yaw_obs):
        obj_yaw_obs = 0.0
    obj_jump = np.linalg.norm(obj_obs - prev_obj)
    alpha_obj = alpha_obj_reach if prev_phase == "Reach" else alpha_obj_default
    if obj_jump > obj_reacquire_jump:
        # Fast object motion (fall/slide): trust current observation more.
        alpha_obj = max(alpha_obj, 0.95)
    s_obj_mean = alpha_obj * obj_obs + (1 - alpha_obj) * prev_obj
    s_target_mean = (
        alpha_target * observation["o_target"] + (1 - alpha_target) * previous_belief["s_target_mean"]
    )

    # Contact hysteresis to avoid sticky/latched grasp state.
    contact_counter = previous_belief.get("contact_counter", 0)
    if observation["o_contact"] == 1:
        contact_counter = min(contact_counter + 1, 10)
    else:
        contact_counter = max(contact_counter - 1, -10)

    prev_grasp = previous_belief["s_grasp"]
    if prev_grasp == 0:
        s_grasp = 1 if contact_counter >= contact_on_count else 0
    else:
        s_grasp = 0 if contact_counter <= -contact_off_count else 1

    # Phase logic (active-inference path):
    # Reach -> Align -> PreGraspHold -> Descend -> CloseHold -> LiftTest
    # -> Transit -> MoveToPlaceAbove -> DescendToPlace -> Open -> Retreat -> Done
    phase = prev_phase
    grasp_timer = previous_belief.get("grasp_timer", 0)
    align_timer = previous_belief.get("align_timer", 0)
    lift_timer = previous_belief.get("lift_timer", 0)
    stable_grasp_counter = previous_belief.get("stable_grasp_counter", 0)
    grasp_side_sign = float(previous_belief.get("grasp_side_sign", 1.0))
    reach_cooldown = int(previous_belief.get("reach_cooldown", 0))
    reach_turn_sign = int(previous_belief.get("reach_turn_sign", 0))
    reach_best_error = float(previous_belief.get("reach_best_error", float("inf")))
    reach_no_progress_steps = int(previous_belief.get("reach_no_progress_steps", 0))
    reach_watchdog_active = int(previous_belief.get("reach_watchdog_active", 0))
    reach_yaw_align_active = int(previous_belief.get("reach_yaw_align_active", 0))
    reach_yaw_align_timer = int(previous_belief.get("reach_yaw_align_timer", 0))
    reach_yaw_align_done = int(previous_belief.get("reach_yaw_align_done", 0))
    pregrasp_hold_timer = int(previous_belief.get("pregrasp_hold_timer", 0))
    descend_timer = int(previous_belief.get("descend_timer", 0))
    descend_best_error = float(previous_belief.get("descend_best_error", float("inf")))
    descend_no_progress_steps = int(previous_belief.get("descend_no_progress_steps", 0))
    descend_timeout_extensions = int(previous_belief.get("descend_timeout_extensions", 0))
    close_hold_timer = int(previous_belief.get("close_hold_timer", 0))
    lift_test_timer = int(previous_belief.get("lift_test_timer", 0))
    transit_timer = int(previous_belief.get("transit_timer", 0))
    open_timer = int(previous_belief.get("open_timer", 0))
    retreat_timer = int(previous_belief.get("retreat_timer", 0))
    lift_test_ref_obj_rel = np.array(
        previous_belief.get("lift_test_ref_obj_rel", s_obj_mean.copy()), dtype=float
    )
    retry_count = int(previous_belief.get("retry_count", 0))
    reach_error = np.linalg.norm(s_obj_mean - reach_obj_rel)
    prev_o_grip = float(previous_belief.get("prev_o_grip", observation["o_grip"]))
    grip_obs = float(observation["o_grip"])
    grip_speed = abs(grip_obs - prev_o_grip)
    gripper_open_ready = (
        abs(grip_obs - grip_open_target) <= grip_ready_width_tol and grip_speed <= grip_ready_speed_tol
    )
    gripper_close_ready = (
        grip_speed <= grip_ready_speed_tol
        and (
            abs(grip_obs - grip_close_target) <= grip_ready_width_tol
            or (grip_obs <= grip_close_ready_max_width and observation["o_contact"] == 1)
        )
    )
    if reach_cooldown > 0:
        reach_cooldown -= 1

    if phase == "Reach":
        # Reach arc direction lock (for side/behind objects): pick CW/CCW once and hold.
        reach_err_vec = s_obj_mean - reach_obj_rel
        xy_err = float(np.linalg.norm(reach_err_vec[:2]))

        # Reach yaw-align substage:
        # briefly hold position and let controller rotate to object yaw before final approach.
        if (
            reach_yaw_align_active == 0
            and reach_yaw_align_done == 0
            and xy_err <= reach_yaw_align_xy_enter
        ):
            reach_yaw_align_active = 1
            reach_yaw_align_timer = reach_yaw_align_steps

        ee_xy = np.asarray(s_ee_mean[:2], dtype=float)
        goal_xy = np.asarray(s_ee_mean[:2] + reach_err_vec[:2], dtype=float)
        r_ee = float(np.linalg.norm(ee_xy))
        r_goal = float(np.linalg.norm(goal_xy))
        if r_ee > 1e-6 and r_goal > 1e-6:
            theta_ee = float(np.arctan2(ee_xy[1], ee_xy[0]))
            theta_goal = float(np.arctan2(goal_xy[1], goal_xy[0]))
            dtheta_short = _wrap_to_pi(theta_goal - theta_ee)
            trigger = float(np.deg2rad(reach_arc_angle_trigger_deg))
            release = float(np.deg2rad(reach_arc_release_deg))
            if reach_turn_sign == 0 and abs(dtheta_short) >= trigger:
                reach_turn_sign = 1 if dtheta_short >= 0.0 else -1
            elif reach_turn_sign != 0 and abs(dtheta_short) <= release:
                reach_turn_sign = 0

        # Reach progress watchdog: if no improvement for long, trigger linear fallback.
        # While yaw-align is active, pause watchdog so intentional hold doesn't look like stall.
        if reach_yaw_align_active == 1:
            reach_yaw_align_timer = max(0, reach_yaw_align_timer - 1)
            reach_no_progress_steps = 0
            reach_watchdog_active = 0
            if reach_yaw_align_timer <= 0:
                reach_yaw_align_active = 0
                reach_yaw_align_done = 1
        else:
            if not np.isfinite(reach_best_error):
                reach_best_error = float(reach_error)
                reach_no_progress_steps = 0
                reach_watchdog_active = 0
            elif float(reach_error) < (reach_best_error - reach_progress_eps):
                reach_best_error = float(reach_error)
                reach_no_progress_steps = 0
                reach_watchdog_active = 0
            else:
                reach_no_progress_steps += 1
                if reach_no_progress_steps >= reach_stall_steps:
                    reach_watchdog_active = 1
                    # Drop turn lock while watchdog is active (fallback to direct approach).
                    reach_turn_sign = 0

        if reach_cooldown == 0 and reach_error < approach_threshold and reach_yaw_align_active == 0:
            phase = "Align"
            # Top-down approach: keep centerline, no side offset.
            grasp_side_sign = 0.0
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
            reach_turn_sign = 0
            reach_best_error = float("inf")
            reach_no_progress_steps = 0
            reach_watchdog_active = 0
            reach_yaw_align_active = 0
            reach_yaw_align_timer = 0
            reach_yaw_align_done = 0
    elif phase == "Align":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        align_timer += 1
        align_error = np.linalg.norm(s_obj_mean - align_obj_rel)
        align_ready = align_timer >= align_min_steps and gripper_open_ready
        align_timeout = align_timer >= align_max_steps
        align_timeout_near = align_error < (1.5 * align_threshold)
        if align_ready and align_error < align_threshold:
            phase = "PreGraspHold"
            pregrasp_hold_timer = 0
            grasp_timer = 0
            stable_grasp_counter = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
        elif align_timeout:
            # Timeout fallback: continue only when still near and gripper is open-ready.
            # Otherwise retry Reach to avoid entering a long hold stall.
            if align_timeout_near and gripper_open_ready:
                phase = "PreGraspHold"
                pregrasp_hold_timer = 0
                grasp_timer = 0
                stable_grasp_counter = 0
                descend_timer = 0
                descend_best_error = float("inf")
                descend_no_progress_steps = 0
                descend_timeout_extensions = 0
                close_hold_timer = 0
                lift_test_timer = 0
            else:
                phase = "Reach"
                reach_cooldown = reach_reentry_cooldown_steps
                align_timer = 0
                pregrasp_hold_timer = 0
                descend_timer = 0
                descend_best_error = float("inf")
                descend_no_progress_steps = 0
                descend_timeout_extensions = 0
                close_hold_timer = 0
                lift_test_timer = 0
    elif phase == "PreGraspHold":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        pregrasp_hold_timer += 1
        hold_ready = pregrasp_hold_timer >= pregrasp_hold_steps
        hold_grace_elapsed = pregrasp_hold_timer >= (
            pregrasp_hold_steps + pregrasp_hold_ready_grace_steps
        )
        if hold_ready and (gripper_open_ready or hold_grace_elapsed):
            phase = "Descend"
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            stable_grasp_counter = 0
    elif phase == "Descend":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        descend_timer += 1
        descend_err_vec = s_obj_mean - descend_obj_rel
        descend_error = float(np.linalg.norm(descend_err_vec))
        x_err = float(abs(descend_err_vec[0]))
        y_err = float(abs(descend_err_vec[1]))
        z_err = float(abs(descend_err_vec[2]))

        xy_axis_ok = x_err <= descend_x_threshold and y_err <= descend_y_threshold
        z_axis_ok = z_err <= descend_z_threshold
        position_ok = (descend_error <= descend_threshold) or (xy_axis_ok and z_axis_ok)

        if not np.isfinite(descend_best_error):
            descend_best_error = descend_error
            descend_no_progress_steps = 0
        elif descend_error < (descend_best_error - descend_progress_eps):
            descend_best_error = descend_error
            descend_no_progress_steps = 0
        else:
            descend_no_progress_steps += 1

        timeout_hit = descend_timer >= descend_max_steps
        timeout_near = (
            x_err <= descend_timeout_x_threshold
            and y_err <= descend_timeout_y_threshold
            and z_err <= descend_timeout_z_threshold
        )

        can_extend = (
            timeout_hit
            and not timeout_near
            and descend_timeout_extensions < descend_max_timeout_extensions
            and descend_no_progress_steps < descend_stall_steps
        )

        if position_ok or (timeout_hit and timeout_near):
            phase = "CloseHold"
            close_hold_timer = 0
            stable_grasp_counter = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
        elif can_extend:
            descend_timeout_extensions += 1
            descend_timer = max(0, descend_timer - descend_timeout_extension_steps)
            descend_no_progress_steps = 0
        elif timeout_hit:
            # Descend timed out while still far: retry instead of forcing close.
            retry_count += 1
            if retry_count > max_retries:
                retry_count = 0
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = reach_reentry_cooldown_steps
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
    elif phase == "CloseHold":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        close_hold_timer += 1
        if s_grasp == 1 and observation["o_contact"] == 1:
            stable_grasp_counter += 1
        else:
            stable_grasp_counter = 0

        if (
            close_hold_timer >= close_hold_steps
            and stable_grasp_counter >= grasp_stable_steps_for_lift
            and gripper_close_ready
        ):
            phase = "LiftTest"
            lift_test_timer = 0
            lift_test_ref_obj_rel = s_obj_mean.copy()
        elif close_hold_timer >= grasp_search_steps:
            # Timed out without reliable grasp confirmation.
            retry_count += 1
            if retry_count > max_retries:
                retry_count = 0
                phase = "Reach"
            else:
                phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = reach_reentry_cooldown_steps
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
    elif phase == "LiftTest":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        lift_test_timer += 1
        obj_rel_drift = np.linalg.norm(s_obj_mean - lift_test_ref_obj_rel)
        if s_grasp == 0:
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = reach_reentry_cooldown_steps
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
        elif obj_rel_drift > lift_test_obj_rel_drift_max:
            retry_count += 1
            if retry_count > max_retries:
                retry_count = 0
            phase = "Reach"
            align_timer = 0
            grasp_timer = 0
            lift_timer = 0
            stable_grasp_counter = 0
            reach_cooldown = reach_reentry_cooldown_steps
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
        else:
            if lift_test_timer >= lift_test_steps:
                retry_count = 0
                phase = "Transit"
                transit_timer = 0
    elif phase == "Transit":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        transit_timer += 1
        ee_z = float(s_ee_mean[2])
        if ee_z >= (transit_height - transit_z_threshold):
            phase = "MoveToPlaceAbove"
        elif s_grasp == 0:
            phase = "Reach"
            reach_cooldown = reach_reentry_cooldown_steps
            align_timer = 0
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
            transit_timer = 0
            open_timer = 0
            retreat_timer = 0
    elif phase == "MoveToPlaceAbove":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        err = s_target_mean - preplace_target_rel
        preplace_xy_ok = float(np.linalg.norm(err[:2])) <= preplace_xy_threshold
        preplace_z_ok = abs(float(err[2])) <= preplace_z_threshold
        if float(np.linalg.norm(err)) <= preplace_threshold or (preplace_xy_ok and preplace_z_ok):
            phase = "DescendToPlace"
        elif s_grasp == 0:
            phase = "Reach"
            reach_cooldown = reach_reentry_cooldown_steps
            align_timer = 0
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
            transit_timer = 0
            open_timer = 0
            retreat_timer = 0
    elif phase == "DescendToPlace":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        err = s_target_mean - place_target_rel
        place_xy_ok = float(np.linalg.norm(err[:2])) <= place_xy_threshold
        place_z_ok = abs(float(err[2])) <= place_z_threshold
        if float(np.linalg.norm(err)) <= place_threshold or (place_xy_ok and place_z_ok):
            phase = "Open"
            open_timer = 0
        elif s_grasp == 0:
            phase = "Reach"
            reach_cooldown = reach_reentry_cooldown_steps
            align_timer = 0
            pregrasp_hold_timer = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
            transit_timer = 0
            open_timer = 0
            retreat_timer = 0
    elif phase == "Open":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        open_timer += 1
        if open_timer >= open_hold_steps:
            phase = "Retreat"
            retreat_timer = 0
    elif phase == "Retreat":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        retreat_timer += 1
        if retreat_timer >= retreat_steps:
            phase = "Done"
    elif phase == "Done":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0

    if phase in ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest"):
        transit_timer = 0
        open_timer = 0
        retreat_timer = 0

    if phase in ("Transit", "MoveToPlaceAbove", "DescendToPlace", "Open", "Retreat", "Done"):
        grasp_timer = 0

    if phase in ("Done",):
        s_grasp = 0

    if phase in ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest", "Transit", "MoveToPlaceAbove", "DescendToPlace"):
        # Ensure lift-test reference remains defined across non-done phases.
        if lift_test_ref_obj_rel is None:
            lift_test_ref_obj_rel = s_obj_mean.copy()

    return {
        "s_ee_mean": s_ee_mean,
        "s_obj_mean": s_obj_mean,
        "s_target_mean": s_target_mean,
        "s_obj_yaw": float(obj_yaw_obs),
        "reach_obj_rel": reach_obj_rel.copy(),
        "align_obj_rel": align_obj_rel.copy(),
        "descend_obj_rel": descend_obj_rel.copy(),
        "preplace_target_rel": preplace_target_rel.copy(),
        "place_target_rel": place_target_rel.copy(),
        "retreat_move": retreat_move.copy(),
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
        "reach_turn_sign": int(reach_turn_sign),
        "reach_best_error": float(reach_best_error),
        "reach_no_progress_steps": int(reach_no_progress_steps),
        "reach_watchdog_active": int(reach_watchdog_active),
        "reach_yaw_align_active": int(reach_yaw_align_active),
        "reach_yaw_align_timer": int(reach_yaw_align_timer),
        "reach_yaw_align_done": int(reach_yaw_align_done),
        "prev_o_grip": grip_obs,
        "pregrasp_hold_timer": pregrasp_hold_timer,
        "descend_timer": descend_timer,
        "descend_best_error": float(descend_best_error),
        "descend_no_progress_steps": int(descend_no_progress_steps),
        "descend_timeout_extensions": int(descend_timeout_extensions),
        "close_hold_timer": close_hold_timer,
        "lift_test_timer": lift_test_timer,
        "lift_test_ref_obj_rel": lift_test_ref_obj_rel,
        "transit_timer": transit_timer,
        "open_timer": open_timer,
        "retreat_timer": retreat_timer,
        "retry_count": retry_count,
    }
