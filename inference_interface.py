import os
from pathlib import Path

import numpy as np

"""
This module bridges Python to Julia (RxInfer).

Responsibilities:
- maintain belief state
- convert observations into inference inputs
- optionally call RxInfer-backed belief update
- fallback safely to Python belief update when Julia/RxInfer is unavailable
- return belief dictionary usable by action selection
"""

# Optional Julia/RxInfer backend for belief updates.
# The runtime always has a pure-Python fallback path.
try:
    from julia import Main as _jl

    _inference_dir = str((Path(__file__).resolve().parent / "inference").as_posix())
    _jl.eval(f'pushfirst!(LOAD_PATH, "{_inference_dir}")')
    _jl.include(str((Path(__file__).resolve().parent / "inference" / "rxinfer_beliefs.jl").as_posix()))
    _RXINFER_JULIA_AVAILABLE = hasattr(_jl, "rxinfer_belief_step")
    _RXINFER_JULIA_ERROR = ""
except Exception as exc:
    _jl = None
    _RXINFER_JULIA_AVAILABLE = False
    _RXINFER_JULIA_ERROR = f"{type(exc).__name__}: {exc}"
    if os.getenv("ACTIVE_INFERENCE_VERBOSE_JULIA", "0") == "1":
        print(
            "Warning: RxInfer Julia initialization failed; "
            f"falling back to Python belief updates ({_RXINFER_JULIA_ERROR})."
        )

_RXINFER_RUNTIME_DISABLED = False


def _rxinfer_step(
    previous_belief: dict,
    *,
    ee_obs: np.ndarray,
    obj_obs: np.ndarray,
    target_obs: np.ndarray,
    ee_vel: np.ndarray,
    obj_vel: np.ndarray,
    target_vel: np.ndarray,
    dt: float,
    process_noise_ee: float,
    process_noise_obj: float,
    process_noise_target: float,
    obs_noise_ee: float,
    obs_noise_obj: float,
    obs_noise_target: float,
    min_variance: float,
) -> dict | None:
    """
    One-step RxInfer-backed belief update.
    Returns None when backend is unavailable or failed, so caller can fallback.
    """
    global _RXINFER_RUNTIME_DISABLED

    if (not _RXINFER_JULIA_AVAILABLE) or _RXINFER_RUNTIME_DISABLED or _jl is None:
        return None

    try:
        prev_ee_mean = np.asarray(previous_belief.get("s_ee_mean", np.zeros(3)), dtype=float).reshape(3)
        prev_obj_mean = np.asarray(previous_belief.get("s_obj_mean", np.zeros(3)), dtype=float).reshape(3)
        prev_target_mean = np.asarray(previous_belief.get("s_target_mean", np.zeros(3)), dtype=float).reshape(3)
        prev_ee_cov = np.asarray(previous_belief.get("s_ee_cov", np.eye(3)), dtype=float)
        prev_obj_cov = np.asarray(previous_belief.get("s_obj_cov", np.eye(3)), dtype=float)
        prev_target_cov = np.asarray(previous_belief.get("s_target_cov", np.eye(3)), dtype=float)
        prev_ee_cov_diag = np.diag(prev_ee_cov) if prev_ee_cov.ndim == 2 else prev_ee_cov.reshape(3)
        prev_obj_cov_diag = np.diag(prev_obj_cov) if prev_obj_cov.ndim == 2 else prev_obj_cov.reshape(3)
        prev_target_cov_diag = (
            np.diag(prev_target_cov) if prev_target_cov.ndim == 2 else prev_target_cov.reshape(3)
        )

        result = _jl.rxinfer_belief_step(
            prev_ee_mean.tolist(),
            prev_obj_mean.tolist(),
            prev_target_mean.tolist(),
            np.asarray(prev_ee_cov_diag, dtype=float).reshape(3).tolist(),
            np.asarray(prev_obj_cov_diag, dtype=float).reshape(3).tolist(),
            np.asarray(prev_target_cov_diag, dtype=float).reshape(3).tolist(),
            np.asarray(ee_obs, dtype=float).reshape(3).tolist(),
            np.asarray(obj_obs, dtype=float).reshape(3).tolist(),
            np.asarray(target_obs, dtype=float).reshape(3).tolist(),
            np.asarray(ee_vel, dtype=float).reshape(3).tolist(),
            np.asarray(obj_vel, dtype=float).reshape(3).tolist(),
            np.asarray(target_vel, dtype=float).reshape(3).tolist(),
            float(dt),
            float(process_noise_ee),
            float(process_noise_obj),
            float(process_noise_target),
            float(obs_noise_ee),
            float(obs_noise_obj),
            float(obs_noise_target),
            float(min_variance),
        )

        s_ee_mean = np.asarray(result.ee_mean, dtype=float).reshape(3)
        s_obj_mean = np.asarray(result.obj_mean, dtype=float).reshape(3)
        s_target_mean = np.asarray(result.target_mean, dtype=float).reshape(3)
        s_ee_cov_diag = np.asarray(result.ee_cov_diag, dtype=float).reshape(3)
        s_obj_cov_diag = np.asarray(result.obj_cov_diag, dtype=float).reshape(3)
        s_target_cov_diag = np.asarray(result.target_cov_diag, dtype=float).reshape(3)

        if not (
            np.all(np.isfinite(s_ee_mean))
            and np.all(np.isfinite(s_obj_mean))
            and np.all(np.isfinite(s_target_mean))
            and np.all(np.isfinite(s_ee_cov_diag))
            and np.all(np.isfinite(s_obj_cov_diag))
            and np.all(np.isfinite(s_target_cov_diag))
        ):
            return None

        return {
            "s_ee_mean": s_ee_mean.copy(),
            "s_obj_mean": s_obj_mean.copy(),
            "s_target_mean": s_target_mean.copy(),
            "s_ee_cov": np.diag(np.maximum(s_ee_cov_diag, float(min_variance))),
            "s_obj_cov": np.diag(np.maximum(s_obj_cov_diag, float(min_variance))),
            "s_target_cov": np.diag(np.maximum(s_target_cov_diag, float(min_variance))),
            "backend_name": str(getattr(result, "backend_name", "rxinfer")),
        }
    except Exception as exc:
        _RXINFER_RUNTIME_DISABLED = True
        if os.getenv("ACTIVE_INFERENCE_VERBOSE_JULIA", "0") == "1":
            print(
                "Warning: RxInfer belief step failed at runtime; "
                f"disabling RxInfer and falling back to Python ({type(exc).__name__}: {exc})."
            )
        return None


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


def _yaw_error(current_yaw: float, desired_yaw: float, pi_symmetric: bool = False) -> float:
    e0 = _wrap_to_pi(float(desired_yaw) - float(current_yaw))
    if not bool(pi_symmetric):
        return float(e0)
    e1 = _wrap_to_pi(float(desired_yaw + np.pi) - float(current_yaw))
    e2 = _wrap_to_pi(float(desired_yaw - np.pi) - float(current_yaw))
    return float(min((e0, e1, e2), key=lambda x: abs(float(x))))


def _safe_scalar(value, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            return float(arr[0])
    except (TypeError, ValueError):
        pass
    return float(default)


def _safe_vec3(value, default=None) -> np.ndarray:
    if default is None:
        default = np.zeros(3, dtype=float)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape[0] >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].astype(float).copy()
    except (TypeError, ValueError):
        pass
    return np.asarray(default, dtype=float).reshape(3).copy()


def _obj_local_rel_to_world(rel_local: np.ndarray, obj_yaw: float) -> np.ndarray:
    """
    Convert object-local XY relation into world-frame XY relation.
    Z stays unchanged (world vertical).
    """
    r = np.asarray(rel_local, dtype=float).copy().reshape(3)
    c = float(np.cos(float(obj_yaw)))
    s = float(np.sin(float(obj_yaw)))
    x_l = float(r[0])
    y_l = float(r[1])
    r[0] = c * x_l - s * y_l
    r[1] = s * x_l + c * y_l
    return r


def _compute_approach_side_sign(
    s_ee_mean: np.ndarray,
    s_obj_mean: np.ndarray,
    obj_yaw: float,
    prev_sign: float,
) -> float:
    """
    Pick a stable +/- side for pregrasp approach along object local +X axis.
    The sign is chosen relative to robot base (origin) and held near boundary.
    """
    c = float(np.cos(float(obj_yaw)))
    s = float(np.sin(float(obj_yaw)))
    approach_axis_world = np.asarray([c, s], dtype=float)  # local +X in world XY
    obj_world_xy = np.asarray(s_ee_mean[:2], dtype=float) + np.asarray(s_obj_mean[:2], dtype=float)
    score = float(np.dot(obj_world_xy, approach_axis_world))
    deadband = 0.015
    if score > deadband:
        return 1.0
    if score < -deadband:
        return -1.0
    return 1.0 if float(prev_sign) >= 0.0 else -1.0


def _compute_phase_vfe(
    phase: str,
    s_ee_mean: np.ndarray,
    s_obj_mean: np.ndarray,
    s_target_mean: np.ndarray,
    s_obj_cov: np.ndarray,
    s_target_cov: np.ndarray,
    reach_obj_rel: np.ndarray,
    align_obj_rel: np.ndarray,
    descend_obj_rel: np.ndarray,
    preplace_target_rel: np.ndarray,
    place_target_rel: np.ndarray,
    transit_height: float,
    vfe_epistemic_weight: float,
) -> tuple[float, float, float]:
    if phase == "Reach":
        pragmatic_err = s_obj_mean - reach_obj_rel
    elif phase in ("Align", "PreGraspHold"):
        pragmatic_err = s_obj_mean - align_obj_rel
    elif phase in ("Descend", "CloseHold", "LiftTest"):
        pragmatic_err = s_obj_mean - descend_obj_rel
    elif phase == "Transit":
        pragmatic_err = np.asarray([float(transit_height) - float(s_ee_mean[2])], dtype=float)
    elif phase == "MoveToPlaceAbove":
        pragmatic_err = s_target_mean - preplace_target_rel
    elif phase in ("DescendToPlace", "Open", "Retreat"):
        pragmatic_err = s_target_mean - place_target_rel
    else:
        pragmatic_err = np.zeros(1, dtype=float)

    vfe_pragmatic = float(np.linalg.norm(pragmatic_err) ** 2)
    vfe_epistemic = float(np.trace(s_obj_cov) + np.trace(s_target_cov))
    vfe_total = float(vfe_pragmatic + float(vfe_epistemic_weight) * vfe_epistemic)
    return vfe_total, vfe_pragmatic, vfe_epistemic


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
    # Pick-side references are configured in object-local XY and converted
    # per-step to world-frame relation using current object yaw.
    reach_obj_rel_local_cfg = _require_vec3(params, "reach_obj_rel")
    align_obj_rel_local_cfg = _require_vec3(params, "align_obj_rel")
    descend_obj_rel_local_cfg = _require_vec3(params, "descend_obj_rel")

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
    confidence_enabled = bool(_require_param(params, "confidence_enabled"))
    confidence_alpha = float(_require_param(params, "confidence_alpha"))
    confidence_min_for_phase_change = float(_require_param(params, "confidence_min_for_phase_change"))
    obj_jump_confidence_scale = float(_require_param(params, "obj_jump_confidence_scale"))
    target_jump_confidence_scale = float(_require_param(params, "target_jump_confidence_scale"))
    contact_flip_confidence_penalty = float(_require_param(params, "contact_flip_confidence_penalty"))
    allow_bt_prior_override = bool(_require_param(params, "allow_bt_prior_override"))
    vfe_enabled = bool(_require_param(params, "vfe_enabled"))
    vfe_epistemic_weight = float(_require_param(params, "vfe_epistemic_weight"))
    vfe_phase_change_enabled = bool(_require_param(params, "vfe_phase_change_enabled"))
    vfe_phase_change_max = float(_require_param(params, "vfe_phase_change_max"))
    vfe_recover_enabled = bool(_require_param(params, "vfe_recover_enabled"))
    vfe_recover_threshold = float(_require_param(params, "vfe_recover_threshold"))
    vfe_recover_steps = int(_require_param(params, "vfe_recover_steps"))
    bt_branch_retry_cap = int(_require_param(params, "bt_branch_retry_cap"))
    bt_global_recovery_cap = int(_require_param(params, "bt_global_recovery_cap"))
    bt_rescan_hold_steps = int(_require_param(params, "bt_rescan_hold_steps"))
    bt_reapproach_offset_xy = float(_require_param(params, "bt_reapproach_offset_xy"))
    bt_safe_backoff_hold_steps = int(_require_param(params, "bt_safe_backoff_hold_steps"))
    bt_safe_backoff_z_boost = float(_require_param(params, "bt_safe_backoff_z_boost"))
    release_contact_warn_steps = int(_require_param(params, "release_contact_warn_steps"))
    release_verify_enabled = bool(_require_param(params, "release_verify_enabled"))
    release_detach_hold_steps = int(_require_param(params, "release_detach_hold_steps"))
    release_stable_hold_steps = int(_require_param(params, "release_stable_hold_steps"))
    release_obj_xy_threshold = float(_require_param(params, "release_obj_xy_threshold"))
    release_obj_z_threshold = float(_require_param(params, "release_obj_z_threshold"))
    release_obj_speed_threshold = float(_require_param(params, "release_obj_speed_threshold"))
    open_max_steps = int(_require_param(params, "open_max_steps"))
    release_reapproach_max_retries = int(_require_param(params, "release_reapproach_max_retries"))

    reach_arc_angle_trigger_deg = float(_require_param(params, "reach_arc_angle_trigger_deg"))
    reach_arc_release_deg = float(_require_param(params, "reach_arc_release_deg"))
    reach_arc_min_radius = float(_require_param(params, "reach_arc_min_radius"))
    reach_progress_eps = float(_require_param(params, "reach_progress_eps"))
    reach_stall_steps = int(_require_param(params, "reach_stall_steps"))
    reach_yaw_align_xy_enter = float(_require_param(params, "reach_yaw_align_xy_enter"))
    reach_yaw_align_steps = int(_require_param(params, "reach_yaw_align_steps"))
    grip_close_ready_max_width = float(_require_param(params, "grip_close_ready_max_width"))
    preplace_target_rel = _require_vec3(params, "preplace_target_rel")
    place_target_rel = _require_vec3(params, "place_target_rel")
    use_world_place_goal_pose = bool(_require_param(params, "use_world_place_goal_pose"))
    place_goal_world_xyz = _require_vec3(params, "place_goal_world_xyz")
    place_goal_world_yaw = float(np.deg2rad(float(_require_param(params, "place_goal_world_yaw_deg"))))
    place_goal_yaw_enabled = bool(_require_param(params, "place_goal_yaw_enabled"))
    place_goal_yaw_threshold = float(
        np.deg2rad(float(_require_param(params, "place_goal_yaw_threshold_deg")))
    )
    place_goal_yaw_pi_symmetric = bool(_require_param(params, "place_goal_yaw_pi_symmetric"))
    place_goal_world_pose6d_deg = None
    place_goal_world_pose6d_deg_raw = params.get("place_goal_world_pose6d_deg", None)
    if place_goal_world_pose6d_deg_raw is not None:
        pose6 = np.asarray(place_goal_world_pose6d_deg_raw, dtype=float).reshape(-1)
        if pose6.shape[0] != 6:
            raise ValueError(
                "Active-inference config key 'place_goal_world_pose6d_deg' must be length-6 "
                "[x,y,z,roll_deg,pitch_deg,yaw_deg]."
            )
        if not np.all(np.isfinite(pose6)):
            raise ValueError(
                "Active-inference config key 'place_goal_world_pose6d_deg' must contain finite values."
            )
        place_goal_world_pose6d_deg = pose6.copy()
        place_goal_world_xyz = pose6[:3].copy()
        place_goal_world_yaw = float(np.deg2rad(float(pose6[5])))
    preplace_threshold = float(_require_param(params, "preplace_threshold"))
    place_threshold = float(_require_param(params, "place_threshold"))
    preplace_xy_threshold = float(_require_param(params, "preplace_xy_threshold"))
    preplace_z_threshold = float(_require_param(params, "preplace_z_threshold"))
    preplace_max_steps = int(_require_param(params, "preplace_max_steps"))
    preplace_progress_eps = float(_require_param(params, "preplace_progress_eps"))
    preplace_stall_steps = int(_require_param(params, "preplace_stall_steps"))
    preplace_timeout_extension_steps = int(_require_param(params, "preplace_timeout_extension_steps"))
    preplace_max_timeout_extensions = int(_require_param(params, "preplace_max_timeout_extensions"))
    preplace_timeout_xy_threshold = float(_require_param(params, "preplace_timeout_xy_threshold"))
    preplace_timeout_z_threshold = float(_require_param(params, "preplace_timeout_z_threshold"))
    place_xy_threshold = float(_require_param(params, "place_xy_threshold"))
    place_z_threshold = float(_require_param(params, "place_z_threshold"))
    place_descend_max_steps = int(_require_param(params, "place_descend_max_steps"))
    place_descend_progress_eps = float(_require_param(params, "place_descend_progress_eps"))
    place_descend_stall_steps = int(_require_param(params, "place_descend_stall_steps"))
    place_descend_timeout_extension_steps = int(
        _require_param(params, "place_descend_timeout_extension_steps")
    )
    place_descend_max_timeout_extensions = int(
        _require_param(params, "place_descend_max_timeout_extensions")
    )
    place_reapproach_max_retries = int(_require_param(params, "place_reapproach_max_retries"))
    transit_height = float(_require_param(params, "transit_height"))
    transit_z_threshold = float(_require_param(params, "transit_z_threshold"))
    open_hold_steps = int(_require_param(params, "open_hold_steps"))
    retreat_steps = int(_require_param(params, "retreat_steps"))
    retreat_move = _require_vec3(params, "retreat_move")

    grip_open_target = float(_require_param(params, "grip_open_target"))
    grip_close_target = float(_require_param(params, "grip_close_target"))
    grip_ready_width_tol = float(_require_param(params, "grip_ready_width_tol"))
    grip_ready_speed_tol = float(_require_param(params, "grip_ready_speed_tol"))

    # Optional stronger belief-update path (RxInfer via Julia).
    rxinfer_enabled = bool(params.get("rxinfer_enabled", False))
    rxinfer_process_noise_ee = float(params.get("rxinfer_process_noise_ee", 0.0025))
    rxinfer_process_noise_obj = float(params.get("rxinfer_process_noise_obj", 0.0040))
    rxinfer_process_noise_target = float(params.get("rxinfer_process_noise_target", 0.0020))
    rxinfer_obs_noise_ee = float(params.get("rxinfer_obs_noise_ee", 0.0016))
    rxinfer_obs_noise_obj = float(params.get("rxinfer_obs_noise_obj", 0.0036))
    rxinfer_obs_noise_target = float(params.get("rxinfer_obs_noise_target", 0.0025))
    rxinfer_min_variance = float(params.get("rxinfer_min_variance", 1e-6))

    # Initial belief (t = 0)
    if previous_belief is None:
        init_s_ee_mean = _safe_vec3(observation.get("o_ee", [0.0, 0.0, 0.0]))
        init_s_obj_mean = _safe_vec3(observation.get("o_obj", [0.0, 0.0, 0.0]))
        if use_world_place_goal_pose:
            init_s_target_mean = np.asarray(place_goal_world_xyz, dtype=float) - np.asarray(
                init_s_ee_mean, dtype=float
            )
        else:
            init_s_target_mean = _safe_vec3(observation.get("o_target", [0.0, 0.0, 0.0]))
        init_obj_yaw = float(observation.get("o_obj_yaw", 0.0))
        if not np.isfinite(init_obj_yaw):
            init_obj_yaw = 0.0
        init_approach_side_sign = _compute_approach_side_sign(
            s_ee_mean=init_s_ee_mean,
            s_obj_mean=init_s_obj_mean,
            obj_yaw=init_obj_yaw,
            prev_sign=1.0,
        )
        reach_local_signed = reach_obj_rel_local_cfg.copy()
        align_local_signed = align_obj_rel_local_cfg.copy()
        reach_local_signed[0] = abs(float(reach_local_signed[0])) * init_approach_side_sign
        align_local_signed[0] = abs(float(align_local_signed[0])) * init_approach_side_sign
        descend_local = descend_obj_rel_local_cfg.copy()

        reach_obj_rel = _obj_local_rel_to_world(reach_local_signed, init_obj_yaw)
        align_obj_rel = _obj_local_rel_to_world(align_local_signed, init_obj_yaw)
        descend_obj_rel = _obj_local_rel_to_world(descend_local, init_obj_yaw)

        init_s_ee_cov = np.eye(3) * 0.05
        init_s_obj_cov = np.eye(3) * 0.1
        init_s_target_cov = np.eye(3) * 0.1
        init_vfe_total, init_vfe_pragmatic, init_vfe_epistemic = _compute_phase_vfe(
            phase="Reach",
            s_ee_mean=init_s_ee_mean,
            s_obj_mean=init_s_obj_mean,
            s_target_mean=init_s_target_mean,
            s_obj_cov=init_s_obj_cov,
            s_target_cov=init_s_target_cov,
            reach_obj_rel=reach_obj_rel,
            align_obj_rel=align_obj_rel,
            descend_obj_rel=descend_obj_rel,
            preplace_target_rel=preplace_target_rel,
            place_target_rel=place_target_rel,
            transit_height=transit_height,
            vfe_epistemic_weight=vfe_epistemic_weight,
        )
        return {
            "s_ee_mean": init_s_ee_mean.copy(),
            "s_obj_mean": init_s_obj_mean.copy(),
            "s_target_mean": init_s_target_mean.copy(),
            "s_ee_cov": init_s_ee_cov,
            "s_obj_cov": init_s_obj_cov,
            "s_target_cov": init_s_target_cov,
            "s_obj_yaw": init_obj_yaw,
            "reach_obj_rel_local": reach_obj_rel_local_cfg.copy(),
            "align_obj_rel_local": align_obj_rel_local_cfg.copy(),
            "descend_obj_rel_local": descend_obj_rel_local_cfg.copy(),
            "approach_side_sign": float(init_approach_side_sign),
            "reach_obj_rel": reach_obj_rel.copy(),
            "align_obj_rel": align_obj_rel.copy(),
            "descend_obj_rel": descend_obj_rel.copy(),
            "preplace_target_rel": preplace_target_rel.copy(),
            "place_target_rel": place_target_rel.copy(),
            "use_world_place_goal_pose": int(use_world_place_goal_pose),
            "place_goal_world_xyz": np.asarray(place_goal_world_xyz, dtype=float).copy(),
            "place_goal_world_pose6d_deg": (
                None
                if place_goal_world_pose6d_deg is None
                else np.asarray(place_goal_world_pose6d_deg, dtype=float).copy()
            ),
            "place_goal_yaw": float(place_goal_world_yaw),
            "place_goal_yaw_enabled": int(place_goal_yaw_enabled),
            "place_goal_yaw_threshold": float(place_goal_yaw_threshold),
            "place_goal_yaw_pi_symmetric": int(place_goal_yaw_pi_symmetric),
            "place_goal_yaw_error": float(
                abs(
                    _yaw_error(
                        current_yaw=init_obj_yaw,
                        desired_yaw=place_goal_world_yaw,
                        pi_symmetric=place_goal_yaw_pi_symmetric,
                    )
                )
            ),
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
            "reach_gate_active": 0,
            "reach_gate_counter": 0,
            "align_gate_active": 0,
            "align_gate_counter": 0,
            "pregrasp_ready_counter": 0,
            "descend_gate_active": 0,
            "descend_gate_counter": 0,
            "descend_yaw_enabled": 0,
            "prev_o_grip": _safe_scalar(observation.get("o_grip", 0.0), default=0.0),
            "prev_o_contact": int(_safe_scalar(observation.get("o_contact", 0), default=0)),
            "prev_o_obj": init_s_obj_mean.copy(),
            "prev_o_target": init_s_target_mean.copy(),
            "obs_confidence": 1.0,
            "phase_conf_ok": 1,
            "phase_vfe_ok": 1,
            "phase_gate_ok": 1,
            "vfe_total": float(init_vfe_total if vfe_enabled else 0.0),
            "vfe_pragmatic": float(init_vfe_pragmatic if vfe_enabled else 0.0),
            "vfe_epistemic": float(init_vfe_epistemic if vfe_enabled else 0.0),
            "vfe_recover_enabled": int(vfe_recover_enabled),
            "vfe_recover_threshold": float(vfe_recover_threshold),
            "vfe_recover_steps": int(vfe_recover_steps),
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
            "release_contact_counter": 0,
            "release_warning": 0,
            "release_detach_counter": 0,
            "release_stable_counter": 0,
            "release_reapproach_count": 0,
            "gripper_open_ready": 1,
            "gripper_close_ready": 0,
            "place_descend_timer": 0,
            "place_descend_best_error": float("inf"),
            "place_descend_no_progress_steps": 0,
            "place_descend_timeout_extensions": 0,
            "preplace_timer": 0,
            "preplace_best_error": float("inf"),
            "preplace_no_progress_steps": 0,
            "preplace_timeout_extensions": 0,
            "place_reapproach_count": 0,
            "recovery_branch": "",
            "recovery_branch_retry": 0,
            "recovery_global_count": 0,
            "bt_branch_retry_cap": int(bt_branch_retry_cap),
            "bt_global_recovery_cap": int(bt_global_recovery_cap),
            "bt_rescan_hold_steps": int(bt_rescan_hold_steps),
            "bt_reapproach_offset_xy": float(bt_reapproach_offset_xy),
            "bt_safe_backoff_hold_steps": int(bt_safe_backoff_hold_steps),
            "bt_safe_backoff_z_boost": float(bt_safe_backoff_z_boost),
            "retry_count": 0,
            "last_retry_reason": "",
            "failure_reason": "",
            "belief_update_backend": "python_init",
            "rxinfer_enabled": int(bool(rxinfer_enabled)),
            "rxinfer_available": int(bool(_RXINFER_JULIA_AVAILABLE and not _RXINFER_RUNTIME_DISABLED)),
        }

    reach_obj_rel_local = reach_obj_rel_local_cfg.copy()
    align_obj_rel_local = align_obj_rel_local_cfg.copy()
    descend_obj_rel_local = descend_obj_rel_local_cfg.copy()
    if allow_bt_prior_override:
        def _maybe_override_ref(keys, current: np.ndarray) -> np.ndarray:
            for key in keys:
                if key not in previous_belief:
                    continue
                candidate = np.asarray(previous_belief.get(key), dtype=float).reshape(-1)
                if candidate.shape[0] == 3 and np.all(np.isfinite(candidate)):
                    return candidate.copy()
            return current

        # Backward-compatible lookup: prefer *_local keys, then legacy key names.
        reach_obj_rel_local = _maybe_override_ref(
            ["reach_obj_rel_local", "reach_obj_rel"], reach_obj_rel_local
        )
        align_obj_rel_local = _maybe_override_ref(
            ["align_obj_rel_local", "align_obj_rel"], align_obj_rel_local
        )
        descend_obj_rel_local = _maybe_override_ref(
            ["descend_obj_rel_local", "descend_obj_rel"], descend_obj_rel_local
        )
        preplace_target_rel = _maybe_override_ref(["preplace_target_rel"], preplace_target_rel)
        place_target_rel = _maybe_override_ref(["place_target_rel"], place_target_rel)
        retreat_move = _maybe_override_ref(["retreat_move"], retreat_move)

    # Belief update path:
    # - default: Python EMA fusion (baseline behavior)
    # - optional: RxInfer-backed one-step Bayesian update (Julia), then smoothed
    #   with the same alpha gains for compatibility with current phase logic.
    ee_obs = _safe_vec3(observation.get("o_ee", [0.0, 0.0, 0.0]))
    obj_obs_raw = _safe_vec3(observation.get("o_obj", [0.0, 0.0, 0.0]))
    if use_world_place_goal_pose:
        # Planner-provided world goal pose is treated as the canonical place target.
        # Convert to relative EE->target relation used by existing place pipeline.
        target_obs_raw = np.asarray(place_goal_world_xyz, dtype=float) - np.asarray(ee_obs, dtype=float)
    else:
        target_obs_raw = _safe_vec3(observation.get("o_target", [0.0, 0.0, 0.0]))
    obj_est = _safe_vec3(observation.get("o_obj_est", obj_obs_raw), default=obj_obs_raw)
    target_est = _safe_vec3(observation.get("o_target_est", target_obs_raw), default=target_obs_raw)
    obj_vel = _safe_vec3(observation.get("o_obj_vel", [0.0, 0.0, 0.0]))
    ee_vel = _safe_vec3(observation.get("o_ee_vel", [0.0, 0.0, 0.0]))
    if use_world_place_goal_pose:
        target_vel = np.zeros(3, dtype=float)
    else:
        target_vel = _safe_vec3(observation.get("o_target_vel", [0.0, 0.0, 0.0]))
    obs_dt = _safe_scalar(observation.get("o_dt", 0.0), default=0.0)
    predict_horizon = float(np.clip(obs_dt, 0.0, 0.04))

    obj_obs = obj_est + obj_vel * predict_horizon
    if not np.all(np.isfinite(obj_obs)):
        obj_obs = obj_est.copy()
    target_obs = target_est + target_vel * predict_horizon
    if not np.all(np.isfinite(target_obs)):
        target_obs = target_est.copy()

    prev_phase = previous_belief["phase"]
    prev_obj = np.asarray(previous_belief["s_obj_mean"], dtype=float).reshape(3)
    prev_ee = np.asarray(previous_belief["s_ee_mean"], dtype=float).reshape(3)
    prev_target = np.asarray(previous_belief["s_target_mean"], dtype=float).reshape(3)
    obj_yaw_obs = float(observation.get("o_obj_yaw", 0.0))
    if not np.isfinite(obj_yaw_obs):
        obj_yaw_obs = 0.0
    obj_jump = np.linalg.norm(obj_obs - prev_obj)
    alpha_obj = alpha_obj_reach if prev_phase == "Reach" else alpha_obj_default
    if obj_jump > obj_reacquire_jump:
        # Fast object motion (fall/slide): trust current observation more.
        alpha_obj = max(alpha_obj, 0.95)
    s_ee_cov = np.asarray(previous_belief["s_ee_cov"], dtype=float)
    s_obj_cov = np.asarray(previous_belief["s_obj_cov"], dtype=float)
    s_target_cov = np.asarray(previous_belief["s_target_cov"], dtype=float)
    belief_update_backend = "python_ema"

    # Increase process noise during strong object jumps so RxInfer can reacquire faster.
    rxinfer_obj_process_noise = float(rxinfer_process_noise_obj)
    if obj_jump > obj_reacquire_jump:
        rxinfer_obj_process_noise = max(
            rxinfer_obj_process_noise,
            float(rxinfer_process_noise_obj) * 2.5,
        )

    rxinfer_state = None
    if rxinfer_enabled:
        rxinfer_state = _rxinfer_step(
            previous_belief,
            ee_obs=ee_obs,
            obj_obs=obj_obs,
            target_obs=target_obs,
            ee_vel=ee_vel,
            obj_vel=obj_vel,
            target_vel=target_vel,
            dt=predict_horizon,
            process_noise_ee=rxinfer_process_noise_ee,
            process_noise_obj=rxinfer_obj_process_noise,
            process_noise_target=rxinfer_process_noise_target,
            obs_noise_ee=rxinfer_obs_noise_ee,
            obs_noise_obj=rxinfer_obs_noise_obj,
            obs_noise_target=rxinfer_obs_noise_target,
            min_variance=rxinfer_min_variance,
        )

    if rxinfer_state is not None:
        # Keep existing alpha gains to preserve established phase behavior.
        s_ee_mean = alpha_ee * rxinfer_state["s_ee_mean"] + (1 - alpha_ee) * prev_ee
        s_obj_mean = alpha_obj * rxinfer_state["s_obj_mean"] + (1 - alpha_obj) * prev_obj
        s_target_mean = alpha_target * rxinfer_state["s_target_mean"] + (1 - alpha_target) * prev_target
        s_ee_cov = np.asarray(rxinfer_state["s_ee_cov"], dtype=float)
        s_obj_cov = np.asarray(rxinfer_state["s_obj_cov"], dtype=float)
        s_target_cov = np.asarray(rxinfer_state["s_target_cov"], dtype=float)
        belief_update_backend = str(rxinfer_state.get("backend_name", "rxinfer"))
    else:
        s_ee_mean = alpha_ee * ee_obs + (1 - alpha_ee) * prev_ee
        s_obj_mean = alpha_obj * obj_obs + (1 - alpha_obj) * prev_obj
        s_target_mean = alpha_target * target_obs + (1 - alpha_target) * prev_target

    prev_approach_side_sign = float(previous_belief.get("approach_side_sign", 1.0))
    approach_side_sign = _compute_approach_side_sign(
        s_ee_mean=np.asarray(s_ee_mean, dtype=float),
        s_obj_mean=np.asarray(s_obj_mean, dtype=float),
        obj_yaw=obj_yaw_obs,
        prev_sign=prev_approach_side_sign,
    )
    # Reach/Align use dynamic side sign (toward safer base-facing side).
    reach_local_signed = reach_obj_rel_local.copy()
    align_local_signed = align_obj_rel_local.copy()
    reach_local_signed[0] = abs(float(reach_local_signed[0])) * approach_side_sign
    align_local_signed[0] = abs(float(align_local_signed[0])) * approach_side_sign
    # Descend keeps configured local relation (typically near-centered XY).
    descend_local = descend_obj_rel_local.copy()

    reach_obj_rel = _obj_local_rel_to_world(reach_local_signed, obj_yaw_obs)
    align_obj_rel = _obj_local_rel_to_world(align_local_signed, obj_yaw_obs)
    descend_obj_rel = _obj_local_rel_to_world(descend_local, obj_yaw_obs)
    prev_o_obj = np.asarray(previous_belief.get("prev_o_obj", obj_obs), dtype=float)
    prev_o_target = np.asarray(
        previous_belief.get("prev_o_target", target_obs),
        dtype=float,
    )
    prev_o_contact = int(previous_belief.get("prev_o_contact", int(observation["o_contact"])))
    obj_obs_jump = float(np.linalg.norm(obj_obs - prev_o_obj))
    target_obs_jump = float(np.linalg.norm(target_obs - prev_o_target))
    contact_flipped = int(int(observation["o_contact"]) != prev_o_contact)

    if confidence_enabled:
        obj_scale = max(obj_jump_confidence_scale, 1e-6)
        target_scale = max(target_jump_confidence_scale, 1e-6)
        obj_conf = float(np.exp(-obj_obs_jump / obj_scale))
        target_conf = float(np.exp(-target_obs_jump / target_scale))
        raw_conf = 0.5 * (obj_conf + target_conf)
        if contact_flipped:
            raw_conf -= float(contact_flip_confidence_penalty)
        raw_conf = float(np.clip(raw_conf, 0.0, 1.0))
        conf_alpha = float(np.clip(confidence_alpha, 0.0, 1.0))
        prev_conf = float(previous_belief.get("obs_confidence", 1.0))
        obs_confidence = (1.0 - conf_alpha) * prev_conf + conf_alpha * raw_conf
        obs_confidence = float(np.clip(obs_confidence, 0.0, 1.0))
    else:
        obs_confidence = 1.0

    phase_conf_ok = (not confidence_enabled) or (
        obs_confidence >= confidence_min_for_phase_change
    )
    vfe_total, vfe_pragmatic, vfe_epistemic = _compute_phase_vfe(
        phase=prev_phase,
        s_ee_mean=np.asarray(s_ee_mean, dtype=float),
        s_obj_mean=np.asarray(s_obj_mean, dtype=float),
        s_target_mean=np.asarray(s_target_mean, dtype=float),
        s_obj_cov=np.asarray(s_obj_cov, dtype=float),
        s_target_cov=np.asarray(s_target_cov, dtype=float),
        reach_obj_rel=reach_obj_rel,
        align_obj_rel=align_obj_rel,
        descend_obj_rel=descend_obj_rel,
        preplace_target_rel=preplace_target_rel,
        place_target_rel=place_target_rel,
        transit_height=transit_height,
        vfe_epistemic_weight=vfe_epistemic_weight,
    )
    phase_vfe_ok = (not vfe_phase_change_enabled) or (vfe_total <= vfe_phase_change_max)
    phase_gate_ok = bool(phase_conf_ok and phase_vfe_ok)
    place_yaw_error = float(
        abs(
            _yaw_error(
                current_yaw=obj_yaw_obs,
                desired_yaw=place_goal_world_yaw,
                pi_symmetric=place_goal_yaw_pi_symmetric,
            )
        )
    )
    place_yaw_ok = (not place_goal_yaw_enabled) or (
        place_yaw_error <= place_goal_yaw_threshold
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
    release_contact_counter = int(previous_belief.get("release_contact_counter", 0))
    release_warning = int(previous_belief.get("release_warning", 0))
    release_detach_counter = int(previous_belief.get("release_detach_counter", 0))
    release_stable_counter = int(previous_belief.get("release_stable_counter", 0))
    release_reapproach_count = int(previous_belief.get("release_reapproach_count", 0))
    reach_gate_active = int(previous_belief.get("reach_gate_active", 0))
    reach_gate_counter = int(previous_belief.get("reach_gate_counter", 0))
    align_gate_active = int(previous_belief.get("align_gate_active", 0))
    align_gate_counter = int(previous_belief.get("align_gate_counter", 0))
    pregrasp_ready_counter = int(previous_belief.get("pregrasp_ready_counter", 0))
    descend_gate_active = int(previous_belief.get("descend_gate_active", 0))
    descend_gate_counter = int(previous_belief.get("descend_gate_counter", 0))
    descend_yaw_enabled = int(previous_belief.get("descend_yaw_enabled", 0))
    place_descend_timer = int(previous_belief.get("place_descend_timer", 0))
    place_descend_best_error = float(previous_belief.get("place_descend_best_error", float("inf")))
    place_descend_no_progress_steps = int(previous_belief.get("place_descend_no_progress_steps", 0))
    place_descend_timeout_extensions = int(previous_belief.get("place_descend_timeout_extensions", 0))
    preplace_timer = int(previous_belief.get("preplace_timer", 0))
    preplace_best_error = float(previous_belief.get("preplace_best_error", float("inf")))
    preplace_no_progress_steps = int(previous_belief.get("preplace_no_progress_steps", 0))
    preplace_timeout_extensions = int(previous_belief.get("preplace_timeout_extensions", 0))
    place_reapproach_count = int(previous_belief.get("place_reapproach_count", 0))
    recovery_branch = str(previous_belief.get("recovery_branch", ""))
    recovery_branch_retry = int(previous_belief.get("recovery_branch_retry", 0))
    recovery_global_count = int(previous_belief.get("recovery_global_count", 0))
    last_retry_reason = str(previous_belief.get("last_retry_reason", ""))
    failure_reason = str(previous_belief.get("failure_reason", ""))
    lift_test_ref_obj_rel = np.array(
        previous_belief.get("lift_test_ref_obj_rel", s_obj_mean.copy()), dtype=float
    )
    retry_count = int(previous_belief.get("retry_count", 0))
    reach_error = np.linalg.norm(s_obj_mean - reach_obj_rel)
    prev_o_grip = float(previous_belief.get("prev_o_grip", observation["o_grip"]))
    grip_obs = float(observation["o_grip"])
    contact_obs = int(observation["o_contact"])
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
    if phase != "Failure":
        failure_reason = ""

    # Hysteresis/debounce gates to reduce threshold chatter around phase transitions.
    reach_enter_threshold = float(approach_threshold)
    reach_exit_threshold = float(max(approach_threshold * 1.35, approach_threshold + 0.005))
    align_enter_threshold = float(align_threshold)
    align_exit_threshold = float(max(align_threshold * 1.35, align_threshold + 0.004))
    descend_exit_x_threshold = float(max(descend_x_threshold * 1.40, descend_x_threshold + 0.004))
    descend_exit_y_threshold = float(max(descend_y_threshold * 1.40, descend_y_threshold + 0.004))
    descend_exit_z_threshold = float(max(descend_z_threshold * 1.40, descend_z_threshold + 0.004))
    reach_gate_stable_steps = 3
    align_gate_stable_steps = 2
    descend_gate_stable_steps = 3
    pregrasp_ready_steps = 4

    if phase == "Reach":
        if reach_gate_active == 1:
            reach_gate_active = int(reach_error <= reach_exit_threshold)
        else:
            reach_gate_active = int(reach_error <= reach_enter_threshold)
        reach_gate_counter = (reach_gate_counter + 1) if reach_gate_active == 1 else 0

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
        arc_radius_gate = max(0.05, float(reach_arc_min_radius))
        # Disable arc when sufficiently near in XY so final approach is linear.
        # This preserves arc for far/side cases but avoids late sideways motion.
        arc_disable_xy = float(max(reach_yaw_align_xy_enter, 0.6 * reach_enter_threshold))
        if r_ee >= arc_radius_gate and r_goal >= arc_radius_gate:
            theta_ee = float(np.arctan2(ee_xy[1], ee_xy[0]))
            theta_goal = float(np.arctan2(goal_xy[1], goal_xy[0]))
            dtheta_short = _wrap_to_pi(theta_goal - theta_ee)
            trigger = float(np.deg2rad(reach_arc_angle_trigger_deg))
            release = float(np.deg2rad(reach_arc_release_deg))
            if reach_turn_sign == 0 and abs(dtheta_short) >= trigger and xy_err > arc_disable_xy:
                reach_turn_sign = 1 if dtheta_short >= 0.0 else -1
            elif reach_turn_sign != 0 and abs(dtheta_short) <= release:
                reach_turn_sign = 0
        else:
            reach_turn_sign = 0
        if xy_err <= arc_disable_xy:
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

        if (
            reach_cooldown == 0
            and reach_gate_counter >= reach_gate_stable_steps
            and reach_yaw_align_active == 0
            and phase_gate_ok
        ):
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
            release_contact_counter = 0
            release_warning = 0
            reach_turn_sign = 0
            reach_best_error = float("inf")
            reach_no_progress_steps = 0
            reach_watchdog_active = 0
            reach_yaw_align_active = 0
            reach_yaw_align_timer = 0
            reach_yaw_align_done = 0
            reach_gate_active = 0
            reach_gate_counter = 0
            align_gate_active = 0
            align_gate_counter = 0
            pregrasp_ready_counter = 0
            descend_gate_active = 0
            descend_gate_counter = 0
            descend_yaw_enabled = 0
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
        if align_gate_active == 1:
            align_gate_active = int(align_error <= align_exit_threshold)
        else:
            align_gate_active = int(align_error <= align_enter_threshold)
        align_gate_counter = (align_gate_counter + 1) if align_gate_active == 1 else 0

        align_ready = (
            align_timer >= align_min_steps
            and gripper_open_ready
            and align_gate_counter >= align_gate_stable_steps
        )
        align_timeout = align_timer >= align_max_steps
        align_timeout_near = align_error <= align_exit_threshold
        if align_ready and align_error < align_threshold and phase_gate_ok:
            phase = "PreGraspHold"
            last_retry_reason = ""
            pregrasp_hold_timer = 0
            pregrasp_ready_counter = 0
            grasp_timer = 0
            stable_grasp_counter = 0
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            close_hold_timer = 0
            lift_test_timer = 0
            release_contact_counter = 0
            release_warning = 0
            align_gate_active = 0
            align_gate_counter = 0
        elif align_timeout:
            # Timeout fallback: continue only when still near and gripper is open-ready.
            # Otherwise retry Reach to avoid entering a long hold stall.
            if align_timeout_near and gripper_open_ready and phase_gate_ok:
                phase = "PreGraspHold"
                last_retry_reason = ""
                pregrasp_hold_timer = 0
                pregrasp_ready_counter = 0
                grasp_timer = 0
                stable_grasp_counter = 0
                descend_timer = 0
                descend_best_error = float("inf")
                descend_no_progress_steps = 0
                descend_timeout_extensions = 0
                close_hold_timer = 0
                lift_test_timer = 0
                release_contact_counter = 0
                release_warning = 0
                align_gate_active = 0
                align_gate_counter = 0
            else:
                phase = "Reach"
                last_retry_reason = "reach_stall"
                reach_cooldown = reach_reentry_cooldown_steps
                align_timer = 0
                pregrasp_hold_timer = 0
                descend_timer = 0
                descend_best_error = float("inf")
                descend_no_progress_steps = 0
                descend_timeout_extensions = 0
                close_hold_timer = 0
                lift_test_timer = 0
                reach_gate_active = 0
                reach_gate_counter = 0
                align_gate_active = 0
                align_gate_counter = 0
                pregrasp_ready_counter = 0
                descend_gate_active = 0
                descend_gate_counter = 0
                descend_yaw_enabled = 0
    elif phase == "PreGraspHold":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        pregrasp_hold_timer += 1
        align_error = np.linalg.norm(s_obj_mean - align_obj_rel)
        if gripper_open_ready and align_error <= align_exit_threshold:
            pregrasp_ready_counter += 1
        else:
            pregrasp_ready_counter = 0

        hold_ready = pregrasp_ready_counter >= pregrasp_ready_steps
        hold_grace_elapsed = pregrasp_hold_timer >= (
            pregrasp_hold_steps + pregrasp_hold_ready_grace_steps
        )
        hold_timeout = pregrasp_hold_timer >= pregrasp_hold_steps
        if (hold_ready or (hold_timeout and (gripper_open_ready or hold_grace_elapsed))) and phase_gate_ok:
            phase = "Descend"
            last_retry_reason = ""
            descend_timer = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            stable_grasp_counter = 0
            pregrasp_ready_counter = 0
            descend_gate_active = 0
            descend_gate_counter = 0
            descend_yaw_enabled = 0
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

        xy_enter_ok = x_err <= descend_x_threshold and y_err <= descend_y_threshold
        z_enter_ok = z_err <= descend_z_threshold
        xy_exit_ok = x_err <= descend_exit_x_threshold and y_err <= descend_exit_y_threshold
        z_exit_ok = z_err <= descend_exit_z_threshold

        if descend_gate_active == 1:
            descend_gate_active = int(xy_exit_ok and z_exit_ok)
        else:
            descend_gate_active = int(xy_enter_ok and z_enter_ok)
        descend_gate_counter = (descend_gate_counter + 1) if descend_gate_active == 1 else 0

        if descend_yaw_enabled == 1:
            descend_yaw_enabled = int(xy_exit_ok)
        else:
            descend_yaw_enabled = int(xy_enter_ok)

        position_ok = (
            (descend_error <= descend_threshold and descend_gate_counter >= 2)
            or (descend_gate_counter >= descend_gate_stable_steps)
        )

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

        if (position_ok or (timeout_hit and timeout_near)) and phase_gate_ok:
            phase = "CloseHold"
            last_retry_reason = ""
            close_hold_timer = 0
            stable_grasp_counter = 0
            descend_best_error = float("inf")
            descend_no_progress_steps = 0
            descend_timeout_extensions = 0
            descend_gate_active = 0
            descend_gate_counter = 0
            descend_yaw_enabled = 0
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
            last_retry_reason = "grasp_failed"
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
            release_contact_counter = 0
            release_warning = 0
            reach_gate_active = 0
            reach_gate_counter = 0
            align_gate_active = 0
            align_gate_counter = 0
            pregrasp_ready_counter = 0
            descend_gate_active = 0
            descend_gate_counter = 0
            descend_yaw_enabled = 0
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
            and phase_gate_ok
        ):
            phase = "LiftTest"
            last_retry_reason = ""
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
            last_retry_reason = "grasp_failed"
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
            release_contact_counter = 0
            release_warning = 0
            reach_gate_active = 0
            reach_gate_counter = 0
            align_gate_active = 0
            align_gate_counter = 0
            pregrasp_ready_counter = 0
            descend_gate_active = 0
            descend_gate_counter = 0
            descend_yaw_enabled = 0
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
            last_retry_reason = "grasp_failed"
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
            last_retry_reason = "grasp_failed"
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
            release_contact_counter = 0
            release_warning = 0
        else:
            if lift_test_timer >= lift_test_steps and phase_gate_ok:
                retry_count = 0
                phase = "Transit"
                last_retry_reason = ""
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
            last_retry_reason = ""
            preplace_timer = 0
            preplace_best_error = float("inf")
            preplace_no_progress_steps = 0
            preplace_timeout_extensions = 0
        elif s_grasp == 0:
            phase = "Reach"
            last_retry_reason = "grasp_failed"
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
            release_contact_counter = 0
            release_warning = 0
    elif phase == "MoveToPlaceAbove":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        preplace_timer += 1
        err = s_target_mean - preplace_target_rel
        preplace_err = float(np.linalg.norm(err))
        preplace_xy_err = float(np.linalg.norm(err[:2]))
        preplace_z_err = abs(float(err[2]))
        preplace_xy_ok = float(np.linalg.norm(err[:2])) <= preplace_xy_threshold
        preplace_z_ok = abs(float(err[2])) <= preplace_z_threshold

        if not np.isfinite(preplace_best_error):
            preplace_best_error = preplace_err
            preplace_no_progress_steps = 0
        elif preplace_err < (preplace_best_error - preplace_progress_eps):
            preplace_best_error = preplace_err
            preplace_no_progress_steps = 0
        else:
            preplace_no_progress_steps += 1

        timeout_hit = preplace_timer >= preplace_max_steps
        timeout_near = (
            preplace_xy_err <= preplace_timeout_xy_threshold
            and preplace_z_err <= preplace_timeout_z_threshold
        )
        can_extend = (
            timeout_hit
            and not timeout_near
            and preplace_timeout_extensions < preplace_max_timeout_extensions
            and preplace_no_progress_steps < preplace_stall_steps
        )

        if ((preplace_xy_ok and preplace_z_ok) or (timeout_hit and timeout_near)) and phase_gate_ok:
            phase = "DescendToPlace"
            last_retry_reason = ""
            place_descend_timer = 0
            place_descend_best_error = float("inf")
            place_descend_no_progress_steps = 0
            place_descend_timeout_extensions = 0
            preplace_timer = 0
            preplace_best_error = float("inf")
            preplace_no_progress_steps = 0
            preplace_timeout_extensions = 0
            place_reapproach_count = 0
        elif can_extend:
            preplace_timeout_extensions += 1
            preplace_timer = max(0, preplace_timer - preplace_timeout_extension_steps)
            preplace_no_progress_steps = 0
        elif timeout_hit:
            last_retry_reason = "place_alignment_failed"
            place_reapproach_count += 1
            preplace_timer = 0
            preplace_best_error = float("inf")
            preplace_no_progress_steps = 0
            preplace_timeout_extensions = 0
            if place_reapproach_count > place_reapproach_max_retries:
                phase = "Failure"
                failure_reason = "place_alignment_failed"
        elif s_grasp == 0:
            phase = "Reach"
            last_retry_reason = "grasp_failed"
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
            release_contact_counter = 0
            release_warning = 0
            preplace_timer = 0
            preplace_best_error = float("inf")
            preplace_no_progress_steps = 0
            preplace_timeout_extensions = 0
    elif phase == "DescendToPlace":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        place_descend_timer += 1
        err = s_target_mean - place_target_rel
        place_err = float(np.linalg.norm(err))
        place_xy_ok = float(np.linalg.norm(err[:2])) <= place_xy_threshold
        place_z_ok = abs(float(err[2])) <= place_z_threshold

        if not np.isfinite(place_descend_best_error):
            place_descend_best_error = place_err
            place_descend_no_progress_steps = 0
        elif place_err < (place_descend_best_error - place_descend_progress_eps):
            place_descend_best_error = place_err
            place_descend_no_progress_steps = 0
        else:
            place_descend_no_progress_steps += 1

        timeout_hit = place_descend_timer >= place_descend_max_steps
        timeout_near = (
            float(np.linalg.norm(err[:2])) <= (1.5 * place_xy_threshold)
            and abs(float(err[2])) <= (1.5 * place_z_threshold)
        )
        can_extend = (
            timeout_hit
            and not timeout_near
            and place_descend_timeout_extensions < place_descend_max_timeout_extensions
            and place_descend_no_progress_steps < place_descend_stall_steps
        )

        if place_xy_ok and place_z_ok and place_yaw_ok and phase_gate_ok:
            phase = "Open"
            last_retry_reason = ""
            open_timer = 0
            release_contact_counter = 0
            release_warning = 0
            release_detach_counter = 0
            release_stable_counter = 0
            place_descend_timer = 0
            place_descend_best_error = float("inf")
            place_descend_no_progress_steps = 0
            place_descend_timeout_extensions = 0
        elif can_extend:
            place_descend_timeout_extensions += 1
            place_descend_timer = max(0, place_descend_timer - place_descend_timeout_extension_steps)
            place_descend_no_progress_steps = 0
        elif timeout_hit:
            if s_grasp == 1:
                place_reapproach_count += 1
                last_retry_reason = "place_alignment_failed"
                place_descend_timer = 0
                place_descend_best_error = float("inf")
                place_descend_no_progress_steps = 0
                place_descend_timeout_extensions = 0
                if place_reapproach_count <= place_reapproach_max_retries:
                    phase = "MoveToPlaceAbove"
                else:
                    phase = "Failure"
                    failure_reason = "place_alignment_failed"
            else:
                phase = "Reach"
                last_retry_reason = "grasp_failed"
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
                release_contact_counter = 0
                release_warning = 0
                place_descend_timer = 0
                place_descend_best_error = float("inf")
                place_descend_no_progress_steps = 0
                place_descend_timeout_extensions = 0
                place_reapproach_count = 0
        elif s_grasp == 0:
            phase = "Reach"
            last_retry_reason = "grasp_failed"
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
            release_contact_counter = 0
            release_warning = 0
            place_descend_timer = 0
            place_descend_best_error = float("inf")
            place_descend_no_progress_steps = 0
            place_descend_timeout_extensions = 0
            place_reapproach_count = 0
    elif phase == "Open":
        reach_turn_sign = 0
        reach_best_error = float("inf")
        reach_no_progress_steps = 0
        reach_watchdog_active = 0
        reach_yaw_align_active = 0
        reach_yaw_align_timer = 0
        reach_yaw_align_done = 0
        if observation["o_contact"] == 1:
            release_contact_counter += 1
        else:
            release_contact_counter = 0
        if release_contact_counter >= release_contact_warn_steps:
            release_warning = 1

        # Post-open release verification:
        # 1) detached from gripper, 2) object stable near place target.
        obj_to_target = np.asarray(s_target_mean - s_obj_mean, dtype=float)
        obj_target_xy_err = float(np.linalg.norm(obj_to_target[:2]))
        obj_target_z_err = float(abs(obj_to_target[2]))
        # Release stability should use object world speed.
        # o_obj is object relative to EE, so:
        #   v_obj_world = v_obj_rel + v_ee_world
        obj_world_vel = np.asarray(obj_vel, dtype=float).reshape(3) + np.asarray(ee_vel, dtype=float).reshape(3)
        obj_speed = float(np.linalg.norm(obj_world_vel))
        detached_now = int(observation["o_contact"] == 0 and gripper_open_ready)
        if detached_now:
            release_detach_counter += 1
        else:
            release_detach_counter = 0
        stable_now = (
            obj_target_xy_err <= release_obj_xy_threshold
            and obj_target_z_err <= release_obj_z_threshold
            and obj_speed <= release_obj_speed_threshold
            and place_yaw_ok
        )
        if stable_now:
            release_stable_counter += 1
        else:
            release_stable_counter = 0

        release_ok = (
            (not release_verify_enabled)
            or (
                release_detach_counter >= release_detach_hold_steps
                and release_stable_counter >= release_stable_hold_steps
            )
        )
        open_timer += 1
        if open_timer >= open_hold_steps and release_ok:
            phase = "Retreat"
            last_retry_reason = ""
            retreat_timer = 0
            release_reapproach_count = 0
        elif open_timer >= open_max_steps and not release_ok:
            last_retry_reason = "release_failed"
            release_reapproach_count += 1
            open_timer = 0
            release_contact_counter = 0
            release_detach_counter = 0
            release_stable_counter = 0
            release_warning = 0
            place_descend_timer = 0
            place_descend_best_error = float("inf")
            place_descend_no_progress_steps = 0
            place_descend_timeout_extensions = 0
            if release_reapproach_count <= release_reapproach_max_retries:
                phase = "MoveToPlaceAbove"
            else:
                phase = "Failure"
                failure_reason = "release_failed"
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

    if phase != "Open":
        release_detach_counter = 0
        release_stable_counter = 0

    if phase != "DescendToPlace":
        place_descend_timer = 0
        place_descend_best_error = float("inf")
        place_descend_no_progress_steps = 0
        place_descend_timeout_extensions = 0
    if phase != "MoveToPlaceAbove":
        preplace_timer = 0
        preplace_best_error = float("inf")
        preplace_no_progress_steps = 0
        preplace_timeout_extensions = 0

    if phase in ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest", "Failure", "Done"):
        place_reapproach_count = 0

    if phase in ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest", "Retreat", "Done", "Failure"):
        release_reapproach_count = 0

    if phase != "Reach":
        reach_gate_active = 0
        reach_gate_counter = 0
    if phase != "Align":
        align_gate_active = 0
        align_gate_counter = 0
    if phase != "PreGraspHold":
        pregrasp_ready_counter = 0
    if phase != "Descend":
        descend_gate_active = 0
        descend_gate_counter = 0
        descend_yaw_enabled = 0

    if phase in ("Reach", "Align", "PreGraspHold", "Descend", "CloseHold", "LiftTest", "Transit", "MoveToPlaceAbove", "DescendToPlace"):
        # Ensure lift-test reference remains defined across non-done phases.
        if lift_test_ref_obj_rel is None:
            lift_test_ref_obj_rel = s_obj_mean.copy()

    return {
        "s_ee_mean": s_ee_mean,
        "s_obj_mean": s_obj_mean,
        "s_target_mean": s_target_mean,
        "s_obj_yaw": float(obj_yaw_obs),
        "reach_obj_rel_local": reach_obj_rel_local.copy(),
        "align_obj_rel_local": align_obj_rel_local.copy(),
        "descend_obj_rel_local": descend_obj_rel_local.copy(),
        "approach_side_sign": float(approach_side_sign),
        "reach_obj_rel": reach_obj_rel.copy(),
        "align_obj_rel": align_obj_rel.copy(),
        "descend_obj_rel": descend_obj_rel.copy(),
        "preplace_target_rel": preplace_target_rel.copy(),
        "place_target_rel": place_target_rel.copy(),
        "use_world_place_goal_pose": int(use_world_place_goal_pose),
        "place_goal_world_xyz": np.asarray(place_goal_world_xyz, dtype=float).copy(),
        "place_goal_world_pose6d_deg": (
            None
            if place_goal_world_pose6d_deg is None
            else np.asarray(place_goal_world_pose6d_deg, dtype=float).copy()
        ),
        "place_goal_yaw": float(place_goal_world_yaw),
        "place_goal_yaw_enabled": int(place_goal_yaw_enabled),
        "place_goal_yaw_threshold": float(place_goal_yaw_threshold),
        "place_goal_yaw_pi_symmetric": int(place_goal_yaw_pi_symmetric),
        "place_goal_yaw_error": float(place_yaw_error),
        "retreat_move": retreat_move.copy(),
        "s_ee_cov": s_ee_cov,
        "s_obj_cov": s_obj_cov,
        "s_target_cov": s_target_cov,
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
        "reach_gate_active": int(reach_gate_active),
        "reach_gate_counter": int(reach_gate_counter),
        "align_gate_active": int(align_gate_active),
        "align_gate_counter": int(align_gate_counter),
        "pregrasp_ready_counter": int(pregrasp_ready_counter),
        "descend_gate_active": int(descend_gate_active),
        "descend_gate_counter": int(descend_gate_counter),
        "descend_yaw_enabled": int(descend_yaw_enabled),
        "prev_o_grip": grip_obs,
        "prev_o_contact": int(contact_obs),
        "prev_o_obj": np.asarray(obj_obs, dtype=float).copy(),
        "prev_o_target": np.asarray(target_obs, dtype=float).copy(),
        "obs_confidence": float(obs_confidence),
        "phase_conf_ok": int(phase_conf_ok),
        "phase_vfe_ok": int(phase_vfe_ok),
        "phase_gate_ok": int(phase_gate_ok),
        "vfe_total": float(vfe_total if vfe_enabled else 0.0),
        "vfe_pragmatic": float(vfe_pragmatic if vfe_enabled else 0.0),
        "vfe_epistemic": float(vfe_epistemic if vfe_enabled else 0.0),
        "vfe_recover_enabled": int(vfe_recover_enabled),
        "vfe_recover_threshold": float(vfe_recover_threshold),
        "vfe_recover_steps": int(vfe_recover_steps),
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
        "release_contact_counter": int(release_contact_counter),
        "release_warning": int(release_warning),
        "release_detach_counter": int(release_detach_counter),
        "release_stable_counter": int(release_stable_counter),
        "release_reapproach_count": int(release_reapproach_count),
        "gripper_open_ready": int(gripper_open_ready),
        "gripper_close_ready": int(gripper_close_ready),
        "place_descend_timer": int(place_descend_timer),
        "place_descend_best_error": float(place_descend_best_error),
        "place_descend_no_progress_steps": int(place_descend_no_progress_steps),
        "place_descend_timeout_extensions": int(place_descend_timeout_extensions),
        "preplace_timer": int(preplace_timer),
        "preplace_best_error": float(preplace_best_error),
        "preplace_no_progress_steps": int(preplace_no_progress_steps),
        "preplace_timeout_extensions": int(preplace_timeout_extensions),
        "place_reapproach_count": int(place_reapproach_count),
        "recovery_branch": str(recovery_branch),
        "recovery_branch_retry": int(recovery_branch_retry),
        "recovery_global_count": int(recovery_global_count),
        "bt_branch_retry_cap": int(bt_branch_retry_cap),
        "bt_global_recovery_cap": int(bt_global_recovery_cap),
        "bt_rescan_hold_steps": int(bt_rescan_hold_steps),
        "bt_reapproach_offset_xy": float(bt_reapproach_offset_xy),
        "bt_safe_backoff_hold_steps": int(bt_safe_backoff_hold_steps),
        "bt_safe_backoff_z_boost": float(bt_safe_backoff_z_boost),
        "retry_count": retry_count,
        "last_retry_reason": str(last_retry_reason),
        "failure_reason": str(failure_reason),
        "belief_update_backend": str(belief_update_backend),
        "rxinfer_enabled": int(bool(rxinfer_enabled)),
        "rxinfer_available": int(bool(_RXINFER_JULIA_AVAILABLE and not _RXINFER_RUNTIME_DISABLED)),
    }
