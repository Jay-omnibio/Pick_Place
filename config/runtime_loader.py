from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

DEFAULT_COMMON_CONFIG_PATH = "config/common_robot.yaml"
DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH = "config/active_inference_config.yaml"

COMMON_REQUIRED_TOP_LEVEL_KEYS = {"run", "controller"}
COMMON_ALLOWED_TOP_LEVEL_KEYS = COMMON_REQUIRED_TOP_LEVEL_KEYS | {"task_shared"}
ACTIVE_TOP_LEVEL_KEYS = {"inference", "action_selection"}

REQUIRED_RUN_KEYS = {"log_every_steps"}
REQUIRED_CONTROLLER_KEYS = {
    "max_step",
    "min_height",
    "max_target_radius",
    "ik_damping",
    "nullspace_gain",
    "nullspace_gain_grasp",
    "max_joint_step",
    "ee_tolerance",
    "move_smoothing",
    "yaw_target",
    "yaw_weight",
    "yaw_weight_grasp",
    "enable_yaw_objective",
    "yaw_axis",
    "enable_topdown_objective",
    "topdown_weight",
    "topdown_weight_grasp",
    "tool_axis",
    "gripper_open_width",
    "gripper_close_width",
    "gripper_rate",
    "gripper_width_tol",
    "gripper_speed_tol",
    "gripper_switch_cooldown_steps",
    "debug_every_steps",
}

REQUIRED_TASK_SHARED_KEYS = {
    "target_world_xyz",
    "target_world_yaw_deg",
}
OPTIONAL_TASK_SHARED_KEYS = {
    "target_world_pose6d_deg",
}

REQUIRED_AI_INFERENCE_KEYS = {
    "reach_obj_rel",
    "align_obj_rel",
    "descend_obj_rel",
    "approach_threshold",
    "align_threshold",
    "descend_threshold",
    "align_max_steps",
    "align_min_steps",
    "pregrasp_hold_steps",
    "pregrasp_hold_ready_grace_steps",
    "descend_max_steps",
    "descend_x_threshold",
    "descend_y_threshold",
    "descend_z_threshold",
    "descend_timeout_x_threshold",
    "descend_timeout_y_threshold",
    "descend_timeout_z_threshold",
    "descend_progress_eps",
    "descend_stall_steps",
    "descend_timeout_extension_steps",
    "descend_max_timeout_extensions",
    "close_hold_steps",
    "grasp_search_steps",
    "grasp_stable_steps_for_lift",
    "reach_reentry_cooldown_steps",
    "lift_test_steps",
    "lift_test_obj_rel_drift_max",
    "max_retries",
    "alpha_ee",
    "alpha_obj_default",
    "alpha_obj_reach",
    "obj_reacquire_jump",
    "contact_on_count",
    "contact_off_count",
    "reach_arc_angle_trigger_deg",
    "reach_arc_release_deg",
    "reach_progress_eps",
    "reach_stall_steps",
    "reach_yaw_align_xy_enter",
    "reach_yaw_align_steps",
    "grip_close_ready_max_width",
    "confidence_enabled",
    "confidence_alpha",
    "confidence_min_for_phase_change",
    "obj_jump_confidence_scale",
    "target_jump_confidence_scale",
    "contact_flip_confidence_penalty",
    "allow_bt_prior_override",
    "bt_set_priors_enabled",
    "bt_retry_reach_z_step",
    "bt_retry_reach_z_max",
    "bt_branch_retry_cap",
    "bt_global_recovery_cap",
    "bt_rescan_hold_steps",
    "bt_reapproach_offset_xy",
    "bt_safe_backoff_hold_steps",
    "bt_safe_backoff_z_boost",
    "vfe_enabled",
    "vfe_epistemic_weight",
    "vfe_phase_change_enabled",
    "vfe_phase_change_max",
    "vfe_recover_enabled",
    "vfe_recover_threshold",
    "vfe_recover_steps",
    "release_contact_warn_steps",
    "release_verify_enabled",
    "release_detach_hold_steps",
    "release_stable_hold_steps",
    "release_obj_xy_threshold",
    "release_obj_z_threshold",
    "release_obj_speed_threshold",
    "open_max_steps",
    "release_reapproach_max_retries",
    "risk_detection_enabled",
    "singularity_dq_ratio_threshold",
    "singularity_no_progress_steps",
    "unintended_contact_warn_steps",
    "alpha_target",
    "preplace_target_rel",
    "place_target_rel",
    "place_goal_yaw_enabled",
    "place_goal_yaw_threshold_deg",
    "place_goal_yaw_pi_symmetric",
    "preplace_threshold",
    "place_threshold",
    "preplace_xy_threshold",
    "preplace_z_threshold",
    "place_xy_threshold",
    "place_z_threshold",
    "place_descend_max_steps",
    "place_descend_progress_eps",
    "place_descend_stall_steps",
    "place_descend_timeout_extension_steps",
    "place_descend_max_timeout_extensions",
    "place_reapproach_max_retries",
    "transit_height",
    "transit_z_threshold",
    "open_hold_steps",
    "retreat_steps",
    "retreat_move",
}

REQUIRED_AI_ACTION_KEYS = {
    "lambda_epistemic",
    "delta",
    "reach_delta",
    "action_effectiveness",
    "reach_axis_threshold_x",
    "reach_axis_threshold_y",
    "reach_z_blend_start",
    "reach_z_blend_full",
    "reach_step_min",
    "reach_arc_enabled",
    "reach_arc_max_theta_step_deg",
    "reach_arc_radial_step_max",
    "reach_arc_min_radius",
    "reach_arc_max_radius",
    "grasp_step",
    "align_step",
    "pregrasp_hold_step",
    "confidence_speed_scaling_enabled",
    "confidence_speed_min_scale",
    "reach_confidence_speed_power",
    "descend_confidence_speed_power",
    "place_goal_gripper_yaw_offset_deg",
}

OPTIONAL_AI_INFERENCE_KEYS = {
    "use_world_place_goal_pose",
    "place_goal_world_xyz",
    "place_goal_world_yaw_deg",
    "place_goal_world_pose6d_deg",
}

AI_VECTOR_KEYS = {
    "reach_obj_rel",
    "align_obj_rel",
    "descend_obj_rel",
    "preplace_target_rel",
    "place_target_rel",
    "place_goal_world_xyz",
    "retreat_move",
}
AI_BOOL_KEYS = {
    "reach_arc_enabled",
    "confidence_enabled",
    "risk_detection_enabled",
    "allow_bt_prior_override",
    "bt_set_priors_enabled",
    "vfe_enabled",
    "vfe_phase_change_enabled",
    "vfe_recover_enabled",
    "release_verify_enabled",
    "use_world_place_goal_pose",
    "place_goal_yaw_enabled",
    "place_goal_yaw_pi_symmetric",
    "confidence_speed_scaling_enabled",
}
AI_INT_KEYS = {
    "align_max_steps",
    "align_min_steps",
    "pregrasp_hold_steps",
    "pregrasp_hold_ready_grace_steps",
    "descend_max_steps",
    "descend_stall_steps",
    "descend_timeout_extension_steps",
    "descend_max_timeout_extensions",
    "close_hold_steps",
    "grasp_search_steps",
    "grasp_stable_steps_for_lift",
    "reach_reentry_cooldown_steps",
    "lift_test_steps",
    "max_retries",
    "contact_on_count",
    "contact_off_count",
    "bt_branch_retry_cap",
    "bt_global_recovery_cap",
    "bt_rescan_hold_steps",
    "bt_safe_backoff_hold_steps",
    "reach_stall_steps",
    "reach_yaw_align_steps",
    "vfe_recover_steps",
    "release_contact_warn_steps",
    "release_detach_hold_steps",
    "release_stable_hold_steps",
    "open_max_steps",
    "release_reapproach_max_retries",
    "singularity_no_progress_steps",
    "unintended_contact_warn_steps",
    "place_descend_max_steps",
    "place_descend_stall_steps",
    "place_descend_timeout_extension_steps",
    "place_descend_max_timeout_extensions",
    "place_reapproach_max_retries",
    "open_hold_steps",
    "retreat_steps",
}


def _load_yaml_dict(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} config not found: {path}")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} config root must be a mapping/object: {path}")
    return loaded


def _ensure_dict(section_name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return value


def _check_unknown_keys(section_name: str, data: Dict[str, Any], allowed_keys: set[str]) -> None:
    unknown = sorted(set(data.keys()) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown keys in section '{section_name}': {unknown}")


def _check_missing_keys(section_name: str, data: Dict[str, Any], required_keys: set[str]) -> None:
    missing = sorted(required_keys - set(data.keys()))
    if missing:
        raise ValueError(f"Missing required keys in section '{section_name}': {missing}")


def _coerce_active_section(
    section_name: str,
    data: Dict[str, Any],
    required_keys: set[str],
    optional_keys: set[str] | None = None,
) -> Dict[str, Any]:
    opt = set(optional_keys or set())
    allowed_keys = set(required_keys) | opt
    _check_unknown_keys(section_name, data, allowed_keys)
    _check_missing_keys(section_name, data, required_keys)

    out: Dict[str, Any] = {}
    for key in required_keys:
        value = data[key]
        if key in AI_VECTOR_KEYS:
            vec = np.asarray(value, dtype=float).reshape(-1)
            if vec.shape[0] != 3:
                raise ValueError(f"Section '{section_name}' key '{key}' must be length-3 vector.")
            out[key] = vec
        elif key in AI_BOOL_KEYS:
            out[key] = bool(value)
        elif key in AI_INT_KEYS:
            out[key] = int(value)
        else:
            out[key] = float(value)
    for key in opt:
        if key not in data:
            continue
        value = data[key]
        if key == "place_goal_world_pose6d_deg":
            vec = np.asarray(value, dtype=float).reshape(-1)
            if vec.shape[0] != 6:
                raise ValueError(
                    f"Section '{section_name}' key '{key}' must be length-6 vector "
                    "[x,y,z,roll_deg,pitch_deg,yaw_deg]."
                )
            out[key] = vec
            continue
        if key in AI_VECTOR_KEYS:
            vec = np.asarray(value, dtype=float).reshape(-1)
            if vec.shape[0] != 3:
                raise ValueError(f"Section '{section_name}' key '{key}' must be length-3 vector.")
            out[key] = vec
        elif key in AI_BOOL_KEYS:
            out[key] = bool(value)
        elif key in AI_INT_KEYS:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def load_runtime_sections(
    common_path: str = DEFAULT_COMMON_CONFIG_PATH,
    active_inference_path: str = DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH,
) -> Dict[str, Any]:
    common_cfg_path = Path(common_path)
    ai_cfg_path = Path(active_inference_path)

    common_raw = _load_yaml_dict(common_cfg_path, "common")
    _check_unknown_keys("common_root", common_raw, COMMON_ALLOWED_TOP_LEVEL_KEYS)
    _check_missing_keys("common_root", common_raw, COMMON_REQUIRED_TOP_LEVEL_KEYS)

    run_cfg = _ensure_dict("run", common_raw["run"])
    _check_unknown_keys("run", run_cfg, REQUIRED_RUN_KEYS)
    _check_missing_keys("run", run_cfg, REQUIRED_RUN_KEYS)
    run_cfg_out = {
        "log_every_steps": int(run_cfg["log_every_steps"]),
    }

    controller_cfg = _ensure_dict("controller", common_raw["controller"])
    _check_unknown_keys("controller", controller_cfg, REQUIRED_CONTROLLER_KEYS)
    _check_missing_keys("controller", controller_cfg, REQUIRED_CONTROLLER_KEYS)
    controller_cfg_out = {
        "max_step": float(controller_cfg["max_step"]),
        "min_height": float(controller_cfg["min_height"]),
        "max_target_radius": float(controller_cfg["max_target_radius"]),
        "ik_damping": float(controller_cfg["ik_damping"]),
        "nullspace_gain": float(controller_cfg["nullspace_gain"]),
        "nullspace_gain_grasp": float(controller_cfg["nullspace_gain_grasp"]),
        "max_joint_step": float(controller_cfg["max_joint_step"]),
        "ee_tolerance": float(controller_cfg["ee_tolerance"]),
        "move_smoothing": float(controller_cfg["move_smoothing"]),
        "yaw_target": controller_cfg["yaw_target"],
        "yaw_weight": float(controller_cfg["yaw_weight"]),
        "yaw_weight_grasp": float(controller_cfg["yaw_weight_grasp"]),
        "enable_yaw_objective": bool(controller_cfg["enable_yaw_objective"]),
        "yaw_axis": int(controller_cfg["yaw_axis"]),
        "enable_topdown_objective": bool(controller_cfg["enable_topdown_objective"]),
        "topdown_weight": float(controller_cfg["topdown_weight"]),
        "topdown_weight_grasp": float(controller_cfg["topdown_weight_grasp"]),
        "tool_axis": int(controller_cfg["tool_axis"]),
        "gripper_open_width": float(controller_cfg["gripper_open_width"]),
        "gripper_close_width": float(controller_cfg["gripper_close_width"]),
        "gripper_rate": float(controller_cfg["gripper_rate"]),
        "gripper_width_tol": float(controller_cfg["gripper_width_tol"]),
        "gripper_speed_tol": float(controller_cfg["gripper_speed_tol"]),
        "gripper_switch_cooldown_steps": int(controller_cfg["gripper_switch_cooldown_steps"]),
        "debug_every_steps": int(controller_cfg["debug_every_steps"]),
    }

    shared_task_cfg: Dict[str, Any] = {}
    if "task_shared" in common_raw:
        task_shared_cfg = _ensure_dict("task_shared", common_raw["task_shared"])
        allowed = REQUIRED_TASK_SHARED_KEYS | OPTIONAL_TASK_SHARED_KEYS
        _check_unknown_keys("task_shared", task_shared_cfg, allowed)
        _check_missing_keys("task_shared", task_shared_cfg, REQUIRED_TASK_SHARED_KEYS)
        target_xyz = np.asarray(task_shared_cfg["target_world_xyz"], dtype=float).reshape(-1)
        if target_xyz.shape[0] != 3:
            raise ValueError("Section 'task_shared' key 'target_world_xyz' must be length-3 vector.")
        target_yaw_deg = float(task_shared_cfg["target_world_yaw_deg"])
        shared_task_cfg = {
            "target_world_xyz": target_xyz[:3].copy(),
            "target_world_yaw_deg": target_yaw_deg,
        }
        if "target_world_pose6d_deg" in task_shared_cfg:
            pose6 = np.asarray(task_shared_cfg["target_world_pose6d_deg"], dtype=float).reshape(-1)
            if pose6.shape[0] != 6:
                raise ValueError(
                    "Section 'task_shared' key 'target_world_pose6d_deg' must be length-6 vector "
                    "[x,y,z,roll_deg,pitch_deg,yaw_deg]."
                )
            if not np.all(np.isfinite(pose6)):
                raise ValueError(
                    "Section 'task_shared' key 'target_world_pose6d_deg' must contain finite values."
                )
            shared_task_cfg["target_world_pose6d_deg"] = pose6.copy()
            shared_task_cfg["target_world_xyz"] = pose6[:3].copy()
            shared_task_cfg["target_world_yaw_deg"] = float(pose6[5])

    ai_raw = _load_yaml_dict(ai_cfg_path, "active_inference")
    _check_unknown_keys("active_inference_root", ai_raw, ACTIVE_TOP_LEVEL_KEYS)
    _check_missing_keys("active_inference_root", ai_raw, ACTIVE_TOP_LEVEL_KEYS)
    ai_inference_cfg = _coerce_active_section(
        "active_inference.inference",
        _ensure_dict("active_inference.inference", ai_raw["inference"]),
        REQUIRED_AI_INFERENCE_KEYS,
        OPTIONAL_AI_INFERENCE_KEYS,
    )
    ai_action_cfg = _coerce_active_section(
        "active_inference.action_selection",
        _ensure_dict("active_inference.action_selection", ai_raw["action_selection"]),
        REQUIRED_AI_ACTION_KEYS,
    )
    overlap = sorted(set(ai_inference_cfg.keys()) & set(ai_action_cfg.keys()))
    if overlap:
        raise ValueError(
            "Overlapping keys between active_inference.inference and "
            f"active_inference.action_selection are not allowed: {overlap}"
        )
    active_inference_cfg: Dict[str, Any] = {}
    active_inference_cfg.update(ai_inference_cfg)
    active_inference_cfg.update(ai_action_cfg)

    return {
        "common_path": str(common_cfg_path),
        "active_inference_path": str(ai_cfg_path),
        "found": True,
        "strict": True,
        "run_cfg": run_cfg_out,
        "controller_cfg": controller_cfg_out,
        "shared_task_cfg": shared_task_cfg,
        "active_inference_cfg": active_inference_cfg,
    }
