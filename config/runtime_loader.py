from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import numpy as np
import yaml

from policies.scripted_pick_place import PolicyConfig
from tasks.pick_place_fsm import TaskConfig

T = TypeVar("T")

DEFAULT_COMMON_CONFIG_PATH = "config/common_robot.yaml"
DEFAULT_FSM_CONFIG_PATH = "config/fsm_config.yaml"
DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH = "config/active_inference_config.yaml"

COMMON_TOP_LEVEL_KEYS = {"run", "controller"}
FSM_TOP_LEVEL_KEYS = {"task", "policy"}
ACTIVE_TOP_LEVEL_KEYS = {"inference", "action_selection"}

REQUIRED_RUN_KEYS = {"control_mode", "log_every_steps"}
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
    "vfe_enabled",
    "vfe_epistemic_weight",
    "vfe_phase_change_enabled",
    "vfe_phase_change_max",
    "vfe_recover_enabled",
    "vfe_recover_threshold",
    "vfe_recover_steps",
    "release_contact_warn_steps",
    "risk_detection_enabled",
    "singularity_dq_ratio_threshold",
    "singularity_no_progress_steps",
    "unintended_contact_warn_steps",
    "alpha_target",
    "preplace_target_rel",
    "place_target_rel",
    "preplace_threshold",
    "place_threshold",
    "preplace_xy_threshold",
    "preplace_z_threshold",
    "place_xy_threshold",
    "place_z_threshold",
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
}

AI_VECTOR_KEYS = {
    "reach_obj_rel",
    "align_obj_rel",
    "descend_obj_rel",
    "preplace_target_rel",
    "place_target_rel",
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
    "reach_stall_steps",
    "reach_yaw_align_steps",
    "vfe_recover_steps",
    "release_contact_warn_steps",
    "singularity_no_progress_steps",
    "unintended_contact_warn_steps",
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


def _coerce_from_type(value: Any, field_type: Any) -> Any:
    field_type_name = str(field_type)
    if field_type is np.ndarray or field_type_name in {"np.ndarray", "numpy.ndarray"}:
        return np.asarray(value, dtype=float)
    if field_type is bool or field_type_name == "bool":
        return bool(value)
    if field_type is int or field_type_name == "int":
        return int(value)
    if field_type is float or field_type_name == "float":
        return float(value)
    return value


def _build_dataclass_from_section(cls: Type[T], section_name: str, data: Dict[str, Any]) -> T:
    field_defs = {f.name: f for f in fields(cls)}
    allowed = set(field_defs.keys())
    _check_unknown_keys(section_name, data, allowed)
    _check_missing_keys(section_name, data, allowed)

    kwargs: Dict[str, Any] = {}
    for name, f in field_defs.items():
        kwargs[name] = _coerce_from_type(data[name], f.type)
    return cls(**kwargs)  # type: ignore[arg-type]


def _coerce_active_section(
    section_name: str,
    data: Dict[str, Any],
    required_keys: set[str],
) -> Dict[str, Any]:
    _check_unknown_keys(section_name, data, required_keys)
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
    return out


def load_runtime_sections(
    common_path: str = DEFAULT_COMMON_CONFIG_PATH,
    fsm_path: str = DEFAULT_FSM_CONFIG_PATH,
    active_inference_path: str = DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH,
) -> Dict[str, Any]:
    common_cfg_path = Path(common_path)
    fsm_cfg_path = Path(fsm_path)
    ai_cfg_path = Path(active_inference_path)

    common_raw = _load_yaml_dict(common_cfg_path, "common")
    _check_unknown_keys("common_root", common_raw, COMMON_TOP_LEVEL_KEYS)
    _check_missing_keys("common_root", common_raw, COMMON_TOP_LEVEL_KEYS)

    run_cfg = _ensure_dict("run", common_raw["run"])
    _check_unknown_keys("run", run_cfg, REQUIRED_RUN_KEYS)
    _check_missing_keys("run", run_cfg, REQUIRED_RUN_KEYS)
    run_cfg_out = {
        "control_mode": str(run_cfg["control_mode"]),
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

    fsm_raw = _load_yaml_dict(fsm_cfg_path, "fsm")
    _check_unknown_keys("fsm_root", fsm_raw, FSM_TOP_LEVEL_KEYS)
    _check_missing_keys("fsm_root", fsm_raw, FSM_TOP_LEVEL_KEYS)
    task_cfg = _build_dataclass_from_section(TaskConfig, "task", _ensure_dict("task", fsm_raw["task"]))
    policy_cfg = _build_dataclass_from_section(
        PolicyConfig, "policy", _ensure_dict("policy", fsm_raw["policy"])
    )

    ai_raw = _load_yaml_dict(ai_cfg_path, "active_inference")
    _check_unknown_keys("active_inference_root", ai_raw, ACTIVE_TOP_LEVEL_KEYS)
    _check_missing_keys("active_inference_root", ai_raw, ACTIVE_TOP_LEVEL_KEYS)
    ai_inference_cfg = _coerce_active_section(
        "active_inference.inference",
        _ensure_dict("active_inference.inference", ai_raw["inference"]),
        REQUIRED_AI_INFERENCE_KEYS,
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
        "fsm_path": str(fsm_cfg_path),
        "active_inference_path": str(ai_cfg_path),
        "found": True,
        "strict": True,
        "run_cfg": run_cfg_out,
        "controller_cfg": controller_cfg_out,
        "task_cfg": task_cfg,
        "policy_cfg": policy_cfg,
        "active_inference_cfg": active_inference_cfg,
    }
