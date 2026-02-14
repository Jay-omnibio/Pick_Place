from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import numpy as np
import yaml

from policies.scripted_pick_place import PolicyConfig
from tasks.pick_place_fsm import TaskConfig

T = TypeVar("T")

ALLOWED_TOP_LEVEL_KEYS = {"run", "task", "policy", "controller"}
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


def _ensure_dict(section_name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object in tuning config.")
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


def load_tuning_sections(path: str = "config/tuning_config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Tuning config not found: {cfg_path}")

    loaded = yaml.safe_load(cfg_path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Tuning config root must be a mapping/object: {cfg_path}")
    raw: Dict[str, Any] = loaded

    _check_unknown_keys("root", raw, ALLOWED_TOP_LEVEL_KEYS)
    _check_missing_keys("root", raw, ALLOWED_TOP_LEVEL_KEYS)

    run_cfg = _ensure_dict("run", raw["run"])
    _check_unknown_keys("run", run_cfg, REQUIRED_RUN_KEYS)
    _check_missing_keys("run", run_cfg, REQUIRED_RUN_KEYS)
    run_cfg_out = {
        "control_mode": str(run_cfg["control_mode"]),
        "log_every_steps": int(run_cfg["log_every_steps"]),
    }

    task_cfg = _build_dataclass_from_section(TaskConfig, "task", _ensure_dict("task", raw["task"]))
    policy_cfg = _build_dataclass_from_section(PolicyConfig, "policy", _ensure_dict("policy", raw["policy"]))

    controller_cfg = _ensure_dict("controller", raw["controller"])
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

    return {
        "path": str(cfg_path),
        "found": True,
        "strict": True,
        "task_cfg": task_cfg,
        "policy_cfg": policy_cfg,
        "controller_cfg": controller_cfg_out,
        "run_cfg": run_cfg_out,
    }
