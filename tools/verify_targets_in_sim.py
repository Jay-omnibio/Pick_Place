import argparse
import math
from pathlib import Path
from typing import Tuple
import sys

import mujoco
import numpy as np

# Allow running as: python tools/verify_targets_in_sim.py
# by ensuring project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.runtime_loader import (
    DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH,
    DEFAULT_COMMON_CONFIG_PATH,
    DEFAULT_FSM_CONFIG_PATH,
    load_runtime_sections,
)
from env.simulator import MujocoSimulator


def _vec3_str(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=float).reshape(3)
    return f"({a[0]:.5f}, {a[1]:.5f}, {a[2]:.5f})"


def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _world_to_object_rel(rel_world: np.ndarray, obj_yaw: float) -> np.ndarray:
    r = np.asarray(rel_world, dtype=float).copy()
    c = float(np.cos(float(obj_yaw)))
    s = float(np.sin(float(obj_yaw)))
    x_w, y_w = float(r[0]), float(r[1])
    # local = Rz(-yaw) * world
    r[0] = c * x_w + s * y_w
    r[1] = -s * x_w + c * y_w
    return r


def _rel_local_to_world(rel_local: np.ndarray, obj_yaw: float) -> np.ndarray:
    r = np.asarray(rel_local, dtype=float).copy()
    c = float(np.cos(float(obj_yaw)))
    s = float(np.sin(float(obj_yaw)))
    x_l, y_l = float(r[0]), float(r[1])
    # world = Rz(yaw) * local
    r[0] = c * x_l - s * y_l
    r[1] = s * x_l + c * y_l
    return r


def _target_from_obj_rel(
    obj_world: np.ndarray,
    rel_cfg: np.ndarray,
    obj_yaw: float,
    use_object_local_xy_errors: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rel_cfg = np.asarray(rel_cfg, dtype=float)
    if bool(use_object_local_xy_errors):
        rel_world = _rel_local_to_world(rel_cfg, obj_yaw)
    else:
        rel_world = rel_cfg.copy()
    ee_target = np.asarray(obj_world, dtype=float) - rel_world
    return ee_target, rel_world


def _relative_error(
    obj_world: np.ndarray,
    ee_world: np.ndarray,
    rel_cfg: np.ndarray,
    obj_yaw: float,
    use_object_local_xy_errors: bool,
) -> np.ndarray:
    o_obj_world = np.asarray(obj_world, dtype=float) - np.asarray(ee_world, dtype=float)
    if bool(use_object_local_xy_errors):
        o_obj_eval = _world_to_object_rel(o_obj_world, obj_yaw)
    else:
        o_obj_eval = o_obj_world
    return o_obj_eval - np.asarray(rel_cfg, dtype=float)


def _set_ee_exact_mocap(sim: MujocoSimulator, target_world: np.ndarray) -> np.ndarray:
    target = np.asarray(target_world, dtype=float).reshape(3)
    sim.data.mocap_pos[sim.mocap_id] = np.clip(target, sim.workspace_min, sim.workspace_max)
    mujoco.mj_forward(sim.model, sim.data)
    return sim.get_ee_position()


def _print_descend_xy_region(
    obj_world: np.ndarray,
    grasp_rel_cfg: np.ndarray,
    obj_yaw: float,
    use_object_local_xy_errors: bool,
    x_thr: float,
    y_thr: float,
) -> None:
    g = np.asarray(grasp_rel_cfg, dtype=float)
    print("Descend XY threshold region (error frame):")
    print(f"  x in [{g[0]-x_thr:.5f}, {g[0]+x_thr:.5f}]")
    print(f"  y in [{g[1]-y_thr:.5f}, {g[1]+y_thr:.5f}]")

    corners_local = [
        np.array([g[0] - x_thr, g[1] - y_thr, g[2]], dtype=float),
        np.array([g[0] + x_thr, g[1] - y_thr, g[2]], dtype=float),
        np.array([g[0] + x_thr, g[1] + y_thr, g[2]], dtype=float),
        np.array([g[0] - x_thr, g[1] + y_thr, g[2]], dtype=float),
    ]
    corners_world = []
    for c in corners_local:
        rel_w = _rel_local_to_world(c, obj_yaw) if use_object_local_xy_errors else c
        corners_world.append(np.asarray(obj_world, dtype=float) - rel_w)

    print("Descend XY threshold box corners (EE world at grasp Z):")
    for i, c in enumerate(corners_world, start=1):
        print(f"  corner{i}: {_vec3_str(c)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ReachAbove/Descend targets in simulator without running the control policy."
    )
    parser.add_argument("--common-config", default=DEFAULT_COMMON_CONFIG_PATH)
    parser.add_argument("--fsm-config", default=DEFAULT_FSM_CONFIG_PATH)
    parser.add_argument("--active-inference-config", default=DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH)
    parser.add_argument("--model-xml", default="assets/pick_and_place.xml")
    parser.add_argument("--settle-steps", type=int, default=600, help="Physics settle steps to get post-fall object pose.")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer.")
    parser.add_argument(
        "--pause-between",
        action="store_true",
        help="Pause between stages for visual inspection (press Enter).",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=120,
        help="Viewer hold steps after each target snap (render mode).",
    )
    parser.add_argument("--pregrasp-rel", type=float, nargs=3, default=None, help="Override pregrasp_obj_rel.")
    parser.add_argument("--grasp-rel", type=float, nargs=3, default=None, help="Override grasp_obj_rel.")
    args = parser.parse_args()

    runtime_cfg = load_runtime_sections(
        common_path=args.common_config,
        fsm_path=args.fsm_config,
        active_inference_path=args.active_inference_config,
    )
    task_cfg = runtime_cfg["task_cfg"]

    pregrasp_rel = np.asarray(args.pregrasp_rel, dtype=float) if args.pregrasp_rel is not None else np.asarray(task_cfg.pregrasp_obj_rel, dtype=float)
    grasp_rel = np.asarray(args.grasp_rel, dtype=float) if args.grasp_rel is not None else np.asarray(task_cfg.grasp_obj_rel, dtype=float)
    use_local = bool(task_cfg.use_object_local_xy_errors)

    sim = MujocoSimulator(model_path=str(Path(args.model_xml)), render=bool(args.render))
    try:
        for _ in range(max(0, int(args.settle_steps))):
            sim.step()

        st = sim.get_state()
        obj_world = np.asarray(st["obj_pos"], dtype=float)
        obj_quat = np.asarray(st["obj_quat_wxyz"], dtype=float)
        obj_yaw = _quat_wxyz_to_yaw(obj_quat)

        reach_target, reach_rel_world = _target_from_obj_rel(obj_world, pregrasp_rel, obj_yaw, use_local)
        descend_target, descend_rel_world = _target_from_obj_rel(obj_world, grasp_rel, obj_yaw, use_local)

        print(
            "Configs: "
            f"common={args.common_config} "
            f"fsm={args.fsm_config} "
            f"active_inference={args.active_inference_config}"
        )
        print(f"Model XML: {args.model_xml}")
        print(f"use_object_local_xy_errors: {int(use_local)}")
        print("-" * 90)
        print(f"Object pose after settle({int(args.settle_steps)}):")
        print(f"  obj_world: {_vec3_str(obj_world)}")
        print(f"  obj_yaw: {np.rad2deg(obj_yaw):.3f} deg")
        print("-" * 90)
        print("Computed targets:")
        print(f"  ReachAbove rel_cfg:  {_vec3_str(pregrasp_rel)}")
        print(f"  ReachAbove rel_world:{_vec3_str(reach_rel_world)}")
        print(f"  ReachAbove target:   {_vec3_str(reach_target)}")
        print(f"  Descend rel_cfg:     {_vec3_str(grasp_rel)}")
        print(f"  Descend rel_world:   {_vec3_str(descend_rel_world)}")
        print(f"  Descend target:      {_vec3_str(descend_target)}")
        print("-" * 90)
        _print_descend_xy_region(
            obj_world=obj_world,
            grasp_rel_cfg=grasp_rel,
            obj_yaw=obj_yaw,
            use_object_local_xy_errors=use_local,
            x_thr=float(task_cfg.descend_x_threshold),
            y_thr=float(task_cfg.descend_y_threshold),
        )
        print("-" * 90)

        if args.pause_between:
            input("Press Enter to snap EE to ReachAbove target...")
        ee_at_reach = _set_ee_exact_mocap(sim, reach_target)
        err_reach = _relative_error(obj_world, ee_at_reach, pregrasp_rel, obj_yaw, use_local)
        print("EE snapped to ReachAbove target:")
        print(f"  ee_world: {_vec3_str(ee_at_reach)}")
        print(f"  relative_error_vs_pregrasp: {_vec3_str(err_reach)}")

        if bool(args.render):
            for _ in range(max(0, int(args.hold_steps))):
                sim.step()

        if args.pause_between:
            input("Press Enter to snap EE to Descend target...")
        ee_at_descend = _set_ee_exact_mocap(sim, descend_target)
        err_descend = _relative_error(obj_world, ee_at_descend, grasp_rel, obj_yaw, use_local)
        print("EE snapped to Descend target:")
        print(f"  ee_world: {_vec3_str(ee_at_descend)}")
        print(f"  relative_error_vs_grasp: {_vec3_str(err_descend)}")

        if bool(args.render):
            for _ in range(max(0, int(args.hold_steps))):
                sim.step()
    finally:
        if getattr(sim, "viewer", None) is not None and hasattr(sim.viewer, "close"):
            sim.viewer.close()


if __name__ == "__main__":
    main()
