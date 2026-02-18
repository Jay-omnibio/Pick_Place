import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _parse_float_list(text: str) -> List[float]:
    clean = str(text).replace(" ", "")
    if not clean:
        return []
    return [float(v) for v in clean.split(",") if v != ""]


def _vec3_str(v: Sequence[float]) -> str:
    a = np.asarray(v, dtype=float)
    return f"({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f})"


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _deg_from_rad_wrapped(rad: float) -> float:
    return float(np.rad2deg(_wrap_pi(float(rad))))


def _rel_local_to_world(rel_local: np.ndarray, obj_yaw_rad: float, use_object_local_xy_errors: bool) -> np.ndarray:
    rel = np.asarray(rel_local, dtype=float).copy()
    if not bool(use_object_local_xy_errors):
        return rel
    c = float(np.cos(float(obj_yaw_rad)))
    s = float(np.sin(float(obj_yaw_rad)))
    x_l, y_l = float(rel[0]), float(rel[1])
    # world = Rz(yaw) * local
    rel[0] = c * x_l - s * y_l
    rel[1] = s * x_l + c * y_l
    return rel


def _ee_target_from_object(
    obj_world: np.ndarray,
    rel_cfg: np.ndarray,
    obj_yaw_rad: float,
    use_object_local_xy_errors: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - ee_target_world
      - rel_world_used
    """
    rel_world = _rel_local_to_world(rel_cfg, obj_yaw_rad, use_object_local_xy_errors)
    ee_target = np.asarray(obj_world, dtype=float) - rel_world
    return ee_target, rel_world


def _load_fsm_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"FSM config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"FSM config root must be mapping/object: {path}")

    task = loaded.get("task", {})
    policy = loaded.get("policy", {})
    if not isinstance(task, dict):
        raise ValueError("Invalid fsm config: task must be mapping/object.")
    if not isinstance(policy, dict):
        raise ValueError("Invalid fsm config: policy must be mapping/object.")

    def _need_vec3(section: Dict[str, object], key: str) -> np.ndarray:
        if key not in section:
            raise ValueError(f"Missing required key in fsm config task section: {key}")
        v = np.asarray(section[key], dtype=float)
        if v.shape != (3,):
            raise ValueError(f"Key '{key}' must be a 3-element vector, got shape {v.shape}.")
        return v

    pregrasp_obj_rel = _need_vec3(task, "pregrasp_obj_rel")
    grasp_obj_rel = _need_vec3(task, "grasp_obj_rel")
    use_local_xy = bool(task.get("use_object_local_xy_errors", True))
    descend_x_threshold = float(task.get("descend_x_threshold", 0.0))
    descend_y_threshold = float(task.get("descend_y_threshold", 0.0))

    enable_yaw_align = bool(policy.get("enable_object_yaw_align", False))
    object_yaw_offset_deg = float(policy.get("object_yaw_offset_deg", 90.0))

    return {
        "pregrasp_obj_rel": pregrasp_obj_rel,
        "grasp_obj_rel": grasp_obj_rel,
        "use_object_local_xy_errors": use_local_xy,
        "descend_x_threshold": descend_x_threshold,
        "descend_y_threshold": descend_y_threshold,
        "enable_object_yaw_align": enable_yaw_align,
        "object_yaw_offset_deg": object_yaw_offset_deg,
    }


def _resolve_bool(mode: str, default_value: bool) -> bool:
    m = str(mode).strip().lower()
    if m == "auto":
        return bool(default_value)
    if m in {"true", "1", "yes", "y"}:
        return True
    if m in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool mode: {mode}")


def _load_object_half_size_from_xml(model_xml: Path) -> np.ndarray:
    if not model_xml.exists():
        raise FileNotFoundError(f"Model XML not found: {model_xml}")
    root = ET.fromstring(model_xml.read_text(encoding="utf-8"))
    for geom in root.iter("geom"):
        if geom.attrib.get("name", "") == "obj_geom":
            size_txt = geom.attrib.get("size", "")
            vals = [float(v) for v in str(size_txt).split() if v.strip() != ""]
            if len(vals) >= 3:
                return np.array(vals[:3], dtype=float)
            raise ValueError(f"obj_geom size must have 3 values, got: {size_txt}")
    raise ValueError("Could not find geom name='obj_geom' in model XML.")


def _build_object_poses(args: argparse.Namespace) -> List[Tuple[float, float, float, float]]:
    if int(args.random_count) > 0:
        rng = np.random.default_rng(int(args.seed))
        out = []
        for _ in range(int(args.random_count)):
            x = float(rng.uniform(float(args.x_range[0]), float(args.x_range[1])))
            y = float(rng.uniform(float(args.y_range[0]), float(args.y_range[1])))
            z = float(rng.uniform(float(args.z_range[0]), float(args.z_range[1])))
            yaw = float(rng.uniform(float(args.yaw_range[0]), float(args.yaw_range[1])))
            out.append((x, y, z, yaw))
        return out

    x_vals = _parse_float_list(args.obj_x_list) if args.obj_x_list else [float(args.obj_x)]
    y_vals = _parse_float_list(args.obj_y_list) if args.obj_y_list else [float(args.obj_y)]
    z_vals = _parse_float_list(args.obj_z_list) if args.obj_z_list else [float(args.obj_z)]
    yaw_vals = _parse_float_list(args.obj_yaw_list) if args.obj_yaw_list else [float(args.obj_yaw_deg)]

    mode = str(args.pose_mode).strip().lower()
    if mode == "grid":
        return [
            (float(x), float(y), float(z), float(yaw))
            for x, y, z, yaw in itertools.product(x_vals, y_vals, z_vals, yaw_vals)
        ]

    if mode == "zip":
        lengths = [len(x_vals), len(y_vals), len(z_vals), len(yaw_vals)]
        n = max(lengths)
        bad = [ln for ln in lengths if ln not in {1, n}]
        if bad:
            raise ValueError(
                "pose_mode=zip requires each list length to be 1 or equal to max length; "
                f"got lengths x/y/z/yaw={lengths}"
            )

        def _at(vals: List[float], i: int) -> float:
            if len(vals) == 1:
                return float(vals[0])
            return float(vals[i])

        return [(_at(x_vals, i), _at(y_vals, i), _at(z_vals, i), _at(yaw_vals, i)) for i in range(n)]

    raise ValueError(f"Unsupported pose_mode: {args.pose_mode}")


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _draw_oriented_box(ax, center: np.ndarray, yaw_rad: float, half_size: np.ndarray, color: str, alpha: float) -> None:
    c = np.asarray(center, dtype=float)
    h = np.asarray(half_size, dtype=float)

    # 8 corners in local frame.
    local = np.array(
        [
            [-h[0], -h[1], -h[2]],
            [+h[0], -h[1], -h[2]],
            [+h[0], +h[1], -h[2]],
            [-h[0], +h[1], -h[2]],
            [-h[0], -h[1], +h[2]],
            [+h[0], -h[1], +h[2]],
            [+h[0], +h[1], +h[2]],
            [-h[0], +h[1], +h[2]],
        ],
        dtype=float,
    )
    c_y = float(np.cos(float(yaw_rad)))
    s_y = float(np.sin(float(yaw_rad)))
    rot = np.array(
        [
            [c_y, -s_y, 0.0],
            [s_y, c_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    world = (local @ rot.T) + c.reshape(1, 3)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    for i, j in edges:
        ax.plot(
            [world[i, 0], world[j, 0]],
            [world[i, 1], world[j, 1]],
            [world[i, 2], world[j, 2]],
            color=color,
            alpha=alpha,
            linewidth=1.1,
        )


def _threshold_ee_points_world(
    obj_world: np.ndarray,
    obj_yaw_rad: float,
    grasp_rel_cfg: np.ndarray,
    use_object_local_xy_errors: bool,
    x_thr: float,
    y_thr: float,
) -> Dict[str, np.ndarray]:
    """
    Build world-space EE points for threshold guides around grasp target.
    Returns x/y axis line endpoints and rectangle corners in the XY plane.
    """
    g = np.asarray(grasp_rel_cfg, dtype=float)
    # X threshold lines (in error frame)
    rel_x_lo = np.array([g[0] - x_thr, g[1], g[2]], dtype=float)
    rel_x_hi = np.array([g[0] + x_thr, g[1], g[2]], dtype=float)
    # Y threshold lines (in error frame)
    rel_y_lo = np.array([g[0], g[1] - y_thr, g[2]], dtype=float)
    rel_y_hi = np.array([g[0], g[1] + y_thr, g[2]], dtype=float)

    # Rectangle corners (x/y box)
    c1 = np.array([g[0] - x_thr, g[1] - y_thr, g[2]], dtype=float)
    c2 = np.array([g[0] + x_thr, g[1] - y_thr, g[2]], dtype=float)
    c3 = np.array([g[0] + x_thr, g[1] + y_thr, g[2]], dtype=float)
    c4 = np.array([g[0] - x_thr, g[1] + y_thr, g[2]], dtype=float)

    def _to_ee_world(rel_cfg: np.ndarray) -> np.ndarray:
        rel_world = _rel_local_to_world(rel_cfg, obj_yaw_rad, use_object_local_xy_errors)
        return np.asarray(obj_world, dtype=float) - rel_world

    return {
        "x_lo": _to_ee_world(rel_x_lo),
        "x_hi": _to_ee_world(rel_x_hi),
        "y_lo": _to_ee_world(rel_y_lo),
        "y_hi": _to_ee_world(rel_y_hi),
        "c1": _to_ee_world(c1),
        "c2": _to_ee_world(c2),
        "c3": _to_ee_world(c3),
        "c4": _to_ee_world(c4),
    }


def _save_3d_plot(
    path: Path,
    rows: List[Dict[str, float]],
    use_object_local_xy_errors: bool,
    x_thr: float,
    y_thr: float,
    threshold_guides_mode: str,
    object_half_size: np.ndarray,
    object_box_mode: str,
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "3D plot requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return

    obj_x = np.array([float(r["obj_x"]) for r in rows], dtype=float)
    obj_y = np.array([float(r["obj_y"]) for r in rows], dtype=float)
    obj_z = np.array([float(r["obj_z"]) for r in rows], dtype=float)

    pre_x = np.array([float(r["pregrasp_ee_x"]) for r in rows], dtype=float)
    pre_y = np.array([float(r["pregrasp_ee_y"]) for r in rows], dtype=float)
    pre_z = np.array([float(r["pregrasp_ee_z"]) for r in rows], dtype=float)

    grasp_x = np.array([float(r["grasp_ee_x"]) for r in rows], dtype=float)
    grasp_y = np.array([float(r["grasp_ee_y"]) for r in rows], dtype=float)
    grasp_z = np.array([float(r["grasp_ee_z"]) for r in rows], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(obj_x, obj_y, obj_z, c="black", s=32, label="object center")
    ax.scatter(pre_x, pre_y, pre_z, c="tab:blue", s=24, label="reach_above target")
    ax.scatter(grasp_x, grasp_y, grasp_z, c="tab:red", s=24, label="descend/grasp target")

    # Draw object box (marker-only scene, no simulator step/run).
    box_mode = str(object_box_mode).strip().lower()
    if box_mode in {"first", "all"}:
        indices = [0] if box_mode == "first" else list(range(len(rows)))
        for idx in indices:
            row = rows[idx]
            center = np.array([row["obj_x"], row["obj_y"], row["obj_z"]], dtype=float)
            yaw = float(np.deg2rad(float(row["obj_yaw_deg"])))
            _draw_oriented_box(
                ax=ax,
                center=center,
                yaw_rad=yaw,
                half_size=np.asarray(object_half_size, dtype=float),
                color="tab:olive",
                alpha=0.7,
            )

    # Geometry guide lines:
    #   - object -> pregrasp
    #   - pregrasp -> grasp (phase trajectory guidance)
    #   - object -> grasp
    for i in range(len(rows)):
        ax.plot(
            [obj_x[i], pre_x[i]],
            [obj_y[i], pre_y[i]],
            [obj_z[i], pre_z[i]],
            color="tab:blue",
            alpha=0.35,
        )
        ax.plot(
            [pre_x[i], grasp_x[i]],
            [pre_y[i], grasp_y[i]],
            [pre_z[i], grasp_z[i]],
            color="tab:purple",
            alpha=0.45,
        )
        ax.plot(
            [obj_x[i], grasp_x[i]],
            [obj_y[i], grasp_y[i]],
            [obj_z[i], grasp_z[i]],
            color="tab:red",
            alpha=0.35,
        )

    # Threshold guide lines around grasp target (X/Y in descend error frame).
    mode = str(threshold_guides_mode).strip().lower()
    if x_thr > 0.0 and y_thr > 0.0 and mode in {"first", "all"}:
        indices = [0] if mode == "first" else list(range(len(rows)))
        for idx in indices:
            row = rows[idx]
            obj = np.array([row["obj_x"], row["obj_y"], row["obj_z"]], dtype=float)
            yaw = float(np.deg2rad(float(row["obj_yaw_deg"])))
            grasp_rel_cfg = np.array(
                [row["grasp_rel_x_cfg"], row["grasp_rel_y_cfg"], row["grasp_rel_z_cfg"]],
                dtype=float,
            )
            pts = _threshold_ee_points_world(
                obj_world=obj,
                obj_yaw_rad=yaw,
                grasp_rel_cfg=grasp_rel_cfg,
                use_object_local_xy_errors=use_object_local_xy_errors,
                x_thr=float(x_thr),
                y_thr=float(y_thr),
            )

            # X and Y axis threshold lines
            ax.plot(
                [pts["x_lo"][0], pts["x_hi"][0]],
                [pts["x_lo"][1], pts["x_hi"][1]],
                [pts["x_lo"][2], pts["x_hi"][2]],
                color="tab:green",
                linewidth=1.8,
                alpha=0.9,
            )
            ax.plot(
                [pts["y_lo"][0], pts["y_hi"][0]],
                [pts["y_lo"][1], pts["y_hi"][1]],
                [pts["y_lo"][2], pts["y_hi"][2]],
                color="tab:orange",
                linewidth=1.8,
                alpha=0.9,
            )

            # Rectangle boundary (XY threshold box)
            corners = [pts["c1"], pts["c2"], pts["c3"], pts["c4"], pts["c1"]]
            ax.plot(
                [p[0] for p in corners],
                [p[1] for p in corners],
                [p[2] for p in corners],
                color="tab:gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Marker-Only 3D Scene: Object + ReachAbove/Descend Targets")
    ax.plot([], [], [], color="tab:purple", label="phase path pregrasp->grasp")
    if box_mode in {"first", "all"}:
        hs = np.asarray(object_half_size, dtype=float)
        ax.plot([], [], [], color="tab:olive", label=f"object box half-size=({hs[0]:.3f},{hs[1]:.3f},{hs[2]:.3f})")
    if x_thr > 0.0 and y_thr > 0.0 and mode in {"first", "all"}:
        ax.plot([], [], [], color="tab:green", label=f"x-threshold line (+/-{x_thr:.3f})")
        ax.plot([], [], [], color="tab:orange", label=f"y-threshold line (+/-{y_thr:.3f})")
        ax.plot([], [], [], color="tab:gray", linestyle="--", label="xy threshold box")
    ax.legend(loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline target-geometry helper (no simulator): object pose -> pregrasp/grasp EE targets."
    )
    parser.add_argument("--fsm-config", default="config/fsm_config.yaml", help="Path to fsm config YAML.")

    # Single pose defaults.
    parser.add_argument("--obj-x", type=float, default=0.40, help="Single object X (m).")
    parser.add_argument("--obj-y", type=float, default=0.00, help="Single object Y (m).")
    parser.add_argument("--obj-z", type=float, default=0.20, help="Single object Z (m).")
    parser.add_argument("--obj-yaw-deg", type=float, default=0.0, help="Single object yaw in degrees.")

    # Multi-pose options.
    parser.add_argument("--obj-x-list", default="", help="Comma-separated X list, e.g. 0.4,0.5,0.6")
    parser.add_argument("--obj-y-list", default="", help="Comma-separated Y list, e.g. -0.1,0,0.1")
    parser.add_argument("--obj-z-list", default="", help="Comma-separated Z list.")
    parser.add_argument("--obj-yaw-list", default="", help="Comma-separated yaw-deg list, e.g. -45,0,45")
    parser.add_argument(
        "--pose-mode",
        default="grid",
        choices=["grid", "zip"],
        help="How to combine x/y/z/yaw lists: grid=cartesian product, zip=pair by index.",
    )

    # Random pose generation.
    parser.add_argument("--random-count", type=int, default=0, help="If >0, sample this many random object poses.")
    parser.add_argument("--x-range", type=float, nargs=2, default=[0.35, 0.60], help="Random X range.")
    parser.add_argument("--y-range", type=float, nargs=2, default=[-0.15, 0.15], help="Random Y range.")
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.20, 0.20], help="Random Z range.")
    parser.add_argument("--yaw-range", type=float, nargs=2, default=[-90.0, 90.0], help="Random yaw-deg range.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for random poses.")

    # Overrides.
    parser.add_argument("--pregrasp-rel", type=float, nargs=3, default=None, help="Override pregrasp_obj_rel.")
    parser.add_argument("--grasp-rel", type=float, nargs=3, default=None, help="Override grasp_obj_rel.")
    parser.add_argument(
        "--use-object-local-xy-errors",
        default="auto",
        choices=["auto", "true", "false"],
        help="Use object-local XY for rel vectors (auto uses fsm config value).",
    )
    parser.add_argument(
        "--enable-yaw-align",
        default="auto",
        choices=["auto", "true", "false"],
        help="Compute gripper yaw setpoint from object yaw (auto uses fsm config value).",
    )
    parser.add_argument(
        "--object-yaw-offset-deg",
        type=float,
        default=None,
        help="Override policy object_yaw_offset_deg used for gripper yaw setpoint.",
    )

    parser.add_argument("--print-limit", type=int, default=200, help="Max rows to print to console.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path.")
    parser.add_argument(
        "--model-xml",
        default="assets/pick_and_place.xml",
        help="Model XML used to read default obj_geom size when --object-half-size is not provided.",
    )
    parser.add_argument(
        "--object-half-size",
        type=float,
        nargs=3,
        default=None,
        help="Override object half-size (x y z). If omitted, read from obj_geom size in model XML.",
    )
    parser.add_argument(
        "--plot-object-box",
        default="first",
        choices=["none", "first", "all"],
        help="Draw object box for none / first pose / all poses.",
    )
    parser.add_argument(
        "--descend-x-threshold",
        type=float,
        default=None,
        help="Override x-threshold guide for 3D plot (default from fsm task.descend_x_threshold).",
    )
    parser.add_argument(
        "--descend-y-threshold",
        type=float,
        default=None,
        help="Override y-threshold guide for 3D plot (default from fsm task.descend_y_threshold).",
    )
    parser.add_argument(
        "--plot-threshold-guides",
        default="first",
        choices=["none", "first", "all"],
        help="Draw threshold guide lines for none / first pose / all poses.",
    )
    parser.add_argument(
        "--plot-3d",
        nargs="?",
        const="logs/target_locations_3d.png",
        default="",
        help="Optional output PNG for 3D geometry plot. If provided without a path, defaults to logs/target_locations_3d.png.",
    )
    args = parser.parse_args()

    fsm_path = Path(args.fsm_config)
    tuning = _load_fsm_config(fsm_path)

    pregrasp_rel = (
        np.asarray(args.pregrasp_rel, dtype=float)
        if args.pregrasp_rel is not None
        else np.asarray(tuning["pregrasp_obj_rel"], dtype=float)
    )
    grasp_rel = (
        np.asarray(args.grasp_rel, dtype=float)
        if args.grasp_rel is not None
        else np.asarray(tuning["grasp_obj_rel"], dtype=float)
    )

    use_local_xy = _resolve_bool(args.use_object_local_xy_errors, bool(tuning["use_object_local_xy_errors"]))
    yaw_align_enabled = _resolve_bool(args.enable_yaw_align, bool(tuning["enable_object_yaw_align"]))
    yaw_offset_deg = (
        float(args.object_yaw_offset_deg)
        if args.object_yaw_offset_deg is not None
        else float(tuning["object_yaw_offset_deg"])
    )
    descend_x_threshold = (
        float(args.descend_x_threshold)
        if args.descend_x_threshold is not None
        else float(tuning["descend_x_threshold"])
    )
    descend_y_threshold = (
        float(args.descend_y_threshold)
        if args.descend_y_threshold is not None
        else float(tuning["descend_y_threshold"])
    )
    if args.object_half_size is not None:
        object_half_size = np.asarray(args.object_half_size, dtype=float)
    else:
        object_half_size = _load_object_half_size_from_xml(Path(args.model_xml))

    poses = _build_object_poses(args)
    if not poses:
        raise ValueError("No object poses generated. Check list/random arguments.")

    print(f"FSM config: {fsm_path}")
    print(f"Poses: {len(poses)}")
    print(f"use_object_local_xy_errors: {int(use_local_xy)}")
    print(f"pregrasp_obj_rel: {_vec3_str(pregrasp_rel)}")
    print(f"grasp_obj_rel: {_vec3_str(grasp_rel)}")
    print(f"descend_x_threshold: {descend_x_threshold:.4f}")
    print(f"descend_y_threshold: {descend_y_threshold:.4f}")
    print(f"object_half_size: {_vec3_str(object_half_size)}")
    print(f"yaw_align_enabled: {int(yaw_align_enabled)} (offset_deg={yaw_offset_deg:.2f})")
    print("-" * 100)

    rows: List[Dict[str, float]] = []
    limit = max(1, int(args.print_limit))

    for i, (obj_x, obj_y, obj_z, obj_yaw_deg) in enumerate(poses, start=1):
        obj_world = np.array([obj_x, obj_y, obj_z], dtype=float)
        obj_yaw_rad = float(np.deg2rad(float(obj_yaw_deg)))

        pregrasp_target, pregrasp_rel_world = _ee_target_from_object(
            obj_world=obj_world,
            rel_cfg=pregrasp_rel,
            obj_yaw_rad=obj_yaw_rad,
            use_object_local_xy_errors=use_local_xy,
        )
        grasp_target, grasp_rel_world = _ee_target_from_object(
            obj_world=obj_world,
            rel_cfg=grasp_rel,
            obj_yaw_rad=obj_yaw_rad,
            use_object_local_xy_errors=use_local_xy,
        )

        gripper_yaw_set_deg = float("nan")
        if yaw_align_enabled:
            gripper_yaw_set_deg = _deg_from_rad_wrapped(obj_yaw_rad + np.deg2rad(yaw_offset_deg))

        rows.append(
            {
                "index": float(i),
                "obj_x": float(obj_x),
                "obj_y": float(obj_y),
                "obj_z": float(obj_z),
                "obj_yaw_deg": float(obj_yaw_deg),
                "gripper_yaw_set_deg": float(gripper_yaw_set_deg),
                "pregrasp_rel_x_cfg": float(pregrasp_rel[0]),
                "pregrasp_rel_y_cfg": float(pregrasp_rel[1]),
                "pregrasp_rel_z_cfg": float(pregrasp_rel[2]),
                "pregrasp_rel_x_world": float(pregrasp_rel_world[0]),
                "pregrasp_rel_y_world": float(pregrasp_rel_world[1]),
                "pregrasp_rel_z_world": float(pregrasp_rel_world[2]),
                "pregrasp_ee_x": float(pregrasp_target[0]),
                "pregrasp_ee_y": float(pregrasp_target[1]),
                "pregrasp_ee_z": float(pregrasp_target[2]),
                "grasp_rel_x_cfg": float(grasp_rel[0]),
                "grasp_rel_y_cfg": float(grasp_rel[1]),
                "grasp_rel_z_cfg": float(grasp_rel[2]),
                "grasp_rel_x_world": float(grasp_rel_world[0]),
                "grasp_rel_y_world": float(grasp_rel_world[1]),
                "grasp_rel_z_world": float(grasp_rel_world[2]),
                "grasp_ee_x": float(grasp_target[0]),
                "grasp_ee_y": float(grasp_target[1]),
                "grasp_ee_z": float(grasp_target[2]),
            }
        )

        if i <= limit:
            gyaw_txt = "n/a" if not yaw_align_enabled else f"{gripper_yaw_set_deg:.1f}deg"
            print(
                f"[{i}] obj={_vec3_str(obj_world)} yaw={obj_yaw_deg:.1f}deg "
                f"gripper_yaw_set={gyaw_txt}"
            )
            print(
                f"     pregrasp_rel_world={_vec3_str(pregrasp_rel_world)} "
                f"pregrasp_target={_vec3_str(pregrasp_target)}"
            )
            print(
                f"     grasp_rel_world={_vec3_str(grasp_rel_world)} "
                f"grasp_target={_vec3_str(grasp_target)}"
            )

    if len(poses) > limit:
        print(f"... ({len(poses) - limit} more rows hidden by --print-limit={limit})")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        _write_csv(csv_path, rows)
        print("-" * 100)
        print(f"Saved CSV: {csv_path}")

    if args.plot_3d:
        plot_path = Path(str(args.plot_3d))
        try:
            _save_3d_plot(
                plot_path,
                rows,
                use_object_local_xy_errors=use_local_xy,
                x_thr=descend_x_threshold,
                y_thr=descend_y_threshold,
                threshold_guides_mode=str(args.plot_threshold_guides),
                object_half_size=object_half_size,
                object_box_mode=str(args.plot_object_box),
            )
            print("-" * 100)
            print(f"Saved 3D plot: {plot_path}")
        except RuntimeError as exc:
            print("-" * 100)
            print(f"3D plot skipped: {exc}")


if __name__ == "__main__":
    main()
