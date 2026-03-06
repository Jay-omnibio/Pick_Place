import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _latest_csv(logs_dir: Path) -> Path:
    csv_files = sorted(logs_dir.glob("run_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No run CSV files found in {logs_dir}")
    return csv_files[0]


def _load_csv(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _as_float_array(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def _as_int_array(rows, key):
    return np.array([int(float(r[key])) for r in rows], dtype=int)


def _as_str_array(rows, key):
    return np.array([str(r[key]) for r in rows], dtype=object)


def _has_key(rows, key):
    return bool(rows) and (key in rows[0])


def _as_optional_float_array(rows, key, default=np.nan):
    values = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            values.append(default)
        else:
            try:
                values.append(float(v))
            except Exception:
                values.append(default)
    return np.array(values, dtype=float)


def _as_optional_int_array(rows, key, default=0):
    values = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            values.append(default)
        else:
            values.append(int(float(v)))
    return np.array(values, dtype=int)


def _save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _first_contiguous_phase_segment(phase_arr: np.ndarray, phase_name: str):
    idx = np.where(phase_arr == phase_name)[0]
    if idx.size == 0:
        return None
    start = int(idx[0])
    end = start
    n = phase_arr.shape[0]
    while (end + 1) < n and phase_arr[end + 1] == phase_name:
        end += 1
    return start, end


def _first_approach_to_descend_segment(phase_arr: np.ndarray):
    """
    Return (start_idx, end_idx, descend_start_idx) for the first pick-side approach
    window: from first Reach/ReachAbove until end of first Descend block.
    If Descend is missing, returns None.
    """
    if phase_arr.size == 0:
        return None
    phase_s = np.asarray(phase_arr, dtype=object)
    start_candidates = np.where((phase_s == "ReachAbove") | (phase_s == "Reach"))[0]
    if start_candidates.size == 0:
        return None
    i0 = int(start_candidates[0])
    desc_rel = np.where(phase_s[i0:] == "Descend")[0]
    if desc_rel.size == 0:
        return None
    i_desc = i0 + int(desc_rel[0])
    i1 = i_desc
    n = phase_s.shape[0]
    while (i1 + 1) < n and phase_s[i1 + 1] == "Descend":
        i1 += 1
    return i0, i1, i_desc


def _contiguous_phase_segments(phase_arr: np.ndarray):
    if phase_arr.size == 0:
        return {}
    segments = {}
    start = 0
    cur = str(phase_arr[0])
    for i in range(1, phase_arr.shape[0]):
        nxt = str(phase_arr[i])
        if nxt != cur:
            segments.setdefault(cur, []).append((start, i - 1))
            start = i
            cur = nxt
    segments.setdefault(cur, []).append((start, phase_arr.shape[0] - 1))
    return segments


def _phase_metric_specs(phase_name: str):
    p = str(phase_name)
    if p in {"Reach", "ReachAbove"}:
        return [
            ("obs_reach_xy_error", "reach_xy_err"),
            ("obs_reach_z_error", "reach_z_err"),
        ]
    if p in {"Align", "Descend", "Close", "CloseHold", "LiftTest"}:
        return [
            ("obs_descend_xy_error", "descend_xy_err"),
            ("obs_descend_z_error", "descend_z_err"),
        ]
    if p == "MoveToPlaceAbove":
        return [
            ("obs_preplace_error", "preplace_norm_err"),
            ("obs_preplace_xy_error", "preplace_xy_err"),
        ]
    if p in {"DescendToPlace", "Open", "Retreat", "Done"}:
        return [
            ("obs_place_error", "place_norm_err"),
            ("obs_place_xy_error", "place_xy_err"),
        ]
    if p == "Transit":
        return [
            ("true_ee_z", "true_ee_z"),
            ("true_obj_z", "true_obj_z"),
        ]
    return [("reach_error", "reach_error")]


def main():
    parser = argparse.ArgumentParser(description="Plot Active Inference run metrics from CSV.")
    parser.add_argument("--csv", type=str, default="", help="Path to run CSV. If omitted, use latest in logs/")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory containing run_*.csv")
    parser.add_argument("--out-dir", type=str, default="", help="Output plot directory")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    csv_path = Path(args.csv) if args.csv else _latest_csv(logs_dir)
    rows = _load_csv(csv_path)

    run_name = csv_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else logs_dir / "plots" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    step = _as_int_array(rows, "step")
    phase = _as_str_array(rows, "phase")
    distance_to_object = _as_float_array(rows, "distance_to_object")
    reach_error = _as_float_array(rows, "reach_error")

    s_ee_x = _as_float_array(rows, "s_ee_x")
    s_ee_y = _as_float_array(rows, "s_ee_y")
    s_ee_z = _as_float_array(rows, "s_ee_z")

    s_obj_x = _as_float_array(rows, "s_obj_rel_x")
    s_obj_y = _as_float_array(rows, "s_obj_rel_y")
    s_obj_z = _as_float_array(rows, "s_obj_rel_z")

    a_x = _as_float_array(rows, "action_move_x")
    a_y = _as_float_array(rows, "action_move_y")
    a_z = _as_float_array(rows, "action_move_z")
    a_grip = _as_int_array(rows, "action_grip")

    t_ee_x = _as_float_array(rows, "true_ee_x")
    t_ee_y = _as_float_array(rows, "true_ee_y")
    t_ee_z = _as_float_array(rows, "true_ee_z")
    t_obj_x = _as_float_array(rows, "true_obj_x")
    t_obj_y = _as_float_array(rows, "true_obj_y")
    t_obj_z = _as_float_array(rows, "true_obj_z")

    obs_contact = _as_int_array(rows, "obs_contact")
    obs_grip = _as_float_array(rows, "obs_grip")
    escape_active = _as_optional_int_array(rows, "escape_active", default=0)
    loop_dt_ms = _as_optional_float_array(rows, "loop_dt_ms")
    loop_dt_ema_ms = _as_optional_float_array(rows, "loop_dt_ema_ms")
    loop_target_ms = _as_optional_float_array(rows, "loop_target_ms")
    obs_age_ms = _as_optional_float_array(rows, "obs_age_ms")
    obs_stale_warn = _as_optional_int_array(rows, "obs_stale_warn", default=0)
    loop_overrun_count = _as_optional_int_array(rows, "loop_overrun_count", default=0)

    ai_obs_conf = _as_optional_float_array(rows, "ai_obs_confidence")
    ai_vfe = _as_optional_float_array(rows, "ai_vfe_total")
    ai_phase_conf_ok = _as_optional_int_array(rows, "ai_phase_conf_ok", default=-1)
    ai_phase_vfe_ok = _as_optional_int_array(rows, "ai_phase_vfe_ok", default=-1)
    ai_phase_gate_ok = _as_optional_int_array(rows, "ai_phase_gate_ok", default=-1)
    ai_retry_flag = np.array(
        [1 if str(r.get("ai_retry_reason", "")).strip() else 0 for r in rows],
        dtype=int,
    )
    ai_fail_flag = np.array(
        [1 if str(r.get("ai_failure_reason", "")).strip() else 0 for r in rows],
        dtype=int,
    )

    true_reach_xy_error = _as_optional_float_array(rows, "true_reach_xy_error")
    true_descend_xy_error = _as_optional_float_array(rows, "true_descend_xy_error")
    active_reach_ref_x = _as_optional_float_array(rows, "active_reach_ref_x")
    active_reach_ref_y = _as_optional_float_array(rows, "active_reach_ref_y")
    active_reach_ref_z = _as_optional_float_array(rows, "active_reach_ref_z")
    active_descend_ref_x = _as_optional_float_array(rows, "active_descend_ref_x")
    active_descend_ref_y = _as_optional_float_array(rows, "active_descend_ref_y")
    active_descend_ref_z = _as_optional_float_array(rows, "active_descend_ref_z")
    phase_step = _as_optional_int_array(rows, "phase_step", default=0)
    ai_belief_ee_x = _as_optional_float_array(rows, "ai_belief_ee_x")
    ai_belief_ee_y = _as_optional_float_array(rows, "ai_belief_ee_y")
    ai_belief_obj_rel_x = _as_optional_float_array(rows, "ai_belief_obj_rel_x")
    ai_belief_obj_rel_y = _as_optional_float_array(rows, "ai_belief_obj_rel_y")
    obs_reach_xy_error = _as_optional_float_array(rows, "obs_reach_xy_error")
    obs_reach_z_error = _as_optional_float_array(rows, "obs_reach_z_error")
    obs_descend_xy_error = _as_optional_float_array(rows, "obs_descend_xy_error")
    obs_descend_z_error = _as_optional_float_array(rows, "obs_descend_z_error")
    obs_preplace_error = _as_optional_float_array(rows, "obs_preplace_error")
    obs_preplace_xy_error = _as_optional_float_array(rows, "obs_preplace_xy_error")
    obs_place_error = _as_optional_float_array(rows, "obs_place_error")
    obs_place_xy_error = _as_optional_float_array(rows, "obs_place_xy_error")
    retry_reason = np.array([str(r.get("ai_retry_reason", "")).strip() for r in rows], dtype=object)

    phase_map = {p: i for i, p in enumerate(sorted(set(phase)))}
    phase_idx = np.array([phase_map[p] for p in phase], dtype=int)

    plt.figure(figsize=(10, 4))
    plt.plot(step, distance_to_object, label="distance_to_object")
    plt.plot(step, reach_error, label="reach_error")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.title("Reach Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(out_dir / "01_reach_metrics.png")

    plt.figure(figsize=(10, 5))
    plt.plot(step, s_obj_x, label="s_obj_rel_x")
    plt.plot(step, s_obj_y, label="s_obj_rel_y")
    plt.plot(step, s_obj_z, label="s_obj_rel_z")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.title("Object Relative Belief Components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(out_dir / "02_object_relative_belief.png")

    plt.figure(figsize=(10, 5))
    plt.plot(step, s_ee_x, label="s_ee_x")
    plt.plot(step, s_ee_y, label="s_ee_y")
    plt.plot(step, s_ee_z, label="s_ee_z")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.title("EE Belief Components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(out_dir / "03_ee_belief.png")

    plt.figure(figsize=(10, 5))
    plt.plot(step, a_x, label="action_move_x")
    plt.plot(step, a_y, label="action_move_y")
    plt.plot(step, a_z, label="action_move_z")
    plt.step(step, a_grip, where="post", label="action_grip")
    plt.xlabel("Step")
    plt.ylabel("Command")
    plt.title("Action Commands")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(out_dir / "04_actions.png")

    plt.figure(figsize=(10, 5))
    plt.plot(step, t_ee_x, label="true_ee_x")
    plt.plot(step, t_ee_y, label="true_ee_y")
    plt.plot(step, t_ee_z, label="true_ee_z")
    plt.plot(step, t_obj_x, "--", label="true_obj_x")
    plt.plot(step, t_obj_y, "--", label="true_obj_y")
    plt.plot(step, t_obj_z, "--", label="true_obj_z")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.title("True World Positions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(out_dir / "05_true_positions.png")

    plt.figure(figsize=(10, 4))
    plt.step(step, phase_idx, where="post", label="phase")
    plt.step(step, obs_contact, where="post", label="obs_contact")
    plt.step(step, escape_active, where="post", label="escape_active")
    plt.plot(step, obs_grip, label="obs_grip")
    plt.yticks(list(phase_map.values()), list(phase_map.keys()))
    plt.xlabel("Step")
    plt.title("Phase, Contact, Gripper Observation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(out_dir / "06_phase_contact_gripper.png")

    # 07) Descend diagnostics (error trend vs commanded motion)
    plt.figure(figsize=(11, 7))
    plt.subplot(3, 1, 1)
    plt.plot(step, _as_float_array(rows, "descend_x_error"), label="descend_x_error")
    plt.plot(step, _as_float_array(rows, "descend_y_error"), label="descend_y_error")
    plt.plot(step, _as_float_array(rows, "descend_z_error"), label="descend_z_error")
    if not np.all(np.isnan(true_descend_xy_error)):
        plt.plot(step, true_descend_xy_error, label="true_descend_xy_error")
    if not np.all(np.isnan(true_reach_xy_error)):
        plt.plot(step, true_reach_xy_error, label="true_reach_xy_error")
    plt.ylabel("Meters")
    plt.title("Descend Drift Diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 2)
    plt.plot(step, a_x, label="action_move_x")
    plt.plot(step, a_y, label="action_move_y")
    plt.plot(step, a_z, label="action_move_z")
    plt.ylabel("Command")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 3)
    plt.plot(step, t_ee_y, label="true_ee_y")
    plt.plot(step, s_obj_y, label="s_obj_rel_y")
    plt.plot(step, s_obj_x, label="s_obj_rel_x")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    _save_fig(out_dir / "07_descend_diagnostics.png")

    # 08) Runtime loop/observation timing health
    if not np.all(np.isnan(loop_dt_ms)):
        plt.figure(figsize=(11, 7))
        plt.subplot(3, 1, 1)
        plt.plot(step, loop_dt_ms, label="loop_dt_ms")
        if not np.all(np.isnan(loop_dt_ema_ms)):
            plt.plot(step, loop_dt_ema_ms, label="loop_dt_ema_ms")
        if not np.all(np.isnan(loop_target_ms)):
            target = loop_target_ms[~np.isnan(loop_target_ms)]
            if target.size > 0:
                plt.axhline(float(target[-1]), linestyle="--", label="loop_target_ms")
        plt.ylabel("ms")
        plt.title("Runtime Timing Health")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.plot(step, obs_age_ms, label="obs_age_ms")
        plt.step(step, obs_stale_warn, where="post", label="obs_stale_warn")
        plt.ylabel("obs")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 3)
        plt.step(step, loop_overrun_count, where="post", label="loop_overrun_count")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        _save_fig(out_dir / "08_runtime_health.png")

    # 09) AI confidence/VFE/gate diagnostics
    ai_present = not np.all(np.isnan(ai_obs_conf)) or not np.all(np.isnan(ai_vfe))
    if ai_present:
        plt.figure(figsize=(11, 8))
        plt.subplot(4, 1, 1)
        if not np.all(np.isnan(ai_obs_conf)):
            plt.plot(step, ai_obs_conf, label="ai_obs_confidence")
        plt.ylim(-0.05, 1.05)
        plt.ylabel("conf")
        plt.title("AI Confidence, VFE, and Gate Diagnostics")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        plt.subplot(4, 1, 2)
        if not np.all(np.isnan(ai_vfe)):
            plt.plot(step, ai_vfe, label="ai_vfe_total")
        plt.ylabel("vfe")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        plt.subplot(4, 1, 3)
        if np.any(ai_phase_conf_ok >= 0):
            plt.step(step, ai_phase_conf_ok, where="post", label="ai_phase_conf_ok")
        if np.any(ai_phase_vfe_ok >= 0):
            plt.step(step, ai_phase_vfe_ok, where="post", label="ai_phase_vfe_ok")
        if np.any(ai_phase_gate_ok >= 0):
            plt.step(step, ai_phase_gate_ok, where="post", label="ai_phase_gate_ok")
        plt.ylim(-0.1, 1.1)
        plt.ylabel("gate")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        plt.subplot(4, 1, 4)
        plt.step(step, ai_retry_flag, where="post", label="retry_flag")
        plt.step(step, ai_fail_flag, where="post", label="failure_flag")
        plt.xlabel("Step")
        plt.ylabel("event")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        _save_fig(out_dir / "09_ai_diagnostics.png")

    # 10) Active references and phase step (new logs)
    refs_present = not np.all(np.isnan(active_reach_ref_x))
    if refs_present:
        plt.figure(figsize=(11, 7))
        plt.subplot(2, 1, 1)
        plt.plot(step, active_reach_ref_x, label="active_reach_ref_x")
        plt.plot(step, active_reach_ref_y, label="active_reach_ref_y")
        plt.plot(step, active_reach_ref_z, label="active_reach_ref_z")
        plt.plot(step, active_descend_ref_x, "--", label="active_descend_ref_x")
        plt.plot(step, active_descend_ref_y, "--", label="active_descend_ref_y")
        plt.plot(step, active_descend_ref_z, "--", label="active_descend_ref_z")
        plt.ylabel("Meters")
        plt.title("Active Reach/Descend References")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.step(step, phase_step, where="post", label="phase_step")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        _save_fig(out_dir / "10_active_refs_phase_step.png")

    # 11) Pick-side XY path (start -> end of first Descend), including belief path if available.
    seg = _first_approach_to_descend_segment(phase)
    if seg is None:
        # Fallback: keep old behavior if Descend is absent.
        reach_phase_name = None
        if np.any(phase == "Reach"):
            reach_phase_name = "Reach"
        elif np.any(phase == "ReachAbove"):
            reach_phase_name = "ReachAbove"
        if reach_phase_name is not None:
            fs = _first_contiguous_phase_segment(phase, reach_phase_name)
            if fs is not None:
                i0, i1 = fs
                i_desc = i1
                seg = (i0, i1, i_desc)

    if seg is not None:
        i0, i1, i_desc = seg
        sl = slice(i0, i1 + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(t_ee_x[sl], t_ee_y[sl], label="true_ee_xy")
        plt.plot(s_ee_x[sl], s_ee_y[sl], label="obs_ee_xy")
        if not np.all(np.isnan(ai_belief_ee_x[sl])) and not np.all(np.isnan(ai_belief_ee_y[sl])):
            plt.plot(ai_belief_ee_x[sl], ai_belief_ee_y[sl], label="belief_ee_xy")
        plt.scatter([t_ee_x[i0]], [t_ee_y[i0]], marker="o", label="start")
        plt.scatter([t_ee_x[i_desc]], [t_ee_y[i_desc]], marker="^", label="desc_start")
        plt.scatter([t_ee_x[i1]], [t_ee_y[i1]], marker="x", label="desc_end")
        plt.scatter([t_obj_x[i0]], [t_obj_y[i0]], marker="s", label="obj_start")
        plt.xlabel("World X")
        plt.ylabel("World Y")
        plt.title("Pick-Side Path XY (Start -> Descend End)")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend(loc="best")

        plt.subplot(1, 2, 2)
        plt.plot(s_obj_x[sl], s_obj_y[sl], label="obs_obj_rel_xy")
        if not np.all(np.isnan(ai_belief_obj_rel_x[sl])) and not np.all(np.isnan(ai_belief_obj_rel_y[sl])):
            plt.plot(ai_belief_obj_rel_x[sl], ai_belief_obj_rel_y[sl], label="belief_obj_rel_xy")
        if not np.all(np.isnan(active_reach_ref_x[sl])) and not np.all(np.isnan(active_reach_ref_y[sl])):
            plt.plot(active_reach_ref_x[sl], active_reach_ref_y[sl], "--", label="reach_ref_rel_xy")
        if not np.all(np.isnan(active_descend_ref_x[sl])) and not np.all(np.isnan(active_descend_ref_y[sl])):
            plt.plot(active_descend_ref_x[sl], active_descend_ref_y[sl], ":", label="desc_ref_rel_xy")
        plt.scatter([s_obj_x[i0]], [s_obj_y[i0]], marker="o", label="start")
        plt.scatter([s_obj_x[i_desc]], [s_obj_y[i_desc]], marker="^", label="desc_start")
        plt.scatter([s_obj_x[i1]], [s_obj_y[i1]], marker="x", label="desc_end")
        plt.xlabel("Rel X")
        plt.ylabel("Rel Y")
        plt.title("Object-Relative XY (Start -> Descend End)")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend(loc="best")
        _save_fig(out_dir / "11_reach_path_xy.png")

    # 12) Phase-split metrics: one panel per phase with per-segment local-step overlays.
    phase_order = []
    for p in phase.tolist():
        if p not in phase_order:
            phase_order.append(p)
    segments_by_phase = _contiguous_phase_segments(phase)
    if len(phase_order) > 0:
        metric_series = {
            "reach_error": reach_error,
            "obs_reach_xy_error": obs_reach_xy_error,
            "obs_reach_z_error": obs_reach_z_error,
            "obs_descend_xy_error": obs_descend_xy_error,
            "obs_descend_z_error": obs_descend_z_error,
            "obs_preplace_error": obs_preplace_error,
            "obs_preplace_xy_error": obs_preplace_xy_error,
            "obs_place_error": obs_place_error,
            "obs_place_xy_error": obs_place_xy_error,
            "true_ee_z": t_ee_z,
            "true_obj_z": t_obj_z,
        }
        n = len(phase_order)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(2.5 * n, 7)), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, ph in zip(axes, phase_order):
            segs = segments_by_phase.get(ph, [])
            specs = _phase_metric_specs(ph)
            for sidx, (i0, i1) in enumerate(segs, start=1):
                local_step = np.arange(i1 - i0 + 1, dtype=int)
                for key, lbl in specs:
                    arr = metric_series.get(key, None)
                    if arr is None:
                        continue
                    y = arr[i0 : i1 + 1]
                    if np.all(np.isnan(y)):
                        continue
                    show_label = f"{lbl}" if sidx == 1 else None
                    ax.plot(local_step, y, alpha=0.9, label=show_label)
                # mark segment end for easier visual comparison across segments
                ax.axvline(local_step[-1], color="k", alpha=0.08)
            ax.set_title(f"{ph} (segments={len(segs)})")
            ax.set_xlabel("Local Step In Segment")
            ax.set_ylabel("Meters")
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper right")
        fig.suptitle("Phase-Split Metrics (Local Step Overlay)")
        _save_fig(out_dir / "12_phase_split_metrics.png")

    # 13) Retry windows: zoomed windows around retry start events.
    retry_start_idx = np.where((retry_reason != "") & np.r_[True, retry_reason[:-1] == ""])[0]
    if retry_start_idx.size > 0:
        max_windows = 4
        retry_start_idx = retry_start_idx[:max_windows]
        window = 220
        n = int(retry_start_idx.size)
        fig, axes = plt.subplots(n, 3, figsize=(15, max(3.2 * n, 4)), sharex=False)
        if n == 1:
            axes = np.asarray([axes])
        for row_i, ridx in enumerate(retry_start_idx):
            j0 = max(0, int(ridx) - window)
            j1 = min(step.shape[0] - 1, int(ridx) + window)
            sl = slice(j0, j1 + 1)
            s = step[sl]

            ax0 = axes[row_i, 0]
            ax0.step(s, phase_idx[sl], where="post")
            ax0.axvline(step[int(ridx)], color="r", linestyle="--", alpha=0.8)
            ax0.set_yticks(list(phase_map.values()))
            ax0.set_yticklabels(list(phase_map.keys()))
            ax0.set_title(f"Retry Window #{row_i + 1}: {retry_reason[int(ridx)]}")
            ax0.set_xlabel("Step")
            ax0.set_ylabel("Phase")
            ax0.grid(True, alpha=0.3)

            ax1 = axes[row_i, 1]
            ax1.plot(s, obs_reach_xy_error[sl], label="reach_xy")
            ax1.plot(s, obs_descend_xy_error[sl], label="desc_xy")
            ax1.plot(s, obs_preplace_error[sl], label="preplace_norm")
            ax1.plot(s, obs_place_error[sl], label="place_norm")
            ax1.axvline(step[int(ridx)], color="r", linestyle="--", alpha=0.8)
            ax1.set_title("Error Norms Around Retry")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Meters")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right")

            ax2 = axes[row_i, 2]
            ax2.step(s, obs_contact[sl], where="post", label="contact")
            ax2.step(s, a_grip[sl], where="post", label="action_grip")
            ax2.plot(s, t_obj_z[sl], label="obj_z")
            ax2.plot(s, t_ee_z[sl], label="ee_z")
            ax2.axvline(step[int(ridx)], color="r", linestyle="--", alpha=0.8)
            ax2.set_title("Contact / Grip / Z Around Retry")
            ax2.set_xlabel("Step")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")

        fig.suptitle("Retry-Centered Windows")
        _save_fig(out_dir / "13_retry_windows.png")

    print(f"CSV: {csv_path}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
