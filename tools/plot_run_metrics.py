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

    print(f"CSV: {csv_path}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
