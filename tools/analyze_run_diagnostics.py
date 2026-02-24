import argparse
import csv
from pathlib import Path

import numpy as np


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


def _latest_csv(logs_dir: Path) -> Path:
    runs = sorted(logs_dir.glob("run_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run_*.csv files found in {logs_dir}")
    return runs[0]


def _load_rows(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _f(row, key, default=np.nan):
    v = row.get(key, "")
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _i(row, key, default=0):
    v = row.get(key, "")
    if v is None or v == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _phase_transitions(rows):
    trans = []
    prev = None
    for r in rows:
        ph = str(r.get("phase", ""))
        if ph != prev:
            trans.append(
                {
                    "step": _i(r, "step", 0),
                    "phase": ph,
                    "retry": str(r.get("ai_retry_reason", "")).strip(),
                    "fail": str(r.get("ai_failure_reason", "")).strip(),
                    "branch": str(r.get("ai_recovery_branch", "")).strip(),
                    "bt_reason": str(r.get("ai_bt_reason", "")).strip(),
                }
            )
        prev = ph
    return trans


def _phase_rows(rows, phase_name):
    return [r for r in rows if str(r.get("phase", "")) == phase_name]


def _arr(rows, key):
    return np.array([_f(r, key) for r in rows], dtype=float)


def _descend_diag(rows, phase_name):
    seg = _phase_rows(rows, phase_name)
    if not seg:
        return None
    step = _arr(seg, "step")
    ex = _arr(seg, "descend_x_error")
    ey = _arr(seg, "descend_y_error")
    ez = _arr(seg, "descend_z_error")
    mx = _arr(seg, "action_move_x")
    my = _arr(seg, "action_move_y")
    mz = _arr(seg, "action_move_z")
    ee_x = _arr(seg, "true_ee_x")
    ee_y = _arr(seg, "true_ee_y")
    ee_z = _arr(seg, "true_ee_z")
    d_ee_y = np.diff(ee_y) if len(ee_y) > 1 else np.array([0.0], dtype=float)
    return {
        "rows": len(seg),
        "step0": int(step[0]),
        "step1": int(step[-1]),
        "x0": float(ex[0]),
        "x1": float(ex[-1]),
        "y0": float(ey[0]),
        "y1": float(ey[-1]),
        "z0": float(ez[0]),
        "z1": float(ez[-1]),
        "x_min": float(np.min(ex)),
        "y_min": float(np.min(ey)),
        "z_min": float(np.min(ez)),
        "mean_abs_mx": float(np.mean(np.abs(mx))),
        "mean_abs_my": float(np.mean(np.abs(my))),
        "mean_abs_mz": float(np.mean(np.abs(mz))),
        "ee_drift_x": float(ee_x[-1] - ee_x[0]),
        "ee_drift_y": float(ee_y[-1] - ee_y[0]),
        "ee_drift_z": float(ee_z[-1] - ee_z[0]),
        "frac_my_neg": float(np.mean(my < 0.0)),
        "frac_d_ee_y_pos": float(np.mean(d_ee_y > 0.0)),
    }


def _fmt_pct(v):
    return f"{100.0 * float(v):.1f}%"


def _make_plots(rows, out_png: Path):
    if not HAS_MPL:
        return False

    step = np.array([_i(r, "step", 0) for r in rows], dtype=int)
    phase = np.array([str(r.get("phase", "")) for r in rows], dtype=object)
    phase_ids = {p: i for i, p in enumerate(sorted(set(phase.tolist())))}
    phase_idx = np.array([phase_ids[p] for p in phase], dtype=int)
    ex = _arr(rows, "descend_x_error")
    ey = _arr(rows, "descend_y_error")
    ez = _arr(rows, "descend_z_error")
    mx = _arr(rows, "action_move_x")
    my = _arr(rows, "action_move_y")
    ee_y = _arr(rows, "true_ee_y")

    plt.figure(figsize=(12, 7))
    plt.subplot(3, 1, 1)
    plt.step(step, phase_idx, where="post")
    plt.yticks(list(phase_ids.values()), list(phase_ids.keys()))
    plt.title("Phase Timeline")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(step, ex, label="descend_x_error")
    plt.plot(step, ey, label="descend_y_error")
    plt.plot(step, ez, label="descend_z_error")
    plt.title("Descend Error Signals")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 3)
    plt.plot(step, mx, label="action_move_x")
    plt.plot(step, my, label="action_move_y")
    plt.plot(step, ee_y, label="true_ee_y")
    plt.title("Command vs Motion (Y drift visibility)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Diagnose one run CSV for phase and drift issues.")
    parser.add_argument("--csv", type=str, default="", help="CSV path. If omitted, latest logs/run_*.csv is used.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory containing run_*.csv.")
    parser.add_argument("--save-report", action="store_true", help="Write markdown report to logs/reports/")
    parser.add_argument("--plot", action="store_true", help="Create a diagnostics plot PNG.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    csv_path = Path(args.csv) if args.csv else _latest_csv(logs_dir)
    rows = _load_rows(csv_path)

    step = np.array([_i(r, "step", 0) for r in rows], dtype=int)
    phase = np.array([str(r.get("phase", "")) for r in rows], dtype=object)
    transitions = _phase_transitions(rows)
    final_phase = str(rows[-1].get("phase", ""))

    tx = _arr(rows, "true_target_x")
    ty = _arr(rows, "true_target_y")
    tz = _arr(rows, "true_target_z")

    phase_names, phase_counts = np.unique(phase, return_counts=True)
    phase_summary = sorted(zip(phase_names.tolist(), phase_counts.tolist()), key=lambda x: x[1], reverse=True)

    descend_diag = _descend_diag(rows, "Descend")
    place_descend_diag = _descend_diag(rows, "DescendToPlace")

    lines = []
    lines.append(f"Run: `{csv_path}`")
    lines.append(f"Rows: {len(rows)} | steps: {int(step[0])} -> {int(step[-1])}")
    lines.append(
        "Target world: "
        f"x={float(tx[0]):.3f} y={float(ty[0]):.3f} z={float(tz[0]):.3f} "
        f"(min/max x={float(np.min(tx)):.3f}/{float(np.max(tx)):.3f}, "
        f"y={float(np.min(ty)):.3f}/{float(np.max(ty)):.3f}, "
        f"z={float(np.min(tz)):.3f}/{float(np.max(tz)):.3f})"
    )
    lines.append(f"Final phase: {final_phase}")
    lines.append("")
    lines.append("Phase durations (rows):")
    for name, count in phase_summary:
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("Transitions:")
    for t in transitions:
        extra = []
        if t["retry"]:
            extra.append(f"retry={t['retry']}")
        if t["fail"]:
            extra.append(f"fail={t['fail']}")
        if t["branch"]:
            extra.append(f"branch={t['branch']}")
        if t["bt_reason"]:
            extra.append(f"bt={t['bt_reason']}")
        suffix = (" | " + ", ".join(extra)) if extra else ""
        lines.append(f"- step={t['step']} -> {t['phase']}{suffix}")
    lines.append("")

    def _append_desc(name, d):
        if d is None:
            lines.append(f"{name}: not reached")
            return
        lines.append(f"{name}: rows={d['rows']} step={d['step0']}->{d['step1']}")
        lines.append(
            f"  error x {d['x0']:.4f}->{d['x1']:.4f} (min {d['x_min']:.4f}) | "
            f"y {d['y0']:.4f}->{d['y1']:.4f} (min {d['y_min']:.4f}) | "
            f"z {d['z0']:.4f}->{d['z1']:.4f} (min {d['z_min']:.4f})"
        )
        lines.append(
            f"  mean |move| x/y/z = {d['mean_abs_mx']:.5f}/{d['mean_abs_my']:.5f}/{d['mean_abs_mz']:.5f}"
        )
        lines.append(
            f"  ee drift x/y/z = {d['ee_drift_x']:+.4f}/{d['ee_drift_y']:+.4f}/{d['ee_drift_z']:+.4f} | "
            f"move_y<0={_fmt_pct(d['frac_my_neg'])} and d(ee_y)>0={_fmt_pct(d['frac_d_ee_y_pos'])}"
        )

    lines.append("Descend diagnostics:")
    _append_desc("Descend", descend_diag)
    _append_desc("DescendToPlace", place_descend_diag)

    reached_place = any(p in set(phase.tolist()) for p in ["Transit", "MoveToPlaceAbove", "DescendToPlace", "Open"])
    if not reached_place:
        lines.append("")
        lines.append("Interpretation: run did not reach place-side phases; issue is still on pick-side progression.")

    text = "\n".join(lines)
    print(text)

    if args.save_report:
        out_dir = logs_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{csv_path.stem}_diagnostics.md"
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved report: {out_path}")

    if args.plot:
        out_png = logs_dir / "plots" / csv_path.stem / "diagnostics_descend.png"
        if _make_plots(rows, out_png):
            print(f"Saved plot: {out_png}")
        else:
            print("Plot skipped: matplotlib not available.")


if __name__ == "__main__":
    main()
