from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np


DONE_RE = re.compile(r"- done_count:\s*(\d+)\s*/\s*(\d+)")
GATE_RE = re.compile(r"-\s+([a-zA-Z0-9_]+):\s*(.+)")


@dataclass
class Scenario:
    label: str
    x: float
    y: float
    z: float
    yaw_deg: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation across multiple object positions/yaws."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="A1:0.40,0.00,0.20,0;A2:0.50,0.00,0.20,0;A3:0.60,0.00,0.20,0",
        help=(
            "Semicolon-separated scenarios in format "
            "'label:x,y,z,yaw_deg'. Example: "
            "'G1:0.4,0,0.2,0;G2:0.6,0,0.2,0'"
        ),
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per scenario.")
    parser.add_argument("--timeout-sec", type=float, default=240.0, help="Per-run timeout in seconds.")
    parser.add_argument("--run-args", type=str, default="--no-render --no-pause", help="Args for run script.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory.")
    parser.add_argument("--save-per-run-report", action="store_true", help="Save per-run diagnostics.")
    parser.add_argument("--plot-per-run", action="store_true", help="Save per-run diagnostics plot.")
    parser.add_argument(
        "--hard-stuck-reach-rows",
        type=int,
        default=1200,
        help="Hard-stuck threshold passed to run_batch_eval.py.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="",
        help="Optional markdown output path for sweep summary.",
    )
    return parser.parse_args()


def _parse_scenarios(text: str) -> List[Scenario]:
    out: List[Scenario] = []
    entries = [e.strip() for e in str(text).split(";") if e.strip()]
    if not entries:
        raise ValueError("No scenarios provided.")
    for idx, entry in enumerate(entries, start=1):
        if ":" in entry:
            label, raw = entry.split(":", 1)
            label = label.strip() or f"S{idx:02d}"
        else:
            label = f"S{idx:02d}"
            raw = entry
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(
                f"Scenario '{entry}' is invalid. Expected 'label:x,y,z,yaw_deg' or 'x,y,z,yaw_deg'."
            )
        x, y, z, yaw = [float(v) for v in parts]
        out.append(Scenario(label=label, x=x, y=y, z=z, yaw_deg=yaw))
    return out


def _yaw_deg_to_quat_wxyz(yaw_deg: float) -> np.ndarray:
    h = float(np.deg2rad(float(yaw_deg)) * 0.5)
    return np.asarray([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=float)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_") or "scenario"


def _parse_batch_summary(path: Path):
    done_count = ""
    gate_lines = {}
    if not path.exists():
        return done_count, gate_lines
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = DONE_RE.search(text)
    if m:
        done_count = f"{m.group(1)}/{m.group(2)}"
    for line in text.splitlines():
        gm = GATE_RE.match(line.strip())
        if not gm:
            continue
        key = gm.group(1).strip()
        if key in (
            "reach_descend",
            "reach_close_hold",
            "reach_stall_total",
            "median_lifttest_step",
            "p90_lifttest_step",
            "hard_stuck_reach_count",
        ):
            gate_lines[key] = gm.group(2).strip()
    return done_count, gate_lines


def main() -> None:
    args = _parse_args()
    scenarios = _parse_scenarios(args.scenarios)
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = (repo_root / args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_out:
        summary_out = Path(args.summary_out).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_out = logs_dir / "reports" / f"position_sweep_{ts}.md"

    print(
        f"[Sweep] scenarios={len(scenarios)} episodes_per_scenario={int(args.episodes)} "
        f"timeout={float(args.timeout_sec):.1f}s"
    )

    rows = []
    for i, sc in enumerate(scenarios, start=1):
        quat = _yaw_deg_to_quat_wxyz(sc.yaw_deg)
        env = os.environ.copy()
        env["OBJ_WORLD_XYZ"] = f"{sc.x:.6f},{sc.y:.6f},{sc.z:.6f}"
        env["OBJ_WORLD_QUAT_WXYZ"] = f"{quat[0]:.8f},{quat[1]:.8f},{quat[2]:.8f},{quat[3]:.8f}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_out = logs_dir / "reports" / f"batch_eval_{_slug(sc.label)}_{ts}.md"
        cmd = [
            str(args.python),
            "tools/run_batch_eval.py",
            "--episodes",
            str(int(args.episodes)),
            "--timeout-sec",
            str(float(args.timeout_sec)),
            "--logs-dir",
            str(args.logs_dir),
            "--run-args",
            str(args.run_args),
            "--hard-stuck-reach-rows",
            str(int(args.hard_stuck_reach_rows)),
            "--summary-out",
            str(batch_out),
        ]
        if args.save_per_run_report:
            cmd.append("--save-per-run-report")
        if args.plot_per_run:
            cmd.append("--plot-per-run")

        print(
            f"[Sweep] {i}/{len(scenarios)} label={sc.label} "
            f"obj=({sc.x:.3f},{sc.y:.3f},{sc.z:.3f}) yaw={sc.yaw_deg:.1f}"
        )
        cp = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        done_count, gates = _parse_batch_summary(batch_out)
        status = "ok" if cp.returncode == 0 else f"error({cp.returncode})"
        rows.append(
            {
                "label": sc.label,
                "x": sc.x,
                "y": sc.y,
                "z": sc.z,
                "yaw_deg": sc.yaw_deg,
                "status": status,
                "done_count": done_count or "-",
                "reach_descend": gates.get("reach_descend", "-"),
                "reach_close_hold": gates.get("reach_close_hold", "-"),
                "reach_stall_total": gates.get("reach_stall_total", "-"),
                "hard_stuck_reach_count": gates.get("hard_stuck_reach_count", "-"),
                "batch_report": str(batch_out).replace("\\", "/"),
                "stdout_tail": "\n".join((cp.stdout or "").splitlines()[-6:]),
                "stderr_tail": "\n".join((cp.stderr or "").splitlines()[-6:]),
            }
        )

    lines = []
    lines.append(f"# Position Sweep Report ({datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- scenarios: {len(scenarios)}")
    lines.append(f"- episodes_per_scenario: {int(args.episodes)}")
    lines.append(f"- timeout_sec: {float(args.timeout_sec):.1f}")
    lines.append(f"- run_args: `{args.run_args}`")
    lines.append("")
    lines.append("## Results")
    lines.append(
        "| label | obj pose (x,y,z) | yaw_deg | status | done_count | reach_descend | "
        "reach_close_hold | reach_stall_total | hard_stuck_reach_count | batch_report |"
    )
    lines.append("|---|---|---:|---|---:|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['label']} | ({r['x']:.3f},{r['y']:.3f},{r['z']:.3f}) | {r['yaw_deg']:.1f} | "
            f"{r['status']} | {r['done_count']} | {r['reach_descend']} | {r['reach_close_hold']} | "
            f"{r['reach_stall_total']} | {r['hard_stuck_reach_count']} | {r['batch_report']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Each row is one full call to `tools/run_batch_eval.py` with object pose overridden via env.")
    lines.append("- The run script itself remains unchanged except reading `OBJ_WORLD_XYZ` and `OBJ_WORLD_QUAT_WXYZ`.")
    lines.append("- If status is error, inspect batch report path and stderr tail.")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Sweep] summary saved: {summary_out}")


if __name__ == "__main__":
    main()
