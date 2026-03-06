from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SAVED_LOG_RE = re.compile(r"Saved run log:\s*(?P<path>\S+\.csv)")

# Phase edges used to evaluate first-try pass behavior.
PHASE_FIRST_TRY_EDGES: List[Tuple[str, str]] = [
    ("Reach", "Align"),
    ("Align", "Descend"),
    ("Descend", "CloseHold"),
    ("CloseHold", "LiftTest"),
    ("LiftTest", "Transit"),
    ("Transit", "MoveToPlaceAbove"),
    ("MoveToPlaceAbove", "DescendToPlace"),
    ("DescendToPlace", "Open"),
    ("Open", "Retreat"),
    ("Retreat", "Done"),
]


@dataclass
class RunResult:
    run_index: int
    status: str
    returncode: int
    timed_out: bool
    duration_s: float
    csv_path: Optional[Path]
    report_path: Optional[Path]
    final_phase: str
    total_rows: int
    total_steps: int
    reached_descend: bool
    reached_close_hold: bool
    reached_lift_test: bool
    step_descend: Optional[int]
    step_close_hold: Optional[int]
    step_lift_test: Optional[int]
    step_done: Optional[int]
    reach_max_consecutive_rows: int
    reach_stall_retries: int
    recovery_events: int
    failure_reason: str
    phase_attempts: Dict[str, int]
    phase_first_try_pass: Dict[str, int]
    stdout_tail: str
    stderr_tail: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched pick-place episodes and summarize pass/fail metrics."
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=180.0,
        help="Per-episode subprocess timeout in seconds.",
    )
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory.")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess runs.",
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default="run_pick_place.py",
        help="Main run script path.",
    )
    parser.add_argument(
        "--run-args",
        type=str,
        default="--no-render --no-pause",
        help="Extra args for run script.",
    )
    parser.add_argument(
        "--save-per-run-report",
        action="store_true",
        help="Run tools/analyze_run_diagnostics.py --save-report for each produced CSV.",
    )
    parser.add_argument(
        "--plot-per-run",
        action="store_true",
        help="Also generate diagnostics plot per run (requires --save-per-run-report).",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="",
        help="Optional output markdown path for batch summary.",
    )
    parser.add_argument(
        "--hard-stuck-reach-rows",
        type=int,
        default=1200,
        help="Threshold for hard stuck in Reach per run.",
    )
    return parser.parse_args()


def _list_run_csvs(logs_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in logs_dir.glob("run_*.csv"):
        out[p.name] = p.resolve()
    return out


def _load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def _phase_first_step(rows: List[dict], phase: str) -> Optional[int]:
    for r in rows:
        if str(r.get("phase", "")) == phase:
            try:
                return int(float(r.get("step", 0)))
            except Exception:
                return None
    return None


def _phase_present(rows: List[dict], phase: str) -> bool:
    return any(str(r.get("phase", "")) == phase for r in rows)


def _max_consecutive_phase_rows(rows: List[dict], phase: str) -> int:
    best = 0
    cur = 0
    for r in rows:
        if str(r.get("phase", "")) == phase:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _count_reason_events(rows: List[dict], key: str, reason: str) -> int:
    count = 0
    prev_active = False
    for r in rows:
        cur_active = str(r.get(key, "")).strip() == reason
        if cur_active and not prev_active:
            count += 1
        prev_active = cur_active
    return count


def _recovery_events(rows: List[dict]) -> int:
    vals = []
    for r in rows:
        raw = str(r.get("ai_recovery_global_count", "")).strip()
        if not raw:
            continue
        try:
            vals.append(int(float(raw)))
        except Exception:
            continue
    return max(vals) if vals else 0


def _last_nonempty(rows: List[dict], key: str) -> str:
    out = ""
    for r in rows:
        v = str(r.get(key, "")).strip()
        if v:
            out = v
    return out


def _phase_attempts_and_first_try(rows: List[dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    attempts = {phase: 0 for phase, _ in PHASE_FIRST_TRY_EDGES}
    pass_transition = {phase: 0 for phase, _ in PHASE_FIRST_TRY_EDGES}
    if not rows:
        return attempts, pass_transition

    transitions: List[Tuple[str, str]] = []
    prev = None
    for r in rows:
        cur = str(r.get("phase", "")).strip()
        if not cur:
            continue
        if prev is None:
            transitions.append(("START", cur))
        elif cur != prev:
            transitions.append((prev, cur))
        prev = cur

    for _, to_phase in transitions:
        if to_phase in attempts:
            attempts[to_phase] += 1
    for from_phase, to_phase in transitions:
        if (from_phase, to_phase) in PHASE_FIRST_TRY_EDGES:
            pass_transition[from_phase] = 1

    first_try = {}
    for phase, _ in PHASE_FIRST_TRY_EDGES:
        first_try[phase] = int(attempts.get(phase, 0) == 1 and pass_transition.get(phase, 0) == 1)
    return attempts, first_try


def _find_csv_from_output(text: str, repo_root: Path, logs_dir: Path) -> Optional[Path]:
    m = SAVED_LOG_RE.search(text)
    if not m:
        return None
    raw = m.group("path").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    if p.exists():
        return p
    # Fallback: if only filename is printed.
    p2 = (logs_dir / Path(raw).name).resolve()
    if p2.exists():
        return p2
    return None


def _discover_new_csv(before: Dict[str, Path], after: Dict[str, Path]) -> Optional[Path]:
    new_keys = sorted(set(after.keys()) - set(before.keys()))
    if new_keys:
        return after[new_keys[-1]]
    if after:
        return sorted(after.values(), key=lambda p: p.stat().st_mtime)[-1]
    return None


def _short_tail(text: str, max_lines: int = 8) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _run_single(
    run_index: int,
    repo_root: Path,
    logs_dir: Path,
    python_exe: str,
    run_script: str,
    run_args: str,
    timeout_sec: float,
    save_per_run_report: bool,
    plot_per_run: bool,
) -> RunResult:
    before = _list_run_csvs(logs_dir)
    cmd = [python_exe, run_script] + shlex.split(run_args)

    t0 = datetime.now().timestamp()
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        timed_out = False
        returncode = int(cp.returncode)
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -9
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

    duration_s = max(0.0, datetime.now().timestamp() - t0)
    after = _list_run_csvs(logs_dir)
    csv_path = _find_csv_from_output(stdout + "\n" + stderr, repo_root=repo_root, logs_dir=logs_dir)
    if csv_path is None:
        csv_path = _discover_new_csv(before, after)

    report_path = None
    final_phase = ""
    total_rows = 0
    total_steps = 0
    reached_descend = False
    reached_close_hold = False
    reached_lift_test = False
    step_descend = None
    step_close_hold = None
    step_lift_test = None
    step_done = None
    reach_max_consecutive_rows = 0
    reach_stall_retries = 0
    recovery_events = 0
    failure_reason = ""
    phase_attempts = {phase: 0 for phase, _ in PHASE_FIRST_TRY_EDGES}
    phase_first_try_pass = {phase: 0 for phase, _ in PHASE_FIRST_TRY_EDGES}

    if csv_path is not None and csv_path.exists():
        rows = _load_rows(csv_path)
        if rows:
            total_rows = len(rows)
            try:
                total_steps = int(float(rows[-1].get("step", total_rows - 1)))
            except Exception:
                total_steps = total_rows - 1
            final_phase = str(rows[-1].get("phase", "")).strip()
            reached_descend = _phase_present(rows, "Descend")
            reached_close_hold = _phase_present(rows, "CloseHold")
            reached_lift_test = _phase_present(rows, "LiftTest")
            step_descend = _phase_first_step(rows, "Descend")
            step_close_hold = _phase_first_step(rows, "CloseHold")
            step_lift_test = _phase_first_step(rows, "LiftTest")
            step_done = _phase_first_step(rows, "Done")
            reach_max_consecutive_rows = _max_consecutive_phase_rows(rows, "Reach")
            reach_stall_retries = _count_reason_events(rows, "ai_retry_reason", "reach_stall")
            recovery_events = _recovery_events(rows)
            failure_reason = _last_nonempty(rows, "ai_failure_reason")
            phase_attempts, phase_first_try_pass = _phase_attempts_and_first_try(rows)

        if save_per_run_report:
            diag_cmd = [
                python_exe,
                "tools/analyze_run_diagnostics.py",
                "--csv",
                str(csv_path),
                "--save-report",
            ]
            if plot_per_run:
                diag_cmd.append("--plot")
            subprocess.run(diag_cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
            report_candidate = logs_dir / "reports" / f"{csv_path.stem}_diagnostics.md"
            if report_candidate.exists():
                report_path = report_candidate.resolve()

    if timed_out:
        status = "timeout"
    elif returncode != 0:
        status = "error"
    elif final_phase == "Done":
        status = "done"
    elif final_phase:
        status = f"terminal:{final_phase}"
    else:
        status = "unknown"

    return RunResult(
        run_index=run_index,
        status=status,
        returncode=returncode,
        timed_out=timed_out,
        duration_s=duration_s,
        csv_path=csv_path,
        report_path=report_path,
        final_phase=final_phase,
        total_rows=total_rows,
        total_steps=total_steps,
        reached_descend=reached_descend,
        reached_close_hold=reached_close_hold,
        reached_lift_test=reached_lift_test,
        step_descend=step_descend,
        step_close_hold=step_close_hold,
        step_lift_test=step_lift_test,
        step_done=step_done,
        reach_max_consecutive_rows=reach_max_consecutive_rows,
        reach_stall_retries=reach_stall_retries,
        recovery_events=recovery_events,
        failure_reason=failure_reason,
        phase_attempts=phase_attempts,
        phase_first_try_pass=phase_first_try_pass,
        stdout_tail=_short_tail(stdout),
        stderr_tail=_short_tail(stderr),
    )


def _fmt_opt(v: Optional[int]) -> str:
    return "-" if v is None else str(int(v))


def _evaluate_gates(results: List[RunResult], hard_stuck_reach_rows: int) -> Dict[str, str]:
    n = len(results)
    reach_descend_ok = sum(1 for r in results if r.reached_descend)
    reach_close_ok = sum(1 for r in results if r.reached_close_hold)
    reach_stall_total = int(sum(r.reach_stall_retries for r in results))
    hard_stuck_count = int(sum(r.reach_max_consecutive_rows > hard_stuck_reach_rows for r in results))
    lifttest_steps = [r.step_lift_test for r in results if r.step_lift_test is not None]

    median_lifttest = float(np.median(lifttest_steps)) if lifttest_steps else float("nan")
    p90_lifttest = float(np.percentile(lifttest_steps, 90)) if lifttest_steps else float("nan")

    return {
        "reach_descend": f"{reach_descend_ok}/{n} ({'PASS' if reach_descend_ok >= 9 else 'FAIL'})",
        "reach_close_hold": f"{reach_close_ok}/{n} ({'PASS' if reach_close_ok >= 8 else 'FAIL'})",
        "reach_stall_total": f"{reach_stall_total} ({'PASS' if reach_stall_total <= 8 else 'FAIL'})",
        "median_lifttest": (
            f"{median_lifttest:.1f} ({'PASS' if np.isfinite(median_lifttest) and median_lifttest <= 2600 else 'FAIL'})"
        ),
        "p90_lifttest": (
            f"{p90_lifttest:.1f} ({'PASS' if np.isfinite(p90_lifttest) and p90_lifttest <= 3400 else 'FAIL'})"
        ),
        "hard_stuck_reach": f"{hard_stuck_count} ({'PASS' if hard_stuck_count == 0 else 'FAIL'})",
    }


def _write_summary(
    out_path: Path,
    args: argparse.Namespace,
    results: List[RunResult],
    hard_stuck_reach_rows: int,
) -> None:
    done_count = sum(1 for r in results if r.final_phase == "Done")
    timeout_count = sum(1 for r in results if r.timed_out)
    error_count = sum(1 for r in results if r.returncode != 0 and not r.timed_out)
    reach_stall_total = sum(r.reach_stall_retries for r in results)
    gates = _evaluate_gates(results, hard_stuck_reach_rows=hard_stuck_reach_rows)

    lines: List[str] = []
    lines.append(f"# Batch Evaluation Report ({datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("## Run Setup")
    lines.append(f"- episodes: {int(args.episodes)}")
    lines.append(f"- timeout_sec: {float(args.timeout_sec):.1f}")
    lines.append(f"- run_cmd: `{args.python} {args.run_script} {args.run_args}`")
    lines.append(f"- save_per_run_report: {int(bool(args.save_per_run_report))}")
    lines.append(f"- plot_per_run: {int(bool(args.plot_per_run))}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(f"- done_count: {done_count}/{len(results)}")
    lines.append(f"- timeout_count: {timeout_count}")
    lines.append(f"- error_count: {error_count}")
    lines.append(f"- total_reach_stall_retries: {int(reach_stall_total)}")
    lines.append("")
    lines.append("## P0 Gate Check")
    lines.append(f"- reach_descend: {gates['reach_descend']}")
    lines.append(f"- reach_close_hold: {gates['reach_close_hold']}")
    lines.append(f"- reach_stall_total: {gates['reach_stall_total']}")
    lines.append(f"- median_lifttest_step: {gates['median_lifttest']}")
    lines.append(f"- p90_lifttest_step: {gates['p90_lifttest']}")
    lines.append(
        f"- hard_stuck_reach_count (>{int(hard_stuck_reach_rows)} consecutive Reach rows): "
        f"{gates['hard_stuck_reach']}"
    )
    lines.append("")
    lines.append("## Phase First-Try Pass")
    lines.append("| phase | entered_runs | first_try_pass | pass_rate | avg_attempts_when_entered |")
    lines.append("|---|---:|---:|---:|---:|")
    for phase, _ in PHASE_FIRST_TRY_EDGES:
        entered_runs = int(sum(1 for r in results if int(r.phase_attempts.get(phase, 0)) > 0))
        first_try_pass = int(sum(int(r.phase_first_try_pass.get(phase, 0)) for r in results))
        if entered_runs > 0:
            pass_rate = float(first_try_pass) / float(entered_runs)
            avg_attempts = float(
                np.mean([int(r.phase_attempts.get(phase, 0)) for r in results if int(r.phase_attempts.get(phase, 0)) > 0])
            )
            pass_rate_s = f"{pass_rate:.2f}"
            avg_attempts_s = f"{avg_attempts:.2f}"
        else:
            pass_rate_s = "-"
            avg_attempts_s = "-"
        lines.append(
            f"| {phase} | {entered_runs} | {first_try_pass} | {pass_rate_s} | {avg_attempts_s} |"
        )
    lines.append("")
    # Compact key stats for sweep parser.
    for phase_key, out_key in (
        ("Reach", "first_try_reach"),
        ("Descend", "first_try_descend"),
        ("MoveToPlaceAbove", "first_try_preplace"),
        ("DescendToPlace", "first_try_place_descend"),
    ):
        entered_runs = int(sum(1 for r in results if int(r.phase_attempts.get(phase_key, 0)) > 0))
        first_try_pass = int(sum(int(r.phase_first_try_pass.get(phase_key, 0)) for r in results))
        lines.append(f"- {out_key}: {first_try_pass}/{entered_runs}")
    lines.append("")
    lines.append("## Per Run")
    lines.append(
        "| run | status | phase | step_done | step_lifttest | reach_max_rows | "
        "reach_stall_retries | recoveries | reach_1st | descend_1st | preplace_1st | place_1st | csv | report |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in results:
        csv_short = str(r.csv_path).replace("\\", "/") if r.csv_path else "-"
        rep_short = str(r.report_path).replace("\\", "/") if r.report_path else "-"
        lines.append(
            f"| {r.run_index} | {r.status} | {r.final_phase or '-'} | {_fmt_opt(r.step_done)} | "
            f"{_fmt_opt(r.step_lift_test)} | {int(r.reach_max_consecutive_rows)} | "
            f"{int(r.reach_stall_retries)} | {int(r.recovery_events)} | "
            f"{int(r.phase_first_try_pass.get('Reach', 0))} | "
            f"{int(r.phase_first_try_pass.get('Descend', 0))} | "
            f"{int(r.phase_first_try_pass.get('MoveToPlaceAbove', 0))} | "
            f"{int(r.phase_first_try_pass.get('DescendToPlace', 0))} | "
            f"{csv_short} | {rep_short} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = (repo_root / args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_out = Path(args.summary_out).resolve() if args.summary_out else None
    if summary_out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_out = logs_dir / "reports" / f"batch_eval_{ts}.md"

    results: List[RunResult] = []
    print(
        f"[Batch] start episodes={int(args.episodes)} timeout={float(args.timeout_sec):.1f}s "
        f"run='{args.python} {args.run_script} {args.run_args}'"
    )
    for i in range(1, int(args.episodes) + 1):
        print(f"[Batch] run {i}/{int(args.episodes)} ...")
        rr = _run_single(
            run_index=i,
            repo_root=repo_root,
            logs_dir=logs_dir,
            python_exe=str(args.python),
            run_script=str(args.run_script),
            run_args=str(args.run_args),
            timeout_sec=float(args.timeout_sec),
            save_per_run_report=bool(args.save_per_run_report),
            plot_per_run=bool(args.plot_per_run),
        )
        results.append(rr)
        print(
            f"[Batch] run {i} status={rr.status} phase={rr.final_phase or '-'} "
            f"step_lifttest={_fmt_opt(rr.step_lift_test)} "
            f"reach_rows_max={rr.reach_max_consecutive_rows} "
            f"reach_stall_retries={rr.reach_stall_retries}"
        )

    _write_summary(
        out_path=summary_out,
        args=args,
        results=results,
        hard_stuck_reach_rows=int(args.hard_stuck_reach_rows),
    )
    print(f"[Batch] summary saved: {summary_out}")


if __name__ == "__main__":
    main()
