# Python3 Run Commands (With Flags)

Use these commands from repo root: `active_inference_pick_place/`.
All commands use `python3` as requested.

## 1) Main Runtime

### Basic run
```bash
python3 run_pick_place.py
```

### Headless (recommended for batch/debug)
```bash
python3 run_pick_place.py --no-render --no-pause
```

### Viewer on
```bash
python3 run_pick_place.py --render
```

### Record video
```bash
python3 run_pick_place.py --no-pause --record-video logs/run.mp4 --record-fps 25 --record-width 640 --record-height 480 --record-every-steps 1 --record-camera watching
```

### Runtime object override via env (no XML edit)
```bash
OBJ_WORLD_XYZ="0.60,0.00,0.20" OBJ_WORLD_QUAT_WXYZ="1,0,0,0" python3 run_pick_place.py --no-render --no-pause
```

### Runtime config override via env
```bash
COMMON_CONFIG_PATH=config/common_robot.yaml ACTIVE_INFERENCE_CONFIG_PATH=config/active_inference_config.yaml python3 run_pick_place.py --no-render --no-pause
```

### Meeting demo run (clean, deterministic-style)
```bash
OBJ_WORLD_XYZ="0.50,0.00,0.20" OBJ_WORLD_QUAT_WXYZ="1,0,0,0" python3 run_pick_place.py --no-render --no-pause
```

## 2) Single-Run Diagnostics

### Analyze latest run CSV
```bash
python3 tools/analyze_run_diagnostics.py
```

### Analyze specific run and save report
```bash
python3 tools/analyze_run_diagnostics.py --csv logs/run_YYYYMMDD_HHMMSS.csv --save-report
```

### Analyze + save report + plot
```bash
python3 tools/analyze_run_diagnostics.py --csv logs/run_YYYYMMDD_HHMMSS.csv --save-report --plot
```

## 3) Plotting

### Plot latest run
```bash
python3 tools/plot_run_metrics.py
```

### Plot specific run to custom folder
```bash
python3 tools/plot_run_metrics.py --csv logs/run_YYYYMMDD_HHMMSS.csv --out-dir logs/plots/custom_run
```

### Plot from alternate logs dir
```bash
python3 tools/plot_run_metrics.py --logs-dir logs
```

## 4) Batch Evaluation

### 10-episode baseline
```bash
python3 tools/run_batch_eval.py --episodes 10 --timeout-sec 240 --run-args "--no-render --no-pause"
```

### Save per-run diagnostics reports
```bash
python3 tools/run_batch_eval.py --episodes 10 --timeout-sec 240 --run-args "--no-render --no-pause" --save-per-run-report
```

### Save per-run diagnostics reports + plots
```bash
python3 tools/run_batch_eval.py --episodes 10 --timeout-sec 240 --run-args "--no-render --no-pause" --save-per-run-report --plot-per-run
```

### Custom summary output path
```bash
python3 tools/run_batch_eval.py --episodes 10 --summary-out logs/reports/my_batch_summary.md
```

### Custom hard-stuck threshold for Reach
```bash
python3 tools/run_batch_eval.py --episodes 10 --hard-stuck-reach-rows 1200
```

## 5) Position Sweep (multi-scenario, no XML edit)

### Standard A1/A2/A3 sweep
```bash
python3 tools/run_position_sweep.py --scenarios "A1:0.40,0.00,0.20,0;A2:0.50,0.00,0.20,0;A3:0.60,0.00,0.20,0" --episodes 10 --timeout-sec 240 --save-per-run-report
```

### With per-run plots
```bash
python3 tools/run_position_sweep.py --scenarios "A1:0.40,0.00,0.20,0;A2:0.50,0.00,0.20,0;A3:0.60,0.00,0.20,0" --episodes 10 --timeout-sec 240 --save-per-run-report --plot-per-run
```

### Custom summary file
```bash
python3 tools/run_position_sweep.py --scenarios "B1:0.50,0.12,0.20,0;B2:0.50,-0.12,0.20,0" --episodes 8 --summary-out logs/reports/sweep_B.md
```

### Pass custom args to run script
```bash
python3 tools/run_position_sweep.py --scenarios "C1:0.50,0.00,0.20,45" --run-args "--no-render --no-pause"
```

## 6) Mermaid Extraction (docs tooling)

### Extract Mermaid blocks from markdown
```bash
python3 tools/extract_mermaid_from_md.py --root . --out docs/mermaid
```

## 7) Quick Push-Readiness Checks

### Syntax check all Python files
Linux/macOS:
```bash
python3 -m py_compile $(find . -name "*.py")
```

PowerShell:
```powershell
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { python3 -m py_compile $_.FullName }
```

### Show git status
```bash
git status --short
```

## Script Flag Summary

1. `run_pick_place.py`
- `--no-pause`
- `--render`
- `--no-render`
- `--record-video`
- `--record-fps`
- `--record-width`
- `--record-height`
- `--record-every-steps`
- `--record-camera`

2. `tools/analyze_run_diagnostics.py`
- `--csv`
- `--logs-dir`
- `--save-report`
- `--plot`

3. `tools/plot_run_metrics.py`
- `--csv`
- `--logs-dir`
- `--out-dir`

4. `tools/run_batch_eval.py`
- `--episodes`
- `--timeout-sec`
- `--logs-dir`
- `--python`
- `--run-script`
- `--run-args`
- `--save-per-run-report`
- `--plot-per-run`
- `--summary-out`
- `--hard-stuck-reach-rows`

5. `tools/run_position_sweep.py`
- `--scenarios`
- `--episodes`
- `--timeout-sec`
- `--run-args`
- `--python`
- `--logs-dir`
- `--save-per-run-report`
- `--plot-per-run`
- `--hard-stuck-reach-rows`
- `--summary-out`

6. `tools/extract_mermaid_from_md.py`
- `--root`
- `--out`
