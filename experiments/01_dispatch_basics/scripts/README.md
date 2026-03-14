# Experiment 01 Scripts

These scripts are local to Experiment 01 and read/write files under:
- `../results/tables/`
- `../results/charts/`
- `../runs/`

## Usage
From repository root:

```powershell
python experiments/01_dispatch_basics/scripts/collect_run.py
python experiments/01_dispatch_basics/scripts/plot_results.py
python experiments/01_dispatch_basics/scripts/analyze_dispatch_basics.py
```

Root-level convenience scripts:

```powershell
python scripts/run_experiment_data_collection.py
python scripts/generate_experiment_artifacts.py
```

Typical workflow for multi-device tracking:
1. Run benchmark and write `benchmark_results.json` for the current device.
2. Collect the run into `runs/<device>/...`:
   - `python experiments/01_dispatch_basics/scripts/collect_run.py`
3. Rebuild aggregated tables/charts across all collected runs:
   - `python experiments/01_dispatch_basics/scripts/analyze_dispatch_basics.py`

## Inputs
- `../results/tables/benchmark_results.json` (row-level data required for dispatch analysis)
- `../runs/**/*.json` (optional, for multi-device aggregation)

## Outputs
- `../results/charts/benchmark_summary.png`
- `../results/charts/dispatch_basics_time_throughput.png`
- `../results/charts/dispatch_basics_cross_device_comparison.png`
- `../results/charts/dispatch_basics_operation_ratio_by_device.png`
- `../results/charts/dispatch_basics_<device>_time_throughput.png`
- `../results/charts/dispatch_basics_representative_points.png`
- `../results/charts/dispatch_basics_ratio_summary.png`
- `../results/charts/dispatch_basics_peak_throughput.png`
- `../results/charts/dispatch_basics_status_overview.png`
- `../results/charts/dispatch_basics_test_setup.png`
- `../results/tables/dispatch_basics_summary.csv`
- `../results/tables/dispatch_basics_best_dispatch.csv`
- `../results/tables/dispatch_basics_gpu_ms_pivot.csv`
- `../results/tables/dispatch_basics_throughput_pivot.csv`
- `../results/tables/dispatch_basics_runs_index.csv`
- `../results/tables/dispatch_basics_multi_run_summary.csv`
- `../results/tables/dispatch_basics_device_summary.csv`
- `../results/tables/dispatch_basics_best_dispatch_by_device.csv`
- `../results/tables/dispatch_basics_operation_ratio_by_device.csv`
- `../results/tables/dispatch_basics_device_chart_index.csv`
- `../results/tables/dispatch_basics_representative_points.csv`
- `../results/tables/dispatch_basics_ratio_summary.csv`
- `../results/tables/dispatch_basics_peak_throughput.csv`
- `../results/tables/dispatch_basics_status_overview.csv`
- `../results/tables/dispatch_basics_test_setup.csv`
