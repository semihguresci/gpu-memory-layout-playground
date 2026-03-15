# Experiment 08 Scripts

## Purpose
Utilities for collecting, aggregating, and plotting Experiment 08 std430/std140/packed runs.

## Scripts
- `collect_run.py`: copy latest benchmark JSON into `runs/<gpu>/timestamp[_label].json`
- `analyze_std430_std140_packed.py`: aggregate multi-run data into CSV tables and charts
- `plot_results.py`: quick single-run plot from `results/tables/benchmark_results.json`

## Typical Workflow
Run benchmark and collect:
```powershell
python scripts/run_experiment_data_collection.py --experiment 08_std430_std140_packed --iterations 10 --warmup 3 --size 128M
```

Regenerate experiment-local artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 08_std430_std140_packed --collect-run
```

Direct collection call:
```powershell
python experiments/08_std430_std140_packed/scripts/collect_run.py --input experiments/08_std430_std140_packed/results/tables/benchmark_results.json
```

Direct analysis call:
```powershell
python experiments/08_std430_std140_packed/scripts/analyze_std430_std140_packed.py
```
