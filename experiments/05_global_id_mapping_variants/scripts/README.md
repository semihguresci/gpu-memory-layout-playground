# Experiment 05 Scripts

## Purpose
Utilities for collecting, aggregating, and plotting Experiment 05 global ID mapping variant runs.

## Scripts
- `collect_run.py`: copy latest benchmark JSON into `runs/<gpu>/timestamp[_label].json`
- `analyze_global_id_mapping_variants.py`: aggregate multi-run data into CSV tables and charts
- `plot_results.py`: quick single-run plot from `results/tables/benchmark_results.json`

## Typical Workflow
Run benchmark and collect:
```powershell
python scripts/run_experiment_data_collection.py --experiment 05_global_id_mapping_variants --iterations 10 --warmup 3 --size 64M
```

Regenerate experiment-local artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 05_global_id_mapping_variants --collect-run
```

Direct analysis call:
```powershell
python experiments/05_global_id_mapping_variants/scripts/analyze_global_id_mapping_variants.py
```
