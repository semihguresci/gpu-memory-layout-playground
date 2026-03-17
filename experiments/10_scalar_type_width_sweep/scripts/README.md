# Experiment 10 scripts

## Collect a benchmark run

```bash
python experiments/10_scalar_type_width_sweep/scripts/collect_run.py --input experiments/10_scalar_type_width_sweep/results/tables/benchmark_results.json
```

This stores timestamped JSON files under `experiments/10_scalar_type_width_sweep/runs/<gpu_slug>/`.

## Generate summary tables and charts

```bash
python experiments/10_scalar_type_width_sweep/scripts/analyze_scalar_type_width_sweep.py
```

This writes:
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_summary.csv`
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_status_overview.csv`
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_largest_size_comparison.csv`
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_speedup_vs_fp32_by_size.csv`
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_scaling_by_size.csv`
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_speedup_crossover.csv`
- `experiments/10_scalar_type_width_sweep/results/tables/scalar_type_width_sweep_stability_overview.csv`
- `experiments/10_scalar_type_width_sweep/results/charts/scalar_type_width_sweep_median_gpu_ms.png`
- `experiments/10_scalar_type_width_sweep/results/charts/scalar_type_width_sweep_median_gbps.png`
- `experiments/10_scalar_type_width_sweep/results/charts/scalar_type_width_sweep_speedup_vs_fp32.png`
- `experiments/10_scalar_type_width_sweep/results/charts/scalar_type_width_sweep_p95_to_median_ratio.png`
- `experiments/10_scalar_type_width_sweep/results/charts/scalar_type_width_sweep_gpu_ms_per_million_elements.png`
- `experiments/10_scalar_type_width_sweep/results/charts/scalar_type_width_sweep_error_metrics.png`
