# Experiment 10 Results

## Run and Test Snapshot
- Benchmark status: latest run completed (`200/200` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure -L unit` passed (`29/29`)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2479554560`)
- Config: `--iterations 5 --warmup 2 --size 128M`
- Raw export timestamp (UTC): `2026-03-25T18:11:21Z`
- Latest collected run: `runs/nvidia_geforce_rtx_2080_super/20260325_181121Z.json`
- Sweep coverage: `5` variants x `8` problem sizes (`2^17` to `2^24`)

## Key Measurements
Largest tested size (`problem_size=16777216`, medians from current run):
- `u8`: `1.533984 ms`, `21.8740 GB/s`, `p95/median=1.217x`, speedup vs `fp32`: `3.923x`
- `u16`: `2.995744 ms`, `22.4014 GB/s`, `p95/median=1.317x`, speedup vs `fp32`: `2.009x`
- `fp16_storage`: `2.999072 ms`, `22.3765 GB/s`, `p95/median=1.004x`, speedup vs `fp32`: `2.006x`
- `u32`: `6.005280 ms`, `22.3500 GB/s`, `p95/median=1.143x`, speedup vs `fp32`: `1.002x`
- `fp32`: `6.017120 ms`, `22.3060 GB/s`, `p95/median=1.172x`, baseline

Largest-size numerical error (medians):
- `fp32`: max abs `0`, mean abs `0`
- `u32`: max abs `0`, mean abs `0`
- `u16`: max abs `1.5e-05`, mean abs `0`
- `fp16_storage`: max abs `4.88e-04`, mean abs `1.63e-04`
- `u8`: max abs `0`, mean abs `0`

## Additional Analysis
Speedup crossover (`scalar_type_width_sweep_speedup_crossover.csv`):
- `u8`: already `>=2x` faster than `fp32` at smallest size (`2^17`), reaching `3.923x` at `2^24`.
- `u16`: `>=1.5x` faster at `2^17`; crosses `>=2x` at `2^23`.
- `fp16_storage`: `>=1.5x` faster at `2^17`; crosses `>=2x` at `2^23`.
- `u32`: effectively tied with `fp32` across the sweep (max `1.009x`).

Scaling efficiency (`gpu_ms_per_million_elements`, first size to largest size):
- `fp32`: `0.45825 -> 0.35865` (`1.277x` improvement)
- `u32`: `0.45923 -> 0.35794` (`1.283x` improvement)
- `fp16_storage`: `0.28149 -> 0.17876` (`1.574x` improvement)
- `u16`: `0.28149 -> 0.17856` (`1.576x` improvement)
- `u8`: `0.16626 -> 0.09143` (`1.818x` improvement)

Stability observations (`p95/median`):
- Steadiest average variant: `fp16_storage` (`avg=1.0149x`, worst `1.0615x` at `2^21`)
- Largest outlier: `u8` at `2^23` (`2.4532x`)
- `u16` also shows a large tail at `2^23` (`1.4959x`), so medians are the safer headline metric.

## Graphics
![Median GPU ms by Problem Size](./results/charts/scalar_type_width_sweep_median_gpu_ms.png)

![Median GB/s by Problem Size](./results/charts/scalar_type_width_sweep_median_gbps.png)

![Speedup vs fp32 by Problem Size](./results/charts/scalar_type_width_sweep_speedup_vs_fp32.png)

![P95-to-Median Stability Ratio by Problem Size](./results/charts/scalar_type_width_sweep_p95_to_median_ratio.png)

![GPU ms per Million Elements by Problem Size](./results/charts/scalar_type_width_sweep_gpu_ms_per_million_elements.png)

![Error Metrics at Largest Size](./results/charts/scalar_type_width_sweep_error_metrics.png)

## Data Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/scalar_type_width_sweep_summary.csv)
- [Status overview](./results/tables/scalar_type_width_sweep_status_overview.csv)
- [Largest-size comparison](./results/tables/scalar_type_width_sweep_largest_size_comparison.csv)
- [Speedup by size](./results/tables/scalar_type_width_sweep_speedup_vs_fp32_by_size.csv)
- [Scaling by size](./results/tables/scalar_type_width_sweep_scaling_by_size.csv)
- [Speedup crossover summary](./results/tables/scalar_type_width_sweep_speedup_crossover.csv)
- [Stability overview](./results/tables/scalar_type_width_sweep_stability_overview.csv)
- [Latest collected run](./runs/nvidia_geforce_rtx_2080_super/20260325_181121Z.json)

## Interpretation and Limits
- This run reinforces the hypothesis: narrower storage (`u8`, `u16`, `fp16_storage`) remains faster than `fp32` across the full `2^17..2^24` sweep.
- At large sizes, all variants converge near the same effective bandwidth ceiling (`~21.9-22.4 GB/s`), so runtime deltas mostly track bytes moved.
- `u8` is consistently the fastest path in this workload, but it also has the strongest tail-latency spike in this run, so median-only reading is incomplete.
- `u16` and `fp16_storage` still give the cleanest middle ground: roughly `2x` faster than `fp32` at the top end with much less volatility than `u8`.
Limits:
- single GPU/driver stack and one run snapshot;
- no cross-device validation yet;
- no artifact-generation integration yet for Experiment 10 in `scripts/generate_experiment_artifacts.py`.
