# Experiment 10 Results

## Run and Test Snapshot
- Benchmark status: latest run completed (`200/200` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure -L unit` passed (`29/29`)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2479554560`)
- Config: `--iterations 5 --warmup 2 --size 128M`
- Raw export timestamp (UTC): `2026-03-17T20:38:04Z`
- Latest collected run: `runs/nvidia_geforce_rtx_2080_super/20260317_203804Z.json`
- Sweep coverage: `5` variants x `8` problem sizes (`2^17` to `2^24`)

## Key Measurements
Largest tested size (`problem_size=16777216`, medians from current run):
- `u8`: `1.494176 ms`, `22.4568 GB/s`, `p95/median=1.001x`, speedup vs `fp32`: `4.024x`
- `u16`: `2.978720 ms`, `22.5294 GB/s`, `p95/median=1.272x`, speedup vs `fp32`: `2.019x`
- `fp16_storage`: `2.988416 ms`, `22.4563 GB/s`, `p95/median=1.001x`, speedup vs `fp32`: `2.012x`
- `u32`: `5.997088 ms`, `22.3805 GB/s`, `p95/median=1.067x`, speedup vs `fp32`: `1.003x`
- `fp32`: `6.013248 ms`, `22.3203 GB/s`, `p95/median=1.082x`, baseline

Largest-size numerical error (medians):
- `fp32`: max abs `0`, mean abs `0`
- `u32`: max abs `0`, mean abs `0`
- `u16`: max abs `1.5e-05`, mean abs `0`
- `fp16_storage`: max abs `4.88e-04`, mean abs `1.63e-04`
- `u8`: max abs `0`, mean abs `0`

## Additional Analysis
Speedup crossover (`scalar_type_width_sweep_speedup_crossover.csv`):
- `u8`: already `>=2x` faster than `fp32` at smallest size (`2^17`), reaching `4.024x` at `2^24`.
- `u16`: `>=1.5x` faster at `2^17`; crosses `>=2x` at `2^24`.
- `fp16_storage`: `>=1.5x` faster at `2^17`; crosses `>=2x` at `2^24`.
- `u32`: effectively tied with `fp32` across the sweep (max `1.003x`).

Scaling efficiency (`gpu_ms_per_million_elements`, first size to largest size):
- `fp32`: `0.45996 -> 0.35842` (`1.283x` improvement)
- `u32`: `0.46045 -> 0.35745` (`1.288x` improvement)
- `fp16_storage`: `0.28345 -> 0.17812` (`1.591x` improvement)
- `u16`: `0.28271 -> 0.17755` (`1.592x` improvement)
- `u8`: `0.16748 -> 0.08906` (`1.881x` improvement)

Stability observations (`p95/median`):
- Steadiest average variant: `u8` (`avg=1.0076x`, worst `1.0221x` at `2^18`)
- Largest outlier: `u16` at `2^24` (`1.2715x`)
- `fp16_storage` remains stable at large size (`2^24`: `1.0013x`) despite one mid-size spike (`2^19`: `1.1354x`)

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
- [Latest collected run](./runs/nvidia_geforce_rtx_2080_super/20260317_203804Z.json)

## Interpretation and Limits
- This run reinforces the hypothesis: narrower storage (`u8`, `u16`, `fp16_storage`) remains faster than `fp32` across the full `2^17..2^24` sweep.
- At large sizes, all variants converge near the same effective bandwidth ceiling (`~22.3-22.5 GB/s`), so runtime deltas mostly track bytes moved.
- `u8` is consistently the fastest path in this workload, while `u16`/`fp16_storage` provide stable `~2x` gains at the largest size.
- `u16` shows a notable tail-latency spike at `2^24`; conclusions should use medians and p95 together, not medians alone.
Limits:
- single GPU/driver stack and one run snapshot;
- no cross-device validation yet;
- no artifact-generation integration yet for Experiment 10 in `scripts/generate_experiment_artifacts.py`.
