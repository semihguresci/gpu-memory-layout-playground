# Experiment 11 Results: Coalesced vs Strided Access

## Run and Artifact Snapshot
- Benchmark status: latest collection completed (`35/35` row correctness pass)
- Analysis status: summary tables and charts generated successfully
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2479554560`)
- Config: `--iterations 5 --warmup 2 --size 128M`
- Validation: disabled in this run (`validation_enabled=false`)
- Raw export timestamp (UTC): `2026-03-25T18:54:32Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260325_185432Z.json](./runs/nvidia_geforce_rtx_2080_super/20260325_185432Z.json)
- Sweep coverage: `7` stride variants x `5` timed iterations each

## Key Measurements
Median results from [the summary table](./results/tables/coalesced_vs_strided_summary.csv):

| Variant | Median GPU ms | Median GB/s | Slowdown vs `stride_1` |
| --- | ---: | ---: | ---: |
| `stride_1` | `0.198912` | `21.086229` | `1.00x` |
| `stride_2` | `19.430400` | `0.215863` | `97.68x` |
| `stride_4` | `14.418400` | `0.290899` | `72.49x` |
| `stride_8` | `5.041632` | `0.831934` | `25.35x` |
| `stride_16` | `2.819392` | `1.487663` | `14.17x` |
| `stride_32` | `3.142784` | `1.334582` | `15.80x` |
| `stride_64` | `3.149888` | `1.331572` | `15.84x` |

Fastest variant on this run:
- `stride_1` at `0.198912 ms` and `21.086229 GB/s`

Slowest variant on this run:
- `stride_2` at `19.430400 ms` and `0.215863 GB/s`

## Additional Analysis
- The penalty curve is non-monotonic after `stride_2`. `stride_16` is the fastest strided case here, and `stride_32` / `stride_64` are very close to each other.
- That shape is consistent with a transaction-granularity effect rather than a simple linear "more stride = proportionally worse" rule. This is an inference from the measured data, not a hardware-counter proof.
- Stability is strongest at `stride_1` (`p95/median=1.001x`, `cv=0.002`) and weakest at `stride_16` (`p95/median=1.340x`, `cv=0.160`). The tail estimate is coarse because each variant has only five timed iterations.
- The footprint table shows that the total allocated span scales linearly with stride: `stride_64` uses `268,435,456` allocated bytes versus `4,194,304` useful bytes, a `64x` multiplier.
- The derived allocation multiplier is a useful context check, but the timing chart is still the primary result because the benchmark keeps logical work constant across variants.

## Graphics
![Median GPU Time by Stride](./results/charts/coalesced_vs_strided_median_gpu_ms.png)

![Median Bandwidth by Stride](./results/charts/coalesced_vs_strided_median_gbps.png)

![Slowdown vs stride_1](./results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)

![GPU Time Stability by Stride](./results/charts/coalesced_vs_strided_stability_ratio.png)

## Data Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/coalesced_vs_strided_summary.csv)
- [Relative table](./results/tables/coalesced_vs_strided_relative.csv)
- [Stability table](./results/tables/coalesced_vs_strided_stability.csv)
- [Footprint table](./results/tables/coalesced_vs_strided_footprint.csv)
- [Latest collected run](./runs/nvidia_geforce_rtx_2080_super/20260325_185432Z.json)

## Interpretation and Limits
- Coalesced access is the clear winner on this GPU and driver stack.
- The worst slowdown is at `stride_2`, not at the largest stride, so the access penalty is not monotonic in stride.
- `stride_16` through `stride_64` form a plateau around `14x-16x` slower than the coalesced baseline.
- This is a single-GPU snapshot with validation layers disabled. The charts and tables are enough to establish the local trend, but not enough to generalize across other devices or driver versions.
