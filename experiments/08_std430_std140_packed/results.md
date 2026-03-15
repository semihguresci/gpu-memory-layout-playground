# Experiment 08 Results

## Run Snapshot
- Status: latest run completed (`12/12` correctness pass), multi-run analysis available
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2479554560`)
- Config: `--iterations 1 --warmup 0 --size 4M`
- Latest run: `runs/nvidia_geforce_rtx_2080_super/20260315_162439Z_2.json`
- Multi-run aggregate: `5` run captures (`sample_count=5` per variant/size in summary)

## Key Measurements (Aggregated)
- Largest size (`problem_size=1048576`, median across 5 runs):
  - `std430`: `44.892576 ms`, `3.7372 GB/s` storage-bandwidth, `2.9898` logical GB/s
  - `std140`: `48.145088 ms`, `4.8786 GB/s` storage-bandwidth, `2.7878` logical GB/s
  - `packed`: `63.221280 ms`, `2.1230 GB/s` storage-bandwidth, `2.1230` logical GB/s
- Relative to `std430` at largest size:
  - `std140`: `+7.25%` slower in `gpu_ms`, `-6.76%` lower logical GB/s
  - `packed`: `+40.83%` slower in `gpu_ms`, `-28.99%` lower logical GB/s

## Layout Footprint Context
- `std140`: `112` storage bytes/particle, `64` logical bytes/particle, alignment waste `75%`
- `std430`: `80` storage bytes/particle, `64` logical bytes/particle, alignment waste `25%`
- `packed`: `64` storage bytes/particle, `64` logical bytes/particle, alignment waste `0%`

## Graphics
![Dispatch Time](./results/charts/std430_std140_packed_gpu_ms_vs_size.png)

![Effective Bandwidth (Storage Bytes)](./results/charts/std430_std140_packed_gbps_vs_size.png)

![Useful Payload Throughput](./results/charts/std430_std140_packed_logical_gbps_vs_size.png)

![Bandwidth Efficiency](./results/charts/std430_std140_packed_bandwidth_efficiency_vs_size.png)

![Layout Footprint](./results/charts/std430_std140_packed_layout_footprint.png)

![Single-run Summary](./results/charts/benchmark_summary.png)

## Data Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/std430_std140_packed_summary.csv)
- [Status overview](./results/tables/std430_std140_packed_status_overview.csv)
- [Layout overview](./results/tables/std430_std140_packed_layout_overview.csv)
- [Relative-to-std430 table](./results/tables/std430_std140_packed_relative_to_std430.csv)
- [Runs index](./results/tables/std430_std140_packed_runs_index.csv)
- [Multi-run summary](./results/tables/std430_std140_packed_multi_run_summary.csv)
- [Latest collected run](./runs/nvidia_geforce_rtx_2080_super/20260315_162439Z_2.json)

## Interpretation and Limits
- `std430` is the fastest variant in dispatch time over the tested size range in the current dataset.
- `std140` reports the highest storage `GB/s` because the metric counts aligned bytes moved; this does not imply better useful work.
- Logical payload throughput and efficiency metrics show the practical tradeoff clearly:
  - `std430` keeps higher useful throughput than `std140` while reducing alignment overhead.
  - `packed` maximizes byte efficiency (`logical/storage = 1.0`) but underperforms in dispatch time on this GPU/workload.
- Dataset limitations:
  - each collected run used `timed_iterations=1`, so per-run variance characterization remains limited;
  - conclusions are for this kernel and hardware only, and should be revalidated on additional GPUs/drivers.
