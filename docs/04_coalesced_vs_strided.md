# Experiment 04 Plan: Coalesced vs Strided Access

## Objective
Show the performance impact of non-coalesced memory access.

## Implementation Tasks
- [ ] Implement kernel with runtime-configurable stride.
- [ ] Ensure bounds-safe indexing for large strides.
- [ ] Benchmark fixed workload with varying stride values.
- [ ] Capture runtime, bandwidth, and slowdown vs stride=1.

## Test Matrix
- Strides: 1, 2, 4, 8, 16, 32, 64, 128.
- Input size: 64MB and 256MB.

## Output
- `results/tables/04_coalesced_vs_strided.csv`
- `results/charts/04_stride_vs_performance.png`

## Success Criteria
- Clear downward trend in throughput as stride increases.
- Reproducible trend across at least two buffer sizes.
