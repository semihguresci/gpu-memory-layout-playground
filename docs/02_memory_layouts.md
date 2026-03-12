# Experiment 02 Plan: AoS vs SoA

## Objective
Compare memory layout efficiency for particle-like workloads.

## Implementation Tasks
- [ ] Define host buffer schemas for AoS and SoA.
- [ ] Create equivalent compute kernels for both layouts.
- [ ] Ensure identical arithmetic work in both paths.
- [ ] Benchmark multiple particle counts.

## Test Matrix
- Data sizes: 1M, 5M, 10M particles.
- Iterations: warmup=5, timed=30.
- Precision: FP32.

## Output
- `results/tables/02_aos_vs_soa.csv`
- `results/charts/02_layout_vs_bandwidth.png`

## Success Criteria
- Measured runtime and bandwidth for both layouts.
- Statistical summary includes average/min/max and relative speedup.
