# Experiment 01 Plan: GPU Execution Model

## Objective
Measure how dispatch size and workgroup shape affect throughput and occupancy behavior.

## Implementation Tasks
- [ ] Add compute shader variant with configurable `local_size_x`.
- [ ] Add host-side parameter sweep for dispatch counts.
- [ ] Record GPU execution time using timestamp queries.
- [ ] Normalize throughput as elements/sec and GB/s.

## Test Matrix
- Dispatch sizes: 1, 32, 64, 256, 1024, 4096, 16384.
- Workgroup sizes: 32, 64, 128, 256.
- Input size fixed at 1M elements for comparability.

## Output
- `results/tables/01_thread_mapping.csv`
- `results/charts/01_threads_vs_throughput.png`

## Success Criteria
- Throughput curve clearly shows scaling behavior and saturation region.
- Repeated runs have <5% variance at steady sizes.
