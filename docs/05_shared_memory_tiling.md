# Experiment 05 Plan: Shared Memory Tiling

## Objective
Compare direct global memory access against shared-memory tiled kernels.

## Implementation Tasks
- [ ] Implement baseline kernel using direct global reads/writes.
- [ ] Implement tiled kernel with `shared` memory and synchronization barriers.
- [ ] Sweep tile sizes and workgroup sizes.
- [ ] Verify correctness parity between kernels.

## Test Matrix
- Tile sizes: 64, 128, 256.
- Workgroup sizes: 64, 128, 256.
- Input sizes: 16MB, 64MB, 256MB.

## Output
- `results/tables/05_shared_memory_tiling.csv`
- `results/charts/05_tiled_vs_global.png`

## Success Criteria
- Tiled kernel demonstrates latency/bandwidth improvement for at least one regime.
- Performance tradeoffs are documented for tile/workgroup combinations.
