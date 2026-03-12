# Experiment 06 Plan: Memory Bandwidth Saturation

## Objective
Estimate practical peak bandwidth and efficiency relative to hardware theoretical max.

## Implementation Tasks
- [ ] Implement pure streaming copy/read-write kernel.
- [ ] Sweep data sizes to identify bandwidth saturation point.
- [ ] Gather theoretical bandwidth from device specs (manual config file).
- [ ] Compute efficiency: achieved/theoretical.

## Test Matrix
- Data sizes: 1MB, 10MB, 100MB, 1GB.
- Passes: 3 repeats per size, report average.

## Output
- `results/tables/06_bandwidth_saturation.csv`
- `results/charts/06_bandwidth_saturation.png`

## Success Criteria
- Bandwidth curve reaches plateau at large sizes.
- Efficiency percentage reported with assumptions (clock, bus width, DDR factor).
