# Experiment 03 Plan: std430 vs Packed

## Objective
Measure cost of alignment padding and structured buffer layout rules.

## Implementation Tasks
- [ ] Define matching host + shader structs for `std430` and packed representation.
- [ ] Add explicit byte-size and stride validation checks.
- [ ] Benchmark read-modify-write kernel across layouts.
- [ ] Compute effective bandwidth and wasted-byte ratio.

## Test Matrix
- Struct cases: `{vec3 + float}`, `{vec2 + vec2 + float}`, custom mixed-size fields.
- Buffer sizes: 4MB, 64MB, 256MB.

## Output
- `results/tables/03_std430_vs_packed.csv`
- `results/charts/03_alignment_overhead.png`

## Success Criteria
- Results quantify padding overhead.
- Docs include clear explanation of expected alignment rules and observed impact.
