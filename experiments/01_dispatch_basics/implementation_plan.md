# Experiment 01 Dispatch Basics: Implementation Plan

Date: 2026-03-14

## Goal
Implement a reliable Vulkan compute dispatch baseline with correctness validation and stable GPU timing outputs.

## Scope
- Problem size sweep: `2^10` to `2^24` (clamped by configured buffer size)
- Dispatch-count sweep: `1, 4, 16, 64, 128, 256, 512, 1024`
- Timing: dispatch GPU ms (timestamp), end-to-end host-side ms
- Validation: deterministic correctness check per measured point

## Work Packages
1. Result model and statistics
- [x] Row-level benchmark records added with iteration granularity.
- [x] Summary stats implemented: median, p95, min, max, average.
- [x] JSON export extended with metadata and row payload.

2. Dispatch Basics experiment module
- [x] `dispatch_basics_experiment.hpp/.cpp` integrated.
- [x] `contiguous_write` variant wired to storage-buffer pipeline.
- [x] `noop` variant added for overhead/reference comparison.

3. Correctness path
- [x] Deterministic sentinels used per variant.
- [x] Host-side validation compares readback against expected values.
- [x] `correctness_pass` emitted per row and enforced at run level.

4. CLI and registry integration
- [x] Canonical id `01_dispatch_basics` registered and runnable.
- [x] Legacy aliases removed; runtime uses canonical ids.

5. Output and artifacts
- [x] Experiment-local results pipeline established:
  - `results/tables/` and `results/charts/`
  - run collection under `runs/`
- [x] Aggregation scripts generate single-run and multi-run/device outputs.
- [x] Results documentation now references generated charts/tables.
- [x] Architecture documentation added with diagrams.

## Verification Snapshot
- [x] Release build succeeds.
- [x] `--experiment 01_dispatch_basics` run succeeds with current sweep.
- [x] Validation-enabled smoke run succeeds.
- [x] Current run status table reports full correctness and timing coverage.

## Remaining Tasks
- [ ] Add at least one additional GPU run to activate meaningful cross-device comparisons.
- [ ] Optional: add script smoke tests in CI for analysis/collection tooling.

## Definition of Done Status
- [x] `--experiment 01_dispatch_basics` completes successfully.
- [x] Every measured point reports correctness and timing values.
- [x] Output contains raw rows + summary stats including median/p95.
- [x] Legacy aliases are removed from CLI/runtime.
