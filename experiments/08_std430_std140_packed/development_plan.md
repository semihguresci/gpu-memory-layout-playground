# Experiment 08 std430 vs std140 vs Packed: Codebase Development Plan

Date: 2026-03-15  
Source spec: `docs/experiment_plans/08_std430_std140_packed.md`

## 1. Scope and Current Status
This plan tracks implementation and analysis workflow setup for Experiment 08 (`08_std430_std140_packed`).

Current status:
- runtime experiment, shader wiring, adapter, and registry entry are integrated
- row-level output and deterministic correctness checks are integrated
- experiment-local scripts are present (`collect`, `analyze`, `plot`)
- first measured dataset is collected (smoke run)
- experiment-local charts/tables are generated from collected runs

## 2. Development Phases
### Phase A: Runtime and Correctness
- [x] Implement `std140`, `std430`, and packed variants with equivalent logical kernel operations.
- [x] Add explicit host-side layout contracts and static layout assertions.
- [x] Add deterministic seed/expected-value logic for correctness.
- [x] Add per-iteration row export with `gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, and notes.
- [x] Enforce correctness pass/fail at adapter boundary.

### Phase B: Experiment Tooling
- [x] Add experiment-local run collection script.
- [x] Add experiment-local multi-run analysis script.
- [x] Add quick single-run plotting script.
- [x] Add results/report scaffolding and output directories.

### Phase C: Root Workflow Integration
- [x] Add `08_std430_std140_packed` support to data collection helper script.
- [x] Add `08_std430_std140_packed` support to artifact generation helper script.
- [x] Produce first benchmark JSON and collect into `runs/`.

### Phase D: Validation and Hardening
- [x] Configure with `windows-tests-vs` preset.
- [x] Build `gpu_memory_layout_experiments` in Debug profile.
- [x] Run `clang-format` on touched C++ files.
- [x] Run `clang-tidy` for touched translation units when compile database is available.
- [ ] Run unit tests and targeted regression checks.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 08_std430_std140_packed` is discoverable and runnable from CLI.
- [x] Row-level output exists for `std140`, `std430`, and `packed` variants.
- [x] Adapter fails run on correctness mismatch.
- [x] Experiment-local collection script and result directories are present.
- [x] Reproducible charts/tables beyond raw JSON are generated under `experiments/08_std430_std140_packed/results/`.
- [x] `results.md` is updated with measured values and explicit limitations.
