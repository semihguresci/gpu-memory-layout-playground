# Experiment 08 std430 vs std140 vs Packed: Implementation Plan

Date: 2026-03-15

## Goal
Implement a correctness-first Vulkan compute experiment that compares three buffer layout variants on matched update logic:
- `std140`: storage buffer with std140 alignment rules
- `std430`: storage buffer with std430 alignment rules
- `packed`: host-packed float stream with explicit field indexing

Logical payload per particle:
- 16 floats (`coeffs[3]`, `position[3]`, `mass`, `velocity[3]`, `dt`, `color[4]`, `scalar`)

## Scope
- Experiment ID: `08_std430_std140_packed`
- Size sweep: preferred `131072`, `262144`, `524288`, `1048576`, `2097152` particles (fallback to smaller sizes when scratch-limited)
- Local size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)

## File Changes
New files:
- `include/experiments/std430_std140_packed_experiment.hpp`
- `src/experiments/std430_std140_packed_experiment.cpp`
- `src/experiments/adapters/std430_std140_packed_adapter.cpp`
- `shaders/08_std430_std140_packed/08_std140.comp`
- `shaders/08_std430_std140_packed/08_std430.comp`
- `shaders/08_std430_std140_packed/08_packed.comp`
- `experiments/08_std430_std140_packed/development_plan.md`
- `experiments/08_std430_std140_packed/implementation_plan.md`
- `experiments/08_std430_std140_packed/architecture.md`
- `experiments/08_std430_std140_packed/results.md`
- `experiments/08_std430_std140_packed/scripts/README.md`
- `experiments/08_std430_std140_packed/scripts/collect_run.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `experiments/08_std430_std140_packed/README.md`

## Work Packages
### 1. Runtime Contract and Adapter
- [x] Add Experiment 08 config/output contract.
- [x] Add adapter wiring and correctness enforcement.

### 2. Vulkan Runtime
- [x] Add explicit setup/teardown for `std140`, `std430`, and packed pipelines and buffers.
- [x] Add mapped buffer initialization and validation paths for all variants.
- [x] Keep dispatch timing based on GPU timestamps.

### 3. Measurement and Data Export
- [x] Emit row-level metrics for each timed iteration and variant.
- [x] Emit per-case summary statistics via `BenchmarkRunner`.
- [x] Include per-row notes for layout size and alignment waste context.

### 4. Experiment-local Tooling
- [x] Add run collection script.
- [x] Add multi-run analysis script and output naming.
- [x] Add quick single-run plotting script.

### 5. Workflow Integration
- [x] Add experiment to `run_experiment_data_collection.py`.
- [x] Add experiment to `generate_experiment_artifacts.py`.

### 6. Verification and Quality Gates
- [x] Build `gpu_memory_layout_experiments` (Debug).
- [x] Run `clang-format` on touched C++ files.
- [x] Run `clang-tidy` on modified translation units when compile database is available.
- [ ] Run unit tests.
