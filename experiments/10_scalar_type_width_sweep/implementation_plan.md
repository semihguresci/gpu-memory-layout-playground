# Experiment 10 Scalar Type Width Sweep: Implementation Plan

Date: 2026-03-17

## Goal
Implement a correctness-first Vulkan compute experiment that sweeps scalar storage width while keeping equivalent logical math.

Primary variants:
- `fp32`: baseline float path
- `fp16_storage_fp32_compute`: half-width storage with explicit convert/load-store path
- `u32`: integer baseline for normalization/control path
- `u16`: narrow integer storage path
- `u8` (optional): only when device features and shader path support are available

## Scope
- Experiment ID: `10_scalar_type_width_sweep`
- Size sweep: preferred `131072`, `262144`, `524288`, `1048576`, `2097152`, `4194304`, `8388608`, `16777216` (`2^24`) elements (fallback to smaller sizes if scratch-limited)
- Local size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)
- Accuracy outputs: max abs error and mean abs error against CPU reference

## File Changes
New files:
- `include/experiments/scalar_type_width_sweep_experiment.hpp`
- `src/experiments/scalar_type_width_sweep_experiment.cpp`
- `src/experiments/adapters/scalar_type_width_sweep_adapter.cpp`
- `shaders/10_scalar_type_width_sweep/10_fp32.comp`
- `shaders/10_scalar_type_width_sweep/10_fp16_storage.comp`
- `shaders/10_scalar_type_width_sweep/10_u32.comp`
- `shaders/10_scalar_type_width_sweep/10_u16.comp`
- `shaders/10_scalar_type_width_sweep/10_u8.comp` (optional)
- `experiments/10_scalar_type_width_sweep/architecture.md`
- `experiments/10_scalar_type_width_sweep/development_plan.md`
- `experiments/10_scalar_type_width_sweep/implementation_plan.md`
- `experiments/10_scalar_type_width_sweep/results.md`
- `experiments/10_scalar_type_width_sweep/scripts/collect_run.py`
- `experiments/10_scalar_type_width_sweep/scripts/README.md`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`

## Work Packages
### 1. Runtime Contract and Adapter
- [x] Add Experiment 10 config/output contract.
- [x] Add adapter wiring and correctness enforcement.
- [x] Emit per-row notes for width/dispatch/error context.

### 2. Vulkan Runtime
- [x] Add per-variant buffer/pipeline setup and teardown.
- [x] Add deterministic host-side seed/initialization for all variants.
- [x] Add conversion path to preserve equivalent logical operations across widths.
- [x] Keep dispatch timing based on GPU timestamps.

### 3. Correctness and Metrics
- [x] Validate every variant against CPU reference outputs.
- [x] Define and enforce tolerance policy for reduced-width variants.
- [x] Emit row-level metrics for each timed iteration and variant.

### 4. Workflow Integration
- [x] Add experiment to manifest and build target list.
- [x] Add experiment to root data-collection helper.
- [ ] Add experiment to root artifact-generation helper.
- [x] Add experiment-local run collection script and docs.

### 5. Verification Gates
- [x] Build release target (`tests-vs-release` preset).
- [x] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [x] Run tests and a benchmark smoke pass for Experiment 10.
