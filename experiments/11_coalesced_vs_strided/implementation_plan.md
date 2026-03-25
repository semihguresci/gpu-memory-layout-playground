# Experiment 11 Coalesced vs Strided Access: Implementation Plan

Date: 2026-03-25

## Goal
Implement a correctness-first Vulkan compute experiment that sweeps access stride while keeping arithmetic, element type, and logical element count constant.

Primary configuration:
- Experiment ID: `11_coalesced_vs_strided`
- Stride sweep: `1, 2, 4, 8, 16, 32, 64`
- Workgroup size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)
- Buffer budget: derive physical allocation from the existing scratch-size option so the largest stride fits without wraparound
- Accuracy outputs: deterministic CPU reference for touched logical indices; untouched padding must not affect correctness checks

## Scope
- Run target: coalesced baseline versus strided access sweep
- Logical element count: fixed per sweep, with physical buffer span scaled by stride
- Reporting: per-stride median and p95 timing, logical throughput in GB/s, and correctness status
- Validation policy: fail the run on any mismatch before emitting benchmark success

## File Changes
New files:
- `include/experiments/coalesced_vs_strided_experiment.hpp`
- `src/experiments/coalesced_vs_strided_experiment.cpp`
- `src/experiments/adapters/coalesced_vs_strided_adapter.cpp`
- `shaders/11_coalesced_vs_strided/11_coalesced_vs_strided.comp`
- `experiments/11_coalesced_vs_strided/architecture.md`
- `experiments/11_coalesced_vs_strided/development_plan.md`
- `experiments/11_coalesced_vs_strided/implementation_plan.md`
- `experiments/11_coalesced_vs_strided/scripts/collect_run.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `experiments/11_coalesced_vs_strided/README.md`

## Work Packages
### 1. Runtime Contract and Kernel
- [x] Add Experiment 11 config/output contract.
- [x] Add a parameterized stride field and buffer-span calculation.
- [x] Add CPU reference generation for the logical element order.
- [x] Keep dispatch timing based on GPU timestamps.

### 2. Vulkan Runtime
- [x] Add buffer, descriptor, and pipeline setup/teardown for the strided kernel.
- [x] Keep address math explicit and bounds-safe for every stride.
- [x] Preserve exact comparison semantics for touched elements.

### 3. Correctness and Metrics
- [x] Validate the strided path against CPU reference outputs.
- [x] Emit row-level metrics for each timed iteration and stride.
- [x] Capture logical bytes processed so GB/s remains comparable across strides.

### 4. Workflow Integration
- [x] Add experiment to manifest and build target list.
- [x] Add experiment to the root data-collection helper.
- [ ] Add experiment to the artifact-generation helper.
- [x] Add experiment-local run collection script.
- [ ] Add experiment-local analysis script.

### 5. Verification Gates
- [x] Build release target (`tests-vs-release` preset).
- [x] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [x] Run tests and a benchmark smoke pass for Experiment 11.
