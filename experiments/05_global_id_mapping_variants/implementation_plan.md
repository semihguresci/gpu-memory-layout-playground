# Experiment 05 Global ID Mapping Variants: Implementation Plan

Date: 2026-03-15

## Goal
Implement a correctness-first Vulkan compute experiment that compares three global index mapping strategies:
- direct mapping
- fixed-offset mapping
- grid-stride mapping

## Scope
- Experiment ID: `05_global_id_mapping_variants`
- Size sweep: powers of two from `2^10` to `2^24` (runtime-clamped)
- Local size: `256`
- Dispatch count sweep: `1, 4, 16, 64, 128, 256, 512, 1024`
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)

## File Changes
New files:
- `include/experiments/global_id_mapping_variants_experiment.hpp`
- `src/experiments/global_id_mapping_variants_experiment.cpp`
- `src/experiments/adapters/global_id_mapping_variants_adapter.cpp`
- `shaders/05_global_id_mapping_variants.comp`
- `experiments/05_global_id_mapping_variants/architecture.md`
- `experiments/05_global_id_mapping_variants/development_plan.md`
- `experiments/05_global_id_mapping_variants/implementation_plan.md`
- `experiments/05_global_id_mapping_variants/results.md`
- `experiments/05_global_id_mapping_variants/scripts/README.md`
- `experiments/05_global_id_mapping_variants/scripts/collect_run.py`
- `experiments/05_global_id_mapping_variants/scripts/analyze_global_id_mapping_variants.py`
- `experiments/05_global_id_mapping_variants/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `experiments/05_global_id_mapping_variants/README.md`

## Work Packages
### 1. Runtime Contract and Entry Points
- [x] Add public experiment contract header.
- [x] Add adapter integration for generated registry.
- [x] Register experiment manifest entry.

### 2. Vulkan Runtime
- [x] Implement explicit resource structs for buffers and pipeline state.
- [x] Keep synchronization local and explicit around upload, dispatch, and readback.
- [x] Destroy Vulkan resources in reverse creation order.

### 3. Execution and Validation
- [x] Implement direct, offset, and grid-stride mapping paths.
- [x] Add deterministic input initialization and CPU validation.
- [x] Add per-row notes and correctness flags.

### 4. Measurement and Data Export
- [x] Collect dispatch GPU timing via timestamp queries.
- [x] Collect end-to-end timing for supporting context.
- [x] Emit row-level throughput and effective GB/s.
- [x] Summarize each case with median/p95 through `BenchmarkRunner`.

### 5. Experiment-local Tooling
- [x] Add run collection script.
- [x] Add analysis script for CSV/chart generation.
- [x] Add quick-plot script from current benchmark JSON.

### 6. Verification and Quality Gates
- [ ] Build `gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` on modified translation units when compile database is available.
- [ ] Run tests (`ctest`) for regressions.

## Run Commands
Build:
```powershell
cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments
```

Collect experiment data:
```powershell
python scripts/run_experiment_data_collection.py --experiment 05_global_id_mapping_variants --iterations 10 --warmup 3 --size 64M
```

Generate experiment-local artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 05_global_id_mapping_variants --collect-run
```
