# Experiment 10 Scalar Type Width Sweep: Codebase Development Plan

Date: 2026-03-17  
Source spec: `docs/experiment_plans/10_scalar_type_width_sweep.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 10 (`10_scalar_type_width_sweep`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime implementation, adapter wiring, and shaders are integrated
- experiment-local run collection script is integrated
- root data collection helper supports Experiment 10
- no measured dataset/report is checked in under `experiments/10_scalar_type_width_sweep/results.md` yet

## 2. Development Phases
### Phase A: Runtime and Correctness
- [x] Implement matched logical variants: `fp32`, `fp16_storage_fp32_compute`, `u32`, `u16`, and optional `u8`.
- [x] Add deterministic CPU reference path and tolerance-aware numeric checks.
- [x] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, error metrics, notes).
- [x] Enforce correctness fail behavior at adapter boundary.

### Phase B: Experiment Tooling
- [x] Add experiment-local `collect_run.py` script.
- [ ] Add experiment-local analysis script for width/accuracy summary tables.
- [ ] Add experiment-local plotting script for throughput and error trends.
- [x] Add results/report scaffolding (`architecture.md`, script docs).

### Phase C: Root Workflow Integration
- [x] Add `10_scalar_type_width_sweep` to `cmake/experiments_manifest.cmake`.
- [x] Add `10_scalar_type_width_sweep` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `10_scalar_type_width_sweep` to `scripts/generate_experiment_artifacts.py`.
- [x] Verify shader auto-compilation resolves experiment 10 shader outputs with unique basenames.

### Phase D: Validation and Hardening
- [x] Configure with `cmake --preset windows-tests-vs`.
- [x] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [x] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [x] Run tests and targeted experiment smoke run.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 10_scalar_type_width_sweep` is discoverable and runnable.
- [x] Row-level output exists for planned width variants.
- [x] Adapter reports failure on correctness mismatch.
- [ ] Root data collection and artifact-generation scripts support Experiment 10.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
