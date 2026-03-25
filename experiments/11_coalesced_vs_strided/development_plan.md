# Experiment 11 Coalesced vs Strided Access: Codebase Development Plan

Date: 2026-03-25
Source spec: `docs/experiment_plans/11_coalesced_vs_strided.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 11 (`11_coalesced_vs_strided`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime implementation, adapter wiring, and shader are present
- experiment-local run collection script is present
- experiment-local analysis/plotting scripts have not been added yet
- no measured dataset/report is checked in under `experiments/11_coalesced_vs_strided/results.md` yet

## 2. Development Phases
### Phase A: Runtime Contract and Correctness
- [x] Decide whether the benchmark uses one parameterized shader with a stride push constant or explicit coalesced/strided shader variants.
- [x] Define the logical element count, physical buffer span, and addressing formula so stride changes only memory access shape.
- [x] Add deterministic CPU reference output and correctness checks for the touched logical elements.
- [x] Define how padding/untouched buffer regions are handled during validation.
- [x] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, notes).

### Phase B: Experiment Tooling
- [x] Add experiment-local `collect_run.py` script.
- [ ] Add experiment-local analysis/plotting script for stride vs throughput trends.
- [ ] Add results/report scaffolding (`results.md`, script docs, and optional architecture notes).

### Phase C: Root Workflow Integration
- [x] Add `11_coalesced_vs_strided` to `cmake/experiments_manifest.cmake`.
- [x] Add `11_coalesced_vs_strided` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `11_coalesced_vs_strided` to `scripts/generate_experiment_artifacts.py`.
- [x] Verify shader auto-compilation resolves experiment 11 shader outputs with unique basenames.

### Phase D: Validation and Hardening
- [x] Configure with `cmake --preset windows-tests-vs`.
- [x] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [x] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [x] Run tests and a benchmark smoke run.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 11_coalesced_vs_strided` is discoverable and runnable.
- [x] Row-level output exists for the planned stride variants and includes logical size metadata.
- [x] Adapter reports failure on correctness mismatch.
- [ ] Root data collection and artifact-generation scripts support Experiment 11.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
