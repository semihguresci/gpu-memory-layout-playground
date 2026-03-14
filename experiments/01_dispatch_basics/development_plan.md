# Experiment 01 Dispatch Basics: Codebase Development Plan

Date: 2026-03-14  
Source spec: `docs/experiment_plans/01_dispatch_basics.md`

## 1. Scope and Status
This plan tracks implementation of canonical Experiment 01 (Dispatch Basics) in the current harness.

Current status: core implementation is complete and running.  
Remaining work is primarily operational (more device datasets), not core feature gaps.

## 2. Development Phase Status
### Phase A: Measurement and Result Schema
- [x] Row-level measurement model added:
  - `experiment_id`, `variant`, `problem_size`, `dispatch_count`, `iteration`
  - `gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, `notes`
- [x] Summary stats implemented: average/min/max/median/p95/sample_count
- [x] Run metadata export implemented:
  - GPU name, Vulkan API version, driver version
  - validation flag, timestamp support, warmup, timed iterations

### Phase B: Experiment 01 Runtime
- [x] `dispatch_basics_experiment.hpp/.cpp` integrated into main flow
- [x] Explicit staging path measured (`upload -> dispatch -> readback`)
- [x] Variants implemented: `contiguous_write` and `noop`
- [x] Sweep implemented:
  - problem size `2^10..2^24` (runtime-clamped)
  - dispatch count `{1, 4, 16, 64, 128, 256, 512, 1024}`
- [x] Deterministic correctness checks implemented for both variants

### Phase C: CLI and Registry
- [x] Canonical experiment IDs enforced in CLI/runtime:
  - `all`, `01_dispatch_basics`, `06_aos_vs_soa`
- [x] Legacy aliases removed from CLI path

### Phase D: Legacy Cleanup
- [x] Legacy Experiment 01 naming/paths removed from runtime flow
- [x] Canonical shader/result naming used for active experiments

### Phase E: Artifacts and Analysis
- [x] JSON run output and row-level tables generated in experiment-local folders
- [x] Analysis scripts moved under `experiments/01_dispatch_basics/scripts/`
- [x] Multi-device run collection/aggregation implemented:
  - `collect_run.py` + `runs/<device>/<timestamp>.json`
  - device/run summary tables and charts
- [x] Results and architecture docs added with generated charts
- [x] Validation-enabled smoke run verified (non-fatal external loader warnings observed)

## 3. Acceptance Criteria Check
- [x] `--experiment 01_dispatch_basics` runs end-to-end without manual edits
- [x] Output includes per-iteration rows and median/p95 summaries
- [x] Correctness is explicit for every row and enforced at run level
- [x] Plot inputs and generated charts are produced from exported data
- [x] Legacy experiment aliases are removed from runtime CLI

## 4. Remaining Work
- [ ] Collect additional runs from different GPUs/drivers for true cross-device comparison
- [ ] Add a short reproducibility checklist for contributors (exact run commands, expected artifacts)
- [ ] Optional: add CI smoke checks for `collect_run.py` and `analyze_dispatch_basics.py`

## 5. Conclusion
There are no blocking implementation tasks left for Experiment 01 core functionality.  
What remains is dataset expansion and automation hardening.
