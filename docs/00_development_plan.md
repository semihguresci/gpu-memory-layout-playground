# Development Plan

## Scope
Build a Vulkan compute benchmark suite to study GPU memory layout and cache behavior through six reproducible experiments.

## Phase 1: Foundation
- [x] Finalize base Vulkan compute pipeline (instance, device, queue, command pool, command buffers).
- [x] Add reusable helpers for shader module loading, descriptor set creation, pipeline creation, and timing.
- [x] Add result writer utilities (JSON exporter + schema versioning).
- [x] Add CLI options: `--experiment`, `--iterations`, `--warmup`, `--size`, `--output`.
- [x] Add validation-layer toggle for debug builds.

## Progress Update (2026-03-12)
- Implemented Vulkan debug utilities support via `VK_EXT_debug_utils` and validation layer integration.
- Added runtime toggle `--validation` and build toggle `ENABLE_VULKAN_VALIDATION`.
- Added command pool/buffer + fence setup in `VulkanContext`.
- Added timestamp query pool support and a `measureGpuTimeMs(...)` helper for GPU-side timing.
- Integrated timed benchmark path (`BenchmarkRunner::runTimed`) to consume GPU timing data.
- Added CLI support for experiment selection, iteration/warmup control, buffer size control, and output path override.
- Migrated CLI parsing to `CLI11` and moved parsing into `ArgumentParser` utility (`app_options.*`).
- Added `nlohmann/json` integration and switched benchmark output to JSON-only.
- Added `JsonExporter` utility class and removed JSON formatting logic from `main.cpp`.
- Added JSON schema versioning with top-level `schema` and `metadata` blocks.
- Added `VulkanComputeUtils` helper module for shader loading, descriptor/pipeline setup, and reusable GPU timestamp timing helpers.
- Enforced GPU timestamp timing path in benchmark execution (no CPU fallback).

## Phase 2: Experiment Implementation
- [ ] Implement experiment 01 thread mapping baseline and throughput scaling.
- [ ] Implement experiment 02 AoS vs SoA memory layout comparison.
- [ ] Implement experiment 03 std430 vs packed alignment overhead.
- [ ] Implement experiment 04 coalesced vs strided access patterns.
- [ ] Implement experiment 05 shared memory tiling optimization.
- [ ] Implement experiment 06 bandwidth saturation against theoretical limits.

## Phase 3: Analysis + Reporting
- [ ] Add Python plotting scripts for per-experiment charts.
- [ ] Add summary report generator (markdown) from JSON result files.
- [ ] Add reproducibility metadata capture (GPU name, driver, Vulkan version, OS, timestamp).

## Phase 4: Quality
- [ ] Add smoke test that runs small problem sizes for all experiments.
- [ ] Add CI build for Windows with Vulkan SDK setup notes.
- [ ] Add guardrails for invalid parameter combinations and out-of-memory conditions.

## Milestones
1. M1 (Foundation): CLI + reusable Vulkan compute framework running one shader end-to-end.
2. M2 (Core Experiments): Experiments 01-04 produce stable JSON outputs.
3. M3 (Optimization Experiments): Experiments 05-06 integrated with plots.
4. M4 (Publishable Results): Analysis docs and final charts complete.

## Definition of Done
- All six experiments run from CLI.
- Each experiment outputs machine-readable JSON results.
- Charts are generated from raw results without manual editing.
- Docs include methodology, assumptions, and limitations.
