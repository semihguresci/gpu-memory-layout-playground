# Development Principles and Plan

## 1. Course Framing
This repository is organized as a staged GPU systems curriculum.

Core claim of the project:
- You can reason about GPU execution behavior.
- You can design controlled Vulkan compute benchmarks.
- You can extract architecture-aware conclusions from measured data.

The learning progression is intentional:
1. Build reliable measurement infrastructure.
2. Isolate one performance concept per experiment.
3. Connect micro-level results to system-level design decisions.

## 2. Development Principles
The project follows these implementation rules across all experiments and utilities:
- Write clear C++20 code with explicit ownership and minimal hidden behavior.
- Keep code warning-clean on MSVC (`/W4`) and GCC/Clang (`-Wall -Wextra -Wpedantic`).
- Keep APIs small and focused; prefer composition over deep inheritance.
- Check every Vulkan call that returns `VkResult` and log actionable failures.
- Destroy Vulkan resources in reverse creation order and reset handles to `VK_NULL_HANDLE`.
- Keep synchronization explicit and local to the operation being measured.
- Keep benchmark output reproducible, machine-readable, and tied to environment metadata.

## 3. Program Structure
The full program is split into two tracks.

### Core Track (Experiments 01-25)
- Purpose: build first-principles understanding of execution, memory layout, access behavior, synchronization, and parallel primitives.
- Detailed plans: `docs/09_core_experiment_plans_index.md`

### Advanced Track (Post-25 Investigations)
- Purpose: transition from benchmark literacy to rendering and GPU systems architecture studies.
- Detailed plans: `docs/10_advanced_investigation_plans_index.md`

## 4. Current Status Snapshot (2026-03-14)
Completed foundation capabilities:
- Vulkan compute setup and reusable helpers
- GPU timestamp-based timing path
- CLI benchmark controls and JSON export
- Experiment 01 implementation path

Planning status:
- Core experiment plans 01-25: complete (detailed lecture notes)
- Advanced investigation plans 01-12: complete (detailed lecture notes)

## 5. Core Syllabus Map (01-25)
### Level 1: Execution and Compute Foundations (01-05)
Learning goal:
- move from "code runs" to "timing and correctness are trustworthy"

### Level 2: Memory Layout Fundamentals (06-10)
Learning goal:
- understand how data representation choices map to bandwidth efficiency

### Level 3: Access Patterns and Cache Behavior (11-15)
Learning goal:
- connect indexing/order patterns to transaction shape and locality

### Level 4: On-Chip Memory, Occupancy, and Control Flow (16-20)
Learning goal:
- reason about reuse, pressure, divergence, and synchronization costs

### Level 5: Parallel Primitives and System Building Blocks (21-24)
Learning goal:
- implement and analyze reductions, scans, atomics, and compaction

### Level 6: Capstone Pipeline (25)
Learning goal:
- combine prior concepts into a rendering-adjacent spatial binning or clustered culling system

## 6. Lecture-Note Experiment Requirements
Every experiment plan should include:
- lecture focus and theory primer
- explicit hypothesis before measurement
- controlled variable design
- benchmark protocol and metadata policy
- analysis prompts and failure modes
- minimum artifact set and follow-up link

## 7. Benchmark Methodology Rules
Measurement integrity rules:
- use GPU timestamp queries for kernel timing
- keep warmup and measured runs separate
- report median and p95 where relevant
- store raw machine-readable outputs
- validate correctness before performance comparisons
- capture environment metadata for reproducibility

Interpretation rules:
- distinguish overhead-bound vs steady-state behavior
- avoid claiming causality from one uncontrolled comparison
- label architecture-specific findings explicitly
- report limitations and confounders

## 8. Milestones
1. M1: Foundation harness validated with one implemented experiment.
2. M2: Experiments 01-10 implemented with charts and reproducibility metadata.
3. M3: Experiments 11-20 implemented with cache/control-flow analysis.
4. M4: Experiments 21-25 implemented with capstone report.
5. M5: Advanced investigations prioritized and selected subset executed.
6. M6: Cross-GPU validation pass for representative benchmark subset.

## 9. Definition of Done (Program-Level)
Core track is done when:
- all 25 experiments run from CLI
- each experiment has correctness checks and raw outputs
- plots can be regenerated from raw data
- each experiment has a concise written analysis with caveats

Extended track is done when:
- selected advanced investigations produce comparative charts and conclusions
- claims are scoped to measured evidence and hardware context
- cross-GPU notes separate robust trends from architecture-specific behavior

## 10. Planning Documents
- `docs/09_core_experiment_plans_index.md`: detailed plans for core experiments 01-25
- `docs/experiment_plans/`: lecture-note files for each core experiment plan
- `docs/08_advanced_investigations_roadmap.md`: advanced roadmap overview and execution phases
- `docs/10_advanced_investigation_plans_index.md`: detailed plans for advanced investigations 01-12
- `docs/advanced_plans/`: lecture-note files for each advanced investigation plan
