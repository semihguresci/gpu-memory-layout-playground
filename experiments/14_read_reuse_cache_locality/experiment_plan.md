# Experiment 14: Read Reuse and Cache Locality

## 1. Lecture Focus
- Concept: Temporal locality and reuse-distance effects.
- Why this matters: Teaches why access order can matter as much as algorithmic complexity on GPU.
- Central question: How much benefit is retained as reuse distance increases?

## 2. Learning Objectives
By the end of this experiment, you should be able to:
- explain the performance mechanism behind the studied concept
- design a controlled Vulkan compute benchmark for this concept
- interpret measured results without over-claiming causality
- document practical rules you would apply in production kernels

## 3. Theory Primer (Lecture Notes)
- Start from the execution model: workgroups, waves/warps, and memory transactions.
- Identify whether the kernel is likely memory-bound, latency-bound, or synchronization-bound.
- Predict how this experiment changes transaction shape, locality, pressure, or control-flow efficiency.
- Record assumptions explicitly before measuring so conclusions can be tested, not guessed.

## 4. Hypothesis
Near-term reuse benefits caches; far-distance reuse approaches single-use cost.

## 5. Experimental Design
### Independent variables
Patterns: single-read, near-reuse loop, far-reuse loop with tunable distance.

### Controlled variables
Same total arithmetic and total reads where possible.

### Workload design
Synthetic reuse kernels over large arrays with configurable reuse distance.

## 6. Implementation Plan
1. Implement or select the shader variant(s) that isolate this concept.
2. Wire host-side sweep parameters into CLI/config.
3. Add correctness checks against deterministic CPU reference outputs.
4. Run warmup iterations before measured iterations.
5. Capture raw timing and metadata for every run.
6. Export results in machine-readable format for plotting.

## 7. Measurement Protocol
- Timing source: GPU timestamp queries for dispatch timing.
- Reporting: median as primary, p95 as stability indicator.
- Run policy: multiple repetitions per point, fixed seeds for reproducibility.
- Metadata: GPU model, driver, Vulkan version, OS, compiler flags, shader options.

## 8. Data to Capture
Runtime, reuse-distance sensitivity, relative speedup vs no-reuse baseline.

Recommended columns:
- experiment_id
- variant
- problem_size
- iteration
- gpu_ms
- throughput
- gbps
- correctness_pass
- notes

## 9. Expected Patterns and Interpretation

Performance declines as reuse distance exceeds effective cache residency window.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Accidental compiler caching effects and unintentional loop invariant elimination.

## 11. Deliverables
Locality vs runtime chart and reuse-distance interpretation.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Apply locality insights to ordering strategies in later culling and GPU-driven pipelines.


