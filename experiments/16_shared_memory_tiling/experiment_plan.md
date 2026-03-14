# Experiment 16: Shared or Workgroup Memory Tiling

## 1. Lecture Focus
- Concept: Staging data in on-chip memory for reuse.
- Why this matters: Fundamental GPU optimization technique for reducing repeated global memory traffic.
- Central question: How much speedup does tiled staging provide relative to direct global reads?

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
Tiling helps when data reuse per tile exceeds barrier/synchronization overhead.

## 5. Experimental Design
### Independent variables
Variants: direct global access vs shared-memory tiled access; tile sizes from feasible range.

### Controlled variables
Equivalent arithmetic and output.

### Workload design
Kernels with reusable neighborhood or repeated local access.

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
Runtime, speedup factor, barrier count impact, shared memory footprint.

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

Positive speedup when reuse is high enough; otherwise overhead may dominate.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Bank conflicts, incorrect barriers, tile boundary handling bugs.

## 11. Deliverables
Speedup chart and barrier-cost commentary.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Follow with tile-size occupancy tradeoff study.


