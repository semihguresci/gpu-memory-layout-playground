# Experiment 15: Bandwidth Saturation Sweep

## 1. Lecture Focus
- Concept: Scaling data volume until practical bandwidth plateau.
- Why this matters: Establishes hardware limits and identifies size regimes where overhead dominates.
- Central question: At what problem size does achieved GB/s stabilize near a sustained plateau?

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
Small sizes are overhead-bound; larger sizes converge to device-specific sustained bandwidth.

## 5. Experimental Design
### Independent variables
Data sizes: 1 MB, 4 MB, 16 MB, 64 MB, 256 MB, 1 GB (if feasible).

### Controlled variables
Fixed kernel type and access pattern.

### Workload design
Copy-like and read+write kernels from prior baselines.

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
GB/s vs size, plateau onset, variability at large sizes.

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

Clear saturation knee in mid-to-large size range.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Memory allocation fragmentation and thermal throttling across long runs.

## 11. Deliverables
Saturation curve, knee-point estimate, practical workload guidance.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use plateau region settings for architecture-aware optimization studies.


