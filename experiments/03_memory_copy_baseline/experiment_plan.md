# Experiment 03: Memory Copy Baseline

## 1. Lecture Focus
- Concept: Raw buffer read/write/copy throughput characterization.
- Why this matters: Provides the roofline-like memory baseline for interpreting later algorithmic kernels.
- Central question: How close can simple kernels get to practical memory throughput limits?

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
Read+write copy will be bounded by memory bandwidth and approach a stable GB/s region at scale.

## 5. Experimental Design
### Independent variables
Modes: read-only, write-only, read+write copy; Sizes: 1 MB to 1 GB.

### Controlled variables
Uniform access pattern, fixed local size (from Exp 02), fixed data type.

### Workload design
Kernel variants with isolated memory behavior.

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
GB/s per mode, kernel ms, bytes moved, saturation point by size.

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

Write-only and read+write can differ from read-only depending on cache/write-combine behavior.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Miscounting bytes moved, hidden host-device transfer time inclusion.

## 11. Deliverables
Mode comparison table, GB/s vs size chart, saturation threshold estimate.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use this baseline as denominator when reporting efficiency in later experiments.


