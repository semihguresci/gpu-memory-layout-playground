# Experiment 01: Dispatch Basics

## 1. Lecture Focus
- Concept: Minimal Vulkan compute dispatch, correctness path, and baseline GPU timing.
- Why this matters: Establishes the benchmark harness and proves that measurement infrastructure is trustworthy before optimization work.
- Central question: Can the project reliably execute upload -> dispatch -> readback with validated outputs and stable timestamp-based timings?

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
As problem size and dispatch count grow, throughput should rise until fixed overheads are amortized, then plateau.

## 5. Experimental Design
### Independent variables
Problem sizes: 2^10 to 2^24 elements; Dispatch count per run: 1, 4, 16, 64.

### Controlled variables
Fixed local size, fixed memory layout, same queue, same timestamp query method, validation mode noted.

### Workload design
Contiguous ID write kernel; Optional no-op kernel for overhead estimation.

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
GPU dispatch ms, end-to-end ms, throughput (elements/s), effective bandwidth (GB/s), correctness pass rate.

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

Small dispatches are overhead-bound; larger runs reveal stable throughput plateau.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
CPU-side timing contamination, missing barriers, non-deterministic initialization.

## 11. Deliverables
Baseline table, threads vs throughput chart, harness validation notes.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use this as baseline reference for all later experiments.


