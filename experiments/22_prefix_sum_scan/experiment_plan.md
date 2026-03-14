# Experiment 22: Prefix Sum or Scan

## 1. Lecture Focus
- Concept: Inclusive/exclusive scan as a foundational parallel primitive.
- Why this matters: Enables compaction, sorting, culling lists, and allocator-style workflows.
- Central question: How does scan performance scale, and what are correctness-sensitive edge cases?

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
Workgroup-local scan plus hierarchical merge scales better than naive approaches.

## 5. Experimental Design
### Independent variables
Inclusive vs exclusive; block sizes; dataset sizes; distribution of flags.

### Controlled variables
Same data types and deterministic input seeds.

### Workload design
Integer scan and optional flag-scan variant for compaction prep.

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
Runtime, throughput, correctness checks on boundary cases.

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

Hierarchical approach dominates at larger sizes.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Off-by-one errors and block prefix carry propagation bugs.

## 11. Deliverables
Scan scaling chart and correctness test report.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Integrate scan into histogram and stream compaction workflows.


