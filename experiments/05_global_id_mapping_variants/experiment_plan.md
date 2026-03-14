# Experiment 05: Global ID Mapping Variants

## 1. Lecture Focus
- Concept: Direct, offset, and grid-stride mapping behavior.
- Why this matters: Teaches scalable indexing patterns used in production kernels with large workloads.
- Central question: How do different indexing formulas affect throughput and scalability?

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
Grid-stride loops improve scalability flexibility but may add loop/control overhead.

## 5. Experimental Design
### Independent variables
Mappings: linear direct, fixed offset, grid-stride loop; Sizes: medium to very large.

### Controlled variables
Same arithmetic and memory footprint, fixed local size selection.

### Workload design
Equivalent math with only index strategy changed.

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
Kernel ms, throughput, instruction-count proxy from shader complexity notes.

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

Direct mapping wins for exact-fit dispatch; grid-stride wins for flexible dispatch sizing.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Unequal total work due to loop bounds, accidental extra bounds checks.

## 11. Deliverables
Mapping strategy comparison with practical guidance.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Adopt a default mapping policy for subsequent complex kernels.


