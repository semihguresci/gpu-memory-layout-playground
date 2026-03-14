# Experiment 21: Parallel Reduction

## 1. Lecture Focus
- Concept: Reduction patterns from naive to tree and shared-memory optimized.
- Why this matters: Reduction is a classic primitive reused throughout GPU systems.
- Central question: How much performance is gained by hierarchical reduction strategies?

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
Shared-memory tree reduction outperforms naive global reduction for large arrays.

## 5. Experimental Design
### Independent variables
Implementations: naive global, tree reduction, shared-memory reduction; sizes from small to large.

### Controlled variables
Same input values and final reduction operator.

### Workload design
Sum reduction; optional min/max follow-up for operator sensitivity.

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
Runtime, GB/s-equivalent throughput, correctness against CPU reference.

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

Hierarchical variants show large gains with growing size.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Numerical order differences and non-associative floating-point surprises.

## 11. Deliverables
Implementation comparison table and scaling charts.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use reduction building blocks for scan and compaction pipeline.


