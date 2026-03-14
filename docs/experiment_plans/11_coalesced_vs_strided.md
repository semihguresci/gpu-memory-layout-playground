# Experiment 11: Coalesced vs Strided Access

## 1. Lecture Focus
- Concept: Contiguous and strided load behavior.
- Why this matters: One of the most important GPU memory lessons for real kernels.
- Central question: How rapidly does performance degrade as stride increases?

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
Throughput drops nonlinearly with larger stride due to increased memory transactions and cache miss rate.

## 5. Experimental Design
### Independent variables
Stride: 1, 2, 4, 8, 16, 32, 64.

### Controlled variables
Same arithmetic and element count; same local size and type.

### Workload design
Indexed reads/writes with stride-scaled global index.

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
Runtime per stride, GB/s, relative efficiency vs stride 1 baseline.

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

Sharp decline beyond cacheline-friendly strides.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Out-of-bounds indexing and mismatch in total touched bytes across strides.

## 11. Deliverables
Stride vs performance chart and interpretation notes.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use strided data points to contextualize gather/scatter penalties.


