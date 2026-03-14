# Experiment 06: AoS vs SoA

## 1. Lecture Focus
- Concept: Array-of-Structures versus Structure-of-Arrays layout efficiency.
- Why this matters: Core GPU memory-layout lesson directly relevant to particle and simulation systems.
- Central question: Which layout yields better memory throughput for component-wise operations?

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
SoA should outperform AoS for per-field vectorized access patterns.

## 5. Experimental Design
### Independent variables
Layouts: AoS, SoA; Sizes: 1M, 5M, 10M+ elements; Access mix: read-heavy vs read+write.

### Controlled variables
Equivalent math and precision, same workgroup settings, same total bytes where possible.

### Workload design
Particle-like update: position += velocity * dt, with optional mass field use.

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
Runtime, effective GB/s, bytes transferred, cache-locality behavior notes.

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

SoA advantage grows with larger datasets and field-selective access.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Unintended padding differences and host packing mismatches.

## 11. Deliverables
AoS vs SoA chart, layout recommendation notes.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Feed preferred layout into AoSoA and padding studies.


