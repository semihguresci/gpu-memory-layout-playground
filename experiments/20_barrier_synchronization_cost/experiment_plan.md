# Experiment 20: Barrier and Synchronization Cost

## 1. Lecture Focus
- Concept: Synchronization overhead characterization.
- Why this matters: Barriers are necessary for correctness but expensive when overused.
- Central question: How does barrier count and placement affect runtime across kernels?

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
Runtime rises with barrier frequency; impact depends on workgroup size and per-phase work.

## 5. Experimental Design
### Independent variables
Barrier counts: 0, 1, 2, 4, 8; placements: flat loop vs tiled regions.

### Controlled variables
Equivalent arithmetic and memory work per total iteration.

### Workload design
Synthetic phased kernels with tunable barrier insertion.

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
Runtime, barrier-cost slope, interaction with local size/tile size.

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

Near-linear overhead trend until other bottlenecks dominate.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Incorrect synchronization semantics and missing memory scope correctness.

## 11. Deliverables
Barrier count vs runtime chart and synchronization policy recommendations.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Carry barrier budget assumptions into reduction/scan/histogram designs.


