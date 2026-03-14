# Experiment 17: Tile Size Sweep

## 1. Lecture Focus
- Concept: Tradeoff between reuse, shared-memory pressure, and occupancy.
- Why this matters: Shows that optimization requires parameter tuning, not one-size-fits-all patterns.
- Central question: Which tile size provides best overall performance for this workload and GPU?

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
Intermediate tile sizes balance reuse gains and occupancy/resource constraints.

## 5. Experimental Design
### Independent variables
Tile sizes: 8, 16, 32, 64 (problem-dependent legal values).

### Controlled variables
Same tiled algorithm and input distribution.

### Workload design
Shared-memory kernel from Exp 16 with tunable tile dimensions.

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
Runtime, speedup vs direct baseline, shared-memory bytes/workgroup, occupancy proxy.

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

Convex-like curve with a best region rather than monotonic behavior.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Illegal shared-memory usage and launch configuration mismatches.

## 11. Deliverables
Tile-size performance map and selected default configuration.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use selected tile size in reduction/scan primitives where relevant.


