# Experiment 19: Branch Divergence

## 1. Lecture Focus
- Concept: Control-flow divergence within warp or wave execution.
- Why this matters: Execution-model fundamental that explains many real workload slowdowns.
- Central question: What is the runtime penalty for uniform, alternating, and random branch patterns?

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
Uniform branches perform best; random divergence creates highest execution inefficiency.

## 5. Experimental Design
### Independent variables
Branch patterns: all-true/all-false, alternating, random with adjustable probability.

### Controlled variables
Same arithmetic in both branches and identical memory access volume.

### Workload design
Branch-heavy kernels with controlled predicate distribution.

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
Runtime, divergence sensitivity curve, variance across seeds for random case.

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

Penalty increases with branch entropy and intra-wave disagreement.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Unequal work between branches and hidden compiler branch flattening.

## 11. Deliverables
Divergence penalty chart and practical branching guidelines.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use divergence findings when designing culling and queue-based workloads.


