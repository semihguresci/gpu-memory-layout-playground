# Experiment 02: Local Size Sweep

## 1. Lecture Focus
- Concept: Workgroup sizing and execution efficiency tradeoffs.
- Why this matters: Introduces occupancy-aware thinking and reveals architecture-specific sweet spots.
- Central question: Which local_size_x values maximize throughput for this kernel and GPU?

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
Very small groups underutilize hardware; very large groups increase pressure; middle values perform best.

## 5. Experimental Design
### Independent variables
local_size_x: 32, 64, 128, 256, 512, 1024 (where legal).

### Controlled variables
Same kernel logic, same data size, same memory access pattern, identical repetitions.

### Workload design
Simple contiguous read-modify-write kernel over large buffers.

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
GPU kernel ms, throughput, occupancy proxy metrics (active groups estimate), variance across runs.

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

One or two local sizes dominate across most buffer sizes.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Device limits for workgroup size, subgroup-size assumptions, compiler unrolling differences.

## 11. Deliverables
Local size vs runtime chart, local size vs throughput chart, configuration recommendation.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Carry best local sizes into subsequent experiments unless a specific study overrides them.


