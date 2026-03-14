# Experiment 08: std430 vs std140 vs Packed

## 1. Lecture Focus
- Concept: Shader buffer layout standards and padding cost.
- Why this matters: Critical Vulkan-specific literacy for reliable and efficient storage buffer design.
- Central question: How much bandwidth and capacity is lost to alignment rules under each layout?

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
std430 improves packing over std140 while remaining standards-compliant; packed host-only data can be risky.

## 5. Experimental Design
### Independent variables
Layouts: std140, std430, tightly packed host variant where valid.

### Controlled variables
Equivalent logical fields and operations.

### Workload design
Read/write kernels over structures containing vec3/vec4/scalars.

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
Bytes per element, bandwidth, correctness mismatch incidents, alignment waste ratio.

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

std140 shows highest padding overhead; std430 generally better practical default.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Host-shader struct mismatch and undefined behavior with invalid packed assumptions.

## 11. Deliverables
Alignment diagrams, bytes-per-element table, best-practice recommendations.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use measured alignment costs to inform structure design in later experiments.


