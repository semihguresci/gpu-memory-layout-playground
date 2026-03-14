# Experiment 09: vec3, vec4, and Padding Costs

## 1. Lecture Focus
- Concept: Impact of vector shape choice on storage efficiency and bandwidth.
- Why this matters: Small type-shape choices frequently create hidden, recurring memory penalties.
- Central question: Is vec3 worth the potential padding overhead compared with vec4 or scalar split arrays?

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
vec4 or split scalars often outperform vec3-based struct layouts in practical bandwidth terms.

## 5. Experimental Design
### Independent variables
Representations: vec3-based struct, vec4-based struct, split scalar arrays.

### Controlled variables
Equivalent semantic workload and arithmetic.

### Workload design
Transform or accumulation kernels with repeated vector loads/stores.

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
Wasted bytes ratio, runtime, GB/s, code complexity note.

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

Layouts avoiding implicit vec3 padding show better effective memory use.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Inconsistent host packing and accidental compiler optimizations removing work.

## 11. Deliverables
Wasted-byte analysis and preferred representation guidance.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Carry shape recommendations into advanced traversal/culling data structures.


