# Experiment 25: Spatial Binning or Clustered Culling Capstone

## 1. Lecture Focus
- Concept: Rendering-style compute pipeline combining prior primitives.
- Why this matters: Demonstrates end-to-end systems thinking with practical graphics relevance.
- Central question: Which list-construction strategy gives best performance across scene distributions?

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
Workgroup-local staging plus compaction should outperform naive global append under contention.

## 5. Experimental Design
### Independent variables
Variants: naive append, local staging, compacted lists, sorted/coherent vs unsorted input.

### Controlled variables
Same scene generator, same object/light counts per test level.

### Workload design
Bin assignment and list build pipeline under sparse, dense, clustered scenes.

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
Total runtime, stage timing table, contention proxies, memory traffic estimate.

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

No single winner across all distributions; data coherence should strongly affect results.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Non-deterministic ordering, overflow handling in list buffers, synchronization bugs.

## 11. Deliverables
Full performance table, charts, pipeline diagram, lessons learned mapping to Exp 01-24.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use capstone conclusions to choose advanced follow-up investigations.


