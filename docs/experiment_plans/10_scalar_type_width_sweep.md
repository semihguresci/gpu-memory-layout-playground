# Experiment 10: Scalar Type Width Sweep

## 1. Lecture Focus
- Concept: Precision-width tradeoffs: 32-bit, 16-bit, and narrower storage.
- Why this matters: Bandwidth-limited kernels often gain from narrower storage if conversion overhead stays controlled.
- Central question: When does narrower scalar width provide net speedup versus conversion and precision costs?

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
16-bit storage improves throughput for memory-bound kernels; benefits vary with conversion load.

## 5. Experimental Design
### Independent variables
Types: float32, float16, uint32, uint16 (and uint8 where practical).

### Controlled variables
Equivalent logical operations and normalization rules.

### Workload design
Memory-bound updates and optional compute-heavy variant to isolate conversion cost.

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
Runtime, GB/s, conversion overhead proxy, numerical error statistics.

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

Memory-bound kernels benefit most; compute-heavy kernels may show reduced gains.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Feature support differences and precision-induced correctness drift.

## 11. Deliverables
Type-width comparison table with accuracy-performance tradeoff notes.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use selected widths for large-scale access pattern studies.


