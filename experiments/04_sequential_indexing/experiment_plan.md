# Experiment 04: Sequential Indexing

## 1. Lecture Focus
- Concept: Ideal contiguous thread-to-data mapping as a good-path baseline.
- Why this matters: Defines the best-case memory access reference before testing degraded patterns.
- Central question: What performance is achieved when invocation i accesses element i?

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
Sequential mapping should maximize coalescing and minimize transaction overhead.

## 5. Experimental Design
### Independent variables
Problem sizes and dispatch counts from Exp 01; optional local sizes from Exp 02 top candidates.

### Controlled variables
Same arithmetic, same data type, same buffer alignment.

### Workload design
Contiguous read/write kernels with deterministic indexing.

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
Kernel ms, GB/s, correctness, scalability curve shape.

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

Near-best bandwidth among non-shared-memory versions.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Off-by-one indexing, bounds masking overhead, compiler dead-code elimination.

## 11. Deliverables
Sequential baseline curve and validation notes.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Compare all non-sequential patterns against this baseline.


