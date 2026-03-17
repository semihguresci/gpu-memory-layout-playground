# Experiment 10: Scalar Type Width Sweep

## 1. Lecture Focus
- Concept: precision-width tradeoffs across `32-bit`, `16-bit`, and `8-bit` storage paths.
- Why this matters: memory-bound kernels often benefit from narrower storage, but conversion/packing overhead can erase gains.
- Central question: where is the crossover point where narrower storage becomes a net win?

## 2. Learning Objectives
By the end of this experiment, you should be able to:
- explain the performance mechanism behind the studied concept
- design a controlled Vulkan compute benchmark for this concept
- define a correctness tolerance policy for reduced-precision variants
- document feature/support assumptions that affect portability
- interpret measured results without over-claiming causality
- document practical rules you would apply in production kernels

## 3. Theory Primer (Lecture Notes)
- Start from the execution model: workgroups, waves/warps, and memory transactions.
- Separate two costs explicitly:
  - storage traffic cost (bytes moved)
  - conversion/packing cost (ALU + bit-manipulation work)
- Identify whether the kernel is likely memory-bound, latency-bound, or synchronization-bound.
- Predict how narrower width changes:
  - transaction count and payload size
  - register pressure due to unpack/pack logic
  - sensitivity to edge handling for packed lanes (`2x` or `4x` per word)
- Record assumptions explicitly before measuring so conclusions can be tested, not guessed.

## 4. Hypothesis
- For large enough problem sizes, narrower storage (`u16`, `fp16_storage`, `u8`) should reduce dispatch time versus `fp32`.
- `u32` (same width as `fp32`) should track `fp32` unless representation/conversion overhead differs.
- If the workload becomes conversion-dominated, speedup from narrower storage will flatten or reverse.

## 5. Experimental Design
### Independent variables
- Data representation variant:
  - `fp32`: direct float baseline
  - `fp16_storage`: packed half storage (`2x f16` in one `u32`)
  - `u32`: 32-bit integer storage path
  - `u16`: packed normalized fixed-point (`2x u16` in one `u32`)
  - `u8`: packed normalized fixed-point (`4x u8` in one `u32`)
- Problem size sweep:
  - preferred: `131072`, `262144`, `524288`, `1048576`, `2097152`, `4194304`, `8388608`, `16777216` (`2^24`)
  - fallback: smaller powers of two if scratch memory is limited

### Dependent variables
- `gpu_ms` (median and p95)
- `throughput` (elements/s)
- `gbps` (effective storage GB/s)
- `max_abs_error` and `mean_abs_error`
- `correctness_pass_rate`

### Controlled variables
- equivalent logical update math across variants
- fixed dispatch policy (`dispatch_count=1`)
- fixed local size (`local_size_x=256`)
- same warmup/timed iteration counts for all variants in one run
- identical host/device setup (queue, timestamp source, memory property flags)

### Workload design
- Primary workload (required): memory-dominant scalar update with one update step per logical element.
- Optional extension (later): a compute-heavier update kernel to isolate conversion overhead from bandwidth effects.

### Feature and portability policy
- Prefer packed representations that do not require enabling optional 8/16-bit storage features.
- If a variant cannot run on a device, report it explicitly in row `notes` and exclude it from relative comparisons.

## 6. Implementation Plan
1. Implement five shader variants (`fp32`, `fp16_storage`, `u32`, `u16`, `u8`) with one common logical update contract.
2. Use deterministic host seeding and deterministic CPU expected-value generation per variant.
3. Implement variant-specific pack/unpack and tolerance-aware correctness checks.
4. Record per-iteration rows including error metrics in `notes`.
5. Run warmup before timed iterations for every variant/size point.
6. Export machine-readable outputs (`benchmark_results.json`, summary CSV).
7. Generate chart artifacts under `experiments/10_scalar_type_width_sweep/results/charts/`.
8. Write/update `experiments/10_scalar_type_width_sweep/results.md` with measured values and limits.

## 7. Measurement Protocol
- Timing source: GPU timestamp queries for dispatch timing.
- Build/run policy:
  - configure: `cmake --preset windows-tests-vs`
  - build: `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`
  - run: `python scripts/run_experiment_data_collection.py --experiment 10_scalar_type_width_sweep --binary <path>`
- Reporting:
  - median `gpu_ms` as primary comparison
  - p95 `gpu_ms` as stability/outlier indicator
- Repetitions:
  - minimum `warmup=2`, `timed=5`
  - preferred for stronger claims: `timed>=10`
- Metadata:
  - GPU model, driver, Vulkan API version
  - validation state
  - exported timestamp and run id
  - warmup/timed iteration counts

## 8. Data to Capture
Runtime, bandwidth, and numerical error statistics for every variant and problem size.

Recommended columns:
- experiment_id
- variant
- problem_size
- dispatch_count
- iteration
- gpu_ms
- end_to_end_ms
- throughput
- gbps
- correctness_pass
- notes

Recommended note fields:
- storage_bytes_per_element
- storage_ratio_vs_fp32
- validation_tolerance
- max_abs_error
- mean_abs_error

## 9. Expected Patterns and Interpretation
- At larger sizes, expect faster `gpu_ms` for narrower storage if memory movement dominates.
- Expect `u32` and `fp32` to be close in dispatch time (same nominal width).
- If `gbps` is similar but `gpu_ms` differs, investigate conversion and scheduling overhead.
- If `p95` diverges from median, treat that point as unstable and avoid strong conclusions.

Interpretation checklist:
- confirm correctness before comparing performance
- separate overhead-bound region from steady-state region
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- highlight both absolute and normalized deltas
- call out architecture-specific behavior explicitly (do not generalize from one GPU)

## 10. Common Failure Modes
- Hidden variant mismatch (non-equivalent update logic between shaders).
- Packed-lane edge bugs for odd element counts (`u16`/`u8` tail handling).
- Overly strict tolerance causing false negatives on reduced precision.
- Overly loose tolerance masking real correctness regressions.
- Outlier-driven conclusions from too few timed samples.
- Stale binary or stale JSON used for report generation.

## 11. Deliverables
Type-width comparison report with accuracy/performance tradeoff notes.

Minimum artifact set:
- one raw results export (`benchmark_results.json`)
- one summary table (`scalar_type_width_sweep_summary.csv`)
- at least two charts (runtime and bandwidth trends)
- one `results.md` section with:
  - run/test status
  - hardware/config metadata
  - measured key values
  - limitations and caveats

## 12. Follow-Up Link
Use selected widths from this experiment to parameterize:
- Experiment 11 (`coalesced_vs_strided`)
- Experiment 12 (`gather_access_pattern`)
- Experiment 13 (`scatter_access_pattern`)
- Experiment 15 (`bandwidth_saturation_sweep`)

Carry forward both performance and error envelopes when choosing a production default width.

## 13. Execution Matrix
Run every valid `variant x problem_size` combination with identical run settings.

| Variant | 131072 | 262144 | 524288 | 1048576 | 2097152 | 4194304 | 8388608 | 16777216 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fp32` | required | required | required | required | required | required | required | required |
| `u32` | required | required | required | required | required | required | required | required |
| `fp16_storage` | required if supported | required if supported | required if supported | required if supported | required if supported | required if supported | required if supported | required if supported |
| `u16` | required | required | required | required | required | required | required | required |
| `u8` | required | required | required | required | required | required | required | required |

If a size cannot run due to memory limits, record the skipped point in `results.md` with the exact reason.

## 14. Success Criteria
- Functional:
  - all required variant/size points produce output rows
  - `correctness_pass` is true for all accepted rows
- Measurement quality:
  - at least `5` timed iterations per accepted row
  - timestamp query results are valid and non-negative
  - no single outlier dominates median/p95 interpretation
- Reporting quality:
  - `results.md` references existing chart/table files only
  - every performance claim includes a numeric value and comparison basis

## 15. Validation Checklist
Before finalizing conclusions, verify:
- `fp32` baseline matches CPU reference within strict tolerance.
- `u32` tracks `fp32` behavior for both runtime trend and correctness.
- packed variants (`fp16_storage`, `u16`, `u8`) use correct lane extraction/insertion for tail elements.
- reported `throughput` and `gbps` are computed from the same `gpu_ms` source used in charts.
- summary CSV row count matches the number of accepted measurements.

## 16. Artifact and Naming Conventions
- Raw run output:
  - `experiments/10_scalar_type_width_sweep/results/benchmark_results.json`
- Aggregated tables:
  - `experiments/10_scalar_type_width_sweep/results/scalar_type_width_sweep_summary.csv`
- Charts (minimum):
  - `experiments/10_scalar_type_width_sweep/results/charts/runtime_vs_size.png`
  - `experiments/10_scalar_type_width_sweep/results/charts/bandwidth_vs_size.png`
- Optional charts:
  - `experiments/10_scalar_type_width_sweep/results/charts/error_vs_size.png`

Use deterministic file names so regenerated artifacts replace previous outputs cleanly.


