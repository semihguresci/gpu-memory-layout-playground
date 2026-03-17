# Experiment 10 Scalar Type Width Sweep: Runtime Architecture

## Objective
Measure storage-width tradeoffs for equivalent scalar update logic across five representations:
- `fp32`: native float storage
- `fp16_storage`: packed half storage (`2x f16` per `u32`)
- `u32`: `u32` storage carrying float bit patterns
- `u16`: packed normalized fixed-point (`2x u16` per `u32`)
- `u8`: packed normalized fixed-point (`4x u8` per `u32`)

## Data Flow
1. Host allocates one mapped storage buffer per variant.
2. Host seeds deterministic scalar values using variant-specific packing.
3. Compute dispatch runs one update step per logical element.
4. Host validates outputs against variant-specific expected values.
5. Per-iteration rows and case summaries are exported to JSON.

## Vulkan Resources
- Shared context: compute queue and timestamp timer from `VulkanContext`
- Per variant:
  - one storage buffer
  - one descriptor set layout + descriptor set
  - one compute pipeline

## Push Constants
- `count` (logical element count)

For packed variants (`fp16_storage`, `u16`, `u8`), each invocation updates one packed word and handles trailing lanes with bounds checks.

## Measurement Contract
- Primary timing: GPU dispatch ms via timestamp queries
- Supporting timing: end-to-end host iteration time
- Row fields:
  - `experiment_id`
  - `variant`
  - `problem_size`
  - `iteration`
  - `gpu_ms`
  - `throughput`
  - `gbps`
  - `correctness_pass`
  - `notes`

`notes` include:
- `storage_bytes_per_element`
- `storage_ratio_vs_fp32`
- `max_abs_error`
- `mean_abs_error`
- `validation_tolerance`

## Teardown Policy
All Vulkan objects are destroyed in reverse creation order and handles are reset to `VK_NULL_HANDLE`.
