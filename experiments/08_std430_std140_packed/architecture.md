# Experiment 08 std430 vs std140 vs Packed: Runtime Architecture

## Objective
Benchmark three layout variants under equivalent per-particle update logic:
- `std140` (highest alignment pressure)
- `std430` (reduced alignment overhead)
- `packed` (tight host-defined float stream with explicit indexing)

Logical payload:
- 16 floats per particle
- Variant storage strides:
  - `std140`: 112 bytes
  - `std430`: 80 bytes
  - `packed`: 64 bytes

## Data Flow
1. Host allocates and maps one storage buffer per variant.
2. Host seeds deterministic particle state in variant-specific memory layout.
3. Compute dispatch executes one update step for each particle.
4. Host validates outputs against deterministic CPU expected values.
5. Row metrics and summary statistics are exported to benchmark JSON.

## Vulkan Resources
- Shared context: one compute queue, timestamp timer, and command buffer from `VulkanContext`.
- Per variant (`std140`, `std430`, `packed`):
  - one storage buffer
  - one descriptor set layout + descriptor set
  - one compute pipeline

## Push Constants
- All three variants use the same push-constant payload:
  - `count`

## Measurement Contract
- Primary timing: GPU dispatch ms from timestamp queries.
- Supporting timing: end-to-end host time per iteration.
- Row schema includes:
  - `experiment_id`
  - `variant`
  - `problem_size`
  - `iteration`
  - `gpu_ms`
  - `throughput`
  - `gbps`
  - `correctness_pass`
  - `notes`

Notes include layout-specific context:
- `storage_bytes_per_particle`
- `logical_bytes_per_particle`
- `alignment_waste_ratio`

## Teardown Policy
All Vulkan objects are destroyed in reverse creation order and handles are reset to `VK_NULL_HANDLE`.
