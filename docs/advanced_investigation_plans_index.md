# Advanced Investigation Plans Index (Lecture Notes Track)

This index points to detailed lecture-note style plans for post-25 advanced investigations.

## Recommended Execution Phases
1. Phase A: core primitives (radix sort, subgroup ops, occupancy modeling).
2. Phase B: rendering data systems (tiled assignment, culling pipelines, GPU-driven blocks).
3. Phase C: architecture depth (BVH layout, ray-friendly layouts, temporal coherence).
4. Phase D: systems and platform depth (persistent queues, async overlap, cross-GPU validation).

## Advanced Investigation Plans
- [Advanced Investigation 01: Radix Sort on GPU](advanced_plans/01_radix_sort_gpu.md): Multi-pass radix sorting for key-only and key-value data.
- [Advanced Investigation 02: BVH Node Layout Experiments](advanced_plans/02_bvh_node_layout.md): Node representation impact on traversal efficiency.
- [Advanced Investigation 03: Frustum Culling vs Clustered Culling](advanced_plans/03_frustum_vs_clustered_culling.md): Comparative rendering-style visibility pipelines.
- [Advanced Investigation 04: Tiled Light Assignment](advanced_plans/04_tiled_light_assignment.md): Light-to-tile list construction strategies.
- [Advanced Investigation 05: Persistent Threads and Work Queues](advanced_plans/05_persistent_threads_work_queues.md): Dynamic scheduling for irregular workloads.
- [Advanced Investigation 06: Subgroup Operations Study](advanced_plans/06_subgroup_operations.md): Warp/wave-level intrinsics versus shared-memory patterns.
- [Advanced Investigation 07: Async Compute Overlap](advanced_plans/07_async_compute_overlap.md): Overlapping compute, transfer, and independent stages.
- [Advanced Investigation 08: Occupancy Modeling Against Vendor Guidance](advanced_plans/08_occupancy_modeling.md): Resource-pressure interpretation of measured trends.
- [Advanced Investigation 09: Memory System Study for Ray-Friendly Layouts](advanced_plans/09_ray_friendly_memory_layouts.md): Traversal-friendly layout behavior under coherent/incoherent query patterns.
- [Advanced Investigation 10: Frame-to-Frame Coherence Studies](advanced_plans/10_frame_to_frame_coherence.md): Temporal stability and ordering reuse effects.
- [Advanced Investigation 11: GPU-Driven Pipeline Building Blocks](advanced_plans/11_gpu_driven_pipeline_blocks.md): Compose culling, compaction, bucketing, and argument-like generation.
- [Advanced Investigation 12: Reproducibility and Cross-GPU Comparison](advanced_plans/12_cross_gpu_reproducibility.md): Trend stability across architectures and vendors.
