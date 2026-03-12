# GPU Memory Layout & Cache Behavior

**Repository:** `gpu-memory-layout-experiments`\
**Goal:** GPU execution, memory hierarchy, and Vulkan compute optimization through reproducible
experiments.

------------------------------------------------------------------------

# Overview

This project investigates **how GPU hardware actually executes code** by
measuring the performance impact of memory layout and access patterns
using **Vulkan compute shaders**.

The project is designed as a **mini research paper + benchmark
framework**, producing charts, performance tables, and documented
conclusions.

------------------------------------------------------------------------

# Key GPU Concepts Covered

  Concept               What It Demonstrates
  --------------------- ---------------------------------
  GPU execution model   Warps, thread groups, occupancy
  Memory layout         AoS vs SoA
  Memory alignment      std430 vs packed
  Memory access         coalesced vs strided
  Cache behavior        bandwidth utilization
  On‑chip memory        shared memory tiling
  Throughput analysis   measuring real GPU bandwidth

------------------------------------------------------------------------

# Repository Structure

    gpu-memory-layout-experiments
    │
    ├─ docs/
    │   01_gpu_execution_model.md
    │   02_memory_layouts.md
    │   03_cache_behavior.md
    │
    ├─ experiments/
    │   01_thread_mapping/
    │   02_aos_vs_soa/
    │   03_std430_vs_packed/
    │   04_coalesced_vs_strided/
    │   05_shared_memory_tiling/
    │   06_bandwidth_saturation/
    │
    ├─ shaders/
    │
    ├─ src/
    │   benchmark_runner.cpp
    │   vulkan_context.cpp
    │   buffer_utils.cpp
    │
    ├─ scripts/
    │   plot_results.py
    │
    ├─ results/
    │   charts/
    │   tables/
    │
    └─ README.md

------------------------------------------------------------------------

# Experiment 1 --- GPU Execution Model

### Goal

Understand how compute threads map to hardware execution groups.

### Shader

``` glsl
#version 450

layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer Data {
    float values[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    values[id] = id;
}
```

### Tests

Dispatch sizes:

    1 thread
    32 threads
    64 threads
    256 threads
    1024 threads

### Output

Chart:

    Threads vs Throughput

### Expected Insight

GPUs execute threads in groups (warps). Performance increases when
dispatch sizes match warp sizes.

------------------------------------------------------------------------

# Experiment 2 --- AoS vs SoA

### Structures

AoS:

``` cpp
struct Particle {
    float x;
    float y;
    float z;
    float mass;
};
```

SoA:

``` cpp
struct Particles {
    float* x;
    float* y;
    float* z;
    float* mass;
};
```

### Shader Pattern

    pos[i] += vel[i] * dt;

### Measurement

    1M particles
    5M particles
    10M particles

### Deliverable

  Layout   Time   Bandwidth
  -------- ------ -----------
  AoS             
  SoA             

### Expected Result

Structure‑of‑Arrays improves memory coalescing.

------------------------------------------------------------------------

# Experiment 3 --- std430 vs Packed Layout

### Buffer

``` glsl
struct Data {
    vec3 position;
    float value;
};
```

Layouts tested:

    std430
    std140
    packed

### Objective

Measure wasted bandwidth due to alignment padding.

### Expected Insight

`std430` reduces padding but still follows GPU alignment rules.

------------------------------------------------------------------------

# Experiment 4 --- Coalesced vs Strided Access

### Coalesced Access

    thread 0 -> data[0]
    thread 1 -> data[1]
    thread 2 -> data[2]

### Strided Access

    thread 0 -> data[0]
    thread 1 -> data[32]
    thread 2 -> data[64]

### Shader

``` glsl
uint idx = gl_GlobalInvocationID.x * stride;
result[idx] += input[idx];
```

### Strides Tested

    1
    2
    4
    8
    16
    32
    64

### Deliverable

Chart:

    Stride vs Performance

### Insight

Non‑coalesced memory accesses require multiple memory transactions.

------------------------------------------------------------------------

# Experiment 5 --- Shared Memory Tiling

### Goal

Demonstrate benefits of on‑chip memory.

### Shader Concept

    global memory -> shared memory -> compute

### Example

``` glsl
shared float tile[256];

tile[localID] = input[globalID];

barrier();

output[globalID] = tile[localID] * 2;
```

### Comparison

    Direct global memory access
    Shared memory tiled access

### Insight

Shared memory significantly reduces global memory latency.

------------------------------------------------------------------------

# Experiment 6 --- Memory Bandwidth Saturation

### Goal

Measure achievable memory bandwidth.

### Data Sizes

    1MB
    10MB
    100MB
    1GB

### Output

    GB/s achieved
    vs
    GPU theoretical bandwidth

### Insight

Shows real hardware limits and efficiency.

------------------------------------------------------------------------

# Benchmark Framework

### Core Loop

``` cpp
for (auto& experiment : experiments)
{
    warmup();

    auto start = now();

    for(int i = 0; i < iterations; i++)
        dispatch(experiment);

    auto end = now();

    record_result();
}
```

Outputs:

    JSON

------------------------------------------------------------------------

# Chart Generation

Python script:

    matplotlib
    pandas

Example plots:

    stride_vs_performance.png
    layout_vs_bandwidth.png
    bandwidth_saturation.png

------------------------------------------------------------------------

# README Structure

    Introduction
    GPU Execution Model
    Memory Layout
    Experiments
    Results
    Analysis
    Conclusions

Example conclusion:

> GPU performance is strongly dependent on memory layout and access
> patterns. Coalesced memory accesses and Structure‑of‑Arrays layouts
> maximize bandwidth utilization and align memory transactions with warp
> execution behavior.

------------------------------------------------------------------------