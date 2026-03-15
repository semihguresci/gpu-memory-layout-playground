# Experiment 05 Architecture

## 1. Purpose
Experiment 05 compares three global index mapping strategies in a matched read+write compute kernel:
- direct (`i -> i`)
- fixed offset (`i -> (i + offset) mod N`)
- grid-stride loop (`i, i + stride, ...`)

All variants apply the same deterministic transform to isolate mapping overhead and scalability behavior.

## 2. Runtime Component Architecture
```mermaid
flowchart LR
    CLI["CLI Options<br/>--experiment --size --warmup --iterations"] --> MAIN["main.cpp"]
    MAIN --> REG["Generated Experiment Registry"]
    REG --> ADP["global_id_mapping_variants_adapter.cpp"]
    ADP --> EXP["GlobalIdMappingVariantsExperiment"]

    EXP --> BUF["BufferUtils<br/>src/dst/staging buffers"]
    EXP --> SHD["05_global_id_mapping_variants.comp.spv shader"]
    EXP --> VCU["VulkanComputeUtils<br/>pipeline + barriers"]
    EXP --> TIM["VulkanContext::measure_gpu_time_ms"]
    EXP --> ROWS["BenchmarkMeasurementRow[]"]
    EXP --> SUM["BenchmarkResult[]"]
    MAIN --> JSON["JsonExporter"]
    ROWS --> JSON
    SUM --> JSON
```

## 3. Resource Ownership Model
Shared buffers:
- `src_device` (device-local storage + transfer)
- `dst_device` (device-local storage + transfer)
- `staging` (host-visible transfer src/dst)

Pipeline resources:
- shader module
- descriptor set layout
- descriptor pool + descriptor set
- pipeline layout
- compute pipeline

Ownership rule:
- experiment function creates and destroys all resources
- teardown is reverse-order
- handles are reset to `VK_NULL_HANDLE`

## 4. Execution Flow
```mermaid
flowchart TD
    A["Resolve shader + clamp sweep sizes"] --> B["Create buffers + pipeline"]
    B --> C["Map staging memory"]
    C --> D["For each problem size"]
    D --> E["For each mapping variant"]
    E --> F["For each dispatch count"]
    F --> G["Warmup iterations"]
    G --> H["Timed iterations"]
    H --> I["Upload src"]
    I --> J["Upload dst sentinel"]
    J --> K["Dispatch mapping kernel"]
    K --> L["Readback dst"]
    L --> M["Validate correctness"]
    M --> N["Append row + notes"]
    N --> O["Summarize dispatch samples"]
    O --> P["Unmap + destroy resources"]
```

## 5. Per-Iteration Command Sequence
```mermaid
sequenceDiagram
    participant CPU as "Host Thread"
    participant GPU as "GPU Queue"

    CPU->>CPU: "Prepare source and destination staging payload"
    CPU->>GPU: "Upload source (staging -> src_device)"
    CPU->>GPU: "Upload destination sentinel (staging -> dst_device)"
    CPU->>GPU: "Dispatch direct/offset/grid-stride kernel"
    CPU->>GPU: "Readback destination (dst_device -> staging)"
    CPU->>CPU: "Validate expected contents"
    CPU->>CPU: "Record row (gpu_ms/end_to_end/gbps)"
```

## 6. Data and Analysis Pipeline
```mermaid
flowchart LR
    RUN["run_experiment_data_collection.py --experiment 05_global_id_mapping_variants"] --> RAW["benchmark_results.json"]
    RAW --> COLLECT["scripts/collect_run.py"]
    RAW --> ANALYZE["scripts/analyze_global_id_mapping_variants.py"]
    RAW --> PLOT["scripts/plot_results.py"]
    ANALYZE --> TABLES["results/tables/*.csv"]
    ANALYZE --> CHARTS["results/charts/*.png"]
```
