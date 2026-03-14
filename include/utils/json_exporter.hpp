#pragma once

#include "benchmark_runner.hpp"

#include <string>
#include <vector>

struct BenchmarkExportMetadata {
    std::string gpu_name;
    std::string vulkan_api_version;
    std::string driver_version;
    bool validation_enabled = false;
    bool gpu_timestamps_supported = false;
    int warmup_iterations = 0;
    int timed_iterations = 0;
};

class JsonExporter {
  public:
    static constexpr const char* kSchemaName = "gpu_memory_layout_results";
    static constexpr const char* kSchemaVersion = "1.1.0";

    static void write_benchmark_results(const std::vector<BenchmarkResult>& results, const std::string& output_path);
    static void write_benchmark_results(const std::vector<BenchmarkResult>& results,
                                        const std::vector<BenchmarkMeasurementRow>& rows,
                                        const BenchmarkExportMetadata& metadata, const std::string& output_path);
};
