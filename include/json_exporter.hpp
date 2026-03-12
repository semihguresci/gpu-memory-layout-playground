#pragma once

#include "benchmark_runner.hpp"

#include <string>
#include <vector>

class JsonExporter {
public:
    static constexpr const char* kSchemaName = "gpu_memory_layout_results";
    static constexpr const char* kSchemaVersion = "1.0.0";

    static void writeBenchmarkResults(const std::vector<BenchmarkResult>& results, const std::string& output_path);
};
