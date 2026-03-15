#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct Std430Std140PackedExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string std140_shader_path;
    std::string std430_shader_path;
    std::string packed_shader_path;
    bool verbose_progress = false;
};

struct Std430Std140PackedExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

Std430Std140PackedExperimentOutput
run_std430_std140_packed_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const Std430Std140PackedExperimentConfig& config);
