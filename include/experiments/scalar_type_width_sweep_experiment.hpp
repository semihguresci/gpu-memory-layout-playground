#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct ScalarTypeWidthSweepExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string fp32_shader_path;
    std::string fp16_storage_shader_path;
    std::string u32_shader_path;
    std::string u16_shader_path;
    std::string u8_shader_path;
    bool verbose_progress = false;
};

struct ScalarTypeWidthSweepExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

ScalarTypeWidthSweepExperimentOutput
run_scalar_type_width_sweep_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const ScalarTypeWidthSweepExperimentConfig& config);
