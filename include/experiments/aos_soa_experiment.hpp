#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct AosSoaExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string aos_shader_path;
    std::string soa_shader_path;
};

std::vector<BenchmarkResult> run_aos_soa_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const AosSoaExperimentConfig& config);
