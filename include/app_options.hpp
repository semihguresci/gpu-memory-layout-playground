#pragma once

#include <string>
#include <vulkan/vulkan.h>

struct AppOptions {
#if defined(DEFAULT_ENABLE_VULKAN_VALIDATION)
    bool enable_validation = true;
#else
    bool enable_validation = false;
#endif
    std::string experiment = "all";
    int timed_iterations = 20;
    int warmup_iterations = 5;
    VkDeviceSize scratch_size_bytes = 4 * 1024 * 1024;
    std::string output_path = "results/tables/benchmark_results.json";
};

class ArgumentParser {
public:
    static AppOptions parse(int argc, char** argv);
};
