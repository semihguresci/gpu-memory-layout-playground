#include "app_options.hpp"
#include "benchmark_runner.hpp"
#include "buffer_utils.hpp"
#include "json_exporter.hpp"
#include "vulkan_context.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

namespace {

double measureGpuOnly(const VulkanContext& context, const std::function<double()>& measured) {
    if (!context.gpuTimestampsSupported()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double ms = measured();
    if (!std::isnan(ms) && std::isfinite(ms)) {
        return ms;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace

int main(int argc, char** argv) {
    const AppOptions options = ArgumentParser::parse(argc, argv);

    VulkanContext context;
    if (!context.initialize(options.enable_validation)) {
        std::cerr << "Vulkan initialization failed.\n";
        return 1;
    }

    std::cout << "Using GPU: " << context.selectedDeviceName() << "\n";
    std::cout << "Validation: " << (options.enable_validation ? "enabled" : "disabled") << "\n";
    std::cout << "GPU timestamps: " << (context.gpuTimestampsSupported() ? "supported" : "not supported") << "\n";
    if (!context.gpuTimestampsSupported()) {
        std::cerr << "GPU timestamp queries are required for this benchmark run.\n";
        context.shutdown();
        return 1;
    }

    BufferResource scratch{};
    const bool scratch_ok = createBufferResource(
        context.physicalDevice(),
        context.device(),
        options.scratch_size_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        scratch);

    if (!scratch_ok) {
        std::cerr << "Failed to create scratch buffer.\n";
        context.shutdown();
        return 1;
    }

    BenchmarkRunner runner({options.warmup_iterations, options.timed_iterations});
    std::vector<BenchmarkResult> results;

    if (options.experiment == "all" || options.experiment == "01_thread_mapping") {
        results.push_back(runner.runTimed("01_thread_mapping", [&]() {
            return measureGpuOnly(context, [&]() {
                return context.measureGpuTimeMs([](VkCommandBuffer) {});
            });
        }));
    }

    if (options.experiment == "all" || options.experiment == "02_aos_vs_soa_baseline") {
        results.push_back(runner.runTimed("02_aos_vs_soa_baseline", [&]() {
            return measureGpuOnly(context, [&]() {
                return context.measureGpuTimeMs([](VkCommandBuffer) {});
            });
        }));
    }

    destroyBufferResource(context.device(), scratch);
    context.shutdown();

    JsonExporter::writeBenchmarkResults(results, options.output_path);
    std::cout << "Wrote results to " << options.output_path << "\n";
    return 0;
}
