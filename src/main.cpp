#include "benchmark_runner.hpp"
#include "experiments/aos_soa_experiment.hpp"
#include "experiments/dispatch_basics_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/json_exporter.hpp"
#include "vulkan_context.hpp"

#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

std::string format_vulkan_version(uint32_t version) {
    if (version == 0U) {
        return {};
    }

    std::ostringstream stream;
    stream << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version);
    return stream.str();
}

} // namespace

int main(int argc, char** argv) noexcept {
    try {
        const AppOptions options = ArgumentParser::parse(argc, argv);

        VulkanContext context;
        if (!context.initialize(options.enable_validation)) {
            std::cerr << "Vulkan initialization failed.\n";
            return 1;
        }

        std::cout << "Using GPU: " << context.selected_device_name() << "\n";
        std::cout << "Validation: " << (options.enable_validation ? "enabled" : "disabled") << "\n";
        std::cout << "GPU timestamps: " << (context.gpu_timestamps_supported() ? "supported" : "not supported") << "\n";
        if (!context.gpu_timestamps_supported()) {
            std::cerr << "GPU timestamp queries are required for this benchmark run.\n";
            context.shutdown();
            return 1;
        }

        BenchmarkRunner runner(
            {.warmup_iterations = options.warmup_iterations, .timed_iterations = options.timed_iterations});
        std::vector<BenchmarkResult> results;
        std::vector<BenchmarkMeasurementRow> rows;

        if (options.experiment == "all" || options.experiment == "01_dispatch_basics") {
            const DispatchBasicsExperimentOutput experiment_output = run_dispatch_basics_experiment(
                context, runner,
                DispatchBasicsExperimentConfig{.max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
                                               .write_shader_path = "",
                                               .noop_shader_path = "",
                                               .include_noop_variant = true});

            if (experiment_output.summary_results.empty()) {
                std::cerr << "Dispatch basics experiment failed.\n";
                context.shutdown();
                return 1;
            }

            if (!experiment_output.all_points_correct) {
                std::cerr << "Dispatch basics experiment reported correctness failures.\n";
                context.shutdown();
                return 1;
            }

            results.insert(results.end(), experiment_output.summary_results.begin(),
                           experiment_output.summary_results.end());
            rows.insert(rows.end(), experiment_output.rows.begin(), experiment_output.rows.end());
        }

        if (options.experiment == "all" || options.experiment == "06_aos_vs_soa") {
            const auto experiment_results = run_aos_soa_experiment(
                context, runner,
                AosSoaExperimentConfig{.max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
                                       .aos_shader_path = "",
                                       .soa_shader_path = ""});

            if (experiment_results.empty()) {
                std::cerr << "AoS vs SoA experiment failed.\n";
                context.shutdown();
                return 1;
            }

            results.insert(results.end(), experiment_results.begin(), experiment_results.end());
        }

        const BenchmarkExportMetadata metadata{
            .gpu_name = context.selected_device_name(),
            .vulkan_api_version = format_vulkan_version(context.selected_device_api_version()),
            .driver_version = std::to_string(context.selected_device_driver_version()),
            .validation_enabled = options.enable_validation,
            .gpu_timestamps_supported = context.gpu_timestamps_supported(),
            .warmup_iterations = options.warmup_iterations,
            .timed_iterations = options.timed_iterations,
        };

        context.shutdown();

        JsonExporter::write_benchmark_results(results, rows, metadata, options.output_path);
        std::cout << "Wrote results to " << options.output_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Unhandled exception: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unhandled non-standard exception.\n";
        return 1;
    }
}
