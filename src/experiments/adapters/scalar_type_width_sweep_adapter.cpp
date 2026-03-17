#include "experiments/experiment_contract.hpp"
#include "experiments/scalar_type_width_sweep_experiment.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_scalar_type_width_sweep_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const AppOptions& options, ExperimentRunOutput& output) {
    ScalarTypeWidthSweepExperimentOutput experiment_output = run_scalar_type_width_sweep_experiment(
        context, runner,
        ScalarTypeWidthSweepExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .fp32_shader_path = "",
            .fp16_storage_shader_path = "",
            .u32_shader_path = "",
            .u16_shader_path = "",
            .u8_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "scalar type width sweep experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "scalar type width sweep experiment reported correctness failures.";
        return false;
    }

    return true;
}
