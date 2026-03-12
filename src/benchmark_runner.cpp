#include "benchmark_runner.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

BenchmarkRunner::BenchmarkRunner(BenchmarkConfig config) : config_(config) {}

BenchmarkResult BenchmarkRunner::run(const std::string& name, const std::function<void()>& fn) const {
    for (int i = 0; i < config_.warmup_iterations; ++i) {
        fn();
    }

    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;

    for (int i = 0; i < config_.timed_iterations; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto end = std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        const double ms = elapsed.count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    BenchmarkResult result{};
    result.experiment_name = name;
    result.average_ms = total_ms / static_cast<double>(std::max(1, config_.timed_iterations));
    result.min_ms = min_ms;
    result.max_ms = max_ms;
    return result;
}

BenchmarkResult BenchmarkRunner::runTimed(const std::string& name, const std::function<double()>& fn_ms) const {
    for (int i = 0; i < config_.warmup_iterations; ++i) {
        static_cast<void>(fn_ms());
    }

    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;

    for (int i = 0; i < config_.timed_iterations; ++i) {
        const double ms = fn_ms();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    BenchmarkResult result{};
    result.experiment_name = name;
    result.average_ms = total_ms / static_cast<double>(std::max(1, config_.timed_iterations));
    result.min_ms = min_ms;
    result.max_ms = max_ms;
    return result;
}
