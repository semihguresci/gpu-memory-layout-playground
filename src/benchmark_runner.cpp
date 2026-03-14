#include "benchmark_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace {

double percentile_from_sorted_samples(const std::vector<double>& samples, double percentile) {
    if (samples.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (samples.size() == 1U) {
        return samples.front();
    }

    const double clamped_percentile = std::clamp(percentile, 0.0, 100.0);
    const double rank = (clamped_percentile / 100.0) * static_cast<double>(samples.size() - 1U);
    const auto low_index = static_cast<std::size_t>(std::floor(rank));
    const auto high_index = static_cast<std::size_t>(std::ceil(rank));

    if (low_index == high_index) {
        return samples[low_index];
    }

    const double fraction = rank - static_cast<double>(low_index);
    return samples[low_index] + ((samples[high_index] - samples[low_index]) * fraction);
}

} // namespace

BenchmarkRunner::BenchmarkRunner(BenchmarkConfig config) : config_(config) {
}

BenchmarkResult BenchmarkRunner::run(const std::string& name, const std::function<void()>& fn) const {
    for (int i = 0; i < config_.warmup_iterations; ++i) {
        fn();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(std::max(0, config_.timed_iterations)));

    for (int i = 0; i < config_.timed_iterations; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto end = std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        samples.push_back(elapsed.count());
    }

    return summarize_samples(name, samples);
}

BenchmarkResult BenchmarkRunner::run_timed(const std::string& name, const std::function<double()>& fn_ms) const {
    for (int i = 0; i < config_.warmup_iterations; ++i) {
        static_cast<void>(fn_ms());
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(std::max(0, config_.timed_iterations)));

    for (int i = 0; i < config_.timed_iterations; ++i) {
        samples.push_back(fn_ms());
    }

    return summarize_samples(name, samples);
}

int BenchmarkRunner::warmup_iterations() const {
    return config_.warmup_iterations;
}

int BenchmarkRunner::timed_iterations() const {
    return config_.timed_iterations;
}

BenchmarkResult BenchmarkRunner::summarize_samples(const std::string& name, const std::vector<double>& samples) {
    BenchmarkResult result{};
    result.experiment_name = name;

    std::vector<double> finite_samples;
    finite_samples.reserve(samples.size());
    for (const double sample : samples) {
        if (std::isfinite(sample)) {
            finite_samples.push_back(sample);
        }
    }

    result.sample_count = static_cast<int>(finite_samples.size());

    if (finite_samples.empty()) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        result.average_ms = nan;
        result.min_ms = nan;
        result.max_ms = nan;
        result.median_ms = nan;
        result.p95_ms = nan;
        return result;
    }

    std::sort(finite_samples.begin(), finite_samples.end());

    double total_ms = 0.0;
    for (const double sample : finite_samples) {
        total_ms += sample;
    }

    result.average_ms = total_ms / static_cast<double>(finite_samples.size());
    result.min_ms = finite_samples.front();
    result.max_ms = finite_samples.back();
    result.median_ms = percentile_from_sorted_samples(finite_samples, 50.0);
    result.p95_ms = percentile_from_sorted_samples(finite_samples, 95.0);
    return result;
}
