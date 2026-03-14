#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string experiment_name;
    double average_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    int sample_count = 0;
};

struct BenchmarkMeasurementRow {
    std::string experiment_id;
    std::string variant;
    uint32_t problem_size = 0;
    uint32_t dispatch_count = 0;
    int iteration = 0;
    double gpu_ms = 0.0;
    double end_to_end_ms = 0.0;
    double throughput = 0.0;
    double gbps = 0.0;
    bool correctness_pass = false;
    std::string notes;
};

struct BenchmarkConfig {
    int warmup_iterations = 5;
    int timed_iterations = 25;
};

class BenchmarkRunner {
  public:
    explicit BenchmarkRunner(BenchmarkConfig config);

    BenchmarkResult run(const std::string& name, const std::function<void()>& fn) const;
    BenchmarkResult run_timed(const std::string& name, const std::function<double()>& fn_ms) const;
    [[nodiscard]] int warmup_iterations() const;
    [[nodiscard]] int timed_iterations() const;
    static BenchmarkResult summarize_samples(const std::string& name, const std::vector<double>& samples);

  private:
    BenchmarkConfig config_;
};
