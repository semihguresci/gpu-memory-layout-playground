#pragma once

#include <functional>
#include <string>

struct BenchmarkResult {
    std::string experiment_name;
    double average_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

struct BenchmarkConfig {
    int warmup_iterations = 5;
    int timed_iterations = 25;
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkConfig config);

    BenchmarkResult run(const std::string& name, const std::function<void()>& fn) const;
    BenchmarkResult runTimed(const std::string& name, const std::function<double()>& fn_ms) const;

private:
    BenchmarkConfig config_;
};
