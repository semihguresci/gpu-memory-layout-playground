#pragma once

#include <cmath>
#include <cstdint>

namespace ExperimentMetrics {

inline double compute_throughput_elements_per_second(uint32_t element_count, uint32_t dispatch_count,
                                                     double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double elements = static_cast<double>(element_count) * static_cast<double>(dispatch_count);
    return (elements * 1000.0) / dispatch_gpu_ms;
}

} // namespace ExperimentMetrics
