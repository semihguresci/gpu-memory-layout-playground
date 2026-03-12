#include "json_exporter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>

namespace {

std::string utcNowIso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm utc_tm{};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

} // namespace

void JsonExporter::writeBenchmarkResults(const std::vector<BenchmarkResult>& results, const std::string& output_path) {
    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    nlohmann::json root = nlohmann::json::object();
    root["schema"] = {
        {"name", kSchemaName},
        {"version", kSchemaVersion}
    };
    root["metadata"] = {
        {"exported_at_utc", utcNowIso8601()},
        {"generator", "gpu_memory_layout_experiments"},
        {"result_count", results.size()}
    };
    root["results"] = nlohmann::json::array();

    for (const auto& result : results) {
        root["results"].push_back({
            {"experiment", result.experiment_name},
            {"average_ms", result.average_ms},
            {"min_ms", result.min_ms},
            {"max_ms", result.max_ms}
        });
    }

    std::ofstream output(output_path);
    output << root.dump(2) << "\n";
}
