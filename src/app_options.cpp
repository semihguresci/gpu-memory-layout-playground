#include "app_options.hpp"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include <CLI/CLI.hpp>

namespace {

bool parseSize(const std::string& value, VkDeviceSize& out_size) {
    if (value.empty()) {
        return false;
    }

    char suffix = '\0';
    std::string number_part = value;
    if (!std::isdigit(static_cast<unsigned char>(value.back()))) {
        suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(value.back())));
        number_part = value.substr(0, value.size() - 1);
    }

    std::size_t parsed = 0;
    unsigned long long base = 0;
    try {
        base = std::stoull(number_part, &parsed, 10);
    } catch (...) {
        return false;
    }

    if (parsed != number_part.size()) {
        return false;
    }

    unsigned long long multiplier = 1;
    if (suffix == 'K') {
        multiplier = 1024ULL;
    } else if (suffix == 'M') {
        multiplier = 1024ULL * 1024ULL;
    } else if (suffix == 'G') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
    } else if (suffix != '\0') {
        return false;
    }

    out_size = static_cast<VkDeviceSize>(base * multiplier);
    return out_size > 0;
}

} // namespace

AppOptions ArgumentParser::parse(int argc, char** argv) {
    AppOptions options{};
    std::string size_text = "4M";
    CLI::App app{"GPU memory layout experiments"};

    app.add_flag("--validation", options.enable_validation, "Enable Vulkan validation layers");
    app.add_option("--experiment", options.experiment, "Experiment: all, 01_thread_mapping, 02_aos_vs_soa_baseline")
        ->check(CLI::IsMember({"all", "01_thread_mapping", "02_aos_vs_soa_baseline"}));
    app.add_option("--iterations", options.timed_iterations, "Timed iterations")->check(CLI::PositiveNumber);
    app.add_option("--warmup", options.warmup_iterations, "Warmup iterations")->check(CLI::NonNegativeNumber);
    app.add_option("--size", size_text, "Scratch buffer size in bytes or with K/M/G suffix");
    app.add_option("--output", options.output_path, "Output JSON path");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    VkDeviceSize parsed_size = 0;
    if (!parseSize(size_text, parsed_size)) {
        std::cerr << "Invalid --size value. Use bytes or N[K|M|G], e.g. 4194304 or 4M.\n";
        std::exit(2);
    }
    options.scratch_size_bytes = parsed_size;

    if (options.output_path.empty()) {
        std::cerr << "Invalid --output value. Path cannot be empty.\n";
        std::exit(2);
    }

    if (!options.output_path.ends_with(".json")) {
        options.output_path += ".json";
    }

    return options;
}
