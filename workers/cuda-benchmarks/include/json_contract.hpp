#pragma once

#include "workload_request.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

inline std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

inline std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());

    for (char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += character;
                break;
        }
    }

    return escaped;
}

inline std::string json_unescape(const std::string& value) {
    std::string unescaped;
    unescaped.reserve(value.size());

    bool escaping = false;
    for (char character : value) {
        if (!escaping) {
            if (character == '\\') {
                escaping = true;
            } else {
                unescaped += character;
            }
            continue;
        }

        switch (character) {
            case '\\':
                unescaped += '\\';
                break;
            case '"':
                unescaped += '"';
                break;
            case 'n':
                unescaped += '\n';
                break;
            case 'r':
                unescaped += '\r';
                break;
            case 't':
                unescaped += '\t';
                break;
            default:
                unescaped += character;
                break;
        }
        escaping = false;
    }

    return unescaped;
}

inline std::string extract_string_field(const std::string& content, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(content, match, pattern)) {
        throw std::runtime_error("Missing string field: " + key);
    }
    return json_unescape(match[1].str());
}

inline int extract_integer_field(const std::string& content, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(content, match, pattern)) {
        throw std::runtime_error("Missing integer field: " + key);
    }
    return std::stoi(match[1].str());
}

inline std::unordered_map<std::string, double> extract_number_object(const std::string& content, const std::string& key) {
    const std::regex object_pattern("\"" + key + "\"\\s*:\\s*\\{([^}]*)\\}");
    std::smatch object_match;
    if (!std::regex_search(content, object_match, object_pattern)) {
        throw std::runtime_error("Missing object field: " + key);
    }

    std::unordered_map<std::string, double> result;
    const std::string object_content = object_match[1].str();
    const std::regex entry_pattern("\"([^\"]+)\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");

    for (std::sregex_iterator iterator(object_content.begin(), object_content.end(), entry_pattern), end; iterator != end;
         ++iterator) {
        result[(*iterator)[1].str()] = std::stod((*iterator)[2].str());
    }

    return result;
}

inline BenchmarkRequest parse_request_file(const std::string& path) {
    const std::string content = read_text_file(path);
    BenchmarkRequest request;
    request.job_id = extract_string_field(content, "job_id");
    request.workload = extract_string_field(content, "workload");
    request.implementation = extract_string_field(content, "implementation");
    request.warmup_runs = extract_integer_field(content, "warmup_runs");
    request.timed_runs = extract_integer_field(content, "timed_runs");
    request.artifacts_dir = extract_string_field(content, "artifacts_dir");
    request.parameters = extract_number_object(content, "parameters");
    return request;
}

inline void write_result_json(const BenchmarkResult& result, const std::string& path) {
    std::filesystem::path output_path(path);
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to create result file: " + path);
    }

    output << "{\n";
    output << "  \"job_id\": \"" << json_escape(result.job_id) << "\",\n";
    output << "  \"run_label\": \"" << json_escape(result.run_label) << "\",\n";
    output << "  \"status\": \"" << json_escape(result.status) << "\",\n";
    output << "  \"workload\": \"" << json_escape(result.workload) << "\",\n";
    output << "  \"implementation\": \"" << json_escape(result.implementation) << "\",\n";
    output << "  \"metrics\": {\n";
    output << "    \"avg_latency_ms\": " << result.metrics.avg_latency_ms << ",\n";
    output << "    \"p50_latency_ms\": " << result.metrics.p50_latency_ms << ",\n";
    output << "    \"p95_latency_ms\": " << result.metrics.p95_latency_ms << ",\n";
    output << "    \"throughput\": " << result.metrics.throughput << ",\n";
    if (result.metrics.speedup_vs_baseline.has_value()) {
        output << "    \"speedup_vs_baseline\": " << result.metrics.speedup_vs_baseline.value() << "\n";
    } else {
        output << "    \"speedup_vs_baseline\": null\n";
    }
    output << "  },\n";

    output << "  \"parameters\": {\n";
    bool first_parameter = true;
    for (const auto& [key, value] : result.parameters) {
        if (!first_parameter) {
            output << ",\n";
        }
        output << "    \"" << json_escape(key) << "\": " << value;
        first_parameter = false;
    }
    output << "\n  },\n";

    output << "  \"timed_run_latencies_ms\": [";
    for (std::size_t index = 0; index < result.timed_run_latencies_ms.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << result.timed_run_latencies_ms[index];
    }
    output << "],\n";

    output << "  \"artifacts\": [\n";
    for (std::size_t index = 0; index < result.artifacts.size(); ++index) {
        const auto& artifact = result.artifacts[index];
        output << "    {\n";
        output << "      \"kind\": \"" << json_escape(artifact.kind) << "\",\n";
        output << "      \"path\": \"" << json_escape(artifact.path) << "\"";
        if (!artifact.description.empty()) {
            output << ",\n      \"description\": \"" << json_escape(artifact.description) << "\"\n";
        } else {
            output << "\n";
        }
        output << "    }";
        if (index + 1 < result.artifacts.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ]";

    if (!result.error_message.empty()) {
        output << ",\n  \"error_message\": \"" << json_escape(result.error_message) << "\"\n";
    } else {
        output << "\n";
    }

    output << "}\n";
}
