#include "json_contract.hpp"
#include "workloads.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string make_run_label(const BenchmarkRequest& request) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm time_parts {};
#ifdef _WIN32
    localtime_s(&time_parts, &now_time);
#else
    localtime_r(&now_time, &time_parts);
#endif

    std::ostringstream stream;
    stream << request.workload << "-" << request.implementation << "-" << std::put_time(&time_parts, "%Y%m%d-%H%M%S");
    return stream.str();
}

BenchmarkResult dispatch_request(const BenchmarkRequest& request) {
    if (request.workload == "vector_add") {
        return run_vector_add(request);
    }
    if (request.workload == "reduction") {
        return run_reduction(request);
    }
    if (request.workload == "gemm") {
        return run_gemm(request);
    }
    if (request.workload == "conv2d") {
        return run_conv2d(request);
    }
    if (request.workload == "softmax") {
        return run_softmax(request);
    }

    throw std::runtime_error("Unsupported workload: " + request.workload);
}

}  // namespace

int main(int argc, char** argv) {
    std::string request_path;
    std::string output_path;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--request" && index + 1 < argc) {
            request_path = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
        }
    }

    if (request_path.empty() || output_path.empty()) {
        std::cerr << "Usage: benchsight_worker --request <request.json> --output <result.json>\n";
        return 1;
    }

    BenchmarkRequest request;
    try {
        request = parse_request_file(request_path);
        BenchmarkResult result = dispatch_request(request);
        result.job_id = request.job_id;
        result.workload = request.workload;
        result.implementation = request.implementation;
        result.parameters = request.parameters;
        if (result.run_label.empty()) {
            result.run_label = make_run_label(request);
        }
        result.artifacts.push_back({"result_json", std::filesystem::path(output_path).generic_string(), "Structured benchmark result"});
        write_result_json(result, output_path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";

        BenchmarkResult failed_result;
        failed_result.job_id = request.job_id;
        failed_result.workload = request.workload;
        failed_result.implementation = request.implementation;
        failed_result.parameters = request.parameters;
        failed_result.run_label = request.workload.empty() ? "failed-benchmark-run" : make_run_label(request);
        failed_result.status = "failed";
        failed_result.error_message = error.what();
        failed_result.artifacts.push_back({"result_json", std::filesystem::path(output_path).generic_string(), "Structured benchmark result"});

        try {
            write_result_json(failed_result, output_path);
        } catch (...) {
        }
        return 1;
    }
}
