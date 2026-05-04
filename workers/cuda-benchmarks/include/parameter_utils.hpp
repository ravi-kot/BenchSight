#pragma once

#include "workload_request.hpp"

#include <stdexcept>
#include <string>

inline int get_required_int_parameter(const BenchmarkRequest& request, const std::string& key) {
    const auto iterator = request.parameters.find(key);
    if (iterator == request.parameters.end()) {
        throw std::runtime_error("Missing parameter: " + key);
    }
    return static_cast<int>(iterator->second);
}

inline int get_optional_int_parameter(const BenchmarkRequest& request, const std::string& key, int fallback) {
    const auto iterator = request.parameters.find(key);
    if (iterator == request.parameters.end()) {
        return fallback;
    }
    return static_cast<int>(iterator->second);
}
