#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

inline double compute_average(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("Cannot compute average of an empty sample.");
    }

    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

inline double compute_percentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        throw std::runtime_error("Cannot compute percentile of an empty sample.");
    }

    std::sort(values.begin(), values.end());
    const double scaled_index = percentile * static_cast<double>(values.size() - 1);
    const std::size_t lower_index = static_cast<std::size_t>(scaled_index);
    const std::size_t upper_index = std::min(lower_index + 1, values.size() - 1);
    const double interpolation = scaled_index - static_cast<double>(lower_index);
    return values[lower_index] * (1.0 - interpolation) + values[upper_index] * interpolation;
}
