#pragma once

#include "stats.hpp"
#include "workload_request.hpp"

inline BenchmarkMetrics summarize_metrics(
    const std::vector<double>& latencies_ms,
    double throughput,
    std::optional<double> speedup = std::nullopt
) {
    BenchmarkMetrics metrics;
    metrics.avg_latency_ms = compute_average(latencies_ms);
    metrics.p50_latency_ms = compute_percentile(latencies_ms, 0.50);
    metrics.p95_latency_ms = compute_percentile(latencies_ms, 0.95);
    metrics.throughput = throughput;
    metrics.speedup_vs_baseline = speedup;
    return metrics;
}
