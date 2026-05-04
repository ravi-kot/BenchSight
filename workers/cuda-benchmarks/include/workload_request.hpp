#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct BenchmarkRequest {
    std::string job_id;
    std::string workload;
    std::string implementation;
    int warmup_runs = 10;
    int timed_runs = 50;
    std::unordered_map<std::string, double> parameters;
    std::string artifacts_dir;
};

struct BenchmarkMetrics {
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double throughput = 0.0;
    std::optional<double> speedup_vs_baseline;
};

struct ArtifactRecord {
    std::string kind;
    std::string path;
    std::string description;
};

struct BenchmarkResult {
    std::string job_id;
    std::string run_label;
    std::string status = "completed";
    std::string workload;
    std::string implementation;
    BenchmarkMetrics metrics;
    std::unordered_map<std::string, double> parameters;
    std::vector<double> timed_run_latencies_ms;
    std::vector<ArtifactRecord> artifacts;
    std::string error_message;
};
