#pragma once

#include "workload_request.hpp"

BenchmarkResult run_vector_add(const BenchmarkRequest& request);
BenchmarkResult run_reduction(const BenchmarkRequest& request);
BenchmarkResult run_gemm(const BenchmarkRequest& request);
BenchmarkResult run_conv2d(const BenchmarkRequest& request);
BenchmarkResult run_softmax(const BenchmarkRequest& request);
