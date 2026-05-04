#include "benchmark_harness.cuh"
#include "cublas_utils.hpp"
#include "parameter_utils.hpp"
#include "result_utils.hpp"
#include "stats.hpp"
#include "workloads.hpp"

#include <stdexcept>
#include <vector>

namespace {

__global__ void vector_add_baseline_kernel(const float* a, const float* b, float* c, int num_elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        c[index] = a[index] + b[index];
    }
}

__global__ void vector_add_optimized_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int num_elements) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_elements; index += blockDim.x * gridDim.x) {
        c[index] = a[index] + b[index];
    }
}

}  // namespace

BenchmarkResult run_vector_add(const BenchmarkRequest& request) {
    NvtxScopedRange total_range("vector_add_total");

    const int num_elements = get_optional_int_parameter(request, "num_elements", 1 << 20);
    const std::size_t bytes = static_cast<std::size_t>(num_elements) * sizeof(float);

    std::vector<float> host_a(num_elements, 1.25f);
    std::vector<float> host_b(num_elements, 2.5f);
    std::vector<float> host_c(num_elements, 0.0f);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    {
        NvtxScopedRange allocation_range("allocation");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), bytes));
        CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));
    }

    const bool use_optimized = request.implementation == "optimized";
    const bool use_cublas = request.implementation == "cublas";
    if (request.implementation != "baseline" && !use_optimized && !use_cublas) {
        throw std::runtime_error("Unsupported vector_add implementation: " + request.implementation);
    }

    cublasHandle_t cublas_handle = nullptr;
    if (use_cublas) {
        cublas_check(cublasCreate(&cublas_handle));
    }

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    auto baseline_launcher = [&]() {
        vector_add_baseline_kernel<<<blocks, threads>>>(device_a, device_b, device_c, num_elements);
        CUDA_CHECK(cudaGetLastError());
    };

    auto optimized_launcher = [&]() {
        vector_add_optimized_kernel<<<blocks, threads>>>(device_a, device_b, device_c, num_elements);
        CUDA_CHECK(cudaGetLastError());
    };

    auto cublas_launcher = [&]() {
        const float alpha = 1.0f;
        cublas_check(cublasScopy(cublas_handle, num_elements, device_a, 1, device_c, 1));
        cublas_check(cublasSaxpy(cublas_handle, num_elements, &alpha, device_b, 1, device_c, 1));
    };

    std::vector<double> selected_latencies;
    if (use_cublas) {
        selected_latencies = measure_cuda_iterations(cublas_launcher, request.warmup_runs, request.timed_runs);
    } else if (use_optimized) {
        selected_latencies = measure_cuda_iterations(optimized_launcher, request.warmup_runs, request.timed_runs);
    } else {
        selected_latencies = measure_cuda_iterations(baseline_launcher, request.warmup_runs, request.timed_runs);
    }

    std::optional<double> speedup;
    if (use_optimized || use_cublas) {
        const std::vector<double> baseline_latencies =
            measure_cuda_iterations(baseline_launcher, request.warmup_runs, request.timed_runs);
        speedup = compute_average(baseline_latencies) / compute_average(selected_latencies);
    }

    {
        NvtxScopedRange cleanup_range("copy_back");
        CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    }

    if (cublas_handle != nullptr) {
        cublas_check(cublasDestroy(cublas_handle));
    }
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    const double avg_latency_ms = compute_average(selected_latencies);
    const double throughput = static_cast<double>(num_elements) / (avg_latency_ms / 1000.0);

    BenchmarkResult result;
    result.metrics = summarize_metrics(selected_latencies, throughput, speedup);
    result.timed_run_latencies_ms = selected_latencies;
    return result;
}
