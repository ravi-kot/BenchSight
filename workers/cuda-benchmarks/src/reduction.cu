#include "benchmark_harness.cuh"
#include "cublas_utils.hpp"
#include "parameter_utils.hpp"
#include "result_utils.hpp"
#include "stats.hpp"
#include "workloads.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

__global__ void reduction_baseline_kernel(const float* input, float* output, int num_elements) {
    extern __shared__ float shared[];

    const unsigned int thread_index = threadIdx.x;
    const unsigned int input_index = blockIdx.x * blockDim.x + thread_index;

    shared[thread_index] = input_index < num_elements ? input[input_index] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((thread_index % (2 * stride)) == 0 && thread_index + stride < blockDim.x) {
            shared[thread_index] += shared[thread_index + stride];
        }
        __syncthreads();
    }

    if (thread_index == 0) {
        atomicAdd(output, shared[0]);
    }
}

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void reduction_optimized_kernel(const float* input, float* output, int num_elements) {
    float thread_sum = 0.0f;
    const int global_stride = blockDim.x * gridDim.x * 2;

    for (int index = blockIdx.x * blockDim.x * 2 + threadIdx.x; index < num_elements; index += global_stride) {
        thread_sum += input[index];
        if (index + blockDim.x < num_elements) {
            thread_sum += input[index + blockDim.x];
        }
    }

    thread_sum = warp_reduce_sum(thread_sum);

    __shared__ float shared[32];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        thread_sum = lane < (blockDim.x / warpSize) ? shared[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        if (lane == 0) {
            atomicAdd(output, thread_sum);
        }
    }
}

}  // namespace

BenchmarkResult run_reduction(const BenchmarkRequest& request) {
    NvtxScopedRange total_range("reduction_total");

    const int num_elements = get_optional_int_parameter(request, "num_elements", 1 << 20);
    const int block_size = std::max(32, get_optional_int_parameter(request, "block_size", 256));
    const int blocks = std::min(1024, (num_elements + block_size - 1) / block_size);
    const std::size_t bytes = static_cast<std::size_t>(num_elements) * sizeof(float);

    std::vector<float> host_input(num_elements);
    for (int index = 0; index < num_elements; ++index) {
        host_input[index] = 1.0f + static_cast<float>(index % 7) * 0.01f;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const bool use_optimized = request.implementation == "optimized";
    const bool use_cublas = request.implementation == "cublas";
    if (request.implementation != "baseline" && !use_optimized && !use_cublas) {
        throw std::runtime_error("Unsupported reduction implementation: " + request.implementation);
    }

    cublasHandle_t cublas_handle = nullptr;
    if (use_cublas) {
        cublas_check(cublasCreate(&cublas_handle));
        cublas_check(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    }

    auto baseline_launcher = [&]() {
        CUDA_CHECK(cudaMemset(device_output, 0, sizeof(float)));
        reduction_baseline_kernel<<<blocks, block_size, static_cast<std::size_t>(block_size) * sizeof(float)>>>(
            device_input, device_output, num_elements);
        CUDA_CHECK(cudaGetLastError());
    };

    auto optimized_launcher = [&]() {
        CUDA_CHECK(cudaMemset(device_output, 0, sizeof(float)));
        reduction_optimized_kernel<<<blocks, block_size>>>(device_input, device_output, num_elements);
        CUDA_CHECK(cudaGetLastError());
    };

    auto cublas_launcher = [&]() {
        cublas_check(cublasSasum(cublas_handle, num_elements, device_input, 1, device_output));
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

    float host_output = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost));
    (void)host_output;

    if (cublas_handle != nullptr) {
        cublas_check(cublasDestroy(cublas_handle));
    }
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    const double avg_latency_ms = compute_average(selected_latencies);
    const double throughput = static_cast<double>(num_elements) / (avg_latency_ms / 1000.0);

    BenchmarkResult result;
    result.metrics = summarize_metrics(selected_latencies, throughput, speedup);
    result.timed_run_latencies_ms = selected_latencies;
    return result;
}
