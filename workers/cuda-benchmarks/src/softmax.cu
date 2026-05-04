#include "benchmark_harness.cuh"
#include "cudnn_utils.hpp"
#include "parameter_utils.hpp"
#include "result_utils.hpp"
#include "stats.hpp"
#include "workloads.hpp"

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

__global__ void softmax_baseline_kernel(const float* input, float* output, int cols) {
    const int row = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float row_max = -FLT_MAX;
    for (int col = 0; col < cols; ++col) {
        row_max = fmaxf(row_max, row_input[col]);
    }

    float row_sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        row_sum += expf(row_input[col] - row_max);
    }

    for (int col = 0; col < cols; ++col) {
        row_output[col] = expf(row_input[col] - row_max) / row_sum;
    }
}

__global__ void softmax_optimized_kernel(const float* input, float* output, int cols) {
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_input[col]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_input[col] - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = shared[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        row_output[col] = expf(row_input[col] - row_max) / row_sum;
    }
}

}  // namespace

BenchmarkResult run_softmax(const BenchmarkRequest& request) {
    NvtxScopedRange total_range("softmax_total");

    const int rows = get_optional_int_parameter(request, "rows", 1024);
    const int cols = get_optional_int_parameter(request, "cols", 1024);
    const std::size_t elements = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

    std::vector<float> host_input(elements);
    std::vector<float> host_output(elements, 0.0f);
    for (std::size_t index = 0; index < elements; ++index) {
        host_input[index] = static_cast<float>((index % 23) * 0.1f);
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), elements * sizeof(float), cudaMemcpyHostToDevice));

    const bool use_optimized = request.implementation == "optimized";
    const bool use_cudnn = request.implementation == "cudnn";
    if (request.implementation != "baseline" && !use_optimized && !use_cudnn) {
        throw std::runtime_error("Unsupported softmax implementation: " + request.implementation);
    }

#if BENCHSIGHT_HAVE_CUDNN
    cudnnHandle_t cudnn_handle = nullptr;
    cudnnTensorDescriptor_t cudnn_tensor = nullptr;
    if (use_cudnn) {
        cudnn_check(cudnnCreate(&cudnn_handle));
        cudnn_check(cudnnCreateTensorDescriptor(&cudnn_tensor));
        cudnn_check(cudnnSetTensor4dDescriptor(
            cudnn_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, cols, 1, 1));
    }
#else
    if (use_cudnn) {
        throw std::runtime_error("Softmax cuDNN implementation requested, but cuDNN support is not enabled in this build.");
    }
#endif

    auto baseline_launcher = [&]() {
        softmax_baseline_kernel<<<rows, 1>>>(device_input, device_output, cols);
        CUDA_CHECK(cudaGetLastError());
    };

    auto optimized_launcher = [&]() {
        constexpr int threads = 256;
        softmax_optimized_kernel<<<rows, threads, threads * sizeof(float)>>>(device_input, device_output, cols);
        CUDA_CHECK(cudaGetLastError());
    };

#if BENCHSIGHT_HAVE_CUDNN
    auto cudnn_launcher = [&]() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnn_check(cudnnSoftmaxForward(
            cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            cudnn_tensor,
            device_input,
            &beta,
            cudnn_tensor,
            device_output));
    };
#endif

    std::vector<double> selected_latencies;
    if (use_cudnn) {
#if BENCHSIGHT_HAVE_CUDNN
        selected_latencies = measure_cuda_iterations(cudnn_launcher, request.warmup_runs, request.timed_runs);
#endif
    } else if (use_optimized) {
        selected_latencies = measure_cuda_iterations(optimized_launcher, request.warmup_runs, request.timed_runs);
    } else {
        selected_latencies = measure_cuda_iterations(baseline_launcher, request.warmup_runs, request.timed_runs);
    }

    std::optional<double> speedup;
    if (use_optimized || use_cudnn) {
        const std::vector<double> baseline_latencies =
            measure_cuda_iterations(baseline_launcher, request.warmup_runs, request.timed_runs);
        speedup = compute_average(baseline_latencies) / compute_average(selected_latencies);
    }

    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, elements * sizeof(float), cudaMemcpyDeviceToHost));

#if BENCHSIGHT_HAVE_CUDNN
    if (cudnn_tensor != nullptr) {
        cudnn_check(cudnnDestroyTensorDescriptor(cudnn_tensor));
    }
    if (cudnn_handle != nullptr) {
        cudnn_check(cudnnDestroy(cudnn_handle));
    }
#endif

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    const double avg_latency_ms = compute_average(selected_latencies);
    const double throughput = static_cast<double>(elements) / (avg_latency_ms / 1000.0);

    BenchmarkResult result;
    result.metrics = summarize_metrics(selected_latencies, throughput, speedup);
    result.timed_run_latencies_ms = selected_latencies;
    return result;
}
