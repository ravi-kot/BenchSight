#include "benchmark_harness.cuh"
#include "cublas_utils.hpp"
#include "parameter_utils.hpp"
#include "result_utils.hpp"
#include "stats.hpp"
#include "workloads.hpp"

#include <stdexcept>
#include <vector>

namespace {

constexpr int kMaxConstantFilterFloats = 4096;
__constant__ float constant_filters[kMaxConstantFilterFloats];

__global__ void conv2d_baseline_kernel(
    const float* input,
    const float* filters,
    float* output,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int out_height,
    int out_width
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_channel = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_channel >= out_channels) {
        return;
    }

    float value = 0.0f;
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int input_index = ((in_channel * height) + (out_y + ky)) * width + (out_x + kx);
                const int filter_index =
                    (((out_channel * in_channels) + in_channel) * kernel_size + ky) * kernel_size + kx;
                value += input[input_index] * filters[filter_index];
            }
        }
    }

    const int output_index = ((out_channel * out_height) + out_y) * out_width + out_x;
    output[output_index] = value;
}

__global__ void conv2d_constant_filter_kernel(
    const float* input,
    float* output,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int out_height,
    int out_width
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_channel = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_channel >= out_channels) {
        return;
    }

    float value = 0.0f;
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int input_index = ((in_channel * height) + (out_y + ky)) * width + (out_x + kx);
                const int filter_index =
                    (((out_channel * in_channels) + in_channel) * kernel_size + ky) * kernel_size + kx;
                value += input[input_index] * constant_filters[filter_index];
            }
        }
    }

    const int output_index = ((out_channel * out_height) + out_y) * out_width + out_x;
    output[output_index] = value;
}

__global__ void conv2d_im2col_kernel(
    const float* input,
    float* im2col,
    int in_channels,
    int height,
    int width,
    int kernel_size,
    int out_height,
    int out_width
) {
    const int output_pixels = out_height * out_width;
    const int patch_elements = in_channels * kernel_size * kernel_size;
    const int total_elements = patch_elements * output_pixels;

    for (int linear_index = blockIdx.x * blockDim.x + threadIdx.x; linear_index < total_elements;
         linear_index += blockDim.x * gridDim.x) {
        const int output_pixel = linear_index % output_pixels;
        const int patch_index = linear_index / output_pixels;

        const int out_y = output_pixel / out_width;
        const int out_x = output_pixel % out_width;
        const int channel_kernel_area = kernel_size * kernel_size;
        const int in_channel = patch_index / channel_kernel_area;
        const int kernel_offset = patch_index % channel_kernel_area;
        const int ky = kernel_offset / kernel_size;
        const int kx = kernel_offset % kernel_size;

        const int input_index = ((in_channel * height) + (out_y + ky)) * width + (out_x + kx);
        im2col[linear_index] = input[input_index];
    }
}

}  // namespace

BenchmarkResult run_conv2d(const BenchmarkRequest& request) {
    NvtxScopedRange total_range("conv2d_total");

    const int in_channels = get_optional_int_parameter(request, "in_channels", 3);
    const int out_channels = get_optional_int_parameter(request, "out_channels", 16);
    const int height = get_optional_int_parameter(request, "height", 224);
    const int width = get_optional_int_parameter(request, "width", 224);
    const int kernel_size = get_optional_int_parameter(request, "kernel_size", 3);

    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    if (out_height <= 0 || out_width <= 0) {
        throw std::runtime_error("Conv2D kernel_size is larger than the input dimensions.");
    }

    const std::size_t input_elements = static_cast<std::size_t>(in_channels) * height * width;
    const std::size_t filter_elements = static_cast<std::size_t>(out_channels) * in_channels * kernel_size * kernel_size;
    const std::size_t output_elements = static_cast<std::size_t>(out_channels) * out_height * out_width;

    std::vector<float> host_input(input_elements);
    std::vector<float> host_filters(filter_elements);
    std::vector<float> host_output(output_elements, 0.0f);

    for (std::size_t index = 0; index < input_elements; ++index) {
        host_input[index] = static_cast<float>((index % 11) * 0.03f);
    }
    for (std::size_t index = 0; index < filter_elements; ++index) {
        host_filters[index] = static_cast<float>((index % 7) * 0.02f);
    }

    float* device_input = nullptr;
    float* device_filters = nullptr;
    float* device_output = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_filters), filter_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), output_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), input_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_filters, host_filters.data(), filter_elements * sizeof(float), cudaMemcpyHostToDevice));

    const bool use_optimized = request.implementation == "optimized";
    const bool use_cublas = request.implementation == "cublas";
    if (request.implementation != "baseline" && !use_optimized && !use_cublas) {
        throw std::runtime_error("Unsupported conv2d implementation: " + request.implementation);
    }

    if (filter_elements > static_cast<std::size_t>(kMaxConstantFilterFloats) && use_optimized) {
        throw std::runtime_error("Conv2D optimized path requires a smaller filter tensor for constant memory.");
    }

    if (use_optimized) {
        CUDA_CHECK(cudaMemcpyToSymbol(constant_filters, host_filters.data(), filter_elements * sizeof(float)));
    }

    const int output_pixels = out_height * out_width;
    const int patch_elements = in_channels * kernel_size * kernel_size;
    const std::size_t im2col_elements = static_cast<std::size_t>(patch_elements) * static_cast<std::size_t>(output_pixels);
    float* device_im2col = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    if (use_cublas) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_im2col), im2col_elements * sizeof(float)));
        cublas_check(cublasCreate(&cublas_handle));
    }

    const dim3 block(16, 16);
    const dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y, out_channels);

    auto baseline_launcher = [&]() {
        conv2d_baseline_kernel<<<grid, block>>>(
            device_input, device_filters, device_output, in_channels, out_channels, height, width, kernel_size, out_height, out_width);
        CUDA_CHECK(cudaGetLastError());
    };

    auto optimized_launcher = [&]() {
        conv2d_constant_filter_kernel<<<grid, block>>>(
            device_input, device_output, in_channels, out_channels, height, width, kernel_size, out_height, out_width);
        CUDA_CHECK(cudaGetLastError());
    };

    auto cublas_launcher = [&]() {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((im2col_elements + threads - 1) / threads);
        conv2d_im2col_kernel<<<blocks, threads>>>(
            device_input, device_im2col, in_channels, height, width, kernel_size, out_height, out_width);
        CUDA_CHECK(cudaGetLastError());

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublas_check(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            output_pixels,
            out_channels,
            patch_elements,
            &alpha,
            device_im2col,
            output_pixels,
            device_filters,
            patch_elements,
            &beta,
            device_output,
            output_pixels));
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

    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, output_elements * sizeof(float), cudaMemcpyDeviceToHost));

    if (cublas_handle != nullptr) {
        cublas_check(cublasDestroy(cublas_handle));
    }
    if (device_im2col != nullptr) {
        CUDA_CHECK(cudaFree(device_im2col));
    }
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_filters));
    CUDA_CHECK(cudaFree(device_output));

    const double avg_latency_ms = compute_average(selected_latencies);
    const double throughput = static_cast<double>(output_elements) / (avg_latency_ms / 1000.0);

    BenchmarkResult result;
    result.metrics = summarize_metrics(selected_latencies, throughput, speedup);
    result.timed_run_latencies_ms = selected_latencies;
    return result;
}
