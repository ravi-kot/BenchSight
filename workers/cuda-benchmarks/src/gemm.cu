#include "benchmark_harness.cuh"
#include "cublas_utils.hpp"
#include "parameter_utils.hpp"
#include "result_utils.hpp"
#include "stats.hpp"
#include "workloads.hpp"

#include <stdexcept>
#include <vector>

namespace {

__global__ void gemm_baseline_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float value = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        value += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = value;
}

template <int TileSize>
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[TileSize][TileSize];
    __shared__ float tile_b[TileSize][TileSize];

    const int row = blockIdx.y * TileSize + threadIdx.y;
    const int col = blockIdx.x * TileSize + threadIdx.x;

    float value = 0.0f;
    const int tiles = (k + TileSize - 1) / TileSize;

    for (int tile = 0; tile < tiles; ++tile) {
        const int tiled_col = tile * TileSize + threadIdx.x;
        const int tiled_row = tile * TileSize + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k) ? a[row * k + tiled_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (tiled_row < k && col < n) ? b[tiled_row * n + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < TileSize; ++inner) {
            value += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = value;
    }
}

}  // namespace

BenchmarkResult run_gemm(const BenchmarkRequest& request) {
    NvtxScopedRange total_range("gemm_total");

    const int m = get_optional_int_parameter(request, "m", 512);
    const int n = get_optional_int_parameter(request, "n", 512);
    const int k = get_optional_int_parameter(request, "k", 512);
    const int tile_size = get_optional_int_parameter(request, "tile_size", 16);
    const bool use_optimized = request.implementation == "optimized";
    const bool use_cublas = request.implementation == "cublas";

    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::runtime_error("GEMM dimensions must be positive.");
    }
    if (tile_size != 16 && tile_size != 32) {
        throw std::runtime_error("GEMM tile_size must be 16 or 32.");
    }
    if (request.implementation != "baseline" && !use_optimized && !use_cublas) {
        throw std::runtime_error("Unsupported GEMM implementation: " + request.implementation);
    }

    const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t c_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> host_a(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    std::vector<float> host_b(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    std::vector<float> host_c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);

    for (std::size_t index = 0; index < host_a.size(); ++index) {
        host_a[index] = static_cast<float>((index % 13) * 0.05f);
    }
    for (std::size_t index = 0; index < host_b.size(); ++index) {
        host_b[index] = static_cast<float>((index % 17) * 0.04f);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), c_bytes));
    CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle = nullptr;
    if (use_cublas) {
        cublas_check(cublasCreate(&cublas_handle));
    }

    const dim3 baseline_block(16, 16);
    const dim3 baseline_grid((n + baseline_block.x - 1) / baseline_block.x, (m + baseline_block.y - 1) / baseline_block.y);

    auto baseline_launcher = [&]() {
        gemm_baseline_kernel<<<baseline_grid, baseline_block>>>(device_a, device_b, device_c, m, n, k);
        CUDA_CHECK(cudaGetLastError());
    };

    auto optimized_launcher = [&]() {
        if (tile_size == 32) {
            const dim3 block(32, 32);
            const dim3 grid((n + 31) / 32, (m + 31) / 32);
            gemm_tiled_kernel<32><<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        } else {
            const dim3 block(16, 16);
            const dim3 grid((n + 15) / 16, (m + 15) / 16);
            gemm_tiled_kernel<16><<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    auto cublas_launcher = [&]() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublas_check(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            device_b,
            n,
            device_a,
            k,
            &beta,
            device_c,
            n));
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

    CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

    if (cublas_handle != nullptr) {
        cublas_check(cublasDestroy(cublas_handle));
    }
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    const double avg_latency_ms = compute_average(selected_latencies);
    const double throughput = (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) /
                              (avg_latency_ms / 1000.0);

    BenchmarkResult result;
    result.metrics = summarize_metrics(selected_latencies, throughput, speedup);
    result.timed_run_latencies_ms = selected_latencies;
    return result;
}
