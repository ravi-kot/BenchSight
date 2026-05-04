#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define BENCHSIGHT_HAVE_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define BENCHSIGHT_HAVE_NVTX 1
#else
#define BENCHSIGHT_HAVE_NVTX 0
#endif

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t error_code__ = (call);                                                                             \
        if (error_code__ != cudaSuccess) {                                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error_code__));                 \
        }                                                                                                              \
    } while (0)

class NvtxScopedRange {
public:
    explicit NvtxScopedRange(const char* label) {
#if BENCHSIGHT_HAVE_NVTX
        nvtxRangePushA(label);
#else
        (void)label;
#endif
    }

    ~NvtxScopedRange() {
#if BENCHSIGHT_HAVE_NVTX
        nvtxRangePop();
#endif
    }
};

inline std::vector<double> measure_cuda_iterations(
    const std::function<void()>& launcher,
    int warmup_runs,
    int timed_runs
) {
    {
        NvtxScopedRange warmup_range("warmup");
        for (int iteration = 0; iteration < warmup_runs; ++iteration) {
            launcher();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    NvtxScopedRange timed_range("timed_runs");
    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(timed_runs));

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    for (int iteration = 0; iteration < timed_runs; ++iteration) {
        CUDA_CHECK(cudaEventRecord(start_event));
        launcher();
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        latencies_ms.push_back(static_cast<double>(elapsed_ms));
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    return latencies_ms;
}
