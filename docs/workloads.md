# BenchSight Workloads

## Current Workloads

BenchSight currently focuses on five workloads:

1. `vector_add`
2. `reduction`
3. `gemm`
4. `conv2d`
5. `softmax`

## Implementation Strategy

Each workload is framed as a baseline-versus-optimized comparison using custom CUDA kernels.

- baseline: straightforward implementation for correctness and reference timing
- optimized: improved implementation using practical CUDA optimization ideas

Vendor-library references are included where they fit the workload:

- `cublas` for `vector_add`, `reduction`, `gemm`, and `conv2d`
- `cudnn` for `softmax` when cuDNN is installed and enabled at build time

## Why These Workloads Matter

### `vector_add`

- simple memory-bound calibration workload
- useful for validating the benchmark flow

### `reduction`

- strong synchronization and memory behavior example
- easy to explain during presentation

### `gemm`

- compute-heavy workload
- one of the strongest credibility workloads in the project

### `conv2d`

- deep learning relevance for vision-style operators
- useful for showing optimized data reuse patterns

### `softmax`

- transformer-style operator
- good for latency and stability discussion

## Benchmark Rules

- use the same input sizes across baseline and optimized variants
- run warmup iterations before timed trials
- report average latency, p50, p95, throughput, and speedup
- use profiler evidence when presenting optimized behavior

## Best Workloads To Present

1. `reduction`
2. `gemm`
3. `conv2d`

These are the most presentation-friendly workloads because the performance story is easier to explain.

## Future Reference Paths

Natural next additions are:

- cuDNN reference for `conv2d`
- cuBLASLt or tensor-core-specific variants for `gemm`
- dashboard profiler-artifact cards for Nsight evidence

These would strengthen the project by comparing custom kernels not only against baseline versions but also against optimized library implementations.
