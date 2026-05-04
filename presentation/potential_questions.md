# BenchSight Potential Questions

## 1. Why not just use Nsight?

Nsight is a profiling tool. BenchSight is the benchmarking and comparison layer that standardizes measurement, compares implementations, and presents results in a structured way.

## 2. What is the main contribution of the project?

The main contribution is the integration of custom CUDA benchmarking, structured result export, visual comparison, and profiler-backed explanation into a coherent workflow.

## 3. Are you using your own kernels or vendor libraries?

The current implementation is built around custom baseline and optimized CUDA kernels. It also includes vendor-library references where appropriate: cuBLAS for BLAS-style workloads and cuDNN for softmax when cuDNN is installed.

## 4. Why is this useful for optimization work?

Because it makes performance experiments repeatable and comparable. That improves confidence in optimization decisions and makes it easier to validate whether a change actually helped.

## 5. Why is p95 latency important?

p95 latency captures tail behavior. A kernel with a good average but poor p95 may be unstable, which makes it less reliable in repeated or production-like settings.

## 6. Why did you choose these workloads?

They cover both classical GPU patterns and modern deep learning operators while keeping the scope manageable. This gives the project both educational value and practical relevance.

## 7. What would you add next?

- cuBLASLt or tensor-core-specific GEMM variants
- cuDNN reference for Conv2D
- more workloads such as LayerNorm or attention
- tighter profiler artifact integration into the dashboard

## 8. Is this a production platform?

Not yet. It is a focused benchmarking and profiling workflow prototype designed for evaluation, comparison, and presentation. The architecture still leaves room for future expansion.

## 9. How do you ensure fair comparison?

By using the same workload parameters across implementations, including warmup runs, repeating timed trials, and computing comparable summary metrics such as average latency, p50, p95, throughput, and speedup.

## 10. What is the strongest evidence that the optimized kernel is better?

The strongest evidence is the combination of dashboard results and Nsight evidence. The dashboard shows lower latency and higher throughput, while Nsight explains the underlying execution differences.
