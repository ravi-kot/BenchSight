# BenchSight Quick Learn

## What BenchSight Is

BenchSight is a GPU benchmarking and profiling project for evaluating custom CUDA kernels using a repeatable benchmark workflow and a presentation-ready dashboard.

## Core Project Flow

1. Implement a baseline and optimized CUDA kernel.
2. Run warmup iterations and timed trials.
3. Record average latency, p50, p95, throughput, and speedup.
4. Export the run as structured JSON.
5. Visualize the results in the Streamlit dashboard.
6. Validate performance differences with Nsight Systems and Nsight Compute.

## What You Should Know Before Presenting

### Technical Contribution

- Custom baseline and optimized CUDA kernels
- CUDA event timing
- NVTX ranges for profiler visibility
- Structured JSON result format
- Streamlit dashboard for benchmark comparison

### Why The Project Is Useful

- Standardizes timing and warmup methodology
- Makes results easier to compare
- Improves reproducibility
- Connects benchmark outputs to profiler evidence
- Makes performance work easier to present

### What BenchSight Is Not

- It is not a replacement for Nsight
- It is not a cluster scheduler
- It is not a large-scale benchmark suite

## Best One-Sentence Explanation

BenchSight is a benchmarking and profiling presentation platform for comparing custom CUDA kernels under a structured measurement workflow.

## Best Short Answer To “Why Not Just Use Nsight?”

Nsight explains why a kernel behaves the way it does. BenchSight measures, compares, and presents performance across runs and implementations.

## Metrics To Remember

- Average latency: overall mean execution time
- p50 latency: typical execution time
- p95 latency: tail latency and stability
- Throughput: useful work per unit time
- Speedup: optimized performance relative to baseline

## Best Workloads To Present

- Reduction
- GEMM
- Conv2D

These are easiest to explain and strongest for baseline-versus-optimized comparisons.
