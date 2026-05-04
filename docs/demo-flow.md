# BenchSight Demo Flow

## Goal

Show that BenchSight is a complete benchmarking workflow, not just a set of isolated CUDA files.

## Recommended 5 Minute Demo

1. Open the Streamlit dashboard.
2. Describe BenchSight in one sentence: custom CUDA kernels, structured benchmarking, dashboard comparison, and Nsight-backed profiling.
3. Show one workload comparison, preferably `reduction` or `gemm`.
4. Point out average latency, throughput, and speedup.
5. Open the run data view and show that each result is backed by structured JSON.
6. Open `presentation/project_flow.html` and explain the end-to-end flow.
7. Show one Nsight screenshot or report and connect it back to the dashboard result.

## Best Workloads To Show

- `reduction`
- `gemm`
- `conv2d`

## Core Talking Points

- repeatable warmup and timed-run methodology
- custom baseline and optimized CUDA kernels
- structured JSON benchmark results
- presentation-ready dashboard views
- profiler-backed explanation of performance differences
