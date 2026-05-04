# BenchSight Presentation Slides

## Slide 1: Title

### Slide Content

- BenchSight
- A GPU Benchmarking and Profiling Workflow for Custom CUDA Kernels
- Presenter name
- Course / date

### Speaker Notes

Today I am presenting BenchSight, a GPU benchmarking and profiling workflow designed to evaluate custom CUDA kernels under a repeatable measurement methodology. The goal of the project is not only to optimize kernels, but also to measure, compare, and present performance results in a structured way.

## Slide 2: Problem Overview

### Slide Content

- Lack of standardized benchmarking methodology
- Inconsistent timing practices
- No persistent or structured result collection
- Difficult to compare runs or reproduce experiments

### Speaker Notes

The core problem is that GPU performance experiments are often done with ad hoc scripts. Timing practices vary, warmup is sometimes skipped, results are scattered, and it becomes difficult to compare baseline and optimized implementations fairly.

## Slide 3: Motivation and Importance

### Slide Content

- Leads to incorrect optimization decisions
- Reduces confidence in experimental results
- Slows down debugging and iteration
- Makes improvements harder to validate

### Speaker Notes

This matters because poor benchmark methodology can produce misleading conclusions. A kernel may appear faster in one run and then perform inconsistently under repetition. Without structured results and reproducible methodology, it becomes difficult to trust performance claims.

## Slide 4: Current Approaches and Limitations

### Slide Content

- Ad hoc benchmark scripts
- Standalone profiling tools such as Nsight
- Large benchmark suites

### Speaker Notes

Current approaches all have trade-offs. Ad hoc scripts are easy to write but not structured. Profiling tools like Nsight are excellent for deep analysis, but they are not designed to manage the full benchmarking workflow. Large benchmark suites are powerful, but often too complex for focused experiments and course-scale projects.

## Slide 5: Proposed Solution

### Slide Content

- BenchSight standardizes GPU benchmark execution
- Compares custom baseline and optimized kernels
- Exports structured JSON benchmark results
- Visualizes metrics in a Streamlit dashboard
- Connects benchmark results to Nsight evidence

### Speaker Notes

Our solution is BenchSight. It provides a cleaner workflow for custom CUDA benchmarking: execute workloads under a consistent methodology, export structured results, compare them visually, and connect those results to profiler evidence from Nsight Systems and Nsight Compute.

## Slide 6: Workload Suite

### Slide Content

- Classical GPU:
  - Vector Add
  - Reduction
  - GEMM
- Deep learning:
  - Conv2D
  - Softmax
- Each workload includes baseline and optimized variants

### Speaker Notes

The selected workload suite covers both classical GPU patterns and deep learning operators. This gives the project a balanced workload mix while keeping the implementation scope realistic. Each workload includes a baseline version and an optimized version so that speedup can be discussed directly.

## Slide 7: Project Workflow

### Slide Content

- Configure workload and parameters
- Warmup runs
- Timed CUDA trials
- Aggregate latency and throughput metrics
- Export results as JSON
- Visualize in the dashboard
- Validate with Nsight

### Speaker Notes

This slide shows the project workflow. We configure the workload, run warmup iterations, run timed trials using CUDA events, aggregate performance metrics such as average latency, p50, p95, throughput, and speedup, export the result as structured JSON, and then visualize everything in the dashboard. Nsight is then used to explain why the optimized kernel behaves differently.

## Slide 8: Initial Results

### Slide Content

- Baseline vs optimized comparisons
- Lower latency for optimized kernels
- Higher throughput on optimized paths
- Visible speedup across selected workloads
- Nsight confirms the performance difference

### Speaker Notes

The initial results show that the optimized kernels improve over baseline in both latency and throughput. The dashboard makes these comparisons easy to present, while Nsight provides lower-level evidence that supports the observed gains.

## Slide 9: Why BenchSight Instead of Only Nsight?

### Slide Content

- Nsight is the profiler
- BenchSight is the benchmarking and comparison layer
- BenchSight standardizes methodology
- BenchSight makes results easier to compare and present

### Speaker Notes

This is an important distinction. Nsight helps us understand why a kernel behaves a certain way. BenchSight helps us measure performance across repeated runs, compare implementations, and present the results in a structured way. So BenchSight is not a replacement for Nsight. It complements it.

## Slide 10: Conclusion

### Slide Content

- BenchSight improves benchmark structure and presentation
- Supports custom CUDA kernel evaluation
- Combines results visualization with profiler-backed analysis
- Provides a clean foundation for future expansion

### Speaker Notes

In conclusion, BenchSight turns GPU benchmarking into a cleaner, more reproducible, and more presentable workflow. It supports custom CUDA kernel evaluation, structured comparison, and profiler-backed analysis, while still remaining realistic and lightweight enough for a focused project.
