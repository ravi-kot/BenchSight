# BenchSight Profiling Workflow

## Purpose

BenchSight uses profiling as supporting evidence for benchmark results. The dashboard shows what improved, and Nsight helps explain why it improved.

## NVTX Strategy

The CUDA worker inserts NVTX ranges around major phases:

- allocation
- warmup
- timed runs
- cleanup

This makes the Nsight timeline easier to map back to the benchmark methodology.

## Recommended Workflow

1. Run the worker normally once.
2. Confirm the result JSON looks correct.
3. Run the same request under Nsight Systems.
4. Run the same request under Nsight Compute.
5. Save profiler outputs and screenshots in `profiling/outputs/`.

## Example Commands

### Nsight Systems

```cmd
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_optimized\reduction_optimized C:\Users\Admin\Workspace\BenchSight\workers\cuda-benchmarks\build\benchsight_worker.exe --request C:\Users\Admin\Workspace\BenchSight\profiling\requests\reduction_optimized.json --output C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_optimized\result.json
```

### Nsight Compute

```cmd
ncu --set full --target-processes all --force-overwrite -o C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_optimized\reduction_optimized_ncu C:\Users\Admin\Workspace\BenchSight\workers\cuda-benchmarks\build\benchsight_worker.exe --request C:\Users\Admin\Workspace\BenchSight\profiling\requests\reduction_optimized.json --output C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_optimized\result.json
```

## Where To View Results

- Nsight Systems report: open the generated `.nsys-rep` file in Nsight Systems
- Nsight Compute report: open the generated `.ncu-rep` file in Nsight Compute
- BenchSight charts: open the Streamlit dashboard in the browser

## Best Workloads To Profile

1. `reduction`
2. `gemm`
3. `conv2d`

These tend to produce clearer baseline-versus-optimized stories for presentation.
