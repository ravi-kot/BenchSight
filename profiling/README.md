# BenchSight Profiling

This folder contains profiler-related request files and output directories for Nsight Systems and Nsight Compute.

## Recommended First Profiling Targets

- `reduction`
- `gemm`

## Suggested Workflow

1. Run the worker normally once with the request file.
2. Profile the same command with Nsight Systems.
3. Profile the same command with Nsight Compute.
4. Save the output files and screenshots in `profiling/outputs/`.

## Example Commands

### Nsight Systems

```cmd
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_baseline\reduction_baseline C:\Users\Admin\Workspace\BenchSight\workers\cuda-benchmarks\build\benchsight_worker.exe --request C:\Users\Admin\Workspace\BenchSight\profiling\requests\reduction_baseline.json --output C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_baseline\result.json
```

### Nsight Compute

```cmd
ncu --set full --target-processes all --force-overwrite -o C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_baseline\reduction_baseline_ncu C:\Users\Admin\Workspace\BenchSight\workers\cuda-benchmarks\build\benchsight_worker.exe --request C:\Users\Admin\Workspace\BenchSight\profiling\requests\reduction_baseline.json --output C:\Users\Admin\Workspace\BenchSight\profiling\outputs\reduction_baseline\result.json
```
