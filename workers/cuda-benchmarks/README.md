# BenchSight CUDA Worker

This directory contains the local CUDA benchmark executable used by the FastAPI backend.

## Responsibilities

- Read a benchmark request JSON file
- Execute one selected workload
- Use CUDA events for timing
- Add NVTX ranges around benchmark phases
- Write a structured `result.json` artifact

## Supported MVP Workloads

- `vector_add`
- `reduction`
- `gemm`
- `conv2d`
- `softmax`

Each workload supports:

- `baseline`
- `optimized`

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The backend expects the executable at:

```text
workers/cuda-benchmarks/build/Release/benchsight_worker.exe
```

## CLI Contract

```powershell
.\build\Release\benchsight_worker.exe --request path\to\request.json --output path\to\result.json
```

## Notes

- The worker uses a file-based contract because it is easy to inspect during debugging.
- NVTX ranges are included so the benchmark phases can be profiled in Nsight workflows later.
- The optimized implementations are intentionally practical MVP versions, not final maximum-tuned kernels.
