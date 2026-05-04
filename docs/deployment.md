# BenchSight Deployment Notes

## Current Deployment Shape

The current BenchSight scope is local and presentation-first:

- CUDA worker runs on the GPU-capable machine
- result files are stored locally as JSON
- Streamlit dashboard runs locally and visualizes those results
- Nsight outputs are stored locally as profiler artifacts

## Why This Scope Is Good

- Simpler to run and demonstrate
- Easier to explain during presentation
- Keeps the focus on benchmark methodology and performance evidence
- Avoids unnecessary infrastructure complexity

## Near-Term Expansion

- add cuDNN reference paths for more deep learning operators
- expand GEMM with cuBLASLt or tensor-core-specific variants
- ingest profiler artifact metadata into the dashboard
- add more workloads such as LayerNorm or attention

## Longer-Term Expansion

- add a backend orchestration layer if remote execution becomes necessary
- support stored result catalogs beyond local JSON files
- package the dashboard and worker workflow for a cleaner demo distribution
