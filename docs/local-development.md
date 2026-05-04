# BenchSight Local Development

## What You Need

- Python 3.11+
- `pip`
- CUDA Toolkit with `nvcc`
- CMake 3.24+
- A supported MSVC compiler for your CUDA installation

## Main Working Folders

- `workers/cuda-benchmarks/` for CUDA workloads and the benchmark worker
- `simple_dashboard/` for the Streamlit visualization app
- `profiling/` for Nsight requests and output artifacts

## Build The CUDA Worker

### Standard CMake

```powershell
cd C:\Users\Admin\Workspace\BenchSight\workers\cuda-benchmarks
cmake -S . -B build
cmake --build build --config Release
```

### Ninja Build On Windows

```powershell
cd C:\Users\Admin\Workspace\BenchSight\workers\cuda-benchmarks
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run The Streamlit Dashboard

```powershell
cd C:\Users\Admin\Workspace\BenchSight\simple_dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open:

- `http://localhost:8501`

## Suggested Local Validation

1. Build the CUDA worker successfully.
2. Confirm the sample JSON result files exist in `simple_dashboard/results/`.
3. Start the Streamlit dashboard.
4. Open the dashboard and verify that the charts load correctly.
5. Profile one workload using the request files in `profiling/requests/`.

## Windows Notes

- Build the worker in `Release` mode for meaningful timing.
- If PowerShell blocks script activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

- If you use Ninja, the worker executable is typically:

```text
workers/cuda-benchmarks/build/benchsight_worker.exe
```
