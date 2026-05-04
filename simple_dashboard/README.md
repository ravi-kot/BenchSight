# BenchSight Simple Dashboard

This is the current BenchSight presentation MVP: one local Streamlit dashboard that reads benchmark result JSON files from the `results/` folder and visualizes latency, throughput, stability, and speedup in a single screen.

## What It Includes

- Executive overview for project presentation metrics
- Workload comparison charts
- Run explorer with raw JSON preview
- Sample benchmark results for:
  - `vector_add`
  - `reduction`
  - `gemm`
  - `conv2d`
  - `softmax`

## Run It

From the repository root:

```cmd
conda activate bench_env
cd C:\Users\Admin\Workspace\BenchSight\simple_dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Add Your Own Results

Drop additional `.json` files into:

```text
simple_dashboard/results/
```

## Result File Shape

Each benchmark result file should look like:

```json
{
  "run_id": "run_softmax_optimized_001",
  "timestamp": "2026-04-02T10:15:00Z",
  "workload": "softmax",
  "category": "deep_learning",
  "implementation": "optimized",
  "input_label": "1024 x 1024",
  "hardware": {
    "gpu_name": "RTX 5070",
    "cuda_version": "12.9"
  },
  "parameters": {
    "rows": 1024,
    "cols": 1024
  },
  "metrics": {
    "avg_latency_ms": 0.948,
    "p50_latency_ms": 0.941,
    "p95_latency_ms": 0.989,
    "throughput": 1106143.460,
    "speedup_vs_baseline": 1.670
  },
  "timed_run_latencies_ms": [0.939, 0.941, 0.936],
  "notes": "Parallel reduction softmax with stronger throughput and better tail behavior."
}
```
