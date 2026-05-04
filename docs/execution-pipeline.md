# BenchSight Execution Pipeline

## Why This Exists

BenchSight focuses on standardized benchmark execution. The execution pipeline defines how a workload moves from custom CUDA kernel code to a presentation-ready result.

## Current Pipeline

1. Choose a workload and implementation.
2. Create or load a JSON request file.
3. Run the CUDA benchmark worker.
4. Execute warmup iterations.
5. Execute timed trials using CUDA events.
6. Aggregate average latency, p50, p95, throughput, and speedup.
7. Export the structured result as JSON.
8. Load the result into the Streamlit dashboard.
9. Re-run the same request under Nsight for profiler evidence.

## Example Worker Request

```json
{
  "job_id": "profile_reduction_optimized",
  "workload": "reduction",
  "implementation": "optimized",
  "warmup_runs": 10,
  "timed_runs": 50,
  "parameters": {
    "num_elements": 1048576,
    "block_size": 256
  },
  "artifacts_dir": "profiling/outputs/reduction_optimized"
}
```

## Example Result Shape

```json
{
  "job_id": "profile_reduction_optimized",
  "run_label": "reduction-optimized-demo",
  "status": "completed",
  "workload": "reduction",
  "implementation": "optimized",
  "metrics": {
    "avg_latency_ms": 0.31,
    "p50_latency_ms": 0.30,
    "p95_latency_ms": 0.34,
    "throughput": 3350000000.0,
    "speedup_vs_baseline": 1.82
  },
  "parameters": {
    "num_elements": 1048576,
    "block_size": 256
  },
  "timed_run_latencies_ms": [0.31, 0.30, 0.32]
}
```

## Why JSON Is Used

- Easy to inspect manually
- Easy to load into Streamlit
- Easy to reuse for Nsight comparisons
- Easy to archive as part of the report or presentation

## Debugging Flow

If a run fails:

1. Inspect the request JSON.
2. Confirm the worker path and build output.
3. Re-run the worker manually with the same request.
4. Profile the same command only after the normal run succeeds.
