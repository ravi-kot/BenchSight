# 🚀 BenchSight

> A streamlined GPU benchmarking and profiling presentation toolkit.

BenchSight simplifies the workflow for comparing custom CUDA kernels against standard implementations (like cuBLAS/cuDNN). It runs benchmarks, exports clean JSON, and provides a single **Streamlit dashboard** to visualize latency, throughput, stability, and speedup—making it presentation-ready out of the box.

## ✨ Features

- **Standard Workloads:** Supports `vector_add`, `reduction`, `gemm`, `conv2d`, and `softmax`.
- **Easy Comparisons:** Compare your `baseline` and `optimized` custom kernels directly with vendor libraries.
- **Profiling-Ready:** Includes Nsight Systems/Compute workflows and example outputs.
- **Visual Dashboard:** An interactive Streamlit app to analyze and present your results instantly.
- **Report Generation:** Automated word document reports.

## 🛠️ Getting Started

### 1. Run the Dashboard

```powershell
conda activate bench_env
cd C:\Users\Admin\Workspace\BenchSight\simple_dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 2. Explore the Project

- **`/workers/cuda-benchmarks/`**: Where the custom CUDA kernel magic happens.
- **`/simple_dashboard/`**: The Streamlit visualization app.
- **`/presentation/`**: Slide content, flow diagrams, and quick-learn notes.
- **`/profiling/`**: Nsight profiling assets and workflows.
- **`/output/doc/`**: Auto-generated project reports.

## 📊 The Workflow

1. **Run** custom CUDA kernels with timed trials.
2. **Export** clean, structured JSON results.
3. **Visualize** the data instantly in the dashboard.
4. **Validate** performance with Nsight Systems & Compute.
5. **Present** quantitative and profiling evidence effortlessly.
