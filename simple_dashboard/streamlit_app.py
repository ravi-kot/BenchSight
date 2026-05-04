from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"
IMPLEMENTATION_COLORS = {
    "baseline": "#94A3B8",
    "optimized": "#2563EB",
    "cublas": "#16A34A",
    "cudnn": "#9333EA",
}
DISPLAY_LABELS = {
    "vector_add": "Vector Add",
    "reduction": "Reduction",
    "gemm": "GEMM",
    "conv2d": "Conv2D",
    "softmax": "Softmax",
    "classical": "Classical GPU",
    "deep_learning": "Deep Learning",
    "baseline": "Baseline",
    "optimized": "Optimized",
    "cublas": "cuBLAS",
    "cudnn": "cuDNN",
}


def display_label(value: str) -> str:
    return DISPLAY_LABELS.get(value, value.replace("_", " ").title())


def compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"

    absolute = abs(float(value))
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}G"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def metric_text(value: float | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}{suffix}"


def gemm_gflops(parameters: dict[str, Any], avg_latency_ms: float | None) -> float | None:
    if avg_latency_ms is None or pd.isna(avg_latency_ms) or avg_latency_ms <= 0:
        return None

    try:
        m = float(parameters["m"])
        n = float(parameters["n"])
        k = float(parameters["k"])
    except (KeyError, TypeError, ValueError):
        return None

    return (2.0 * m * n * k) / (avg_latency_ms / 1000.0) / 1_000_000_000.0


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F6F8FC;
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background-color: #F9FBFF;
            border-right: 1px solid #E5EAF3;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E6ECF5;
            border-radius: 18px;
            padding: 0.65rem 0.85rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .hero-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #EEF5FF 100%);
            border: 1px solid #D9E8FF;
            border-radius: 22px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
        }
        .hero-kicker {
            color: #2563EB;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            color: #0F172A;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .hero-body {
            color: #334155;
            font-size: 1.02rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .hero-pill {
            background: #FFFFFF;
            border: 1px solid #E3ECFA;
            border-radius: 16px;
            padding: 0.85rem 1rem;
        }
        .hero-pill strong {
            display: block;
            color: #0F172A;
            font-size: 1.2rem;
            margin-bottom: 0.15rem;
        }
        .hero-pill span {
            color: #475569;
            font-size: 0.92rem;
        }
        .section-note {
            background: #FFFFFF;
            border: 1px solid #E6ECF5;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
        }
        .section-note strong {
            color: #0F172A;
        }
        .guide-card {
            background: #FFFFFF;
            border: 1px solid #E6ECF5;
            border-radius: 18px;
            padding: 1rem;
            min-height: 132px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .guide-card h4 {
            color: #0F172A;
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
        }
        .guide-card p {
            color: #475569;
            margin: 0;
            line-height: 1.55;
            font-size: 0.92rem;
        }
        .takeaway-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
            border: 1px solid #DFE9F7;
            border-radius: 18px;
            padding: 1rem;
            min-height: 132px;
        }
        .takeaway-card h4 {
            color: #0F172A;
            margin: 0 0 0.45rem 0;
        }
        .takeaway-card p {
            color: #475569;
            margin: 0;
            line-height: 1.55;
            font-size: 0.93rem;
        }
        .small-caption {
            color: #64748B;
            font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #EAF2FF;
            border-radius: 999px;
            padding: 0.5rem 1rem;
            color: #1E3A8A;
            font-weight: 600;
            border: 1px solid #D7E7FF;
        }
        .stTabs [aria-selected="true"] {
            background: #2563EB !important;
            color: #FFFFFF !important;
            border-color: #2563EB !important;
        }
        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    raw_payloads: dict[str, dict[str, Any]] = {}

    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        raw_payloads[path.name] = payload

        workload = payload.get("workload", "unknown")
        category = payload.get("category", "unknown")
        implementation = payload.get("implementation", "unknown")
        metrics = payload.get("metrics", {})
        hardware = payload.get("hardware", {})
        parameters = payload.get("parameters", {})
        avg_latency_ms = metrics.get("avg_latency_ms")
        p50_latency_ms = metrics.get("p50_latency_ms")
        p95_latency_ms = metrics.get("p95_latency_ms")
        speedup_vs_baseline = metrics.get("speedup_vs_baseline")
        throughput = metrics.get("throughput")
        computed_gemm_gflops = gemm_gflops(parameters, avg_latency_ms) if workload == "gemm" else None
        workload_label = display_label(workload)
        implementation_label = display_label(implementation)
        category_label = display_label(category)
        input_label = payload.get("input_label", "")
        display_name = " | ".join(
            part
            for part in [
                workload_label,
                implementation_label,
                input_label,
            ]
            if part
        )

        rows.append(
            {
                "run_id": payload.get("run_id", path.stem),
                "timestamp": payload.get("timestamp"),
                "workload": workload,
                "workload_label": workload_label,
                "category": category,
                "category_label": category_label,
                "implementation": implementation,
                "implementation_label": implementation_label,
                "input_label": input_label,
                "avg_latency_ms": avg_latency_ms,
                "p50_latency_ms": p50_latency_ms,
                "p95_latency_ms": p95_latency_ms,
                "throughput": throughput,
                "speedup_vs_baseline": speedup_vs_baseline,
                "gemm_gflops": computed_gemm_gflops,
                "gpu_name": hardware.get("gpu_name", "unknown"),
                "cuda_version": hardware.get("cuda_version", "unknown"),
                "parameters": json.dumps(parameters, indent=2),
                "notes": payload.get("notes", ""),
                "source_file": path.name,
                "display_label": display_name,
                "consistency_gap_ms": (
                    p95_latency_ms - p50_latency_ms
                    if p95_latency_ms is not None and p50_latency_ms is not None
                    else None
                ),
            }
        )

        for latency in payload.get("timed_run_latencies_ms", []):
            latency_rows.append(
                {
                    "source_file": path.name,
                    "display_label": display_name,
                    "workload": workload,
                    "workload_label": workload_label,
                    "implementation": implementation,
                    "implementation_label": implementation_label,
                    "latency_ms": latency,
                }
            )

    results_df = pd.DataFrame(rows)
    latencies_df = pd.DataFrame(latency_rows)

    if not results_df.empty:
        results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], errors="coerce")
        results_df = results_df.sort_values(
            by=["workload_label", "implementation_label", "timestamp"],
            ascending=[True, True, False],
        )

    return results_df, latencies_df, raw_payloads


st.set_page_config(page_title="BenchSight Performance Dashboard", layout="wide")
inject_styles()

if not RESULTS_DIR.exists():
    st.error(f"Results folder not found: {RESULTS_DIR}")
    st.stop()

results_df, latencies_df, raw_payloads = load_results(RESULTS_DIR)

if results_df.empty:
    st.warning("No valid benchmark JSON files were found in the results folder.")
    st.stop()

with st.sidebar:
    st.header("BenchSight")
    st.write(f"Results folder: `{RESULTS_DIR}`")

    if st.button("Refresh results"):
        st.cache_data.clear()
        st.rerun()

    st.header("Filters")
    category_options = sorted(results_df["category"].dropna().unique().tolist())
    workload_options = sorted(results_df["workload"].dropna().unique().tolist())
    implementation_options = sorted(results_df["implementation"].dropna().unique().tolist())

    selected_categories = st.multiselect(
        "Categories",
        category_options,
        default=category_options,
        format_func=display_label,
    )
    selected_workloads = st.multiselect(
        "Workloads",
        workload_options,
        default=workload_options,
        format_func=display_label,
    )
    selected_implementations = st.multiselect(
        "Variants",
        implementation_options,
        default=implementation_options,
        format_func=display_label,
    )
    with st.expander("Advanced"):
        show_raw_json = st.checkbox("Show raw technical JSON", value=False)
        st.caption("Use all results for the full platform view, or narrow to one workload for focused analysis.")

filtered_df = results_df[
    results_df["category"].isin(selected_categories)
    & results_df["workload"].isin(selected_workloads)
    & results_df["implementation"].isin(selected_implementations)
].copy()

if filtered_df.empty:
    st.warning("No runs match the current filters.")
    st.stop()

filtered_latency_df = latencies_df[latencies_df["source_file"].isin(filtered_df["source_file"])].copy()
runs_count = len(filtered_df)
workload_count = filtered_df["workload"].nunique()
category_count = filtered_df["category"].nunique()
fastest_avg_latency = filtered_df["avg_latency_ms"].min()
best_speedup = filtered_df["speedup_vs_baseline"].max()

fastest_row = filtered_df.loc[filtered_df["avg_latency_ms"].idxmin()]
throughput_row = filtered_df.loc[filtered_df["throughput"].idxmax()]
speedup_rows = filtered_df[filtered_df["speedup_vs_baseline"].notna()]
best_speedup_row = speedup_rows.loc[speedup_rows["speedup_vs_baseline"].idxmax()] if not speedup_rows.empty else None
stability_row = filtered_df.loc[filtered_df["consistency_gap_ms"].fillna(float("inf")).idxmin()]
overview_df = (
    filtered_df.groupby(["workload_label", "implementation"], as_index=False)[
        ["avg_latency_ms", "throughput", "speedup_vs_baseline", "p95_latency_ms"]
    ]
    .mean(numeric_only=True)
    .sort_values(by=["workload_label", "implementation"])
)
overview_speedup_df = overview_df[overview_df["speedup_vs_baseline"].notna()].copy()

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">BenchSight Performance Dashboard</div>
        <div class="hero-title">CUDA Workload Benchmark Results</div>
        <div class="hero-body">
            Custom baseline and optimized CUDA kernels benchmarked on RTX 5070 across representative classical GPU and deep learning workloads.
        </div>
        <div class="hero-grid">
            <div class="hero-pill">
                <strong>{runs_count}</strong>
                <span>runs in the current view</span>
            </div>
            <div class="hero-pill">
                <strong>{metric_text(fastest_avg_latency, " ms")}</strong>
                <span>fastest observed average latency</span>
            </div>
            <div class="hero-pill">
                <strong>{compact_number(throughput_row['throughput'])}</strong>
                <span>highest observed throughput</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
summary_col_1.metric("Runs Loaded", f"{runs_count}")
summary_col_2.metric("Workloads", f"{workload_count}")
summary_col_3.metric("Categories", f"{category_count}")
summary_col_4.metric("Best Speedup", metric_text(best_speedup, "x"))

story_tab, compare_tab, gemm_tab, explorer_tab = st.tabs(
    ["Overview", "Comparison", "GEMM Deep Dive", "Runs"]
)

with story_tab:
    st.caption("Lower latency is better. Higher throughput and speedup are better.")

    takeaway_col_1, takeaway_col_2, takeaway_col_3, takeaway_col_4 = st.columns(4)
    with takeaway_col_1:
        st.markdown(
            f"""
            <div class="takeaway-card">
                <h4>Fastest Run</h4>
                <p>
                    <strong>{fastest_row['display_label']}</strong><br>
                    <strong>{metric_text(fastest_row['avg_latency_ms'], ' ms')}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with takeaway_col_2:
        speedup_text = (
            f"<strong>{best_speedup_row['display_label']}</strong><br>"
            f"<strong>{metric_text(best_speedup_row['speedup_vs_baseline'], 'x')}</strong> over baseline"
            if best_speedup_row is not None
            else "No speedup data available"
        )
        st.markdown(
            f"""
            <div class="takeaway-card">
                <h4>Best Speedup</h4>
                <p>{speedup_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with takeaway_col_3:
        st.markdown(
            f"""
            <div class="takeaway-card">
                <h4>Highest Throughput</h4>
                <p>
                    <strong>{throughput_row['display_label']}</strong><br>
                    <strong>{compact_number(throughput_row['throughput'])}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with takeaway_col_4:
        st.markdown(
            f"""
            <div class="takeaway-card">
                <h4>Best Stability</h4>
                <p>
                    <strong>{stability_row['display_label']}</strong><br>
                    <strong>{metric_text(stability_row['consistency_gap_ms'], ' ms')}</strong> p95-p50 gap
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        latency_fig = px.bar(
            overview_df,
            x="workload_label",
            y="avg_latency_ms",
            color="implementation",
            color_discrete_map=IMPLEMENTATION_COLORS,
            barmode="group",
            title="Average Latency by Workload",
            hover_data=["workload_label"],
        )
        latency_fig.update_layout(xaxis_title="", yaxis_title="Average Latency (ms)")
        st.plotly_chart(latency_fig, use_container_width=True)

    with right_chart:
        throughput_fig = px.bar(
            overview_df,
            x="workload_label",
            y="throughput",
            color="implementation",
            color_discrete_map=IMPLEMENTATION_COLORS,
            barmode="group",
            title="Throughput by Workload",
            hover_data=["workload_label"],
        )
        throughput_fig.update_layout(xaxis_title="", yaxis_title="Throughput")
        st.plotly_chart(throughput_fig, use_container_width=True)

    lower_left, lower_right = st.columns(2)

    with lower_left:
        if not overview_speedup_df.empty:
            speedup_fig = px.bar(
                overview_speedup_df,
                x="workload_label",
                y="speedup_vs_baseline",
                color="implementation",
                color_discrete_map=IMPLEMENTATION_COLORS,
                barmode="group",
                title="Speedup by Workload",
                hover_data=["workload_label"],
            )
            speedup_fig.update_layout(xaxis_title="", yaxis_title="Speedup (x)")
            st.plotly_chart(speedup_fig, use_container_width=True)

    with lower_right:
        if not filtered_latency_df.empty:
            latency_dist_fig = px.box(
                filtered_latency_df,
                x="workload_label",
                y="latency_ms",
                color="implementation",
                color_discrete_map=IMPLEMENTATION_COLORS,
                points=False,
                title="Latency Distribution",
            )
            latency_dist_fig.update_layout(xaxis_title="", yaxis_title="Timed Latency (ms)")
            st.plotly_chart(latency_dist_fig, use_container_width=True)

    with st.expander("Per-workload summary"):
        best_per_workload = (
            filtered_df.sort_values(by=["workload_label", "speedup_vs_baseline"], ascending=[True, False])
            .drop_duplicates(subset=["workload"], keep="first")
            .loc[:, ["workload_label", "implementation_label", "avg_latency_ms", "p95_latency_ms", "throughput", "speedup_vs_baseline"]]
            .rename(
                columns={
                    "workload_label": "Workload",
                    "implementation_label": "Best Variant",
                    "avg_latency_ms": "Average Latency (ms)",
                    "p95_latency_ms": "p95 Latency (ms)",
                    "throughput": "Throughput",
                    "speedup_vs_baseline": "Speedup (x)",
                }
            )
        )
        best_per_workload["Throughput"] = best_per_workload["Throughput"].apply(compact_number)
        st.dataframe(best_per_workload, use_container_width=True, hide_index=True)

with compare_tab:
    st.subheader("Comparison")

    metric_options = {
        "Average Latency (lower is better)": "avg_latency_ms",
        "p50 Latency (lower is better)": "p50_latency_ms",
        "p95 Latency (lower is better)": "p95_latency_ms",
        "Throughput (higher is better)": "throughput",
        "Speedup vs Baseline (higher is better)": "speedup_vs_baseline",
        "GEMM GFLOP/s (higher is better)": "gemm_gflops",
    }
    selected_metric_label = st.selectbox("Comparison metric", list(metric_options.keys()))
    selected_metric = metric_options[selected_metric_label]

    comparison_df = (
        filtered_df.groupby(["workload_label", "implementation"], as_index=False)[selected_metric]
        .mean(numeric_only=True)
        .sort_values(by=["workload_label", "implementation"])
    )

    comparison_fig = px.bar(
        comparison_df,
        x="workload_label",
        y=selected_metric,
        color="implementation",
        color_discrete_map=IMPLEMENTATION_COLORS,
        barmode="group",
        title=selected_metric_label,
    )
    comparison_fig.update_layout(xaxis_title="", yaxis_title=selected_metric_label)
    st.plotly_chart(comparison_fig, use_container_width=True)

    scatter_fig = px.scatter(
        filtered_df,
        x="avg_latency_ms",
        y="throughput",
        color="workload_label",
        symbol="implementation_label",
        size="p95_latency_ms",
        hover_name="display_label",
        title="Latency vs Throughput Positioning",
    )
    scatter_fig.update_layout(xaxis_title="Average Latency (ms)", yaxis_title="Throughput")
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.caption("Bottom-right is the most favorable region: low latency and high throughput.")

with gemm_tab:
    st.subheader("GEMM Deep Dive")

    gemm_df = filtered_df[filtered_df["workload"] == "gemm"].copy()
    if gemm_df.empty:
        st.info("No GEMM results are available in the current filter selection.")
    else:
        implementation_order = {"baseline": 0, "optimized": 1, "cublas": 2}
        gemm_df["implementation_order"] = gemm_df["implementation"].map(implementation_order).fillna(99)
        gemm_summary_df = (
            gemm_df.groupby(["implementation", "implementation_label", "implementation_order"], as_index=False)
            .agg(
                avg_latency_ms=("avg_latency_ms", "mean"),
                p95_latency_ms=("p95_latency_ms", "mean"),
                speedup_vs_baseline=("speedup_vs_baseline", "mean"),
                gemm_gflops=("gemm_gflops", "mean"),
            )
            .sort_values(by=["implementation_order", "implementation_label"])
        )

        optimized_rows = gemm_summary_df[gemm_summary_df["implementation"] == "optimized"]
        cublas_rows = gemm_summary_df[gemm_summary_df["implementation"] == "cublas"]
        custom_candidates = gemm_summary_df[gemm_summary_df["implementation"] != "cublas"].copy()
        custom_row = None
        if not optimized_rows.empty:
            custom_row = optimized_rows.iloc[0]
        elif not custom_candidates.empty:
            custom_row = custom_candidates.sort_values(by="gemm_gflops", ascending=False).iloc[0]
        cublas_row = cublas_rows.iloc[0] if not cublas_rows.empty else None
        custom_gflops = None if custom_row is None else custom_row["gemm_gflops"]
        cublas_gflops = None if cublas_row is None else cublas_row["gemm_gflops"]
        custom_vs_cublas = (
            None
            if custom_gflops is None
            or cublas_gflops is None
            or pd.isna(custom_gflops)
            or pd.isna(cublas_gflops)
            or cublas_gflops == 0
            else (custom_gflops / cublas_gflops) * 100.0
        )
        custom_speedup = None if custom_row is None else custom_row["speedup_vs_baseline"]

        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric("Custom GFLOP/s", metric_text(custom_gflops))
        metric_col_2.metric("cuBLAS GFLOP/s", metric_text(cublas_gflops))
        metric_col_3.metric("Custom / cuBLAS", metric_text(custom_vs_cublas, "%"))
        metric_col_4.metric("Custom Speedup", metric_text(custom_speedup, "x"))

        gemm_chart_left, gemm_chart_right = st.columns(2)
        with gemm_chart_left:
            gflops_fig = px.bar(
                gemm_summary_df,
                x="implementation_label",
                y="gemm_gflops",
                color="implementation",
                color_discrete_map=IMPLEMENTATION_COLORS,
                title="GEMM Compute Throughput",
                text="gemm_gflops",
            )
            gflops_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            gflops_fig.update_layout(xaxis_title="", yaxis_title="GFLOP/s")
            st.plotly_chart(gflops_fig, use_container_width=True)

        with gemm_chart_right:
            gemm_latency_fig = px.bar(
                gemm_summary_df,
                x="implementation_label",
                y="avg_latency_ms",
                color="implementation",
                color_discrete_map=IMPLEMENTATION_COLORS,
                title="GEMM Average Latency",
                text="avg_latency_ms",
            )
            gemm_latency_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            gemm_latency_fig.update_layout(xaxis_title="", yaxis_title="Average Latency (ms)")
            st.plotly_chart(gemm_latency_fig, use_container_width=True)

        gemm_latency_df = filtered_latency_df[filtered_latency_df["workload"] == "gemm"].copy()
        if not gemm_latency_df.empty:
            gemm_distribution_fig = px.box(
                gemm_latency_df,
                x="implementation_label",
                y="latency_ms",
                color="implementation",
                color_discrete_map=IMPLEMENTATION_COLORS,
                points=False,
                title="GEMM Timed-Run Stability",
            )
            gemm_distribution_fig.update_layout(xaxis_title="", yaxis_title="Timed Latency (ms)")
            st.plotly_chart(gemm_distribution_fig, use_container_width=True)

        gemm_table = gemm_summary_df[
            [
                "implementation_label",
                "avg_latency_ms",
                "p95_latency_ms",
                "gemm_gflops",
                "speedup_vs_baseline",
            ]
        ].rename(
            columns={
                "implementation_label": "Variant",
                "avg_latency_ms": "Average Latency (ms)",
                "p95_latency_ms": "p95 Latency (ms)",
                "gemm_gflops": "GFLOP/s",
                "speedup_vs_baseline": "Speedup (x)",
            }
        )
        for column in ["Average Latency (ms)", "p95 Latency (ms)", "GFLOP/s", "Speedup (x)"]:
            gemm_table[column] = gemm_table[column].map(lambda value: None if pd.isna(value) else round(float(value), 3))
        st.dataframe(gemm_table, use_container_width=True, hide_index=True)

with explorer_tab:
    st.subheader("Run Data")

    display_table = filtered_df[
        [
            "run_id",
            "workload_label",
            "implementation_label",
            "input_label",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "throughput",
            "gemm_gflops",
            "speedup_vs_baseline",
            "gpu_name",
            "source_file",
        ]
    ].copy()
    display_table = display_table.rename(
        columns={
            "run_id": "Run ID",
            "workload_label": "Workload",
            "implementation_label": "Variant",
            "input_label": "Input",
            "avg_latency_ms": "Average Latency (ms)",
            "p50_latency_ms": "p50 Latency (ms)",
            "p95_latency_ms": "p95 Latency (ms)",
            "throughput": "Throughput",
            "gemm_gflops": "GEMM GFLOP/s",
            "speedup_vs_baseline": "Speedup (x)",
            "gpu_name": "GPU",
            "source_file": "Source File",
        }
    )
    display_table["Throughput"] = display_table["Throughput"].apply(compact_number)
    for column in ["Average Latency (ms)", "p50 Latency (ms)", "p95 Latency (ms)", "GEMM GFLOP/s", "Speedup (x)"]:
        display_table[column] = display_table[column].map(
            lambda value: None if pd.isna(value) else round(float(value), 3)
        )

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    selected_file = st.selectbox("Inspect a run", filtered_df["source_file"].tolist())
    selected_row = filtered_df[filtered_df["source_file"] == selected_file].iloc[0]

    detail_col_1, detail_col_2 = st.columns(2)
    with detail_col_1:
        st.markdown(f"**Run ID:** `{selected_row['run_id']}`")
        st.markdown(f"**Workload:** `{selected_row['workload_label']}`")
        st.markdown(f"**Variant:** `{selected_row['implementation_label']}`")
        st.markdown(f"**Input:** `{selected_row['input_label']}`")
        st.markdown(f"**GPU:** `{selected_row['gpu_name']}`")
        st.markdown(f"**CUDA:** `{selected_row['cuda_version']}`")

    with detail_col_2:
        st.markdown(f"**Average Latency:** `{metric_text(selected_row['avg_latency_ms'], ' ms')}`")
        st.markdown(f"**p50 Latency:** `{metric_text(selected_row['p50_latency_ms'], ' ms')}`")
        st.markdown(f"**p95 Latency:** `{metric_text(selected_row['p95_latency_ms'], ' ms')}`")
        throughput_value = selected_row["throughput"]
        throughput_text = "n/a" if pd.isna(throughput_value) else compact_number(float(throughput_value))
        st.markdown(f"**Throughput:** `{throughput_text}`")
        st.markdown(f"**GEMM GFLOP/s:** `{metric_text(selected_row['gemm_gflops'])}`")
        st.markdown(f"**Speedup:** `{metric_text(selected_row['speedup_vs_baseline'], 'x')}`")

    if selected_row["notes"]:
        st.markdown(f"**Why this run matters:** {selected_row['notes']}")

    if show_raw_json:
        with st.expander("Raw Result JSON"):
            st.json(raw_payloads[selected_file])
