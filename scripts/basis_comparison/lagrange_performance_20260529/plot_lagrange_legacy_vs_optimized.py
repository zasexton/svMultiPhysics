"""Focused legacy-vs-optimized Lagrange basis performance report.

The input CSVs are produced by the existing basis comparison test harness:

  SVMP_BASIS_COMPARE_OUT=<this-folder>/data \
    build/svMultiPhysics-build/bin/run_all_unit_tests \
    --gtest_filter='LagrangeBasisComparison.*'

  SVMP_FE_RUN_PERF_TESTS=1 \
  SVMP_BASIS_COMPARE_OUT=<this-folder>/data \
    build/svMultiPhysics-build/bin/run_all_unit_tests \
    --gtest_filter='LagrangeBasisPerformance.*'

Only element/function pairs with legacy equivalents are plotted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from _helpers import ELEMENT_DIM, ELEMENT_ORDER, filter_present  # noqa: E402


DATA_DIR = SCRIPT_DIR / "data"
PERF_DIR = DATA_DIR / "perf"
FIG_DIR = SCRIPT_DIR / "figures"
BYTES_PER_DOUBLE = 8

LEGACY_COLOR = "#264653"
OPT_COLOR = "#2a9d8f"
ALT_COLOR = "#e9c46a"
ACCENT_COLOR = "#e76f51"
NEUTRAL_COLOR = "#6c757d"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input CSV is missing: {path}. Run the comparison and "
            "performance tests first."
        )


def _read_csv(path: Path) -> pd.DataFrame:
    _require_file(path)
    return pd.read_csv(path)


def _present_elements(*dfs: pd.DataFrame) -> list[str]:
    present = set(ELEMENT_ORDER)
    for df in dfs:
        if "elem_type" in df.columns:
            present &= set(df["elem_type"].unique())
    return [elem for elem in ELEMENT_ORDER if elem in present]


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    svg = FIG_DIR / f"{name}.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


def _safe_log_values(values: pd.Series | np.ndarray, floor: float = 1.0e-18) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.maximum(arr, floor)


def _bar_offsets(count: int, width: float) -> list[float]:
    return [(i - (count - 1) / 2.0) * width for i in range(count)]


def build_accuracy_summary() -> pd.DataFrame:
    summary = _read_csv(DATA_DIR / "summary.csv")
    mass = _read_csv(DATA_DIR / "mass_matrix.csv")
    pou = _read_csv(DATA_DIR / "partition_of_unity.csv")

    elems = _present_elements(summary, mass, pou)
    mass_max = mass.groupby("elem_type")["abs_diff"].max()
    pou_abs = pou.copy()
    pou_abs["abs_pou_legacy"] = (pou_abs["sum_N_legacy"] - 1.0).abs()
    pou_abs["abs_pou_oop"] = (pou_abs["sum_N_oop"] - 1.0).abs()
    pou_abs["max_abs_pou_sample"] = pou_abs[["abs_pou_legacy", "abs_pou_oop"]].max(axis=1)
    pou_max = pou_abs.groupby("elem_type")["max_abs_pou_sample"].max()
    grad_sum_max = pou_abs.groupby("elem_type")[
        ["grad_sum_norm_legacy", "grad_sum_norm_oop"]
    ].max().max(axis=1)

    rows = []
    for elem in elems:
        sub = summary[summary["elem_type"] == elem].iloc[0]
        rows.append(
            {
                "elem_type": elem,
                "eNoN": int(sub["eNoN"]),
                "n_samples": int(sub["n_samples"]),
                "max_abs_value_error": float(sub["max_abs_err_value"]),
                "max_abs_gradient_error": float(sub["max_abs_err_grad"]),
                "max_abs_partition_error": float(pou_max.get(elem, math.nan)),
                "max_gradient_sum_norm": float(grad_sum_max.get(elem, math.nan)),
                "max_abs_mass_matrix_diff": float(mass_max.get(elem, math.nan)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "lagrange_accuracy_summary.csv", index=False)
    return out


def plot_accuracy(df: pd.DataFrame) -> None:
    elems = filter_present(df, "elem_type", ELEMENT_ORDER)
    indexed = df.set_index("elem_type").reindex(elems)
    x = np.arange(len(elems))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        _safe_log_values(indexed["max_abs_value_error"]),
        width,
        label="values",
        color=LEGACY_COLOR,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].bar(
        x + width / 2,
        _safe_log_values(indexed["max_abs_gradient_error"]),
        width,
        label="gradients",
        color=OPT_COLOR,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(elems)
    axes[0].set_ylabel("max absolute error, floor 1e-18")
    axes[0].set_title("Basis value and gradient agreement")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[0].legend()

    invariant_cols = [
        ("max_abs_partition_error", "partition of unity", ALT_COLOR),
        ("max_gradient_sum_norm", "gradient sum", ACCENT_COLOR),
        ("max_abs_mass_matrix_diff", "mass matrix", NEUTRAL_COLOR),
    ]
    width = 0.24
    for offset, (col, label, color) in zip(_bar_offsets(len(invariant_cols), width), invariant_cols):
        axes[1].bar(
            x + offset,
            _safe_log_values(indexed[col]),
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(elems)
    axes[1].set_ylabel("max diagnostic error, floor 1e-18")
    axes[1].set_title("Accuracy diagnostics")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Legacy solver vs optimized LagrangeBasis accuracy")
    _save(fig, "lagrange_accuracy_errors")


def build_pointwise_comparison() -> pd.DataFrame:
    micro = _read_csv(PERF_DIR / "perf_microbench_pointwise.csv")
    elems = filter_present(micro, "elem_type", ELEMENT_ORDER)
    rows = []
    for elem in elems:
        sub = micro[micro["elem_type"] == elem]
        legacy_vg = sub[
            (sub["operation"] == "values_and_gradients")
            & (sub["implementation"] == "legacy")
        ]
        oop_values = sub[(sub["operation"] == "values") & (sub["implementation"] == "oop")]
        oop_grad = sub[(sub["operation"] == "gradients") & (sub["implementation"] == "oop")]
        if not legacy_vg.empty and not oop_values.empty and not oop_grad.empty:
            legacy_ns = float(legacy_vg.iloc[0]["ns_per_call"])
            oop_ns = float(oop_values.iloc[0]["ns_per_call"]) + float(
                oop_grad.iloc[0]["ns_per_call"]
            )
            rows.append(
                {
                    "elem_type": elem,
                    "workload": "values_plus_gradients",
                    "legacy_ns_per_call": legacy_ns,
                    "optimized_ns_per_call": oop_ns,
                    "speedup_legacy_over_optimized": legacy_ns / oop_ns,
                }
            )

        legacy_h = sub[(sub["operation"] == "hessians") & (sub["implementation"] == "legacy")]
        oop_h = sub[(sub["operation"] == "hessians") & (sub["implementation"] == "oop")]
        if not legacy_h.empty and not oop_h.empty:
            legacy_ns = float(legacy_h.iloc[0]["ns_per_call"])
            oop_ns = float(oop_h.iloc[0]["ns_per_call"])
            rows.append(
                {
                    "elem_type": elem,
                    "workload": "hessians",
                    "legacy_ns_per_call": legacy_ns,
                    "optimized_ns_per_call": oop_ns,
                    "speedup_legacy_over_optimized": legacy_ns / oop_ns,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "lagrange_pointwise_fair_comparison.csv", index=False)
    return out


def plot_pointwise(df: pd.DataFrame) -> None:
    elems = filter_present(df, "elem_type", ELEMENT_ORDER)
    workloads = [w for w in ["values_plus_gradients", "hessians"] if w in set(df["workload"])]
    fig, axes = plt.subplots(1, len(workloads), figsize=(7 * len(workloads), 5.5), constrained_layout=True)
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        sub = df[df["workload"] == workload].set_index("elem_type").reindex(elems)
        sub = sub.dropna(subset=["legacy_ns_per_call", "optimized_ns_per_call"], how="all")
        sub_elems = list(sub.index)
        x = np.arange(len(sub_elems))
        width = 0.34
        ax.bar(
            x - width / 2,
            sub["legacy_ns_per_call"],
            width,
            label="legacy",
            color=LEGACY_COLOR,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + width / 2,
            sub["optimized_ns_per_call"],
            width,
            label="optimized",
            color=OPT_COLOR,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub_elems)
        ax.set_yscale("log")
        ax.set_ylabel("ns / call, log scale")
        readable = workload.replace("_plus_", " + ").replace("_", " ")
        ax.set_title(readable)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend()
        for xpos, speedup in zip(x, sub["speedup_legacy_over_optimized"]):
            if np.isfinite(speedup):
                ax.text(
                    xpos,
                    max(sub.loc[sub_elems[xpos], ["legacy_ns_per_call", "optimized_ns_per_call"]]) * 1.12,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    fig.suptitle("Pointwise Lagrange basis runtime on equivalent legacy functions")
    _save(fig, "lagrange_pointwise_runtime")


def build_setup_steady_comparison() -> pd.DataFrame:
    setup = _read_csv(PERF_DIR / "perf_setup_cost.csv")
    steady = _read_csv(PERF_DIR / "perf_steady_state_access.csv")

    rows = []
    setup_pairs = [
        ("setup_cold_cache", "legacy_full_fill", "oop_cache_cold", "ns_per_setup"),
        ("setup_warm_cache", "legacy_full_fill", "oop_cache_warm", "ns_per_setup"),
    ]
    for workload, legacy_name, opt_name, metric in setup_pairs:
        for elem in filter_present(setup, "elem_type", ELEMENT_ORDER):
            sub = setup[setup["elem_type"] == elem]
            legacy = sub[sub["implementation"] == legacy_name]
            opt = sub[sub["implementation"] == opt_name]
            if legacy.empty or opt.empty:
                continue
            legacy_ns = float(legacy.iloc[0][metric])
            opt_ns = float(opt.iloc[0][metric])
            rows.append(
                {
                    "elem_type": elem,
                    "workload": workload,
                    "legacy_label": legacy_name,
                    "optimized_label": opt_name,
                    "legacy_ns": legacy_ns,
                    "optimized_ns": opt_ns,
                    "speedup_legacy_over_optimized": legacy_ns / opt_ns,
                }
            )

    steady_pairs = [
        ("steady_cache_span", "legacy_array", "oop_cache_span", "ns_per_element"),
        ("steady_batch_aligned", "legacy_array", "oop_batch_aligned", "ns_per_element"),
    ]
    for workload, legacy_name, opt_name, metric in steady_pairs:
        for elem in filter_present(steady, "elem_type", ELEMENT_ORDER):
            sub = steady[steady["elem_type"] == elem]
            legacy = sub[sub["access_pattern"] == legacy_name]
            opt = sub[sub["access_pattern"] == opt_name]
            if legacy.empty or opt.empty:
                continue
            legacy_ns = float(legacy.iloc[0][metric])
            opt_ns = float(opt.iloc[0][metric])
            rows.append(
                {
                    "elem_type": elem,
                    "workload": workload,
                    "legacy_label": legacy_name,
                    "optimized_label": opt_name,
                    "legacy_ns": legacy_ns,
                    "optimized_ns": opt_ns,
                    "speedup_legacy_over_optimized": legacy_ns / opt_ns,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "lagrange_setup_steady_fair_comparison.csv", index=False)
    return out


def plot_setup_steady(df: pd.DataFrame) -> None:
    elems = filter_present(df, "elem_type", ELEMENT_ORDER)
    workloads = [
        "setup_cold_cache",
        "setup_warm_cache",
        "steady_cache_span",
        "steady_batch_aligned",
    ]
    workloads = [w for w in workloads if w in set(df["workload"])]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    x = np.arange(len(elems))
    width = min(0.18, 0.76 / max(1, len(workloads)))
    for offset, workload in zip(_bar_offsets(len(workloads), width), workloads):
        sub = df[df["workload"] == workload].set_index("elem_type").reindex(elems)
        axes[0].bar(
            x + offset,
            sub["speedup_legacy_over_optimized"],
            width,
            label=workload.replace("_", " "),
            edgecolor="black",
            linewidth=0.4,
        )
        axes[1].bar(
            x + offset,
            sub["optimized_ns"],
            width,
            label=workload.replace("_", " "),
            edgecolor="black",
            linewidth=0.4,
        )

    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("legacy ns / optimized ns")
    axes[0].set_title("Relative speed, values above 1 mean optimized is faster")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("optimized ns, log scale")
    axes[1].set_title("Optimized absolute runtime for each access path")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(elems)

    fig.suptitle("Setup and steady-state access comparisons")
    _save(fig, "lagrange_setup_and_steady_state")


def build_kernel_comparison() -> pd.DataFrame:
    kernels = _read_csv(PERF_DIR / "perf_fused_kernels.csv")
    rows = []
    for elem in filter_present(kernels, "elem_type", ELEMENT_ORDER):
        elem_df = kernels[kernels["elem_type"] == elem]
        for operator in sorted(elem_df["operator"].unique()):
            op_df = elem_df[elem_df["operator"] == operator]
            legacy = op_df[op_df["implementation"] == "legacy_manual"]
            if legacy.empty:
                continue
            legacy_ns = float(legacy.iloc[0]["ns_per_element"])
            for impl in ["oop_manual", "oop_fused"]:
                opt = op_df[op_df["implementation"] == impl]
                if opt.empty:
                    continue
                opt_ns = float(opt.iloc[0]["ns_per_element"])
                rows.append(
                    {
                        "elem_type": elem,
                        "operator": operator,
                        "optimized_path": impl,
                        "legacy_ns_per_element": legacy_ns,
                        "optimized_ns_per_element": opt_ns,
                        "speedup_legacy_over_optimized": legacy_ns / opt_ns,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "lagrange_kernel_fair_comparison.csv", index=False)
    return out


def plot_kernels(df: pd.DataFrame) -> None:
    elems = filter_present(df, "elem_type", ELEMENT_ORDER)
    labels = []
    for operator in ["mass", "stiffness", "convection"]:
        for path in ["oop_manual", "oop_fused"]:
            if not df[(df["operator"] == operator) & (df["optimized_path"] == path)].empty:
                labels.append((operator, path))

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(elems))
    width = min(0.14, 0.78 / max(1, len(labels)))
    for offset, (operator, path) in zip(_bar_offsets(len(labels), width), labels):
        sub = df[(df["operator"] == operator) & (df["optimized_path"] == path)]
        sub = sub.set_index("elem_type").reindex(elems)
        label = f"{operator} {path.replace('oop_', '')}"
        ax.bar(
            x + offset,
            sub["speedup_legacy_over_optimized"],
            width,
            label=label,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(elems)
    ax.set_ylabel("legacy ns / optimized ns")
    ax.set_title("Element kernel relative runtime, values above 1 mean optimized is faster")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    _save(fig, "lagrange_kernel_runtime")


def _element_sizes() -> pd.DataFrame:
    summary = _read_csv(DATA_DIR / "summary.csv")
    steady = _read_csv(PERF_DIR / "perf_steady_state_access.csv")
    qpts = steady.groupby("elem_type")["nG"].max()
    rows = []
    for elem in filter_present(summary, "elem_type", ELEMENT_ORDER):
        sub = summary[summary["elem_type"] == elem].iloc[0]
        rows.append(
            {
                "elem_type": elem,
                "eNoN": int(sub["eNoN"]),
                "dim": int(ELEMENT_DIM[elem]),
                "nG": int(qpts.get(elem, 1)),
            }
        )
    return pd.DataFrame(rows)


def build_memory_estimates(pointwise: pd.DataFrame) -> pd.DataFrame:
    sizes = _element_sizes()
    hessian_elems = set(
        pointwise[pointwise["workload"] == "hessians"]["elem_type"].unique()
        if not pointwise.empty
        else []
    )

    rows = []
    for row in sizes.itertuples(index=False):
        elem = row.elem_type
        eNoN = int(row.eNoN)
        dim = int(row.dim)
        nG = int(row.nG)
        legacy_value_grad = eNoN * (1 + dim) * BYTES_PER_DOUBLE
        optimized_value_grad = eNoN * (1 + 3) * BYTES_PER_DOUBLE
        legacy_quadrature = eNoN * nG * (1 + dim) * BYTES_PER_DOUBLE
        optimized_quadrature = eNoN * nG * (1 + 3) * BYTES_PER_DOUBLE
        rows.extend(
            [
                {
                    "elem_type": elem,
                    "scope": "pointwise_values_gradients",
                    "legacy_bytes": legacy_value_grad,
                    "optimized_bytes": optimized_value_grad,
                    "optimized_over_legacy": optimized_value_grad / legacy_value_grad,
                    "basis": "derived_result_arrays",
                },
                {
                    "elem_type": elem,
                    "scope": "quadrature_values_gradients",
                    "legacy_bytes": legacy_quadrature,
                    "optimized_bytes": optimized_quadrature,
                    "optimized_over_legacy": optimized_quadrature / legacy_quadrature,
                    "basis": "derived_benchmark_working_set",
                },
            ]
        )
        if elem in hessian_elems:
            legacy_hessian_components = 3 if dim == 2 else 6
            legacy_hessian = eNoN * legacy_hessian_components * BYTES_PER_DOUBLE
            optimized_hessian = eNoN * 9 * BYTES_PER_DOUBLE
            rows.append(
                {
                    "elem_type": elem,
                    "scope": "pointwise_hessians",
                    "legacy_bytes": legacy_hessian,
                    "optimized_bytes": optimized_hessian,
                    "optimized_over_legacy": optimized_hessian / legacy_hessian,
                    "basis": "derived_result_arrays",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "lagrange_memory_estimates.csv", index=False)
    return out


def plot_memory(df: pd.DataFrame) -> None:
    elems = filter_present(df, "elem_type", ELEMENT_ORDER)
    scopes = [
        "pointwise_values_gradients",
        "quadrature_values_gradients",
        "pointwise_hessians",
    ]
    scopes = [s for s in scopes if s in set(df["scope"])]

    fig, axes = plt.subplots(len(scopes), 1, figsize=(12, 4.2 * len(scopes)), constrained_layout=True, sharex=True)
    if len(scopes) == 1:
        axes = [axes]

    for ax, scope in zip(axes, scopes):
        sub = df[df["scope"] == scope].set_index("elem_type").reindex(elems)
        sub = sub.dropna(subset=["legacy_bytes", "optimized_bytes"], how="all")
        sub_elems = list(sub.index)
        x = np.arange(len(sub_elems))
        width = 0.34
        ax.bar(
            x - width / 2,
            sub["legacy_bytes"] / 1024.0,
            width,
            label="legacy",
            color=LEGACY_COLOR,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + width / 2,
            sub["optimized_bytes"] / 1024.0,
            width,
            label="optimized",
            color=OPT_COLOR,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_yscale("log")
        ax.set_ylabel("KiB, log scale")
        ax.set_title(scope.replace("_", " "))
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(sub_elems)
        for xpos, ratio in zip(x, sub["optimized_over_legacy"]):
            if np.isfinite(ratio):
                ymax = max(sub.iloc[xpos]["legacy_bytes"], sub.iloc[xpos]["optimized_bytes"]) / 1024.0
                ax.text(xpos, ymax * 1.15, f"{ratio:.2f}x", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Derived memory working-set estimates for equivalent Lagrange data")
    _save(fig, "lagrange_memory_working_set")


def build_parallel_summary() -> pd.DataFrame:
    parallel = _read_csv(PERF_DIR / "perf_parallel_scaling.csv")
    rows = []
    for elem in filter_present(parallel, "elem_type", ELEMENT_ORDER):
        elem_df = parallel[parallel["elem_type"] == elem]
        for threads in sorted(elem_df["n_threads"].unique()):
            legacy = elem_df[
                (elem_df["implementation"] == "legacy") & (elem_df["n_threads"] == threads)
            ]
            opt = elem_df[
                (elem_df["implementation"] == "oop_cache_span")
                & (elem_df["n_threads"] == threads)
            ]
            if legacy.empty or opt.empty:
                continue
            rows.append(
                {
                    "elem_type": elem,
                    "n_threads": int(threads),
                    "legacy_elements_per_second": float(legacy.iloc[0]["elements_per_second_total"]),
                    "optimized_elements_per_second": float(opt.iloc[0]["elements_per_second_total"]),
                    "optimized_over_legacy": float(opt.iloc[0]["elements_per_second_total"])
                    / float(legacy.iloc[0]["elements_per_second_total"]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        one_thread = out[out["n_threads"] == 1].set_index("elem_type")
        scaling_rows = []
        for row in out.itertuples(index=False):
            base = one_thread.loc[row.elem_type]
            scaling_rows.append(
                {
                    **row._asdict(),
                    "legacy_parallel_speedup": row.legacy_elements_per_second
                    / float(base["legacy_elements_per_second"]),
                    "optimized_parallel_speedup": row.optimized_elements_per_second
                    / float(base["optimized_elements_per_second"]),
                }
            )
        out = pd.DataFrame(scaling_rows)
    out.to_csv(DATA_DIR / "lagrange_parallel_summary.csv", index=False)
    return out


def plot_parallel(df: pd.DataFrame) -> None:
    elems = filter_present(df, "elem_type", ELEMENT_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True, sharex=True)
    axes = axes.ravel()

    for idx, elem in enumerate(elems):
        ax = axes[idx]
        sub = df[df["elem_type"] == elem].sort_values("n_threads")
        threads = sub["n_threads"].to_numpy(dtype=float)
        ax.plot(
            threads,
            sub["legacy_elements_per_second"],
            "o-",
            color=LEGACY_COLOR,
            label="legacy",
            linewidth=1.5,
        )
        ax.plot(
            threads,
            sub["optimized_elements_per_second"],
            "o-",
            color=OPT_COLOR,
            label="optimized",
            linewidth=1.5,
        )
        if not sub.empty:
            ax.plot(
                threads,
                float(sub.iloc[0]["legacy_elements_per_second"]) * threads,
                "--",
                color=LEGACY_COLOR,
                alpha=0.35,
                linewidth=0.9,
            )
            ax.plot(
                threads,
                float(sub.iloc[0]["optimized_elements_per_second"]) * threads,
                "--",
                color=OPT_COLOR,
                alpha=0.35,
                linewidth=0.9,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_title(elem)
        ax.grid(True, which="both", alpha=0.25)
        if idx % 3 == 0:
            ax.set_ylabel("elements / sec, log scale")
        ax.set_xlabel("threads")
        ax.legend(fontsize=8)

    for idx in range(len(elems), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Parallel scaling: solid measured throughput, dashed ideal from 1 thread")
    _save(fig, "lagrange_parallel_scaling")

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(elems))
    thread_counts = sorted(df["n_threads"].unique())
    width = min(0.16, 0.76 / max(1, len(thread_counts)))
    for offset, threads in zip(_bar_offsets(len(thread_counts), width), thread_counts):
        sub = df[df["n_threads"] == threads].set_index("elem_type").reindex(elems)
        ax.bar(
            x + offset,
            sub["optimized_over_legacy"],
            width,
            label=f"{threads} thread{'s' if threads != 1 else ''}",
            edgecolor="black",
            linewidth=0.4,
        )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(elems)
    ax.set_ylabel("optimized throughput / legacy throughput")
    ax.set_title("Parallel throughput ratio by thread count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    _save(fig, "lagrange_parallel_throughput_ratio")


def write_summary(
    accuracy: pd.DataFrame,
    pointwise: pd.DataFrame,
    setup_steady: pd.DataFrame,
    kernels: pd.DataFrame,
    memory: pd.DataFrame,
    parallel: pd.DataFrame,
) -> None:
    lines = [
        "# Lagrange Basis Legacy vs Optimized Report",
        "",
        "Scope: TRI3, TRI6, TET4, TET10, HEX8, and HEX27. These are the "
        "Lagrange element types covered by equivalent legacy and optimized "
        "basis functions in the existing comparison harness.",
        "",
        "Legacy source files were not modified. The report consumes CSV output "
        "from the test harness and writes derived CSVs plus PNG/SVG figures.",
        "",
        "## Figures",
        "",
    ]
    figures = sorted(p.name for p in FIG_DIR.glob("lagrange_*.png"))
    lines.extend(f"- `{name}`" for name in figures)
    lines.extend(["", "## Key Derived Metrics", ""])

    if not accuracy.empty:
        max_value = accuracy["max_abs_value_error"].max()
        max_grad = accuracy["max_abs_gradient_error"].max()
        max_mass = accuracy["max_abs_mass_matrix_diff"].max()
        lines.append(f"- Max basis value error: `{max_value:.3e}`")
        lines.append(f"- Max basis gradient error: `{max_grad:.3e}`")
        lines.append(f"- Max mass-matrix absolute difference: `{max_mass:.3e}`")
    if not pointwise.empty:
        vg = pointwise[pointwise["workload"] == "values_plus_gradients"]
        if not vg.empty:
            lines.append(
                "- Pointwise values+gradients optimized/legacy speed ratio range: "
                f"`{vg['speedup_legacy_over_optimized'].min():.3f}x` to "
                f"`{vg['speedup_legacy_over_optimized'].max():.3f}x` "
                "(reported as legacy ns / optimized ns)."
            )
    if not parallel.empty:
        max_threads = parallel["n_threads"].max()
        pmax = parallel[parallel["n_threads"] == max_threads]
        lines.append(
            f"- {max_threads}-thread optimized/legacy throughput ratio range: "
            f"`{pmax['optimized_over_legacy'].min():.3f}x` to "
            f"`{pmax['optimized_over_legacy'].max():.3f}x`."
        )
    if not memory.empty:
        quad = memory[memory["scope"] == "quadrature_values_gradients"]
        if not quad.empty:
            lines.append(
                "- Quadrature values+gradients optimized/legacy memory ratio: "
                f"`{quad['optimized_over_legacy'].min():.3f}x` to "
                f"`{quad['optimized_over_legacy'].max():.3f}x`."
            )

    lines.extend(
        [
            "",
            "## Derived CSVs",
            "",
        ]
    )
    derived_csvs = sorted(p.name for p in DATA_DIR.glob("lagrange_*.csv"))
    lines.extend(f"- `data/{name}`" for name in derived_csvs)
    lines.append("")

    (SCRIPT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SCRIPT_DIR / 'SUMMARY.md'}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    accuracy = build_accuracy_summary()
    pointwise = build_pointwise_comparison()
    setup_steady = build_setup_steady_comparison()
    kernels = build_kernel_comparison()
    memory = build_memory_estimates(pointwise)
    parallel = build_parallel_summary()

    plot_accuracy(accuracy)
    plot_pointwise(pointwise)
    plot_setup_steady(setup_steady)
    plot_kernels(kernels)
    plot_memory(memory)
    plot_parallel(parallel)
    write_summary(accuracy, pointwise, setup_steady, kernels, memory, parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
