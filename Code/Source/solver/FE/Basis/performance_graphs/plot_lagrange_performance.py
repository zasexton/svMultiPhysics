#!/usr/bin/env python3
"""Plot LagrangeBasis benchmark rows emitted by basis_perf_microbench."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TOPOLOGIES = ["line", "triangle", "quad", "tet", "hex", "wedge", "pyramid"]
OPERATIONS = ["values", "gradients", "hessians", "all"]
COLORS = {
    "line": "#1f77b4",
    "triangle": "#d62728",
    "quad": "#2ca02c",
    "tet": "#9467bd",
    "hex": "#ff7f0e",
    "wedge": "#17becf",
    "pyramid": "#4d4d4d",
}
MARKERS = {
    "line": "o",
    "triangle": "s",
    "quad": "^",
    "tet": "D",
    "hex": "P",
    "wedge": "v",
    "pyramid": "X",
}


def basis_dofs(topology: str, order: int) -> int:
    if topology == "line":
        return order + 1
    if topology == "triangle":
        return (order + 1) * (order + 2) // 2
    if topology == "quad":
        return (order + 1) ** 2
    if topology == "tet":
        return (order + 1) * (order + 2) * (order + 3) // 6
    if topology == "hex":
        return (order + 1) ** 3
    if topology == "wedge":
        return ((order + 1) * (order + 2) // 2) * (order + 1)
    if topology == "pyramid":
        return (order + 1) * (order + 2) * (2 * order + 3) // 6
    raise ValueError(f"Unknown topology: {topology}")


def normalized_ideal(rows: pd.DataFrame, metric_column: str) -> pd.Series | None:
    metric = rows[metric_column].astype(float)
    positive = metric > 0.0
    if rows.empty or not positive.any():
        return None
    anchor_index = rows[positive].index[0]
    anchor_metric = float(rows.loc[anchor_index, metric_column])
    anchor_ns = float(rows.loc[anchor_index, "ns_per_call"])
    if anchor_metric <= 0.0 or anchor_ns <= 0.0:
        return None
    return anchor_ns * metric / anchor_metric


PEAK_RE = re.compile(
    r"^lagrange_(?P<topology>line|triangle|quad|tet|hex|wedge|pyramid)"
    r"_order(?P<order>\d+)_(?:(?P<mode>point|strided)_(?P<operation>values|gradients|hessians|all)"
    r"|(?P<construction>construction))$"
)
PARALLEL_RE = re.compile(
    r"^lagrange_parallel_hex_order4_(?P<mode>schedule_only|strided_all)"
    r"_threads(?P<threads>\d+)$"
)


def savefig(output_dir: Path, stem: str) -> None:
    for ext in ("png", "svg"):
        plt.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=180)
    plt.close()


def parse_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_records = []
    parallel_records = []

    for row in df.itertuples(index=False):
        case = getattr(row, "case")
        peak_match = PEAK_RE.match(case)
        if peak_match:
            groups = peak_match.groupdict()
            mode = groups["mode"] or "construction"
            peak_records.append(
                {
                    "case": case,
                    "category": getattr(row, "category"),
                    "topology": groups["topology"],
                    "order": int(groups["order"]),
                    "mode": mode,
                    "operation": groups["operation"] or "construction",
                    "iterations": getattr(row, "iterations"),
                    "seconds": getattr(row, "seconds"),
                    "ns_per_call": getattr(row, "ns_per_call"),
                    "min_ns_per_call": getattr(row, "min_ns_per_call"),
                    "max_ns_per_call": getattr(row, "max_ns_per_call"),
                    "allocations_per_call": getattr(row, "allocations_per_call"),
                    "model_lower_bound_ns": getattr(row, "model_lower_bound_ns"),
                    "measured_to_model_bound": getattr(row, "measured_to_model_bound"),
                    "modeled_flops_per_call": getattr(row, "modeled_flops_per_call"),
                    "estimated_bytes_per_call": getattr(row, "estimated_bytes_per_call"),
                }
            )
            continue

        parallel_match = PARALLEL_RE.match(case)
        if parallel_match:
            groups = parallel_match.groupdict()
            parallel_records.append(
                {
                    "case": case,
                    "category": getattr(row, "category"),
                    "mode": groups["mode"],
                    "threads": int(groups["threads"]),
                    "bench_threads": getattr(row, "bench_threads"),
                    "iterations": getattr(row, "iterations"),
                    "seconds": getattr(row, "seconds"),
                    "ns_per_call": getattr(row, "ns_per_call"),
                    "min_ns_per_call": getattr(row, "min_ns_per_call"),
                    "max_ns_per_call": getattr(row, "max_ns_per_call"),
                    "allocations_per_call": getattr(row, "allocations_per_call"),
                    "model_lower_bound_ns": getattr(row, "model_lower_bound_ns"),
                    "measured_to_model_bound": getattr(row, "measured_to_model_bound"),
                    "modeled_flops_per_call": getattr(row, "modeled_flops_per_call"),
                    "estimated_bytes_per_call": getattr(row, "estimated_bytes_per_call"),
                }
            )

    return pd.DataFrame.from_records(peak_records), pd.DataFrame.from_records(parallel_records)


def plot_order_sweep(peak: pd.DataFrame, output_dir: Path, mode: str, stem: str, title: str) -> None:
    subset = peak[peak["mode"] == mode]
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    axes = axes.ravel()

    for ax, operation in zip(axes, OPERATIONS):
        op_rows = subset[subset["operation"] == operation]
        for topology in TOPOLOGIES:
            rows = op_rows[op_rows["topology"] == topology].sort_values("order")
            if rows.empty:
                continue
            ax.plot(
                rows["order"],
                rows["ns_per_call"],
                marker=MARKERS[topology],
                color=COLORS[topology],
                linewidth=1.8,
                markersize=4.5,
                label=topology,
            )
            ideal = normalized_ideal(rows, "modeled_flops_per_call")
            if ideal is not None:
                ax.plot(
                    rows["order"],
                    ideal,
                    color=COLORS[topology],
                    linestyle="--",
                    linewidth=1.15,
                    alpha=0.55,
                )
        ax.set_title(operation)
        ax.set_yscale("log")
        ax.set_ylabel("ns per call")
        ax.grid(True, which="both", linewidth=0.45, alpha=0.45)

    for ax in axes[-2:]:
        ax.set_xlabel("Polynomial order")

    handles, labels = axes[0].get_legend_handles_labels()
    reference_handle = plt.Line2D(
        [0],
        [0],
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label="modeled work reference",
    )
    fig.legend(handles + [reference_handle],
               labels + ["modeled work reference"],
               loc="upper center",
               ncol=4,
               frameon=False)
    fig.suptitle(title, y=1.03, fontsize=14)
    fig.tight_layout()
    savefig(output_dir, stem)


def plot_construction(peak: pd.DataFrame, output_dir: Path) -> None:
    subset = peak[peak["mode"] == "construction"]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for topology in TOPOLOGIES:
        rows = subset[subset["topology"] == topology].sort_values("order").copy()
        if rows.empty:
            continue
        rows["dofs"] = [basis_dofs(topology, order) for order in rows["order"]]
        ax.plot(
            rows["order"],
            rows["ns_per_call"],
            marker=MARKERS[topology],
            color=COLORS[topology],
            linewidth=1.8,
            label=topology,
        )
        ideal = normalized_ideal(rows, "dofs")
        if ideal is not None:
            ax.plot(
                rows["order"],
                ideal,
                color=COLORS[topology],
                linestyle="--",
                linewidth=1.15,
                alpha=0.55,
            )
    ax.set_title("LagrangeBasis construction cost by order")
    ax.set_xlabel("Polynomial order")
    ax.set_ylabel("ns per construction")
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.45)
    handles, labels = ax.get_legend_handles_labels()
    reference_handle = plt.Line2D(
        [0],
        [0],
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label="O(dofs) reference",
    )
    ax.legend(handles + [reference_handle], labels + ["O(dofs) reference"], ncol=4, frameon=False)
    fig.tight_layout()
    savefig(output_dir, "lagrange_construction_order_sweep")


def plot_model_ratio(peak: pd.DataFrame, output_dir: Path) -> None:
    subset = peak[
        (peak["mode"] == "strided")
        & (peak["operation"] == "all")
        & (peak["measured_to_model_bound"] > 0.0)
    ]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for topology in TOPOLOGIES:
        rows = subset[subset["topology"] == topology].sort_values("order")
        if rows.empty:
            continue
        ax.plot(
            rows["order"],
            rows["measured_to_model_bound"],
            marker=MARKERS[topology],
            color=COLORS[topology],
            linewidth=1.8,
            label=topology,
        )
    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--", label="model lower bound")
    ax.set_title("Strided all measured time vs modeled lower bound")
    ax.set_xlabel("Polynomial order")
    ax.set_ylabel("measured / model lower bound")
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.45)
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    savefig(output_dir, "lagrange_strided_all_model_ratio")


def plot_parallel(parallel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if parallel.empty:
        return pd.DataFrame()

    eval_rows = parallel[parallel["mode"] == "strided_all"].sort_values("threads").copy()
    schedule_rows = parallel[parallel["mode"] == "schedule_only"].sort_values("threads").copy()
    if eval_rows.empty:
        return pd.DataFrame()

    base_ns = float(eval_rows[eval_rows["threads"] == 1]["ns_per_call"].iloc[0])
    eval_rows["speedup"] = base_ns / eval_rows["ns_per_call"]
    eval_rows["parallel_efficiency"] = eval_rows["speedup"] / eval_rows["threads"]
    eval_rows["million_calls_per_second"] = 1000.0 / eval_rows["ns_per_call"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    axes[0].plot(
        eval_rows["threads"],
        eval_rows["speedup"],
        marker="o",
        color="#1f77b4",
        linewidth=2.0,
        label="measured eval speedup",
    )
    axes[0].plot(
        eval_rows["threads"],
        eval_rows["threads"],
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        label="ideal linear speedup",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(eval_rows["threads"])
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_xlabel("Worker threads")
    axes[0].set_ylabel("Speedup vs 1 thread")
    axes[0].set_title("Hex order 4 strided all speedup")
    axes[0].grid(True, which="both", linewidth=0.45, alpha=0.45)
    axes[0].legend(frameon=False)

    axes[1].plot(
        eval_rows["threads"],
        eval_rows["parallel_efficiency"],
        marker="s",
        color="#2ca02c",
        linewidth=2.0,
        label="eval efficiency",
    )
    axes[1].axhline(
        1.0,
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        label="ideal 100% efficiency",
    )
    if not schedule_rows.empty:
        axes[1].plot(
            schedule_rows["threads"],
            schedule_rows["ns_per_call"] / schedule_rows["ns_per_call"].iloc[0],
            marker="D",
            color="#d62728",
            linewidth=1.7,
            label="schedule-only relative cost",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(eval_rows["threads"])
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].set_xlabel("Worker threads")
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Efficiency and scheduling overhead")
    axes[1].grid(True, which="both", linewidth=0.45, alpha=0.45)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    savefig(output_dir, "lagrange_parallel_scaling")
    return eval_rows


def write_summary(
    df: pd.DataFrame,
    peak: pd.DataFrame,
    parallel: pd.DataFrame,
    scaling: pd.DataFrame,
    csv_path: Path,
    output_dir: Path,
) -> None:
    lines = [
        "LagrangeBasis performance graph summary",
        "======================================",
        "",
        f"Source CSV: {csv_path}",
        f"Total benchmark rows: {len(df)}",
        f"Parsed Lagrange order-sweep rows: {len(peak)}",
        f"Parsed Lagrange parallel rows: {len(parallel)}",
        "",
    ]

    if not df.empty:
        first = df.iloc[0]
        lines.extend(
            [
                f"Compiler: {first['compiler_id']} {first['compiler_version']}",
                f"Build flags token: {first['build_flags']}",
                f"CPU: {first['cpu_model']}",
                f"Hardware threads reported: {first['hardware_threads']}",
                f"SIMD width bytes: {first['simd_width_bytes']}",
                "",
            ]
        )

    if not scaling.empty:
        lines.append("Parallel scaling, hex order 4 strided all:")
        for row in scaling.sort_values("threads").itertuples(index=False):
            lines.append(
                f"  threads={row.threads}: ns_per_call={row.ns_per_call:.3f}, "
                f"speedup={row.speedup:.3f}, efficiency={row.parallel_efficiency:.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "Reference line definitions:",
            "  Scalar/strided order sweeps: dashed lines are the benchmark's modeled",
            "  flop-count growth, normalized to the first measured positive-flop order",
            "  for that topology and operation. They are not hardware roofline lower",
            "  bounds.",
            "  Construction sweep: dashed lines are normalized O(dofs) growth. This is",
            "  a coarse reference only; construction may include topology-specific",
            "  setup with higher complexity.",
            "  Parallel scaling: dashed lines are true ideal references: speedup =",
            "  thread_count and efficiency = 1.0.",
            "",
        ]
    )

    lines.append("Generated plots:")
    for png in sorted(output_dir.glob("lagrange_*.png")):
        svg = png.with_suffix(".svg")
        if svg.exists():
            lines.append(f"  {png.name}/.svg")
        else:
            lines.append(f"  {png.name}")
    lines.extend(
        [
            "",
            "Interpretation note: these are local wall-clock microbenchmarks. CPU",
            "frequency scaling, thermal state, and background load can move the",
            "absolute numbers; the order trends and relative scaling are usually",
            "the more stable signal.",
        ]
    )

    (output_dir / "lagrange_performance_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path("lagrange_basis_benchmark.csv")
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = csv_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    numeric_columns = [
        "iterations",
        "seconds",
        "ns_per_call",
        "allocations_per_call",
        "estimated_bytes_per_call",
        "modeled_flops_per_call",
        "model_lower_bound_ns",
        "measured_to_model_bound",
        "min_ns_per_call",
        "max_ns_per_call",
        "hardware_threads",
        "bench_threads",
        "simd_width_bytes",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    peak, parallel = parse_rows(df)
    if peak.empty and parallel.empty:
        raise RuntimeError("No Lagrange benchmark rows were found in the CSV")

    if not peak.empty:
        peak.to_csv(output_dir / "lagrange_peak_rows.csv", index=False)
    if not parallel.empty:
        parallel.to_csv(output_dir / "lagrange_parallel_rows.csv", index=False)

    plot_order_sweep(
        peak,
        output_dir,
        "point",
        "lagrange_scalar_order_sweep",
        "LagrangeBasis scalar point evaluation by order",
    )
    plot_order_sweep(
        peak,
        output_dir,
        "strided",
        "lagrange_strided_order_sweep",
        "LagrangeBasis strided batch evaluation by order",
    )
    plot_construction(peak, output_dir)
    plot_model_ratio(peak, output_dir)
    scaling = plot_parallel(parallel, output_dir)
    if not scaling.empty:
        scaling.to_csv(output_dir / "lagrange_parallel_scaling.csv", index=False)

    write_summary(df, peak, parallel, scaling, csv_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
