"""Plot pointwise basis-value and gradient errors between legacy and OOP Lagrange bases.

Reads CSVs produced by run_all_unit_tests --gtest_filter='LagrangeBasisComparison.*'
and writes PNG figures next to them.

Usage:
    python3 plot_pointwise_error.py [data_dir] [figures_dir]

Defaults: data_dir = ./data, figures_dir = ./figures relative to this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ELEMENT_ORDER, default_dirs


def _safe_log10(x):
    eps = 1e-300
    return np.log10(np.maximum(np.asarray(x, dtype=float), eps))


def plot_value_heatmaps(values_df: pd.DataFrame, out_path: Path) -> None:
    elements = [e for e in ELEMENT_ORDER if e in values_df["elem_type"].unique()]
    n = len(elements)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, elem in enumerate(elements):
        sub = values_df[values_df["elem_type"] == elem]
        # Pivot: rows = OOP dof, cols = sample idx, values = abs_err
        pivot = sub.pivot_table(
            index="oop_dof_index",
            columns="sample_idx",
            values="abs_err",
            aggfunc="max",
        )
        ax = axes[i]
        log_err = _safe_log10(pivot.values)
        im = ax.imshow(
            log_err, aspect="auto", cmap="viridis", vmin=-18, vmax=0,
            interpolation="nearest",
        )
        ax.set_title(f"{elem}: log10|N_legacy - N_oop|")
        ax.set_xlabel("sample index")
        ax.set_ylabel("OOP dof index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for j in range(len(elements), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Pointwise basis-value error: legacy vs OOP Lagrange (after node permutation)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_gradient_heatmaps(grad_df: pd.DataFrame, out_path: Path) -> None:
    elements = [e for e in ELEMENT_ORDER if e in grad_df["elem_type"].unique()]
    n = len(elements)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, elem in enumerate(elements):
        sub = grad_df[grad_df["elem_type"] == elem]
        # Aggregate across the dim component (max over dimensions)
        agg = sub.groupby(["oop_dof_index", "sample_idx"])["abs_err"].max().reset_index()
        pivot = agg.pivot_table(
            index="oop_dof_index",
            columns="sample_idx",
            values="abs_err",
            aggfunc="max",
        )
        ax = axes[i]
        log_err = _safe_log10(pivot.values)
        im = ax.imshow(
            log_err, aspect="auto", cmap="magma", vmin=-18, vmax=0,
            interpolation="nearest",
        )
        ax.set_title(f"{elem}: log10|grad_legacy - grad_oop|_inf")
        ax.set_xlabel("sample index")
        ax.set_ylabel("OOP dof index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for j in range(len(elements), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Pointwise gradient error (inf-norm over dimensions): legacy vs OOP Lagrange",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_summary_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    summary = summary_df.set_index("elem_type").reindex(
        [e for e in ELEMENT_ORDER if e in summary_df["elem_type"].values]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    x = np.arange(len(summary.index))
    width = 0.35

    # Plot error magnitudes (clipped at machine eps for log scale)
    eps = 1e-18
    val = np.maximum(summary["max_abs_err_value"].astype(float).values, eps)
    grd = np.maximum(summary["max_abs_err_grad"].astype(float).values, eps)

    axes[0].bar(x, val, width, color="#3a86ff")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary.index)
    axes[0].set_ylabel("max abs error  (log scale)")
    axes[0].set_title("Basis values: max |N_legacy - N_oop|")
    axes[0].axhline(1e-12, color="red", linestyle="--", label="1e-12 tolerance")
    axes[0].legend()

    axes[1].bar(x, grd, width, color="#f72585")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary.index)
    axes[1].set_ylabel("max abs error  (log scale)")
    axes[1].set_title("Gradients: max |dN_legacy - dN_oop|")
    axes[1].axhline(1e-12, color="red", linestyle="--", label="1e-12 tolerance")
    axes[1].legend()

    fig.suptitle("Legacy vs OOP Lagrange: maximum pointwise disagreement (after permutation)")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_pou_diagnostic(pou_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    elements = [e for e in ELEMENT_ORDER if e in pou_df["elem_type"].unique()]
    n = len(elements)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    eps = 1e-18
    for i, elem in enumerate(elements):
        sub = pou_df[pou_df["elem_type"] == elem]
        leg_dev = np.maximum(np.abs(sub["sum_N_legacy"].values - 1.0), eps)
        oop_dev = np.maximum(np.abs(sub["sum_N_oop"].values - 1.0), eps)
        ax = axes[i]
        ax.hist(np.log10(leg_dev), bins=30, alpha=0.55, label="legacy", color="#3a86ff")
        ax.hist(np.log10(oop_dev), bins=30, alpha=0.55, label="OOP", color="#f72585")
        ax.set_title(f"{elem}: log10|sum N_i - 1| histogram")
        ax.set_xlabel("log10 deviation")
        ax.set_ylabel("samples")
        ax.legend()
    for j in range(len(elements), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Partition of unity diagnostic: distribution of |Sum N_i - 1| at sampled points",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    default_data_dir, default_fig_dir = default_dirs(here)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default=str(default_data_dir))
    parser.add_argument("figures_dir", nargs="?", default=str(default_fig_dir))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    expected = ["basis_values.csv", "basis_gradients.csv", "summary.csv", "partition_of_unity.csv"]
    missing = [f for f in expected if not (data_dir / f).exists()]
    if missing:
        print(f"ERROR: missing CSVs in {data_dir}: {missing}", file=sys.stderr)
        return 2

    values_df = pd.read_csv(data_dir / "basis_values.csv")
    grad_df = pd.read_csv(data_dir / "basis_gradients.csv")
    summary_df = pd.read_csv(data_dir / "summary.csv")
    pou_df = pd.read_csv(data_dir / "partition_of_unity.csv")

    plot_summary_bars(summary_df, fig_dir / "summary_max_error_bars.png")
    plot_pou_diagnostic(pou_df, summary_df, fig_dir / "partition_of_unity_diagnostic.png")
    # Heatmap variants intentionally omitted: when errors are at machine
    # epsilon they render as flat low-saturation backgrounds; the scatter,
    # ULP histogram, and CDF figures convey the same information far better.

    print(f"Wrote figures to {fig_dir}")
    print("Summary:")
    cols = ["elem_type", "eNoN", "n_samples", "max_abs_err_value", "max_abs_err_grad", "permutation"]
    print(summary_df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
