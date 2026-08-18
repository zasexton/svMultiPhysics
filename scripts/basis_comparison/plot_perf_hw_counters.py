"""Pη1, Pη2, Pη3: hardware-counter analysis from `perf stat`.

Reads data/perf/perf_hw_counters.csv (produced by run_perf_stat.sh) and emits:
  perf_roofline.png        — achieved FLOPs/cycle vs theoretical peak (Pη1)
  perf_cache_misses.png    — L1d / LLC miss rates (Pη2)
  perf_branch_misses.png   — branch misprediction rate per element (Pη3)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ELEMENT_ORDER, default_dirs


# Approximate operation counts per evaluate_*. These are reasonable estimates
# of the FLOPs (multiplies + adds + subtracts) the OOP code executes per call,
# pre-vectorization. The compiler may fuse some pairs into FMAs (counted as
# 2 FLOPs); these numbers are unfused-equivalent so the achieved/peak ratio is
# slightly conservative.
#
# Counts derived from inspection of the fast paths in LagrangeBasis.cpp:
#  - TRI3 D1 fast path: 2 subs + 0 mul = 2 ops
#  - TRI6 E1 fast path: ~11 setup + ~12 per-node = ~23 ops
#  - TET4 D1 fast path: 3 subs + 0 mul = 3 ops
#  - TET10 E1 fast path: ~15 setup + ~30 per-node = ~45 ops
#  - HEX8 E4 fast path: 6 sub + 6 mul + 4 yz + 8 final = 24 ops
#  - HEX27 Horner path: 18 axis + 9 yz + 27 final = 54 ops
FLOPS_PER_CALL = {
    "values": {
        "TRI3":  2.0,
        "TRI6":  23.0,
        "TET4":  3.0,
        "TET10": 45.0,
        "HEX8":  24.0,
        "HEX27": 54.0,
    },
    "gradients": {
        "TRI3":   6.0,    # constants — barely any FLOPs
        "TRI6":   60.0,   # phi + dphi + per-node 3 grad components
        "TET4":   12.0,
        "TET10":  120.0,
        "HEX8":   60.0,   # 8 nodes * 3 components, ~2 mul per slot
        "HEX27":  150.0,
    },
    "hessians": {
        "TRI3":   3.0,    # zeros
        "TRI6":   90.0,
        "TET4":   4.0,
        "TET10":  180.0,
        "HEX8":   100.0,  # 8 nodes * 3 unique mixed * (~3 mul each) + signs
        "HEX27":  240.0,
    },
    "all": {
        "TRI3":   8.0,
        "TRI6":   170.0,
        "TET4":   15.0,
        "TET10":  340.0,
        "HEX8":   180.0,
        "HEX27":  430.0,
    },
}

# Theoretical peak for AVX2 + FMA (i7-8565U class):
#   2 FMA ports × 4 doubles per AVX2 register × 2 FLOPs/FMA = 16 FLOPs/cycle.
PEAK_FLOPS_PER_CYCLE = 16.0


def _to_num(s):
    """Coerce a column with possible empty/None entries to float."""
    return pd.to_numeric(s, errors="coerce")


def plot_roofline(df: pd.DataFrame, out_path: Path) -> None:
    """Pη1 — achieved FLOPs/cycle vs peak."""
    elements = [e for e in ELEMENT_ORDER if e in df["elem_type"].unique()]
    ops = ["values", "gradients", "hessians", "all"]

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(elements))
    width = 0.20
    colors = {"values": "#264653", "gradients": "#2a9d8f",
              "hessians": "#e76f51", "all": "#f4a261"}

    for k, op in enumerate(ops):
        sub = df[df["op"] == op].set_index("elem_type").reindex(elements)
        cycles = _to_num(sub["cycles"]).to_numpy()
        iters = _to_num(sub["iterations"]).to_numpy()
        flops_per_call = np.array([FLOPS_PER_CALL[op].get(e, np.nan) for e in elements])
        achieved = (flops_per_call * iters) / cycles
        offset = (k - 1.5) * width
        ax.bar(x + offset, achieved, width, color=colors[op], label=op,
               edgecolor="black", linewidth=0.4)
        for xi, val in zip(x + offset, achieved):
            if np.isfinite(val):
                ax.annotate(f"{val:.2f}", xy=(xi, val * 1.02), ha="center", fontsize=7)

    ax.axhline(PEAK_FLOPS_PER_CYCLE, color="red", linestyle="--", linewidth=1.2,
               label=f"AVX2+FMA peak = {PEAK_FLOPS_PER_CYCLE:.0f} FLOPs/cycle")
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel("achieved FLOPs / cycle")
    ax.set_xlabel("element")
    ax.set_yscale("log")
    ax.set_title(
        "Pη1: achieved arithmetic throughput per CPU cycle vs theoretical AVX2+FMA peak\n"
        "Low utilization indicates dispatch/memory-bound code; high utilization indicates compute-bound"
    )
    ax.legend(fontsize=9, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(True, alpha=0.25, axis="y", which="both")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_cache_misses(df: pd.DataFrame, out_path: Path) -> None:
    """Pη2 — L1d / LLC miss rates per (element, op)."""
    elements = [e for e in ELEMENT_ORDER if e in df["elem_type"].unique()]
    ops = ["values", "gradients", "hessians", "all"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(elements))
    width = 0.20
    colors = {"values": "#264653", "gradients": "#2a9d8f",
              "hessians": "#e76f51", "all": "#f4a261"}

    for ax_idx, (label, miss_col, load_col) in enumerate([
        ("L1d cache miss rate (% of L1d loads)", "l1d_load_misses", "l1d_loads"),
        ("LLC (last-level) cache miss rate (% of LLC loads)", "llc_load_misses", "llc_loads"),
    ]):
        ax = axes[ax_idx]
        for k, op in enumerate(ops):
            sub = df[df["op"] == op].set_index("elem_type").reindex(elements)
            misses = _to_num(sub[miss_col]).to_numpy()
            loads = _to_num(sub[load_col]).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = 100.0 * misses / loads
            offset = (k - 1.5) * width
            ax.bar(x + offset, rate, width, color=colors[op], label=op,
                   edgecolor="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(elements)
        ax.set_ylabel("miss rate (%)")
        ax.set_xlabel("element")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Pη2: cache miss rates during a hot evaluate loop\n"
        "L1d miss rate identifies basis-data working-set pressure; LLC miss rate flags main-memory traffic",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_branch_misses(df: pd.DataFrame, out_path: Path) -> None:
    """Pη3 — branch misprediction rate."""
    elements = [e for e in ELEMENT_ORDER if e in df["elem_type"].unique()]
    ops = ["values", "gradients", "hessians", "all"]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(elements))
    width = 0.20
    colors = {"values": "#264653", "gradients": "#2a9d8f",
              "hessians": "#e76f51", "all": "#f4a261"}

    for k, op in enumerate(ops):
        sub = df[df["op"] == op].set_index("elem_type").reindex(elements)
        misses = _to_num(sub["branch_misses"]).to_numpy()
        branches = _to_num(sub["branches"]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = 100.0 * misses / branches
        offset = (k - 1.5) * width
        ax.bar(x + offset, rate, width, color=colors[op], label=op,
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel("branch misprediction rate (%)")
    ax.set_xlabel("element")
    ax.set_title(
        "Pη3: branch misprediction rate per (element, op)\n"
        "High rates indicate the topology+order dispatch switch is unpredictable;"
        " low rates confirm the BTB has learned the pattern"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_hw_counters.csv")

    plot_roofline(df, fig_dir / "perf_roofline.png")
    print(f"Wrote {fig_dir / 'perf_roofline.png'}")

    plot_cache_misses(df, fig_dir / "perf_cache_misses.png")
    print(f"Wrote {fig_dir / 'perf_cache_misses.png'}")

    plot_branch_misses(df, fig_dir / "perf_branch_misses.png")
    print(f"Wrote {fig_dir / 'perf_branch_misses.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
