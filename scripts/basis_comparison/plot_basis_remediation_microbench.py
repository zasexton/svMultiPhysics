"""Post-remediation basis microbenchmarks.

Reads data/perf/basis_remediation_microbench.csv from the standalone
basis_perf_microbench target and plots the timing, allocation, and estimated
byte footprint for the paths touched by the basis remediation work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import default_dirs


CATEGORY_ORDER = {
    "scalar_point": 0,
    "batched_quadrature": 1,
    "cache_construction": 2,
    "cache_reuse": 3,
    "spectral_high_order": 4,
    "pyramid_modal_to_nodal": 5,
    "vector_rt_generated": 6,
    "vector_rt_nodal": 7,
}

CATEGORY_COLORS = {
    "scalar_point": "#264653",
    "batched_quadrature": "#2a9d8f",
    "cache_construction": "#e9c46a",
    "cache_reuse": "#f4a261",
    "spectral_high_order": "#8ab17d",
    "pyramid_modal_to_nodal": "#577590",
    "vector_rt_generated": "#e76f51",
    "vector_rt_nodal": "#9d4edd",
}

CASE_LABELS = {
    "lagrange_hex_order2_values": "HEX p2 scalar values",
    "batch_hex_order2_weighted_sum": "HEX p2 batched weighted sum",
    "cache_hex_order2_uncached": "HEX p2 cache construction",
    "cache_hex_order2_reuse": "HEX p2 cache reuse",
    "spectral_hex_order6_all": "Spectral HEX p6 all fields",
    "spectral_pyramid_order4_all": "Pyramid p4 modal-to-nodal",
    "rt_wedge_order3_values_jac_div": "RT wedge p3 values/J/div",
    "rt_tetra_order2_values_jac_div": "RT tetra p2 values/J/div",
}


def _format_ns(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.1f} us"
    return f"{value:.0f} ns"


def _format_bytes(value: float) -> str:
    if value >= 1024.0:
        return f"{value / 1024.0:.1f} KiB"
    return f"{value:.0f} B"


def _annotate(ax: plt.Axes, values: np.ndarray, labels: list[str], pad: int = 4) -> None:
    for y, (value, label) in enumerate(zip(values, labels)):
        if not np.isfinite(value):
            continue
        xpos = value if value != 0.0 else 0.0
        ax.annotate(label, xy=(xpos, y), xytext=(pad, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["label"] = df["case"].map(CASE_LABELS).fillna(df["case"])
    df["category_rank"] = df["category"].map(CATEGORY_ORDER).fillna(99)
    df = df.sort_values(["category_rank", "case"]).reset_index(drop=True)

    y = np.arange(len(df))
    colors = df["category"].map(CATEGORY_COLORS).fillna("#6c757d").to_list()

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True, constrained_layout=True)

    ns = df["ns_per_call"].to_numpy(dtype=float)
    axes[0].barh(y, ns, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ns / call (log)")
    axes[0].set_title("Runtime")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["label"])
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.25, axis="x", which="both")
    _annotate(axes[0], ns, [_format_ns(v) for v in ns])

    allocs = df["allocations_per_call"].to_numpy(dtype=float)
    axes[1].barh(y, allocs, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("allocations / call")
    axes[1].set_title("Allocator Traffic")
    axes[1].set_xlim(left=0.0, right=max(3.5, float(np.nanmax(allocs)) * 1.35))
    axes[1].grid(True, alpha=0.25, axis="x")
    _annotate(axes[1], allocs, [f"{v:.0f}" if v.is_integer() else f"{v:.2f}" for v in allocs])

    bytes_per_call = df["estimated_bytes_per_call"].to_numpy(dtype=float)
    axes[2].barh(y, bytes_per_call, color=colors, edgecolor="black", linewidth=0.4)
    axes[2].set_xscale("symlog", linthresh=1.0)
    axes[2].set_xlabel("estimated bytes / call (symlog)")
    axes[2].set_title("Estimated Footprint")
    axes[2].grid(True, alpha=0.25, axis="x", which="both")
    _annotate(axes[2], bytes_per_call, [_format_bytes(v) for v in bytes_per_call])

    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.4, label=category.replace("_", " "))
        for category, color in CATEGORY_COLORS.items()
        if category in set(df["category"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Post-remediation basis microbenchmarks", fontsize=13)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    csv_path = data_dir / "perf" / "basis_remediation_microbench.csv"
    if not csv_path.exists():
        print(f"SKIP basis remediation microbench: {csv_path} not present")
        return 0

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    plot(df, fig_dir / "basis_remediation_microbench.png")
    print(f"Wrote {fig_dir / 'basis_remediation_microbench.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
