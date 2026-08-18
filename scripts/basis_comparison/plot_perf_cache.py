"""Cache-effectiveness benchmark: OOP-only cold vs warm latency."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ELEMENT_ORDER, default_dirs, filter_present


def plot(df: pd.DataFrame, out_path: Path) -> None:
    elements = [e for e in ELEMENT_ORDER if e in df["elem_type"].unique()]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    x = np.arange(len(elements))
    width = 0.40
    cold = df[df["scenario"] == "cold"].set_index("elem_type").reindex(elements)
    warm = df[df["scenario"] == "warm"].set_index("elem_type").reindex(elements)
    ax.bar(x - width / 2, cold["ns_per_call"], width, color="#e76f51",
           label="cold (cache cleared each call)", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, warm["ns_per_call"], width, color="#2a9d8f",
           label="warm (cache hit)", edgecolor="black", linewidth=0.4)

    # Annotate ratio
    for xi, c, w in zip(x, cold["ns_per_call"].to_numpy(), warm["ns_per_call"].to_numpy()):
        if w > 0 and c > 0:
            ratio = c / w
            ax.annotate(f"{ratio:.0f}x", xy=(xi, c * 1.1), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_yscale("log")
    ax.set_ylabel("ns / call  (log)")
    ax.set_xlabel("element")

    # Mixed-mesh result as a horizontal text annotation
    mixed = df[df["scenario"] == "mixed_round_robin"]
    title = ("3.6 OOP cache effectiveness:  cold (compute) vs warm (lookup)\n"
             "Numbers above cold bars: speedup factor (cold / warm)")
    if not mixed.empty:
        m = mixed.iloc[0]
        title += f"\nMixed-mesh round-robin (Tri3+Tet4+Hex8): {m['ns_per_call']:.1f} ns/call"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", which="both")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_cache_effectiveness.csv")
    plot(df, fig_dir / "perf_cache_effectiveness.png")
    print(f"Wrote {fig_dir / 'perf_cache_effectiveness.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
