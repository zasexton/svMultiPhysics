"""Steady-state element-loop access: 4 access patterns per element."""

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
    elements = filter_present(df, "elem_type", ELEMENT_ORDER)
    patterns = ["legacy_array", "oop_cache_random", "oop_cache_span", "oop_batch_aligned"]
    colors = {
        "legacy_array":      "#264653",
        "oop_cache_random":  "#e76f51",
        "oop_cache_span":    "#f4a261",
        "oop_batch_aligned": "#2a9d8f",
    }

    x = np.arange(len(elements))
    width = 0.20
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    for k, p in enumerate(patterns):
        sub = df[df["access_pattern"] == p].set_index("elem_type")
        sub = sub.reindex(elements)
        ys = sub["ns_per_element"].to_numpy()
        offset = (k - 1.5) * width
        ax.bar(x + offset, ys, width, color=colors[p], label=p,
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_yscale("log")
    ax.set_ylabel("ns / element  (log)")
    ax.set_xlabel("element")
    ax.set_title("3.3 Steady-state element loop: ns / element by access pattern\n"
                 "(legacy mesh.N array vs OOP cache random/span vs OOP BatchEvaluator SIMD-aligned)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", which="both")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_steady_state_access.csv")
    plot(df, fig_dir / "perf_steady_state_access.png")
    print(f"Wrote {fig_dir / 'perf_steady_state_access.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
