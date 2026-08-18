"""Setup-cost benchmark: legacy mesh-init vs OOP cold/warm cache."""

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
    impls = ["legacy_full_fill", "oop_cache_cold", "oop_cache_warm"]
    colors = {"legacy_full_fill": "#264653",
              "oop_cache_cold":   "#e76f51",
              "oop_cache_warm":   "#2a9d8f"}

    x = np.arange(len(elements))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    for k, impl in enumerate(impls):
        sub = df[df["implementation"] == impl].set_index("elem_type")
        sub = sub.reindex(elements)
        ys = sub["ns_per_setup"].to_numpy()
        # Legacy may be missing for TRI6/TET10 (mesh-free gip lacks them).
        offset = (k - 1) * width
        ax.bar(x + offset, ys, width, color=colors[impl], label=impl,
               edgecolor="black", linewidth=0.4)
        for xi, yi in zip(x + offset, ys):
            if not np.isnan(yi) and yi > 0:
                ax.text(xi, yi * 1.05, f"{yi:.0f}", ha="center",
                        fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_yscale("log")
    ax.set_ylabel("ns / setup  (log)")
    ax.set_xlabel("element")
    ax.set_title("3.2 Setup cost: legacy full mesh-init vs OOP cache cold / warm\n"
                 "(legacy bars missing for TRI6 and TET10 — they require the "
                 "mesh-bound API, out of scope for this bench)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", which="both")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_setup_cost.csv")
    plot(df, fig_dir / "perf_setup_cost.png")
    print(f"Wrote {fig_dir / 'perf_setup_cost.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
