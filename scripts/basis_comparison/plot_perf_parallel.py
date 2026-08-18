"""Parallel scaling: elements/second vs thread count."""

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
    impls = sorted(df["implementation"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True,
                             sharex=True)
    axes = axes.ravel()

    colors = {"legacy": "#264653", "oop_cache_span": "#2a9d8f"}

    for i, elem in enumerate(elements):
        ax = axes[i]
        sub_e = df[df["elem_type"] == elem]
        for impl in impls:
            sub = sub_e[sub_e["implementation"] == impl].sort_values("n_threads")
            t = sub["n_threads"].to_numpy()
            eps = sub["elements_per_second_total"].to_numpy()
            ax.plot(t, eps, "o-", color=colors.get(impl), label=impl, linewidth=1.5)

        # Ideal scaling lines (linear extrapolation from 1-thread baseline)
        for impl in impls:
            sub = sub_e[sub_e["implementation"] == impl].sort_values("n_threads")
            if sub.empty:
                continue
            base = float(sub.iloc[0]["elements_per_second_total"])
            t_arr = sub["n_threads"].to_numpy(dtype=float)
            ax.plot(t_arr, base * t_arr, "--", color=colors.get(impl), alpha=0.4,
                    linewidth=0.9)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(elem, fontsize=10)
        ax.set_xlabel("n_threads")
        if i % 3 == 0:
            ax.set_ylabel("elements / sec (total)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, which="both")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])

    fig.suptitle(
        "3.7 Parallel scaling: total throughput vs thread count.\n"
        "Solid: measured.  Dashed: ideal linear scaling from 1-thread baseline.",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_parallel_scaling.csv")
    plot(df, fig_dir / "perf_parallel_scaling.png")
    print(f"Wrote {fig_dir / 'perf_parallel_scaling.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
