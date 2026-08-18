"""Pζ1: evaluate_all (fused) vs three separate evaluate_* calls."""

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
    sep = df[df["path"] == "separate"].set_index("elem_type").reindex(elements)
    fused = df[df["path"] == "evaluate_all"].set_index("elem_type").reindex(elements)

    x = np.arange(len(elements))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    bar1 = ax.bar(x - width / 2, sep["ns_per_call"], width,
                  color="#e76f51", label="evaluate_values + evaluate_gradients + evaluate_hessians",
                  edgecolor="black", linewidth=0.4)
    bar2 = ax.bar(x + width / 2, fused["ns_per_call"], width,
                  color="#2a9d8f", label="evaluate_all (fused)",
                  edgecolor="black", linewidth=0.4)

    # Annotate speedup per element.
    for xi, sv, fv in zip(x, sep["ns_per_call"].to_numpy(), fused["ns_per_call"].to_numpy()):
        if fv > 0:
            speedup = sv / fv
            top = max(sv, fv) * 1.05
            ax.annotate(f"{speedup:.2f}x", xy=(xi, top), ha="center", fontsize=9,
                        weight="bold", color="#264653")

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_yscale("log")
    ax.set_ylabel("ns / call  (log)")
    ax.set_xlabel("element")
    ax.set_title(
        "Pζ1: evaluate_all (fused) vs three separate evaluate_* calls\n"
        "Bars above each pair: speedup factor of the fused path (validates B4)"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y", which="both")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_evaluate_all_vs_separate.csv")
    plot(df, fig_dir / "perf_evaluate_all_vs_separate.png")
    print(f"Wrote {fig_dir / 'perf_evaluate_all_vs_separate.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
