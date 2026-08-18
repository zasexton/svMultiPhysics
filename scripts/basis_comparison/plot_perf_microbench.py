"""Pointwise microbenchmark: ns / call per element, operation, implementation."""

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
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    df["label"] = df["operation"].astype(str) + " (" + df["implementation"] + ")"
    labels = sorted(df["label"].unique(),
                    key=lambda s: (0 if "legacy" in s else 1, s))
    color_map = {
        "values_and_gradients (legacy)": "#264653",
        "values (oop)":                  "#2a9d8f",
        "gradients (oop)":               "#e9c46a",
        "hessians (oop)":                "#f4a261",
        "hessians (legacy)":             "#e76f51",
    }
    x = np.arange(len(elements))
    width = 0.16

    for k, label in enumerate(labels):
        sub = df[df["label"] == label].set_index("elem_type")
        sub = sub.reindex(elements)
        ys = sub["ns_per_call"].to_numpy()
        offset = (k - (len(labels) - 1) / 2.0) * width
        ax.bar(x + offset, ys, width, color=color_map.get(label, None),
               label=label, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_yscale("log")
    ax.set_ylabel("ns / call  (log)")
    ax.set_xlabel("element")
    ax.set_title("3.1 Pointwise microbenchmark: ns / call (lower is better)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25, axis="y", which="both")

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)


    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_microbench_pointwise.csv")
    plot(df, fig_dir / "perf_microbench_pointwise.png")
    print(f"Wrote {fig_dir / 'perf_microbench_pointwise.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
