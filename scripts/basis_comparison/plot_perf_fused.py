"""Fused-kernel benchmark: mass / stiffness / convection per element."""

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
    operators = ["mass", "stiffness", "convection"]
    colors = {
        "legacy_manual": "#264653",
        "oop_manual":    "#e76f51",
        "oop_fused":     "#2a9d8f",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True,
                             sharey=True)

    x = np.arange(len(elements))
    width = 0.27

    for col, op in enumerate(operators):
        ax = axes[col]
        sub_op = df[df["operator"] == op]
        impls = sorted(sub_op["implementation"].unique(),
                       key=lambda s: ("legacy_manual", "oop_manual", "oop_fused").index(s)
                       if s in ("legacy_manual", "oop_manual", "oop_fused") else 99)
        for k, impl in enumerate(impls):
            sub = sub_op[sub_op["implementation"] == impl].set_index("elem_type")
            sub = sub.reindex(elements)
            ys = sub["ns_per_element"].to_numpy()
            offset = (k - (len(impls) - 1) / 2.0) * width
            ax.bar(x + offset, ys, width, color=colors.get(impl), label=impl,
                   edgecolor="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(elements)
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("ns / element  (log)")
        ax.set_xlabel("element")
        ax.set_title(f"{op}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y", which="both")

    fig.suptitle(
        "3.4 Element matrix assembly: ns / element per operator and implementation\n"
        "Mass = sum_q w_q N_i N_j;  Stiffness = sum_q w_q grad N_i . grad N_j;  "
        "Convection = sum_q w_q N_i (b . grad N_j)",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_fused_kernels.csv")
    plot(df, fig_dir / "perf_fused_kernels.png")
    print(f"Wrote {fig_dir / 'perf_fused_kernels.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
