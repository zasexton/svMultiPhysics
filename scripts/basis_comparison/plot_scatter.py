"""Scatter N_legacy vs N_oop. If they agree, all points lie on y=x."""

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


def plot(values_df: pd.DataFrame, grad_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(values_df, "elem_type", ELEMENT_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    axes = axes.ravel()

    for i, elem in enumerate(elements):
        ax = axes[i]
        sub = values_df[values_df["elem_type"] == elem]
        leg = sub["N_legacy"].to_numpy()
        oop = sub["N_oop"].to_numpy()

        ax.scatter(leg, oop, s=8, alpha=0.5, color="#3a86ff", edgecolors="none",
                   label="basis values")

        # Overlay gradient component pairs as smaller orange dots
        gsub = grad_df[grad_df["elem_type"] == elem]
        gleg = gsub["dN_legacy"].to_numpy()
        goop = gsub["dN_oop"].to_numpy()
        ax.scatter(gleg, goop, s=4, alpha=0.35, color="#fb5607", edgecolors="none",
                   label="gradient components")

        lo = min(leg.min(), gleg.min(), -0.05) - 0.1
        hi = max(leg.max(), gleg.max(), 1.05) + 0.1
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.7, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{elem}: legacy vs OOP (n_pairs={len(leg) + len(gleg)})")
        ax.set_xlabel("legacy value")
        ax.set_ylabel("OOP value")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)

    for j in range(len(elements), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Per-implementation parity scatter: each dot is a basis value or gradient "
        "component pair (after node permutation).\n"
        "Perfect agreement = all dots on y = x.",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    values_df = pd.read_csv(data_dir / "basis_values.csv")
    grad_df = pd.read_csv(data_dir / "basis_gradients.csv")
    plot(values_df, grad_df, fig_dir / "scatter_legacy_vs_oop.png")
    print(f"Wrote {fig_dir / 'scatter_legacy_vs_oop.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
