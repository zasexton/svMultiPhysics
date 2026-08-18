"""Empirical CDF of |error| (basis values and gradients) per element type.

Log-scaled x-axis from 1e-20 to 1. Curve drops off a cliff at machine epsilon.
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
from _helpers import ELEMENT_ORDER, default_dirs, filter_present


COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#06a77d", "#9b5de5"]


def cdf(x: np.ndarray, floor: float = 1e-20) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(np.asarray(x, dtype=float), floor)
    x = np.sort(x)
    p = np.linspace(0.0, 1.0, len(x), endpoint=True)
    return x, p


def plot(values_df: pd.DataFrame, grad_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(values_df, "elem_type", ELEMENT_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    eps = np.finfo(float).eps

    for k, (df, col, title) in enumerate([
        (values_df, "abs_err", "Basis values: |N_legacy - N_oop|"),
        (grad_df,   "abs_err", "Gradients: |dN_legacy - dN_oop|"),
    ]):
        ax = axes[k]
        for i, elem in enumerate(elements):
            sub = df[df["elem_type"] == elem]
            x, p = cdf(sub[col].to_numpy())
            ax.plot(x, p, label=elem, color=COLORS[i % len(COLORS)], linewidth=1.6)
        ax.axvline(eps, color="red", linestyle="--", alpha=0.6, label="float64 eps")
        ax.set_xscale("log")
        ax.set_xlim(1e-20, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("|error|  (log scale)")
        ax.set_ylabel("empirical CDF")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Distribution of pointwise disagreement across all sampled (point, dof) pairs",
                 fontsize=11)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    values_df = pd.read_csv(data_dir / "basis_values.csv")
    grad_df = pd.read_csv(data_dir / "basis_gradients.csv")
    plot(values_df, grad_df, fig_dir / "error_cdf.png")
    print(f"Wrote {fig_dir / 'error_cdf.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
