"""Lagrange interpolation error on smooth analytic test functions.

Tests how well the basis interpolates non-polynomial functions sampled at
random interior points. The legacy/OOP interpolation errors should be
indistinguishable; the magnitude tells you basis quality on transcendental
targets (smaller for higher-order bases).

We render two complementary views:
  - CDF of |error| per (element, function), legacy and OOP overlaid.
  - Per-element 'agreement' bars: max |err_legacy - err_oop| across all samples.
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


def _cdf(x: np.ndarray, floor: float = 1e-20):
    x = np.maximum(np.asarray(x, dtype=float), floor)
    x = np.sort(x)
    p = np.linspace(0.0, 1.0, len(x), endpoint=True)
    return x, p


def plot_cdfs(df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(df, "elem_type", ELEMENT_ORDER)
    fns = sorted(df["function_name"].unique())
    n_fns = len(fns)
    n_elem = len(elements)

    fig, axes = plt.subplots(n_fns, n_elem, figsize=(2.7 * n_elem, 2.6 * n_fns),
                             constrained_layout=True, sharey=True)
    if n_fns == 1:
        axes = axes.reshape(1, -1)
    if n_elem == 1:
        axes = axes.reshape(-1, 1)

    for r, fn in enumerate(fns):
        for c, elem in enumerate(elements):
            ax = axes[r, c]
            sub = df[(df["elem_type"] == elem) & (df["function_name"] == fn)]
            if sub.empty:
                ax.axis("off")
                continue
            xl, pl = _cdf(sub["err_legacy"].to_numpy())
            xo, po = _cdf(sub["err_oop"].to_numpy())
            ax.plot(xl, pl, "-", color="#3a86ff", linewidth=1.3, label="legacy")
            ax.plot(xo, po, "--", color="#fb5607", linewidth=1.3, label="OOP")
            ax.set_xscale("log")
            ax.set_xlim(1e-20, 2.0)
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(elem, fontsize=10)
            if c == 0:
                ax.set_ylabel(fn, fontsize=8)
            if r == n_fns - 1:
                ax.set_xlabel("|f_recon - f_true|", fontsize=8)
            ax.grid(True, which="both", alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Interpolation-error CDFs per element and test function.\n"
        "Legacy (blue, solid) and OOP (orange, dashed) curves should overlap.\n"
        "The horizontal position tells the basis's interpolation quality on each function.",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_legacy_vs_oop_agreement(df: pd.DataFrame, out_path: Path) -> None:
    """Strict agreement: max |err_legacy - err_oop| across samples per (element, function)."""
    df["agreement"] = np.abs(df["err_legacy"] - df["err_oop"])
    pivot = (df.groupby(["elem_type", "function_name"])["agreement"]
             .max()
             .reset_index()
             .pivot(index="function_name", columns="elem_type", values="agreement"))
    pivot = pivot.reindex(columns=[e for e in ELEMENT_ORDER if e in pivot.columns])

    eps = 1e-20
    arr = np.maximum(pivot.values.astype(float), eps)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(pivot.index) + 2)),
                           constrained_layout=True)
    im = ax.imshow(np.log10(arr), aspect="auto", cmap="magma",
                   vmin=-18, vmax=-12, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("element")
    ax.set_title("max |err_legacy - err_oop| per (element, function)\n"
                 "(should be at machine epsilon)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                        label="log10 max | |f-f|_legacy - |f-f|_oop |")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "interpolation_error.csv")
    plot_cdfs(df, fig_dir / "interpolation_error_cdfs.png")
    plot_legacy_vs_oop_agreement(df, fig_dir / "interpolation_error_agreement.png")
    print(f"Wrote {fig_dir / 'interpolation_error_cdfs.png'}")
    print(f"Wrote {fig_dir / 'interpolation_error_agreement.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
