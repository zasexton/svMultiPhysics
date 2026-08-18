"""Polynomial reproduction error per (element, monomial).

For each monomial m(xi) = x^a * y^b * z^c, we reconstruct it from nodal
values using each implementation's basis: f_recon(xi) = Sum_i m(node_i) * N_i(xi).
The reconstruction is exact (to machine precision) for monomials within the
basis's reproduction set, and inexact beyond.

We plot a heatmap of log10(max reconstruction error) per (element, monomial),
side by side for legacy vs OOP. The "wall" at the basis order is the figure.
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


def _nice_label(a: int, b: int, c: int) -> str:
    parts = []
    if a == 0 and b == 0 and c == 0:
        return "1"
    for sym, e in [("x", a), ("y", b), ("z", c)]:
        if e == 0:
            continue
        parts.append(sym if e == 1 else f"{sym}^{e}")
    return "·".join(parts)


def plot(poly_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(poly_df, "elem_type", ELEMENT_ORDER)
    # Aggregate: per (element, monomial), take the max reconstruction error
    # across all sample points.
    agg = (
        poly_df.groupby(["elem_type", "monomial_label", "a", "b", "c",
                         "total_degree", "max_axis_degree", "in_reproduction_set"])
        [["err_legacy", "err_oop"]]
        .max()
        .reset_index()
    )

    # Order monomials by total degree, then per-axis
    agg["monomial_pretty"] = agg.apply(
        lambda r: _nice_label(int(r.a), int(r.b), int(r.c)), axis=1
    )
    agg = agg.sort_values(["total_degree", "max_axis_degree", "a", "b", "c"]).reset_index(drop=True)

    monomial_order = list(dict.fromkeys(agg["monomial_pretty"].tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.22 * len(monomial_order))),
                             constrained_layout=True)
    eps = 1e-20

    for k, (col, title, cmap) in enumerate([
        ("err_legacy", "Legacy reconstruction error", "magma"),
        ("err_oop",    "OOP reconstruction error",    "magma"),
    ]):
        ax = axes[k]
        pivot = agg.pivot_table(index="monomial_pretty", columns="elem_type",
                                values=col, aggfunc="max")
        pivot = pivot.reindex(index=monomial_order, columns=elements)
        log_err = np.log10(np.maximum(pivot.values.astype(float), eps))
        im = ax.imshow(log_err, aspect="auto", cmap=cmap, vmin=-18, vmax=0,
                       interpolation="nearest")
        ax.set_xticks(range(len(elements)))
        ax.set_xticklabels(elements, rotation=20)
        ax.set_yticks(range(len(monomial_order)))
        ax.set_yticklabels(monomial_order, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                     label="log10 max |f_recon - f_true|")

        # Mark monomials in the reproduction set with a green dot
        for r, mon in enumerate(monomial_order):
            for c, elem in enumerate(elements):
                row_match = agg[(agg["monomial_pretty"] == mon)
                                & (agg["elem_type"] == elem)]
                if not row_match.empty and bool(row_match.iloc[0]["in_reproduction_set"]):
                    ax.plot(c, r, marker=".", color="#06ffa5", markersize=4)

    fig.suptitle(
        "Polynomial reproduction error per element and monomial.\n"
        "Light dots mark monomials inside each basis's reproduction set "
        "(simplex: total deg <= p; tensor: every axis deg <= p).\n"
        "Inside the set: error at machine epsilon. Outside: O(1) - the basis "
        "cannot represent the monomial; that is the basis order, not a bug.",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    poly_df = pd.read_csv(data_dir / "polynomial_reproduction.csv")
    plot(poly_df, fig_dir / "polynomial_reproduction.png")
    print(f"Wrote {fig_dir / 'polynomial_reproduction.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
