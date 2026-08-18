"""Element mass matrix comparison.

For each element type the harness assembled
    M_ij = sum_q w_q N_i(xi_q) N_j(xi_q)
using the same quadrature for both implementations. We render four panels
per element:
  - M_legacy (heatmap of values)
  - M_oop    (heatmap of values)
  - log10|M_legacy - M_oop|  (the agreement story)
  - Per-row max-error bar chart (where in the matrix is disagreement
    concentrated, if any?)
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


SIGN_BIT = np.uint64(0x8000000000000000)
ALL_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    bb = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    finite = np.isfinite(aa) & np.isfinite(bb)
    out = np.zeros(aa.shape, dtype=np.int64)
    if finite.any():
        ua = aa[finite].view(np.uint64).copy()
        ub = bb[finite].view(np.uint64).copy()
        sa = (ua >> np.uint64(63)) == np.uint64(1)
        sb = (ub >> np.uint64(63)) == np.uint64(1)
        ma = np.where(sa, ALL_ONES - ua, ua | SIGN_BIT)
        mb = np.where(sb, ALL_ONES - ub, ub | SIGN_BIT)
        diff = np.where(ma > mb, ma - mb, mb - ma)
        cap = np.uint64(np.iinfo(np.int64).max)
        out[finite] = np.minimum(diff, cap).astype(np.int64)
    return out


def _build_matrices(df: pd.DataFrame, elem: str):
    sub = df[df["elem_type"] == elem]
    n = int(sub["i"].max()) + 1
    M_leg = np.zeros((n, n))
    M_oop = np.zeros((n, n))
    for _, row in sub.iterrows():
        i = int(row["i"]); j = int(row["j"])
        M_leg[i, j] = row["M_legacy"]
        M_oop[i, j] = row["M_oop"]
    return M_leg, M_oop, n


def plot(df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(df, "elem_type", ELEMENT_ORDER)
    n_elem = len(elements)
    fig, axes = plt.subplots(n_elem, 4, figsize=(15, 3.2 * n_elem),
                             constrained_layout=True)
    if n_elem == 1:
        axes = axes.reshape(1, -1)

    for r, elem in enumerate(elements):
        M_leg, M_oop, n = _build_matrices(df, elem)
        diff = np.abs(M_leg - M_oop)
        ulp = ulp_distance(M_leg.flatten(), M_oop.flatten()).reshape(n, n)

        # Panels
        for c, (M, title, cmap, vargs) in enumerate([
            (M_leg, "Legacy M",       "viridis", {}),
            (M_oop, "OOP M",           "viridis", {}),
            (np.log10(np.maximum(diff, 1e-20)),
             "log10|M_legacy - M_oop|", "magma",
             {"vmin": -18, "vmax": -10}),
        ]):
            ax = axes[r, c]
            im = ax.imshow(M, cmap=cmap, interpolation="nearest", **vargs)
            ax.set_title(f"{elem}: {title}", fontsize=9)
            if n <= 12:
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
            else:
                ax.set_xticks([0, n // 2, n - 1])
                ax.set_yticks([0, n // 2, n - 1])
            ax.set_xlabel("j")
            if c == 0:
                ax.set_ylabel("i")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

        # 4th panel: ULP histogram for the matrix entries
        ax = axes[r, 3]
        bins = np.array([-0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 32.5, 1e15])
        labels = ["0", "1", "2", "3-4", "5-8", "9-32", ">32"]
        counts, _ = np.histogram(ulp.flatten(), bins=bins)
        pct = 100.0 * counts / max(counts.sum(), 1)
        ax.bar(range(len(labels)), pct, color="#3a86ff")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("% of entries")
        ax.set_xlabel("ULP distance")
        ax.set_ylim(0, 105)
        max_abs = float(diff.max())
        ax.set_title(f"{elem}: ULP distribution\n"
                     f"max |Mleg-Moop|={max_abs:.2e}", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Element mass matrix M_ij = sum_q w_q N_i(xi_q) N_j(xi_q): "
        "legacy vs OOP.\n"
        "Quadrature: TRI3/TET4/HEX8/HEX27 use legacy mesh-free rules; "
        "TRI6 7-pt and TET10 15-pt are hardcoded from nn_elem_gip.h.",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "mass_matrix.csv")
    plot(df, fig_dir / "mass_matrix_comparison.png")
    print(f"Wrote {fig_dir / 'mass_matrix_comparison.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
