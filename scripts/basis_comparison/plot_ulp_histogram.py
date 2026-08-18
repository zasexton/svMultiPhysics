"""ULP-distance histogram: how many doubles apart are the two implementations?

For each (sample, dof) pair, compute the integer ULP distance between
N_legacy and N_oop (and similarly for gradient components). Most pairs
should be 0 ULP (bit-identical) or 1 ULP (one floating-point step apart).
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


def _ord_uint(x: np.ndarray) -> np.ndarray:
    """Map float64 values to a monotonic uint64 ordering.

    Standard IEEE-754 trick:
      positive: u | SIGN_BIT  (lift above all negatives)
      negative: ~u            (flip all bits so larger magnitude -> smaller order)
    """
    u = x.view(np.uint64).copy()
    sign = (u >> np.uint64(63)) == np.uint64(1)
    out = np.where(sign, ALL_ONES - u, u | SIGN_BIT)
    return out.astype(np.uint64)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer ULP distance between two arrays of float64."""
    aa = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    bb = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    out = np.full(aa.shape, np.iinfo(np.int64).max, dtype=np.int64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if finite.any():
        oa = _ord_uint(aa[finite])
        ob = _ord_uint(bb[finite])
        diff = np.where(oa > ob, oa - ob, ob - oa)
        cap = np.uint64(np.iinfo(np.int64).max)
        diff = np.minimum(diff, cap).astype(np.int64)
        out[finite] = diff
    return out


def plot(values_df: pd.DataFrame, grad_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(values_df, "elem_type", ELEMENT_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    axes = axes.ravel()

    bins = np.array([-0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 64.5, 1e15])
    bin_labels = ["0", "1", "2", "3-4", "5-8", "9-16", "17-64", ">64"]

    for i, elem in enumerate(elements):
        ax = axes[i]
        v_sub = values_df[values_df["elem_type"] == elem]
        g_sub = grad_df[grad_df["elem_type"] == elem]
        ulp_v = ulp_distance(v_sub["N_legacy"].to_numpy(),
                             v_sub["N_oop"].to_numpy())
        ulp_g = ulp_distance(g_sub["dN_legacy"].to_numpy(),
                             g_sub["dN_oop"].to_numpy())

        v_counts, _ = np.histogram(ulp_v, bins=bins)
        g_counts, _ = np.histogram(ulp_g, bins=bins)
        v_pct = 100.0 * v_counts / max(v_counts.sum(), 1)
        g_pct = 100.0 * g_counts / max(g_counts.sum(), 1)

        x = np.arange(len(bin_labels))
        w = 0.4
        ax.bar(x - w / 2, v_pct, w, color="#3a86ff", label="basis values")
        ax.bar(x + w / 2, g_pct, w, color="#fb5607", label="gradients")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=0, fontsize=8)
        ax.set_ylabel("% of pairs")
        ax.set_xlabel("ULP distance |legacy - OOP|")
        ax.set_ylim(0, 105)
        ax.set_title(
            f"{elem}: bit-identical = {v_pct[0]:.1f}% values, {g_pct[0]:.1f}% grads"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")

    for j in range(len(elements), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Floating-point agreement in ULPs (units in the last place).\n"
        "0 ULP means bit-identical doubles; 1 ULP means adjacent representable values.",
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
    plot(values_df, grad_df, fig_dir / "ulp_distance_histogram.png")
    print(f"Wrote {fig_dir / 'ulp_distance_histogram.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
