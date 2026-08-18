"""Kronecker delta heatmap: each basis function evaluated at every node.

Should be the identity matrix in both implementations. Off-diagonal mass
would reveal node-ordering bugs at a glance.

Data source: basis_values.csv rows where sample_idx < eNoN (those samples
are the OOP nodes themselves, since the harness puts oop_nodes first in
the test_points list).
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


def kron_matrix(values_df: pd.DataFrame, elem: str, eNoN: int, value_col: str):
    sub = values_df[(values_df["elem_type"] == elem)
                    & (values_df["sample_idx"] < eNoN)]
    pivot = sub.pivot_table(
        index="oop_dof_index",
        columns="sample_idx",
        values=value_col,
        aggfunc="first",
    )
    # Reindex to ensure 0..eNoN-1 ordering
    pivot = pivot.reindex(index=range(eNoN), columns=range(eNoN))
    return pivot.to_numpy()


def plot(values_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(values_df, "elem_type", ELEMENT_ORDER)
    eNoN_lookup = dict(zip(summary_df["elem_type"], summary_df["eNoN"]))

    n = len(elements)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.8 * n), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, elem in enumerate(elements):
        eNoN = int(eNoN_lookup[elem])
        K_oop = kron_matrix(values_df, elem, eNoN, "N_oop")
        K_leg = kron_matrix(values_df, elem, eNoN, "N_legacy")
        diff = np.abs(K_oop - K_leg)

        # Row 1: OOP, Row 2: legacy, Row 3: difference
        for ax, M, title, cmap, vmin, vmax in [
            (axes[i, 0], K_oop, "OOP",   "RdBu_r", -1.05, 1.05),
            (axes[i, 1], K_leg, "Legacy", "RdBu_r", -1.05, 1.05),
            (axes[i, 2], diff,  "|OOP - Legacy|", "magma", 0.0, 1e-15),
        ]:
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(f"{elem}: {title}", fontsize=10)
            if eNoN <= 10:
                ax.set_xticks(range(eNoN))
                ax.set_yticks(range(eNoN))
                # Annotate values in small grids
                for r in range(eNoN):
                    for c in range(eNoN):
                        v = M[r, c]
                        if title.startswith("|"):
                            txt = "0" if v < 1e-18 else f"{v:.0e}"
                        else:
                            txt = f"{v:.2f}" if abs(v) > 0.01 else "0"
                        color = "white" if abs(v) > 0.5 and not title.startswith("|") else "black"
                        ax.text(c, r, txt, ha="center", va="center",
                                fontsize=6, color=color)
            else:
                ax.set_xticks([0, eNoN - 1])
                ax.set_yticks([0, eNoN - 1])
            ax.set_xlabel("node (sample) index" if i == n - 1 else "")
            if title == "OOP":
                ax.set_ylabel("OOP basis-fn index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

    fig.suptitle(
        "Kronecker delta diagnostic: N_i evaluated at every reference node.\n"
        "Both implementations should produce the identity matrix.",
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
    summary_df = pd.read_csv(data_dir / "summary.csv")
    plot(values_df, summary_df, fig_dir / "kronecker_delta.png")
    print(f"Wrote {fig_dir / 'kronecker_delta.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
