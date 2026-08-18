"""Vandermonde and mass-matrix conditioning bars per element.

Vandermonde: V_ij = N_i(node_j). By construction this is the identity, so
cond(V) = 1 for any correct Lagrange basis. Useful as a triple-check.

Mass matrix: M_ij = integral N_i N_j dV (assembled by the harness in
mass_matrix.csv). The condition number measures how the basis amplifies
floating-point error if/when the mass matrix is inverted in downstream solvers.
Both implementations should give identical numbers.
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


def _vandermonde(values_df: pd.DataFrame, summary_df: pd.DataFrame, elem: str, value_col: str):
    eNoN = int(summary_df[summary_df["elem_type"] == elem]["eNoN"].iloc[0])
    sub = values_df[(values_df["elem_type"] == elem)
                    & (values_df["sample_idx"] < eNoN)]
    pivot = sub.pivot_table(
        index="oop_dof_index",
        columns="sample_idx",
        values=value_col,
        aggfunc="first",
    ).reindex(index=range(eNoN), columns=range(eNoN))
    return pivot.to_numpy()


def _mass(mass_df: pd.DataFrame, elem: str, col: str):
    sub = mass_df[mass_df["elem_type"] == elem]
    n = int(sub["i"].max()) + 1
    M = np.zeros((n, n))
    for _, r in sub.iterrows():
        M[int(r["i"]), int(r["j"])] = r[col]
    return M


def plot(values_df: pd.DataFrame, summary_df: pd.DataFrame, mass_df: pd.DataFrame,
         out_path: Path) -> None:
    elements = filter_present(summary_df, "elem_type", ELEMENT_ORDER)

    rows = []
    for elem in elements:
        V_oop = _vandermonde(values_df, summary_df, elem, "N_oop")
        V_leg = _vandermonde(values_df, summary_df, elem, "N_legacy")
        M_oop = _mass(mass_df, elem, "M_oop")
        M_leg = _mass(mass_df, elem, "M_legacy")
        rows.append({
            "elem_type":  elem,
            "cond_V_oop": float(np.linalg.cond(V_oop)),
            "cond_V_leg": float(np.linalg.cond(V_leg)),
            "cond_M_oop": float(np.linalg.cond(M_oop)),
            "cond_M_leg": float(np.linalg.cond(M_leg)),
        })
    cond = pd.DataFrame(rows)
    cond.to_csv(out_path.with_suffix(".csv"), index=False)

    x = np.arange(len(elements))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left: Vandermonde
    ax = axes[0]
    ax.bar(x - w / 2, cond["cond_V_leg"], w, color="#3a86ff", label="legacy")
    ax.bar(x + w / 2, cond["cond_V_oop"], w, color="#fb5607", label="OOP")
    ax.set_xticks(x)
    ax.set_xticklabels(cond["elem_type"])
    ax.set_ylabel("cond(V) = cond(N_i(node_j))")
    ax.set_title("Vandermonde matrix conditioning\n(should be 1.0 for both)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    # Right: mass matrix (log scale)
    ax = axes[1]
    ax.bar(x - w / 2, cond["cond_M_leg"], w, color="#3a86ff", label="legacy")
    ax.bar(x + w / 2, cond["cond_M_oop"], w, color="#fb5607", label="OOP")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cond["elem_type"])
    ax.set_ylabel("cond(M) = cond(int N_i N_j)  (log scale)")
    ax.set_title("Mass matrix conditioning\n(higher = more error amplification under inversion)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", which="both")

    # Annotate values
    for i, row in cond.iterrows():
        axes[0].text(i, max(row["cond_V_leg"], row["cond_V_oop"]) * 1.02,
                     f"{row['cond_V_oop']:.3g}", ha="center", fontsize=7)
        axes[1].text(i, row["cond_M_oop"] * 1.15,
                     f"{row['cond_M_oop']:.2g}", ha="center", fontsize=7)

    fig.suptitle("Per-element matrix conditioning: legacy vs OOP",
                 fontsize=11)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("Conditioning summary:")
    print(cond.to_string(index=False))


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    values_df = pd.read_csv(data_dir / "basis_values.csv")
    summary_df = pd.read_csv(data_dir / "summary.csv")
    mass_df = pd.read_csv(data_dir / "mass_matrix.csv")
    plot(values_df, summary_df, mass_df, fig_dir / "conditioning_bars.png")
    print(f"Wrote {fig_dir / 'conditioning_bars.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
