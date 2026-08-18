"""Permutation visualization: NxN permutation matrix per element + 3D scatter
showing the geometric position of each renumbered node.

The permutation matrix gives a complete, compact view: black on the diagonal
means an identity permutation; off-diagonal entries highlight which DOFs
were renumbered. The 3D scatter shows where the renumbered nodes physically sit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ELEMENT_ORDER, ELEMENT_SIMPLEX, default_dirs, filter_present


def plot(nodes_df: pd.DataFrame, perm_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(nodes_df, "elem_type", ELEMENT_ORDER)

    n_elem = len(elements)
    # Layout: 3 rows x 4 cols. Each element occupies 2 cols (matrix | 3D scatter).
    n_rows = (n_elem + 1) // 2
    fig = plt.figure(figsize=(16, 5.0 * n_rows), constrained_layout=True)

    for i, elem in enumerate(elements):
        sub_nodes = nodes_df[nodes_df["elem_type"] == elem]
        sub_perm = perm_df[perm_df["elem_type"] == elem]
        eNoN = len(sub_perm)
        grid_row = i // 2
        col_offset = (i % 2) * 2  # 0 or 2

        # Permutation matrix view (col 1 of pair)
        ax_mat = fig.add_subplot(n_rows, 4, grid_row * 4 + col_offset + 1)
        M = np.zeros((eNoN, eNoN), dtype=int)
        for _, perm_row in sub_perm.iterrows():
            M[int(perm_row["legacy_dof_index"]), int(perm_row["oop_dof_index"])] = 1
        ax_mat.imshow(M, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        ax_mat.set_xticks(range(eNoN))
        ax_mat.set_yticks(range(eNoN))
        if eNoN > 12:
            ax_mat.set_xticks([0, eNoN // 2, eNoN - 1])
            ax_mat.set_yticks([0, eNoN // 2, eNoN - 1])
        ax_mat.set_xlabel("OOP index")
        ax_mat.set_ylabel("legacy index")
        n_off = int(eNoN - np.trace(M))
        ax_mat.set_title(
            f"{elem}: permutation matrix  ({n_off} off-diagonal entries)",
            fontsize=10,
        )
        # Diagonal reference line
        ax_mat.plot([-0.5, eNoN - 0.5], [-0.5, eNoN - 0.5],
                    color="#06ffa5", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_mat.set_xlim(-0.5, eNoN - 0.5)
        ax_mat.set_ylim(eNoN - 0.5, -0.5)
        ax_mat.set_aspect("equal", adjustable="box")
        # Highlight off-diagonal entries
        for _, perm_row in sub_perm.iterrows():
            j_leg = int(perm_row["legacy_dof_index"])
            k_oop = int(perm_row["oop_dof_index"])
            if j_leg != k_oop:
                ax_mat.add_patch(plt.Rectangle((k_oop - 0.5, j_leg - 0.5), 1, 1,
                                                fill=False, edgecolor="#d62828",
                                                linewidth=1.2))

        # 3D scatter view
        if ELEMENT_SIMPLEX[elem]:
            ref_lim = (-0.05, 1.1)
        else:
            ref_lim = (-1.15, 1.15)

        ax_geo = fig.add_subplot(n_rows, 4, grid_row * 4 + col_offset + 2, projection="3d")
        coords = sub_nodes[["oop_dof_index", "xi_x", "xi_y", "xi_z"]].to_numpy()

        # For each node identify whether the permutation is identity at it
        legacy_to_oop = dict(zip(sub_perm["legacy_dof_index"], sub_perm["oop_dof_index"]))
        oop_to_legacy = {v: k for k, v in legacy_to_oop.items()}
        for node_row in coords:
            oop_idx = int(node_row[0])
            x, y, z = node_row[1], node_row[2], node_row[3]
            legacy_idx = oop_to_legacy.get(oop_idx, -1)
            mismatch = (legacy_idx != oop_idx)
            color = "#d62828" if mismatch else "#06a77d"
            ax_geo.scatter([x], [y], [z], c=color, s=50, edgecolors="black",
                           linewidths=0.6, depthshade=False)
            label = f"L{legacy_idx}/{oop_idx}" if mismatch else f"{oop_idx}"
            ax_geo.text(x, y, z, " " + label, fontsize=6,
                        color="black" if not mismatch else "#7a0000")

        # Outline the reference cell
        if ELEMENT_SIMPLEX[elem]:
            verts = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
            ])
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for a, b in edges:
                ax_geo.plot(*zip(verts[a], verts[b]), color="black", linewidth=0.6)
        else:
            verts = np.array([[s1, s2, s3]
                              for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)])
            edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                     (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            for a, b in edges:
                ax_geo.plot(*zip(verts[a], verts[b]), color="black", linewidth=0.6)

        ax_geo.set_xlim(*ref_lim)
        ax_geo.set_ylim(*ref_lim)
        ax_geo.set_zlim(*ref_lim)
        ax_geo.set_xlabel("xi", labelpad=-3)
        ax_geo.set_ylabel("eta", labelpad=-3)
        ax_geo.set_zlabel("zeta", labelpad=-3)
        ax_geo.tick_params(axis="both", which="major", pad=0, labelsize=7)
        ax_geo.set_title(
            f"{elem}: green = identity, red = renumbered",
            fontsize=10,
        )
        ax_geo.view_init(elev=20, azim=-55)

    fig.suptitle(
        "Legacy <-> OOP node numbering: permutation matrix (left) and reference-element layout (right).\n"
        "Permutation matrix:  black squares show legacy_idx -> oop_idx; "
        "green dashed line is the identity diagonal; red boxes mark off-diagonal entries.\n"
        "3D scatter:  green points where legacy index equals OOP index; "
        "red points are renumbered (label shows L<legacy>/<oop>).",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    nodes_df = pd.read_csv(data_dir / "node_locations_oop.csv")
    perm_df = pd.read_csv(data_dir / "node_permutation.csv")
    plot(nodes_df, perm_df, fig_dir / "node_permutation_diagram.png")
    print(f"Wrote {fig_dir / 'node_permutation_diagram.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
