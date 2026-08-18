"""Reference-element contour plots: visualize the actual basis function shapes.

For each element type, picks a representative basis function (by default the
last DOF, which tends to be the most interesting interior/face mode for
higher-order elements) and renders contours of N(xi, eta) in 2D.

For 3D elements, the harness has already produced data on a zeta = 0 slice.

OOP values are shown as filled contours; legacy values are overlaid as dashed
contour lines. If they agree, the dashed lines coincide with the colored bands.
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
from _helpers import ELEMENT_ORDER, ELEMENT_SIMPLEX, default_dirs, filter_present


# DOFs to render per element type. Indices are OOP DOF indices.
# For 3D elements we plot a zeta=0 slice; we therefore pick DOFs whose 1D
# zeta factor is non-zero at zeta=0 so the contours have visible structure.
DOF_PICKS = {
    "TRI3":  [0, 1, 2],
    "TRI6":  [0, 3, 5],          # corner, edge 0-1, edge 2-0
    "TET4":  [0, 1, 2],          # 3 corners on zeta=0 face
    "TET10": [0, 4, 6],          # corner, mid-edge 0-1, mid-edge 0-2 (all on zeta=0 face)
    "HEX8":  [0, 2, 6],          # 3 corners (bilinear in zeta -> visible at zeta=0)
    # For HEX27 corners/edges along zeta vanish at zeta=0; pick nodes lying on
    # the slice: vertical-edge midpoint (16), vertical-face center (22), body (26).
    "HEX27": [16, 22, 26],
}


def plot(grid_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(grid_df, "elem_type", ELEMENT_ORDER)

    n_elem = len(elements)
    n_dofs_per = max(len(DOF_PICKS.get(e, [0])) for e in elements)

    fig, axes = plt.subplots(n_elem, n_dofs_per,
                             figsize=(4.0 * n_dofs_per, 3.5 * n_elem),
                             constrained_layout=True)
    if n_elem == 1:
        axes = axes.reshape(1, -1)
    if n_dofs_per == 1:
        axes = axes.reshape(-1, 1)

    for i, elem in enumerate(elements):
        sub_elem = grid_df[grid_df["elem_type"] == elem]
        dof_picks = DOF_PICKS.get(elem, [0])
        for j_pick, dof in enumerate(dof_picks):
            ax = axes[i, j_pick]
            sub = sub_elem[sub_elem["oop_dof_index"] == dof]
            if sub.empty:
                ax.text(0.5, 0.5, f"DOF {dof} not present", ha="center",
                        va="center", transform=ax.transAxes)
                ax.set_title(f"{elem}: dof {dof}")
                continue

            x = sub["xi_x"].to_numpy()
            y = sub["xi_y"].to_numpy()
            z_oop = sub["N_oop"].to_numpy()
            z_leg = sub["N_legacy"].to_numpy()

            # Filled contour from OOP.
            tcf = ax.tricontourf(x, y, z_oop, levels=12, cmap="viridis")
            # Overlay dashed contour lines from legacy at the same levels.
            try:
                ax.tricontour(x, y, z_leg, levels=tcf.levels,
                              colors="white", linestyles="--",
                              linewidths=0.7, alpha=0.85)
            except Exception:
                pass

            # Outline reference element.
            simplex = ELEMENT_SIMPLEX[elem]
            if simplex:
                ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-", linewidth=1.0)
                ax.set_xlim(-0.05, 1.1)
                ax.set_ylim(-0.05, 1.1)
            else:
                ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", linewidth=1.0)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{elem}: N_{{oop dof {dof}}}", fontsize=10)
            if j_pick == 0:
                ax.set_ylabel("eta")
            if i == n_elem - 1:
                ax.set_xlabel("xi")
            fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

        # Hide unused subplot columns
        for j in range(len(dof_picks), n_dofs_per):
            axes[i, j].axis("off")

    fig.suptitle(
        "Basis function shapes on the reference element.\n"
        "OOP = filled contours; legacy = white dashed lines.\n"
        "Agreement = dashed lines lie within their corresponding colored bands.\n"
        "(3D elements show a zeta = 0 slice.)",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid_df = pd.read_csv(data_dir / "contour_grid.csv")
    plot(grid_df, fig_dir / "reference_element_contours.png")
    print(f"Wrote {fig_dir / 'reference_element_contours.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
