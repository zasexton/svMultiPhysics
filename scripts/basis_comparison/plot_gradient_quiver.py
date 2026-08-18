"""Gradient field quiver overlay for selected basis functions.

OOP gradient field rendered as colored arrows; legacy gradient field overlaid
as dashed black arrows. Visual agreement = arrows coincide arrowhead-for-
arrowhead. We subsample the dense reference-element grid to keep the quiver
density readable.

For 3D elements we project to the same zeta = 0 slice used by the contour
plots. The zeta-component of the gradient is dropped on that projection.
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


# Reuse the same DOF picks as the contour plot for visual coherence.
DOF_PICKS = {
    "TRI3":  [0, 1, 2],
    "TRI6":  [0, 3, 5],
    "TET4":  [0, 1, 2],
    "TET10": [0, 4, 6],
    "HEX8":  [0, 2, 6],
    "HEX27": [16, 22, 26],
}

QUIVER_STRIDE = 4  # subsample the 61x61 grid down to ~15x15 arrows


def _stride_mask(xs, ys, stride: int):
    # Coarsen by selecting roughly every `stride`-th point along each axis.
    keep = np.zeros(len(xs), dtype=bool)
    if len(xs) == 0:
        return keep
    # Map (x,y) onto a coarse grid.
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    nx = max(1, int(round(np.sqrt(len(xs)) / stride)))
    ny = nx
    if x_max == x_min or y_max == y_min:
        return np.ones(len(xs), dtype=bool)
    ix = np.clip(((xs - x_min) / (x_max - x_min) * (nx - 1)).round().astype(int), 0, nx - 1)
    iy = np.clip(((ys - y_min) / (y_max - y_min) * (ny - 1)).round().astype(int), 0, ny - 1)
    seen = set()
    for n, (a, b) in enumerate(zip(ix, iy)):
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        keep[n] = True
    return keep


def plot(grid_df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(grid_df, "elem_type", ELEMENT_ORDER)
    n_elem = len(elements)
    n_dofs = max(len(DOF_PICKS.get(e, [0])) for e in elements)

    fig, axes = plt.subplots(n_elem, n_dofs,
                             figsize=(4.0 * n_dofs, 3.6 * n_elem),
                             constrained_layout=True)
    if n_elem == 1:
        axes = axes.reshape(1, -1)
    if n_dofs == 1:
        axes = axes.reshape(-1, 1)

    for i, elem in enumerate(elements):
        sub_e = grid_df[grid_df["elem_type"] == elem]
        picks = DOF_PICKS.get(elem, [0])
        for j_pick, dof in enumerate(picks):
            ax = axes[i, j_pick]
            sub = sub_e[sub_e["oop_dof_index"] == dof]
            if sub.empty:
                ax.text(0.5, 0.5, f"DOF {dof} not present", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{elem}: dof {dof}")
                continue

            x = sub["xi_x"].to_numpy()
            y = sub["xi_y"].to_numpy()
            mask = _stride_mask(x, y, QUIVER_STRIDE)
            xs = x[mask]
            ys = y[mask]
            uo = sub["dN_oop_x"].to_numpy()[mask]
            vo = sub["dN_oop_y"].to_numpy()[mask]
            ul = sub["dN_legacy_x"].to_numpy()[mask]
            vl = sub["dN_legacy_y"].to_numpy()[mask]
            mag = np.sqrt(uo * uo + vo * vo)

            # OOP quiver: colored by magnitude
            qo = ax.quiver(xs, ys, uo, vo, mag, cmap="viridis",
                           scale_units="xy", angles="xy", scale=None,
                           width=0.005, headwidth=3.5)
            # Legacy quiver: thinner black dashed
            ax.quiver(xs, ys, ul, vl, color="black",
                      scale_units="xy", angles="xy", scale=None,
                      width=0.0015, headwidth=2.5, alpha=0.85)
            fig.colorbar(qo, ax=ax, fraction=0.046, pad=0.04, shrink=0.82,
                         label="|grad| (OOP)")

            # Reference outline.
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
            ax.set_title(f"{elem}: grad N (oop dof {dof})", fontsize=10)
            if j_pick == 0:
                ax.set_ylabel("eta")
            if i == n_elem - 1:
                ax.set_xlabel("xi")

        for j in range(len(picks), n_dofs):
            axes[i, j].axis("off")

    fig.suptitle(
        "Reference-space gradient field of selected basis functions.\n"
        "Colored arrows: OOP gradient (color = magnitude). "
        "Black thin arrows: legacy gradient. "
        "Agreement = arrows coincide.\n"
        "(3D elements show the zeta = 0 slice; zeta-component of grad is omitted.)",
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
    plot(grid_df, fig_dir / "gradient_field_quiver.png")
    print(f"Wrote {fig_dir / 'gradient_field_quiver.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
