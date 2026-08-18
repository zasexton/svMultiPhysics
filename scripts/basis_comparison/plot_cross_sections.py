"""1D cross-section line plots through the reference element.

For each (element, path) plot N_i(t) for all DOFs i. OOP as solid colored
lines, legacy as black dashed overlay.
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


def plot(df: pd.DataFrame, out_path: Path) -> None:
    elements = filter_present(df, "elem_type", ELEMENT_ORDER)
    n_elem = len(elements)
    paths_per = max(df[df["elem_type"] == e]["path_name"].nunique() for e in elements)
    fig, axes = plt.subplots(n_elem, paths_per, figsize=(2.6 * paths_per, 2.4 * n_elem),
                             constrained_layout=True)
    if n_elem == 1:
        axes = axes.reshape(1, -1)
    if paths_per == 1:
        axes = axes.reshape(-1, 1)

    for r, elem in enumerate(elements):
        sub_e = df[df["elem_type"] == elem]
        paths = list(dict.fromkeys(sub_e["path_name"].tolist()))  # preserve order
        eNoN = sub_e["oop_dof_index"].max() + 1
        eNoN = int(eNoN)
        cmap = plt.get_cmap("tab20")
        for c, path in enumerate(paths):
            ax = axes[r, c]
            sub = sub_e[sub_e["path_name"] == path]
            for dof in range(eNoN):
                d_sub = sub[sub["oop_dof_index"] == dof].sort_values("t")
                if d_sub.empty:
                    continue
                color = cmap(dof % cmap.N)
                ax.plot(d_sub["t"], d_sub["N_oop"], color=color, linewidth=1.0,
                        alpha=0.85)
                ax.plot(d_sub["t"], d_sub["N_legacy"], color="black",
                        linestyle="--", linewidth=0.6, alpha=0.6)
            ax.set_title(f"{elem}: {path}", fontsize=8)
            ax.set_xlim(0, 1)
            ax.tick_params(axis="both", which="major", labelsize=7)
            if r == n_elem - 1:
                ax.set_xlabel("path parameter t", fontsize=8)
            if c == 0:
                ax.set_ylabel("N_i", fontsize=8)
            ax.grid(True, alpha=0.25)

        # Hide unused
        for c in range(len(paths), paths_per):
            axes[r, c].axis("off")

    fig.suptitle(
        "Basis function 1D cross-sections.  Colored lines: OOP (one per DOF).  "
        "Black dashed overlay: legacy.\n"
        "Visual agreement = dashed lines overlay their colored counterparts.",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "cross_sections.csv")
    plot(df, fig_dir / "cross_sections.png")
    print(f"Wrote {fig_dir / 'cross_sections.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
