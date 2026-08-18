"""Pζ2: evaluate_at_quadrature_points (multi-QP) vs per-QP loop."""

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
    per_qp = df[df["path"] == "per_qp_loop"].set_index("elem_type").reindex(elements)
    multi = df[df["path"] == "multi_qp_entry"].set_index("elem_type").reindex(elements)
    n_qpts = per_qp["n_qpts"].fillna(0).astype(int).to_numpy()

    x = np.arange(len(elements))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    ax.bar(x - width / 2, per_qp["ns_per_qp"], width,
           color="#e76f51", label="per-QP loop calling evaluate_all",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, multi["ns_per_qp"], width,
           color="#2a9d8f", label="evaluate_at_quadrature_points (multi-QP entry)",
           edgecolor="black", linewidth=0.4)

    for xi, p, m in zip(x, per_qp["ns_per_qp"].to_numpy(), multi["ns_per_qp"].to_numpy()):
        if m > 0:
            speedup = p / m
            top = max(p, m) * 1.05
            color = "#2a7d6f" if speedup >= 1.0 else "#b04848"
            ax.annotate(f"{speedup:.2f}x", xy=(xi, top), ha="center", fontsize=9,
                        weight="bold", color=color)

    # Annotate n_qpts under each element label.
    labels = [f"{e}\n({nq} QPs)" for e, nq in zip(elements, n_qpts)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("ns / quadrature point  (log)")
    ax.set_xlabel("element  (number of QPs in parentheses)")
    ax.set_title(
        "Pζ2: per-QP loop vs evaluate_at_quadrature_points (multi-QP entry)\n"
        "Bars above each pair: speedup factor of the multi-QP path (validates E3-partial)"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y", which="both")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir, fig_dir = default_dirs(here)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "perf" / "perf_multi_qp_vs_per_qp.csv")
    plot(df, fig_dir / "perf_multi_qp_vs_per_qp.png")
    print(f"Wrote {fig_dir / 'perf_multi_qp_vs_per_qp.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
