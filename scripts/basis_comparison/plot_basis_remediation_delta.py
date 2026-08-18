"""Before/after performance delta for the basis remediation run.

Set SVMP_BASIS_COMPARE_BASELINE_DIR to a previous data directory and
SVMP_BASIS_COMPARE_DATA_DIR to the new data directory. The plot reports
baseline/new timing ratios, so values above 1.0 are faster after the change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ELEMENT_ORDER, default_dirs


def _load_ratio(
    baseline_dir: Path,
    current_dir: Path,
    rel_path: str,
    metric: str,
    keys: list[str],
    path_labels: dict[str, str],
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_dir / rel_path)
    current = pd.read_csv(current_dir / rel_path)
    merged = baseline.merge(current, on=keys, suffixes=("_baseline", "_current"))
    merged["speed_ratio"] = merged[f"{metric}_baseline"] / merged[f"{metric}_current"]
    merged["metric_label"] = merged["elem_type"] + " " + merged["path"].map(path_labels).fillna(merged["path"])
    merged["elem_rank"] = merged["elem_type"].map({elem: i for i, elem in enumerate(ELEMENT_ORDER)})
    merged["path_rank"] = merged["path"].map({path: i for i, path in enumerate(path_labels)})
    return merged.sort_values(["elem_rank", "path_rank"]).reset_index(drop=True)


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    y = np.arange(len(df))
    values = df["speed_ratio"].to_numpy(dtype=float)
    colors = np.where(values >= 1.0, "#2a9d8f", "#e76f51")

    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["metric_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("baseline / after timing ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="x")

    xmax = max(1.2, float(np.nanmax(values)) * 1.15)
    ax.set_xlim(0.0, xmax)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:.2f}x", xy=(value, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=7)


def plot(baseline_dir: Path, current_dir: Path, out_path: Path) -> None:
    eval_all = _load_ratio(
        baseline_dir,
        current_dir,
        "perf/perf_evaluate_all_vs_separate.csv",
        "ns_per_call",
        ["elem_type", "path"],
        {"separate": "separate", "evaluate_all": "evaluate_all"},
    )
    multi_qp = _load_ratio(
        baseline_dir,
        current_dir,
        "perf/perf_multi_qp_vs_per_qp.csv",
        "ns_per_qp",
        ["elem_type", "path"],
        {"per_qp_loop": "per-qp", "multi_qp_entry": "multi-qp"},
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    _plot_panel(axes[0], eval_all, "evaluate_all vs separate")
    _plot_panel(axes[1], multi_qp, "multi-qp entry vs per-qp loop")
    fig.suptitle("Basis remediation performance delta (>1.0 is faster after)", fontsize=13)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    current_dir, fig_dir = default_dirs(here)
    baseline_env = os.environ.get("SVMP_BASIS_COMPARE_BASELINE_DIR")
    if not baseline_env:
        print("SKIP basis remediation delta: set SVMP_BASIS_COMPARE_BASELINE_DIR")
        return 0

    baseline_dir = Path(baseline_env)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot(baseline_dir, current_dir, fig_dir / "basis_remediation_perf_delta.png")
    print(f"Wrote {fig_dir / 'basis_remediation_perf_delta.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
