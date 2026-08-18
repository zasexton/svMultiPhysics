"""Shared helpers for basis comparison plotting scripts."""

from __future__ import annotations

import os
from pathlib import Path

ELEMENT_ORDER = ["TRI3", "TRI6", "TET4", "TET10", "HEX8", "HEX27"]
ELEMENT_DIM = {"TRI3": 2, "TRI6": 2, "TET4": 3, "TET10": 3, "HEX8": 3, "HEX27": 3}
ELEMENT_SIMPLEX = {"TRI3": True, "TRI6": True, "TET4": True, "TET10": True,
                   "HEX8": False, "HEX27": False}


def default_dirs(here: Path):
    data_dir = os.environ.get("SVMP_BASIS_COMPARE_DATA_DIR")
    if data_dir is None:
        data_dir = os.environ.get("SVMP_BASIS_COMPARE_OUT")
    fig_dir = os.environ.get("SVMP_BASIS_COMPARE_FIGURES_DIR")
    return (
        Path(data_dir) if data_dir else here / "data",
        Path(fig_dir) if fig_dir else here / "figures",
    )


def filter_present(df, col, order):
    present = set(df[col].unique())
    return [x for x in order if x in present]
