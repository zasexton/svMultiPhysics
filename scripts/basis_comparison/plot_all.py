"""Run every figure script in this folder and print a summary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def run_script(script_path: Path) -> int:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return int(mod.main())


def main() -> int:
    here = Path(__file__).resolve().parent
    scripts = [
        "plot_pointwise_error.py",       # original summary bars + PoU
        "plot_scatter.py",
        "plot_ulp_histogram.py",
        "plot_error_cdf.py",
        "plot_permutation_diagram.py",
        "plot_kronecker.py",
        "plot_reference_contours.py",
        "plot_polynomial_reproduction.py",
        "plot_gradient_quiver.py",
        "plot_interpolation_error.py",
        "plot_mass_matrix.py",
        "plot_cross_sections.py",
        "plot_conditioning.py",
        # Performance figures (require SVMP_FE_RUN_PERF_TESTS=1 harness CSVs)
        "plot_perf_microbench.py",
        "plot_perf_setup.py",
        "plot_perf_steady_state.py",
        "plot_perf_fused.py",
        "plot_perf_cache.py",
        "plot_perf_parallel.py",
        "plot_basis_remediation_microbench.py",
        # Pζ + Pη diagnostic figures (require evaluate_all_vs_separate,
        # multi_qp_vs_per_qp, and hw_counters CSVs).
        "plot_evaluate_all_vs_separate.py",
        "plot_multi_qp_vs_per_qp.py",
        "plot_perf_hw_counters.py",
    ]
    rc = 0
    for s in scripts:
        p = here / s
        if not p.exists():
            print(f"  SKIP {s}: not present")
            continue
        try:
            r = run_script(p)
            print(f"  OK   {s}: rc={r}")
            rc = max(rc, r)
        except Exception as e:
            print(f"  FAIL {s}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
