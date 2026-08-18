# Lagrange Basis Performance Report

This folder is for fresh legacy-vs-optimized Lagrange basis comparison output.

Scope:

- Legacy-equivalent elements only: `TRI3`, `TRI6`, `TET4`, `TET10`, `HEX8`, `HEX27`.
- Accuracy from the existing `LagrangeBasisComparison` harness.
- Runtime and parallel throughput from the existing `LagrangeBasisPerformance` harness.
- Memory as a derived working-set estimate from the benchmark element sizes, quadrature counts, and API result tensor widths.

Generate fresh data:

```bash
SVMP_BASIS_COMPARE_OUT=$PWD/scripts/basis_comparison/lagrange_performance_20260529/data \
  build/svMultiPhysics-build/bin/run_all_unit_tests \
  --gtest_filter='LagrangeBasisComparison.*'

SVMP_FE_RUN_PERF_TESTS=1 \
SVMP_BASIS_COMPARE_OUT=$PWD/scripts/basis_comparison/lagrange_performance_20260529/data \
  build/svMultiPhysics-build/bin/run_all_unit_tests \
  --gtest_filter='LagrangeBasisPerformance.*'
```

Generate the focused plots:

```bash
python3 scripts/basis_comparison/lagrange_performance_20260529/plot_lagrange_legacy_vs_optimized.py
```

The script writes PNG/SVG figures to `figures/`, derived CSVs to `data/`, and a short generated `SUMMARY.md`.
