# LagrangeBasis Automated Test and Benchmark Gate Outline

Date: 2026-06-01

This document outlines the automated correctness and performance gates needed
to make `LagrangeBasis` production-grade as a maintained FE library component.
The goal is to compare values, gradients, and Hessians for every supported
canonical topology/order across scalar vector-returning APIs, pointer-output
`*_to` APIs, and strided batched APIs, then fail automated gates on hot-path heap
allocations or significant performance regressions.

## Scope

Covered implementation surface:

- `LagrangeBasis` scalar vector APIs: `evaluate_values`,
  `evaluate_gradients`, `evaluate_hessians`, and `evaluate_all`.
- Pointer-output APIs: `evaluate_values_to`, `evaluate_gradients_to`,
  `evaluate_hessians_to`, and `evaluate_all_to`.
- Batched strided API: `evaluate_at_quadrature_points_strided`.
- Complete canonical topologies: `Line2`, `Triangle3`, `Quad4`, `Tetra4`,
  `Hex8`, `Wedge6`, and `Pyramid5`.
- Complete low-order aliases that normalize to canonical paths: `Line3`,
  `Triangle6`, `Quad9`, `Tetra10`, `Hex27`, `Wedge18`, and `Pyramid14`.
- Explicit rejection behavior for serendipity variants handled by
  `SerendipityBasis`: `Quad8`, `Hex20`, `Wedge15`, and `Pyramid13`.

Because complete Lagrange constructors support arbitrary order, automation must
be generated from a configured order range rather than handwritten case lists.
The default production gate should cover orders `0..8`, matching the full
range used by the accepted Lagrange benchmark baseline.

## Correctness Test Matrix

Each canonical topology/order pair must run the same API-equivalence matrix.

| Output request | Scalar vector API | Pointer-output API | Strided API |
|---|---|---|---|
| values only | `evaluate_values` | `evaluate_values_to` | values pointer only |
| gradients only | `evaluate_gradients` | `evaluate_gradients_to` | gradients pointer only |
| Hessians only | `evaluate_hessians` | `evaluate_hessians_to` | Hessians pointer only |
| values + gradients + Hessians | `evaluate_all` | `evaluate_all_to` | all pointers |
| mixed subsets | scalar individual calls | individual `*_to` calls | every nonempty pointer subset |

The scalar vector API is the primary API-consistency reference for the pointer
and strided paths. Independent mathematical checks, listed below, keep that
from becoming only self-consistency.

## Sample Points

Use deterministic sample sets per topology so failures are reproducible:

- Interior points away from boundaries for every topology.
- Boundary points on faces and edges where derivatives are finite.
- Near-node points for high-order cancellation sensitivity.
- Quadrature-rule points for common assembly paths.
- Exact interpolation nodes for value-only Kronecker checks.
- Exact pyramid apex for values and expected derivative throws only.
- Near-apex pyramid points for directional-limit coverage, excluding exact apex
  from gradient/Hessian equivalence tests.

For strided tests, use at least these batch shapes:

| Batch | Purpose |
|---|---|
| `qpts = 1`, `stride = 1` | scalar-equivalent minimum batch |
| `qpts = 4`, `stride = 4` | optimized SIMD-like hot path |
| `qpts = 4`, `stride = 7` | padded stride and untouched padding |
| `qpts = 9`, `stride = 11` | non-specialized fallback coverage |

## API Equivalence Tests

Add a parameterized suite in
`Code/Source/solver/FE/Tests/Unit/Basis/test_LagrangeBasis.cpp` with a generated
case table:

```text
for topology in canonical_lagrange_topologies:
  for order in [0, max_order]:
    for sample_set(topology, order):
      compare scalar APIs
      compare pointer-output APIs against scalar APIs
      compare strided APIs against scalar APIs
      verify dirty-output and padding behavior
```

Required checks:

- `evaluate_all` must match the three individual scalar vector calls.
- `evaluate_all_to` must match `evaluate_all`.
- Individual `*_to` calls must match their scalar vector equivalents.
- Strided values, gradients, and Hessians must match scalar pointwise results
  for each requested output subset.
- Mixed strided requests must not require unrelated output pointers to be
  present.
- Output buffers initialized with sentinels must have every active entry
  overwritten and every padded stride entry left untouched.
- Components outside the reference dimension must be explicitly zero.
- Hessians must remain symmetric in the stored components where the analytical
  basis is smooth.
- All API paths must agree for fixed low-order fast paths and generic high-order
  fallback paths.

The existing helpers `expect_strided_matches_pointwise` and
`expect_raw_to_matches_vector_evaluation` should be generalized instead of
duplicating per-topology tests.

## Independent Mathematical Oracles

API equivalence alone cannot detect a common wrong implementation. The
production suite should include independent checks that do not call the same
optimized path as the implementation under test.

| Oracle | Topologies | Orders | Required outputs |
|---|---|---:|---|
| Kronecker delta at nodes | all canonical | `0..max_order` | values |
| Partition of unity | all canonical | `0..max_order` | values |
| Zero derivative sums | all canonical smooth points | `0..max_order` | gradients, Hessians |
| Polynomial reproduction | line, triangle, quad, tet, hex, wedge | `1..max_order` | values, gradients, Hessians |
| Tensor-product oracle | line, quad, hex | `0..max_order` | values, gradients, Hessians |
| Simplex barycentric oracle | triangle, tet | `0..max_order` | values, gradients, Hessians |
| Wedge product oracle | wedge | `0..max_order` | values, gradients, Hessians |
| Pyramid rational oracle | pyramid | `1..max_order` | values, gradients, Hessians away from apex |
| Finite-difference derivative spot checks | all canonical | representative orders | gradients, Hessians |

Polynomial reproduction should construct nodal coefficients from analytic
monomials whose total degree is less than or equal to the basis order. The
reconstructed value, gradient, and Hessian at sample points must match the
analytic monomial derivatives. For wedge, use triangular total-degree monomials
times the line coordinate basis. For pyramid, use the documented rational
complete-family space and keep apex derivative behavior in a separate test.

## Tolerances

Use relative-plus-absolute tolerances scaled by order and derivative level:

```text
tol = base_tol(topology, output) *
      max(1, abs(reference)) *
      max(1, order_scale(order, derivative_level))
```

Initial values:

| Topology class | Values | Gradients | Hessians |
|---|---:|---:|---:|
| tensor and simplex | `1e-12` | `5e-12` | `2e-11` |
| wedge | `2e-12` | `1e-11` | `5e-11` |
| pyramid away from apex | `5e-10` | `5e-9` | `5e-8` |

High-order and near-boundary cases may use a documented multiplier. Do not
silently widen tolerances in individual assertions; route all tolerance choices
through one helper so changes are reviewable.

## Alias and Rejection Tests

Alias tests should verify that complete low-order public element names normalize
to the same canonical basis path:

- `Line3` order 2 equals `Line2` order 2.
- `Triangle6` order 2 equals `Triangle3` order 2.
- `Quad9` order 2 equals `Quad4` order 2.
- `Tetra10` order 2 equals `Tetra4` order 2.
- `Hex27` order 2 equals `Hex8` order 2.
- `Wedge18` order 2 equals `Wedge6` order 2.
- `Pyramid14` order 2 equals `Pyramid5` order 2, with the pyramid apex
  derivative exception preserved.

Rejection tests should continue to assert that serendipity variants are not
accepted by `LagrangeBasis` and are routed to the serendipity implementation
surface instead.

## Benchmark Gate

The benchmark gate should wrap `basis_perf_microbench` and parse its CSV output.
The gate should be implemented as a small script, for example:

```text
tools/check_lagrange_basis_benchmark_gate.py
```

The script inputs should be:

- Current benchmark CSV.
- Optional baseline CSV for the same machine/compiler/build class.
- Maximum allowed per-case slowdown.
- Maximum allowed category geometric-mean slowdown.
- Whether to enforce roofline thresholds.

Recommended benchmark command:

```bash
cmake --build build-fe-eigen-check --target basis_perf_microbench -j 4

SVMP_BASIS_BENCH_LAGRANGE_PEAK=1 \
SVMP_BASIS_BENCH_LAGRANGE_PARALLEL=1 \
SVMP_BASIS_BENCH_LAGRANGE_PEAK_MAX_ORDER=8 \
SVMP_BASIS_BENCH_REPEATS=5 \
./build-fe-eigen-check/basis_perf_microbench 0 \
  > /tmp/lagrange_basis_benchmark_current.csv
```

Hard allocation failures:

- Any non-construction Lagrange row with `allocations_per_call != 0` fails.
- Any hot scalar, `*_to`, or strided evaluation row missing from the CSV fails.
- Construction rows may allocate, but their allocation counts should be tracked
  separately and compared against construction baselines.

Hard performance failures:

- Any required hot case slower than baseline by more than 25% fails.
- Any required hot category geometric mean slower than baseline by more than
  10% fails.
- Any case with `measured_to_model_bound` above the documented roofline
  threshold fails when hardware calibration inputs are supplied.
- Missing or changed benchmark keys fail unless the baseline update explicitly
  records the expected rename or model change.

Soft warnings:

- Any individual hot case slower than baseline by more than 10%.
- Any case with unusually high variance across repeats.
- Any pyramid high-order case whose model is still unclassified by the dense
  modal-to-nodal work model.

Baseline updates must be explicit. Store the command line, compiler ID,
compiler version, CPU model, thread count, SIMD width, and benchmark CSV next to
the accepted baseline.

Accepted baseline update workflow:

1. Rebuild the benchmark target from the exact tree being accepted:

   ```bash
   cmake --build build-fe-eigen-check --target basis_perf_microbench -j 4
   ```

2. Capture the current machine/compiler benchmark with full Lagrange coverage:

   ```bash
   SVMP_BASIS_BENCH_LAGRANGE_PEAK=1 \
   SVMP_BASIS_BENCH_LAGRANGE_PARALLEL=1 \
   SVMP_BASIS_BENCH_LAGRANGE_PEAK_MAX_ORDER=8 \
   SVMP_BASIS_BENCH_REPEATS=5 \
   ./build-fe-eigen-check/basis_perf_microbench 0 \
     > /tmp/lagrange_basis_benchmark_current.csv
   ```

3. Gate the candidate against the current accepted baseline:

   ```bash
   python3 tools/check_lagrange_basis_benchmark_gate.py \
     --current /tmp/lagrange_basis_benchmark_current.csv \
     --baseline Code/Source/solver/FE/Basis/performance_graphs/lagrange_basis_benchmark_baseline.csv \
     --fail-on-hot-allocations \
     --max-case-slowdown 1.25 \
     --max-category-geomean-slowdown 1.10
   ```

4. If the change intentionally updates the baseline, copy the accepted CSV to
   `Code/Source/solver/FE/Basis/performance_graphs/lagrange_basis_benchmark_baseline.csv`
   and keep the matching current report at
   `Code/Source/solver/FE/Basis/performance_graphs/lagrange_basis_benchmark.csv`.
   The benchmark rows record compiler ID, compiler version, build flags, CPU
   model, hardware threads, benchmark threads, SIMD width, repeat count, and
   variance columns. The update rationale must be recorded in the review or
   commit message together with any expected benchmark key/model rename.

## Suggested Commands

```bash
cmake --build build-fe-eigen-check --target test_fe_basis basis_perf_microbench -j 4

ctest --test-dir build-fe-eigen-check \
  -R '^FE_Basis_Tests$' \
  --output-on-failure

python3 tools/check_lagrange_basis_benchmark_gate.py \
  --current /tmp/lagrange_basis_benchmark_current.csv \
  --baseline Code/Source/solver/FE/Basis/performance_graphs/lagrange_basis_benchmark_baseline.csv \
  --fail-on-hot-allocations \
  --max-case-slowdown 1.25 \
  --max-category-geomean-slowdown 1.10
```

## Implementation Sequence

1. Centralize topology/order enumeration helpers for Lagrange tests.
2. Generalize existing scalar, `*_to`, and strided comparison helpers.
3. Add dirty-buffer and padded-stride checks to every API-equivalence path.
4. Add independent polynomial, tensor, simplex, wedge, and pyramid oracles.
5. Add explicit pyramid apex value and derivative-throw cases.
6. Add the benchmark CSV gate script and baseline schema.
7. Document the accepted baseline update workflow in this outline next to the
   benchmark gate command.

## Acceptance Checklist

- [x] Every canonical topology/order in the configured range is generated by
  the test matrix.
- [x] Values, gradients, and Hessians agree across scalar vector, `*_to`, and
  strided APIs for every output subset.
- [x] Independent mathematical oracles cover values, gradients, and Hessians.
- [x] Pyramid apex value behavior and derivative exception behavior are tested.
- [x] Dirty-output tests prove full active-buffer overwrite and untouched
  padding.
- [x] Non-construction Lagrange benchmark rows have zero hot-path allocations.
- [x] Benchmark gate fails on missing rows, hot allocations, and major runtime
  regressions.
- [x] Baseline update procedure records machine, compiler, build flags, and
  rationale.
