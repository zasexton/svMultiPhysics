# Lagrange Basis: Legacy vs OOP Accuracy Comparison

End-to-end harness that drives both the legacy `nn::` basis evaluations and the
new `svmp::FE::basis::LagrangeBasis` at the same reference-element points,
dumps CSVs, and renders comparison figures.

The two implementations are expected to agree to machine precision — figures
are designed to showcase the *kind* of agreement (bit-identical vs 1 ULP vs
machine eps) and the structural relationship (node permutations, basis-function
shapes, polynomial reproduction sets) rather than just to confirm "max error is
small."

## What is compared

For each of {TRI3, TRI6, TET4, TET10, HEX8, HEX27}:

- **Pointwise basis values & gradients** at OOP node coords + 40 random interior points.
- **Partition of unity** `Sum_i N_i(xi)` at every sampled point.
- **Dense reference-element grid** (61 x 61) for contour plots.
- **Polynomial reproduction**: nodal-coefficient reconstruction of monomials up to total degree 4.

The harness auto-discovers the legacy <-> OOP node permutation by evaluating
the legacy basis at each OOP node coordinate. All comparison figures apply
that permutation transparently.

## How to run

```bash
# 1. Build (compiles all legacy CSRCS + svfe lib; ~5 min cold)
cmake --build build-unit/svMultiPhysics-build --target run_all_unit_tests -j$(nproc)

# 2. Run harness, dumping CSVs
SVMP_BASIS_COMPARE_OUT=$PWD/scripts/basis_comparison/data \
  build-unit/svMultiPhysics-build/bin/run_all_unit_tests \
  --gtest_filter='LagrangeBasisComparison.*'

# 3. Render all figures
python3 scripts/basis_comparison/plot_all.py
```

## Output: data/ (CSVs)

| CSV | Contents |
|-----|----------|
| `basis_values.csv` | Per (elem, sample, dof) basis values from both sides + abs error |
| `basis_gradients.csv` | Per (elem, sample, dof, dim) gradient components + abs error |
| `partition_of_unity.csv` | `Sum N_i` and `||Sum grad N_i||` for both sides per sample |
| `node_locations_oop.csv` | OOP node reference coordinates per element |
| `node_permutation.csv` | Discovered legacy_dof -> oop_dof mapping per element |
| `summary.csv` | Per-element max errors and the discovered permutation |
| `contour_grid.csv` | Dense reference-element grid with values + gradients per dof |
| `polynomial_reproduction.csv` | Monomial reconstruction error per element & monomial |
| `interpolation_error.csv` | Lagrange interpolation error on smooth analytic test functions |
| `mass_matrix.csv` | Element mass matrix `M_ij = int N_i N_j dV` from both sides |
| `cross_sections.csv` | 1D `N_i(t)` along reference-element paths |

## Output: figures/ (PNGs)

| PNG | What it shows |
|-----|---------------|
| `summary_max_error_bars.png` | Headline: max\|N_legacy - N_oop\| and gradient analogue per element. Quick sanity check: all bars below the 1e-12 tolerance line. |
| `scatter_legacy_vs_oop.png` | Per-element scatter of N_legacy vs N_oop (and gradient pairs). Perfect agreement = all dots on y=x. Visual proof that no node-ordering bug remains. |
| `ulp_distance_histogram.png` | For each element, fraction of (sample, dof) pairs that are 0 / 1 / 2 / 3-4 / ... ULPs apart. Sharper statement than "max error < eps": e.g. HEX27 is 80% bit-identical, 18% within 1 ULP. |
| `error_cdf.png` | Empirical CDF of \|error\| per element with float64 eps marked. Curve drops off a cliff at machine precision; reads off "what fraction of pairs are within X." |
| `node_permutation_diagram.png` | Permutation matrix (legacy_idx vs oop_idx, off-diagonal entries highlighted) plus 3D scatter of nodes coloured green (identity match) or red (renumbered). Migration documentation artifact. |
| `kronecker_delta.png` | N_i evaluated at every reference node. Both implementations should produce the identity matrix; difference panel confirms agreement. Standard FE textbook diagnostic. |
| `reference_element_contours.png` | OOP-as-filled-contours / legacy-as-dashed-overlay basis function shapes for selected DOFs per element. Agreement = dashed lines lie within their corresponding colored bands. |
| `polynomial_reproduction.png` | log10 of max reconstruction error for each (element, monomial) pair, side by side legacy vs OOP. Light-green dots mark monomials inside the basis's reproduction set. The "wall" at the basis order is visible as the boundary between dark (machine eps) and light (O(1) error). |
| `partition_of_unity_diagnostic.png` | Histogram of \|Sum N_i - 1\| per sample point. Both implementations have identical distributions, all clustered at machine precision. |
| `gradient_field_quiver.png` | Reference-space gradient field for selected basis functions, OOP rendered as colored arrows (color = magnitude), legacy overlaid as black arrows. Agreement = arrows coincide. (3D elements show zeta=0 slice; zeta-component dropped.) |
| `interpolation_error_cdfs.png` | CDF of \|f_recon - f_true\| per (element, function) for four analytic test functions (sin·cos·exp, gauss bump, tanh step, oscillatory). Legacy and OOP curves overlay perfectly; horizontal position tells you basis interpolation quality on each function. |
| `interpolation_error_agreement.png` | Heatmap of `max\|err_legacy - err_oop\|` per (element, function). All cells at machine eps confirms reconstructions are identical. |
| `mass_matrix_comparison.png` | For each element: M_legacy, M_oop, `log10\|diff\|` heatmaps + ULP histogram of matrix entries. Tells you how 1-ULP pointwise basis disagreement propagates to assembled operators. |
| `cross_sections.png` | N_i(t) along edges and diagonals of the reference element for all DOFs. OOP solid colored, legacy dashed black overlay. 1D inspection complement to the 2D contours. |
| `conditioning_bars.png` | cond(V) and cond(M) per element, legacy vs OOP. cond(V)=1 by construction; cond(M) tells you how much each basis amplifies floating-point error under matrix inversion (HEX27: ~326, TRI3: ~4). |

## Headline findings

From a representative run, max\|legacy - OOP\| basis-value disagreement per element:

| Element | bit-identical | within 1 ULP | max abs error |
|---|---|---|---|
| TRI3  | 100.0% | 100% | 0 |
| TRI6  | 100.0% | 100% | 0 |
| TET4  | 100.0% | 100% | 0 |
| TET10 | 100.0% | 100% | 0 |
| HEX8  | 92.7%  | ~100% | 0 |
| HEX27 | 79.5%  | 97.5% | 1.7e-16 |

The HEX27 1.7e-16 (≈ 1 ULP) is consistent with two algebraically equivalent
formulations differing in operation ordering. Nothing exceeds 4 ULPs.

### Mass matrix disagreement

| Element | max\|M_leg - M_oop\| | sum\|M\| | relative |
|---|---|---|---|
| TRI3  | 0 | 0.50  | 0 |
| TRI6  | 0 | 0.67  | 0 |
| TET4  | 0 | 0.17  | 0 |
| TET10 | 0 | 0.36  | 0 |
| HEX8  | 0 | 8.00  | 0 |
| HEX27 | 2.78e-17 | 11.65 | 3.66e-16 |

HEX27 27-term quadrature summation actually *reduces* the pointwise 1.7e-16
disagreement to 2.8e-17 via partial cancellation. Pointwise basis disagreement
does not amplify under operator assembly.

### Mass matrix conditioning (identical legacy and OOP)

| Element | cond(M) |
|---|---|
| TRI3  | 4   |
| TRI6  | 17  |
| TET4  | 5   |
| TET10 | 36  |
| HEX8  | 27  |
| HEX27 | 326 |

## Discovered permutations (legacy_dof -> oop_dof)

| Element | Permutation | Off-diagonal entries |
|---------|-------------|----------------------|
| TRI3    | `[1, 2, 0]` | 3 |
| TRI6    | `[1, 2, 0, 4, 5, 3]` | 6 |
| TET4    | `[1, 2, 3, 0]` | 4 |
| TET10   | `[1, 2, 3, 0, 5, 9, 8, 4, 6, 7]` | 10 |
| HEX8    | identity | 0 |
| HEX27   | identity for 22 nodes; face nodes 20-25 reorder as `[25, 23, 22, 24, 20, 21]` | 5 |

## Scope and limitations

- Pointwise accuracy + monomial reproduction. **Not** a manufactured-solution
  PDE convergence harness (separate workstream).
- No timing — performance comparison is a separate workstream.
- Pyramid5/14, Wedge18: out of scope (legacy or OOP gaps).
- Hessians: out of scope (legacy lacks them for HEX8/HEX20/HEX27, see `nn.cpp:167`).

## Files

```
scripts/basis_comparison/
├── README.md                              this file
├── _helpers.py                            shared element-ordering tables
├── plot_all.py                            run every figure script
├── plot_pointwise_error.py                summary bars + PoU histogram
├── plot_scatter.py                        N_legacy vs N_oop scatter
├── plot_ulp_histogram.py                  ULP-distance histogram
├── plot_error_cdf.py                      empirical CDF of |error|
├── plot_permutation_diagram.py            permutation matrix + 3D node scatter
├── plot_kronecker.py                      Kronecker delta heatmap
├── plot_reference_contours.py             reference-element basis-function contours
├── plot_polynomial_reproduction.py        per-monomial reproduction error
├── plot_gradient_quiver.py                gradient-field arrow overlay
├── plot_interpolation_error.py            interpolation error on smooth functions
├── plot_mass_matrix.py                    element mass matrix comparison
├── plot_cross_sections.py                 1D N_i(t) line plots
├── plot_conditioning.py                   Vandermonde + mass matrix conditioning
├── data/                                  CSV outputs from the harness
└── figures/                               PNG outputs from plot_all.py
```

The harness (six gtest cases) lives at
`tests/unitTests/basis_comparison_tests/test_lagrange_basis_comparison.cpp`.
