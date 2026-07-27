# WP-3/WP-7 symmetric-Nitsche finite-sample prerequisite

## Status and claim boundary

This document defines a finite serial low-level prerequisite. The only accepted
claim is `joint_low_level_prerequisite`.

The method-wide coercivity lower bound is deliberately `null`, with status
`UNFROZEN_NO_BOUND_INVENTED`. FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open.
The evidence does not replace either parent qualification matrix.

The implementing source revision is
`f127ce715f5d9042af3fa409d197667bc289e03f`. The frozen matrix byte-locks that
revision's implementation files, both parent matrices and runners, the WP-7
derivation, and the shared qualification base.

## Discrete operators

For constant dynamic viscosity, let `K` be the production viscous velocity
block integrated on the selected active cut volume. Let `S` be the sum of the
two symmetric Nitsche consistency terms on the selected generated active
boundary, and let `P` be the production Nitsche penalty on that same generated
boundary.

The diagnostic installs four independently assembled velocity operators:

1. `K`;
2. `K + S`;
3. `A = K + S + P`;
4. `H = K + P`.

For each non-dry sample, the test solves the finite generalized eigenproblem

`A x = lambda H x`.

The recorded sampled margin is the smallest computed eigenvalue minus the
dense eigensolver tolerance. This is a finite fixture result. It is not an
analytic or mesh-uniform lower bound.

Variable constitutive viscosity is excluded because its tangent contains
viscosity derivatives and is not the constant-viscosity bilinear decomposition
above. Enabling the diagnostic with a variable-viscosity model fails before
the finite-element system is mutated. The environment switch is also
insufficient by itself: a C++ caller must explicitly select
`JointLowLevelPrerequisite`.

## Native fixture

The fixture uses a strip of five cubes split into thirty linear tetrahedra,
three velocity components, and one triangular exterior wall face. A separate
strong velocity boundary anchors the bulk operator. The free surface cuts the
wall under:

- wall fractions `0`, `1e-8`, `1e-6`, `1e-4`, `1e-2`, `0.1`, `0.25`,
  `0.49`, and `1`;
- axis-aligned and oblique orientations;
- physical scales `1/2`, `1/3`, and `1/4`;
- negative and positive active-side selection.

This gives 108 cases: 96 positive-wet samples and 12 dry samples.

Every case verifies the installed structure directly. The bulk operator has
no boundary term. The other three operators have no ordinary whole-face
boundary term, have nonempty interface-face terms, and use only the selected
generated active-boundary marker present in the cut context. The same marker
route is used by the production and diagnostic forms.

## Exact dry contract

For a completely dry selected wall, the generated active-boundary rule count
is exactly zero. The reduced matrices must satisfy bit-for-bit equality:

`K + S == K`, `A == K`, and `H == K`.

The test separately requires the anchored `K` to be nonzero and to admit a
Cholesky factorization. A vacuous zero matrix therefore cannot satisfy the dry
certificate.

## Cut-volume tangent cache contract

The symbolic tangent cache key includes the cut-volume side both on expression
nodes and on compiled integral terms. A regression compiles the same nonlinear
integrand and marker first on the negative side and then on the positive side,
and requires each tangent to retain its requested side. This prevents reuse of
a negative-side tangent for a positive-side form.

## Aggregation guard envelope

The production default maximum root path remains eight. A fixed oblique
`1e-4` sample is required to reject that default with an observed path of nine.
The finite fixture explicitly selects a maximum path of twelve and records the
following accepted-sweep guards:

- maximum root path: `12`, observed maximum: `9`;
- maximum reference extrapolation distance: `4`;
- maximum absolute coefficient: `16`;
- maximum row L1 norm: `32`;
- zero accepted-sweep guard rejections.

The raised path limit belongs only to this frozen fixture. It does not change
the production default or establish a general amplification bound.

## Evidence that remains required

WP-3 still requires the joint scaling result for every supported sharp
boundary operator, including unsymmetric Nitsche, traction, Robin, outflow,
pressure flux, and wall slip, under active-side reversal and MPI
repartitioning.

WP-7 still requires mixed velocity-pressure stability, manufactured error and
divergence rates, viscous/transient/advection regimes, two and four-or-more MPI
ranks, cut-relative reference conditioning, expected `h` scaling, connected
and disconnected topology policies, continuous node crossing, and production
preconditioner and iteration-spread evidence.

Because those exits are absent, the finite positive eigenvalues in this child
matrix cannot close WP-3, WP-7, FSR-16, FSR-07, or Q1.

## Runner

Validate the immutable schema, dependencies, and claim boundary without
building:

```bash
python3 tests/cases/fluid/run_free_surface_wp3_wp7_nitsche_coercivity_qualification.py \
  --validate-only
```

The full runner reuses the repository's established qualification base. It
requires the eight standard binary arguments, performs clean source-root
preflight and builds, verifies exact test discovery and quantitative
properties, refuses to replace an existing output directory, and writes
checksum-bound final provenance. Closure claims and unknown claims are
rejected before output creation.
