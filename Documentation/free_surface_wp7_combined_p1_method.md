# WP-7 finite qualification basis for the combined P1 cut method

## Status and scope

This document describes the method exercised by
`free_surface_wp7_cut_stability_v2`. It is a matching derivation for the
implemented finite fixtures, not a general coercivity, inf-sup, approximation,
or condition-number theorem. The matrix remains release blocking while its
declared prospective evidence is absent.

The selected scope is a one-phase unfitted liquid domain with continuous P1
velocity and pressure, transient VMS/PSPG bulk stabilization, common
velocity/pressure aggregate constraints, and a pressure first-derivative jump
term on cut-adjacent facets. No velocity ghost penalty is present. Higher-order
spaces are outside this selected qualification even though parts of the
generic aggregation infrastructure support additional isoparametric nodal
layouts.

## Discrete active space

Let the retained liquid domain be

\[
\Omega_h^-=\{x:\phi_h(x)<0\}.
\]

The background P1 spaces are restricted to active support. A nodal degree of
freedom is an aggregate candidate when it touches a cut cell but no
fully-active cell. For each rooted candidate \(s\), production aggregation
installs the componentwise relation

\[
u_s=\sum_{j\in K_r}N_j^{K_r}(\xi_{K_r}(x_s))u_j,\qquad
p_s=\sum_{j\in K_r}N_j^{K_r}(\xi_{K_r}(x_s))p_j.
\]

The root \(K_r\) is chosen by deterministic breadth-first traversal through
the face-adjacent cut band. The configured production guards are:

| Quantity | Maximum |
|---|---:|
| root path length | 8 |
| reference extrapolation distance | 4 |
| absolute coefficient | 16 |
| row \(L^1\) norm | 32 |

Partition of unity is checked directly in the finite tests. Strong boundary
constraints take precedence over aggregate lines. Distributed declarations
are reconciled by global degree-of-freedom identity. Equal-distance roots are
ordered by globally unique physical cell identity rather than partition-owned
algebraic master numbering. Provider rank is used only after the physical root
matches, and equivalent providers must agree on the line weights. Missing
global cell identities, incomplete overlap, or inconsistent providers fail
closed.

A candidate without a full-active root is homogeneously removed by the
current production policy. That is a subresolution feature-deletion event,
not a conservative representation of liquid and not evidence of stability for
a resolved disconnected component. The matrix therefore records rootless
events separately and keeps resolved disconnected-feature exits release
blocking.

## Bulk and facet terms

The momentum and continuity equations use the production transient
equal-order VMS/PSPG form. The WP-7 fixtures assemble that form through
`IncompressibleNavierStokesVMSModule`; they do not substitute a separate model
matrix.

For the selected P1 pressure space, the additional cut-adjacent facet term is

\[
g_p(p_h,q_h)=
\sum_{F\in\mathcal F_h^\Gamma}
\int_F
s_F\,\gamma_p\,
\frac{0.01h_F^3}
{\mu+\rho h_F^2/\Delta t+\varepsilon}
[\nabla p_h]\mathbin{\cdot}[\nabla q_h]\,\mathrm dS .
\]

Here \(h_F\) is the averaged normal cell size, \(\gamma_p=1\) in the frozen
fixture, and \(s_F\) is the generated cut-metadata scale capped at 100. The
three frozen regimes vary \(\mu\), \(\Delta t\), and convection while retaining
the same nondimensional facet calibration:

| Regime | \(\rho\) | \(\mu\) | \(\Delta t\) | convection | imposed advective speed |
|---|---:|---:|---:|---|---:|
| viscous | 1 | 1 | 0.1 | off | 0 |
| transient | 1 | 0.01 | 0.001 | off | 0 |
| advection | 1 | 0.001 | 0.1 | on | 1 |

The pressure jump term and aggregate constraints have distinct roles.
Aggregation defines the reduced velocity and pressure trial/test spaces.
VMS/PSPG supplies equal-order bulk pressure control. The pressure facet term
adds cut-band pressure control. Their coexistence is what the matrix tests;
individual motivations are not treated as a proof for their combination.

Absolute pressure is anchored by the physical natural-traction path in the
fixture. The test verifies that no algebraic gauge is added. Disconnected
components and any componentwise nullspaces must be handled according to the
explicit topology policy rather than hidden by a global pin.

## Finite diagnostics

For each assembled reduced mixed matrix \(A\), the test removes every
registered constraint line. In the traction-only velocity fixture these
include the registered componentwise velocity gauge lines as well as
active-side and aggregate lines. The pressure field remains naturally
anchored and has no algebraic gauge. The test then records:

- zero free-pressure rows;
- rank and retained singular values;
- a stabilized pressure-control surrogate from the pressure mass, mixed
  coupling, PSPG, and pressure-jump blocks;
- exact infinity-norm condition after deterministic row/column equilibration;
- aggregate row amplification and geometric reach; and
- a deterministic Jacobi-preconditioned BiCGSTAB solve of \(Ax=b\), with
  \(b\) constructed from a known dense vector.

The iterative diagnostic requires relative residual at most \(2\times10^{-9}\)
and records iterations, diagonal fallbacks, breakdown, and solution error. It
is not the production preconditioner, so no production iteration-spread bound
is inferred from it.

The frozen finite cross-product contains seven designated cut fractions,
axis-aligned and oblique planes, three background spacings, three regimes, P1,
and 1/2/4 ranks: 378 rank-specific cases. The serial test executes 126 cases.
The distributed tests are defined to execute the same 126 physical cases at
two and four ranks and compare canonical mixed, pressure-jump, and PSPG
operators against serial and across block/METIS ownership. Those tests remain
unqualified until the frozen matrix completes from immutable source.

The superseded v1 registry (SHA-256
`fc028d52bf1b175b414d246e647bcb0502a2ece4e05fd9389da2350cc68342ec`)
capped each distributed group at 14,400 seconds. Its compiler-pinned Eigen
two-rank execution exhausted that ceiling after 4:00:01 with exit status 124,
328 of 504 equation timing headers, and 656 successful distributed halo
validations. Teardown did not complete, so it produced no per-rank JSON. The
terminal log has SHA-256
`7cfd9ecebce80de7bd0fae2e50ae378cee64692f33cfbc1f45cdcf87363e215e`.
No assertion, numerical failure, MPI abort, solver breakdown, or halo failure
was detected before termination; the timeout remains nonpassing resource
evidence.

Version 2 changes only the two- and four-rank wall ceilings to 43,200 seconds.
It does not remove cases, loosen numerical thresholds, change the selected
method, or alter any closure claim. Both v2 distributed groups must start
afresh from frozen source and complete inside the new envelope.

## Canonical transition telemetry source slice

`SmallCutAggregationConstraint` now retains a production-owned transition
ledger between successive successful publications of the same field,
interface marker, and active side when no failed refresh intervenes. A failed
refresh deliberately leaves the completed report empty, so the next success
starts a new comparison sequence rather than bridging the failure. The ledger
is derived from the already communicator-canonical active-feature and
aggregate-slave sets. Those canonical subfields are repeated identically on
every participating rank; the report also carries explicitly rank-local
lineage and therefore is not wholly communicator-canonical. It records feature
entry, exit, persistence, full/cut classification change, rooted/rootless
disposition change, aggregate-slave entry and exit, and the rootless
retained-volume delta.

Each completed report retains a successful-publication ordinal and an explicit
geometry-identity slot. When available, the identity is rank-invariant. An
authoritative free-surface snapshot uses its source/domain state and
distributed snapshot/revision keys. The generated-
publication fallback uses only the source, domain, isovalue, source revisions,
and quadrature-policy fields whose publication contract is communicator-wide;
request mesh epochs are excluded from that identity. Cut-context, snapshot-
local mesh, publication-request mesh, live mesh, FE-layout, and affine-
constraint stamps are segregated into the rank-local lineage. The final
collective checks 64-bit fingerprints of the geometry identity, class-tagged
features, canonical slaves, and ordinal. Its
`communicator_fingerprint_consensus_validated` state is collision-prone audit
evidence, not an exact all-rank comparison of every string and scalar, and an
unavailable source identity remains explicitly unavailable.

The complete post-resolver tail is one local guarded region: patch formation,
constraint-line insertion, diagnostics, copies, transition/report formation,
and pending-prolongation formation all finish before exactly one all-rank
readiness/fingerprint reduction. Constraint lines are inserted into a copied
`AffineConstraints`; a coordinated failure discards that copy and leaves the
caller-owned object untouched by this constraint. After consensus, only the
class's statically checked no-throw move assignment, optional/shared-pointer
moves, vector swap, and scalar writes publish constraint and lifecycle state.
The completed report and successful ordinal are captured and restored together
by the lifecycle checkpoint using an allocate-first, no-throw-commit sequence.

Feature entry/exit, counts, dispositions, and aggregate-slave set changes are
exact within those canonical ledgers. Each feature now has separate
domain-tagged 64-bit full-active and cut-cell GID digests in addition to the
whole-membership digest, so an equal-count full/cut swap is no longer hidden by
unchanged totals and membership. Those digests can still collide;
`canonical_topology_changed` is therefore auditable transition telemetry, not
a collision-free topology oracle.

The serial production-fixture regression
`FreeSurfaceCutStability.ContinuousNodeCrossingReportsCanonicalAggregationTopologyTransitions`
translates a P1 level-set plane on both sides of an interior node plane and
requires the velocity and pressure instances of the combined method to report
the same feature transition, source-publication sequence, and component-scaled
slave churn. The direct constraint regression
`SmallCutAggregationConstraint.SameCountFullCutSwapHasClassSensitiveTransitionAndProvenance`
holds feature membership and class counts fixed while swapping the physical
full/cut cell identities, and checks the class digests plus before/after source
and ordinal provenance. Existing rollback coverage additionally checks that
the checkpoint restores the ordinal with the report. This source slice was
added under the repository memory gate and has not been built or executed in
this worktree. It is deliberately not the frozen prospective test
`FreeSurfaceCutStability.ContinuousNodeCrossingHasNoUnreportedOperatorOrSolutionJump`:
the ledger compares neither assembled operators nor solved states, so the full
node-crossing exit and every qualification claim remain open.

## Closure boundary

Passing the finite cross-product can establish only that the named P1 fixtures
meet their recorded algebraic and partition contracts. WP-7 still requires:

1. expected manufactured velocity, pressure, and divergence convergence;
2. a frozen production-preconditioner spectrum and iteration-spread bound;
3. resolved connected, disconnected, and node-crossing behavior with explicit
   rootless-event accounting; and
4. static-cap, translating-drop, filament, and D18/D38 exits without an
   unreported mesh-relative numerical jump.

Until those items are executable and pass from immutable source, this method
must not be described as uniformly cut stable or WP-7 complete.
