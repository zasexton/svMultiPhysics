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
the checkpoint restores the ordinal with the report.

The response-continuity source slice adds the formerly prospective regression
`FreeSurfaceCutStability.ContinuousNodeCrossingHasNoUnreportedOperatorOrSolutionJump`.
It evaluates four independent pairs at symmetric distances `0.08`, `0.04`,
`0.02`, and `0.01` from the interior node plane `x=4/3` on the three-cell mesh.
Every member starts from the same deterministic affine P1 velocity and pressure
state in a nonconvective transient Stokes regime. The fixture assembles the
unscaled constrained mixed operator and residual, computes the exact dense
linear response on the retained free space, and compares only common physical
degrees of freedom across the left and right topology. A separate persistent
left-to-right pair must report the crossed aggregate topology change.

The limits were fixed before execution: each of the eight dense solves has
relative residual at most `1e-10`; the finest common-space operator, residual,
and solved-state relative differences are each at most `0.1`; and each
finest-to-coarsest ratio is at most `0.5`. Before the final result was
observed, the gate was strengthened to require every adjacent offset-halving
ratio to be at most `0.75`. These finite limits reject a hidden nonvanishing
jump but are not a smooth-parameter theorem, a production-preconditioner
result, or a simulation exit.

The first immutable focused execution, job `41265718`, failed numerically
despite all eight dense solves reaching relative residual at most
`2.070216710331556e-15` and the persistent pair reporting exactly one topology
change. At the finest offset, the operator, residual, and solved-state
differences were `0.66988818148593277`, `0.77857463897704082`, and
`0.18804222079955726`; their finest-to-coarsest ratios were
`1.0992459269515737`, `0.99285930834981029`, and `0.95811454707512655`.
None of the adjacent ratios met the predeclared `0.75` contraction limit.

Job `41269422` reran the unchanged response with stabilization-block and field-
block diagnostics. From offsets `0.08` through `0.01`, the relative pressure-
ghost operator difference was `0.97307941524760511`,
`0.98247236791864268`, `0.98373757165775411`, and
`0.98420843042668993`; the pressure PSPG operator difference was
`0.50248075907081491`, `0.53929178053539573`,
`0.55658494296529437`, and `0.56495185065523701`. At the finest offset,
the velocity and pressure residual differences were `0.79858068161174101`
and `0.57342030954104461`, while the velocity and pressure solved-state
differences were `0.023646448833889888` and `0.47415493834706302`.
This localizes a discontinuous facet-stabilization block and a pressure-led
state response, while also showing that PSPG and the constrained velocity
residual cannot yet be assumed continuous.

The strong whole-cut-band aggregation ablation in job `41268899` did not
repair the response. Its finest operator, residual, and solved-state
differences worsened to `0.780146`, `0.862462`, and `0.227646`, with
finest-to-coarsest ratios `0.960079`, `1.001266`, and `0.981122`. That mode is
retained as failed ablation evidence and is not a selected method.

The exact pressure-ghost-removal counterfactual subtracts only the separately
assembled pressure-ghost operator and residual from each already assembled
mixed response. It leaves aggregation, VMS/PSPG, the affine state, topology,
and common-DOF selection unchanged and subjects the adjusted dense solves to
the same predeclared `1e-10`, `0.1`, `0.5`, and `0.75` limits. Job `41270464`
failed decisively despite an exact-solve relative residual no larger than
`9.106019225462806e-16`. Its finest operator, residual, and solved-state
differences were `0.66934512178152927`, `0.77857463897704082`, and
`0.17825037579845354`; the corresponding coarse-to-finest ratios were
`1.0990466415161013`, `0.99285930834981029`, and
`0.93430647511155041`. Maximum adjacent ratios were
`1.058631642852865`, `0.99990715145582587`, and
`0.9927692935966993`. Pressure-ghost removal therefore neither removes nor
makes the jump contract under offset refinement and is rejected as a method
change.

The next bounded counterfactual disables only small-cut aggregation while
retaining pressure-ghost stabilization, VMS/PSPG, the deterministic affine
state, the independent symmetric problems, common-DOF comparison, and every
predeclared response limit. Both field blocks must report zero aggregate
slave constraints, both retained stabilization blocks must remain nonzero,
and every dense solve must meet the unchanged `1e-10` residual limit. This
isolates whether the aggregate trial/test-space switch is the persistent jump
source; it does not select an aggregation-free production method or change any
qualification or closure state before execution and the complete frozen
matrix. Initial job `41272546` stopped before the counterfactual response was
formed because the no-aggregation fixture still requested an aggregation-only
transition record; it produced no method metric and is not physical evidence.
The retry with that irrelevant query disabled, job `41273182`, formed every
response and reached a maximum dense-solve relative residual of
`1.1737989171366622e-15`. With aggregation disabled, the finest operator,
residual, and solved-state differences were `0.066963038496607258`,
`0.025175066644206524`, and `0.069176964899505158`. Their coarse-to-finest
ratios were `0.39029413586112305`, `0.13335548124667368`, and
`0.5968579939926022`; the maximum adjacent ratios were
`0.89682848891260314`, `0.51929368485842009`, and
`0.94270202811795833`. Removing aggregation therefore makes the operator and
residual responses much smaller, but the solved state still approaches a
nonzero fixed-mesh jump too slowly to satisfy the predeclared response gate.
This is failed ablation evidence, not a selected method.

The three fixed-mesh ablations also expose a flaw in the acceptance sequence:
offset refinement at constant `h` repeatedly crosses the same discontinuous
aggregate-space switch. It cannot establish whether that switch vanishes
under physical mesh refinement. The fixed-mesh test is retained as a disabled
diagnostic, while the acceptance test crosses the same physical node on a
sequence of refined meshes, uses an offset proportional to `h`, and requires
the operator, residual, and solved-state jumps to converge while every actual
aggregate-topology change remains reported.

The rooted physical-refinement test uses `4`, `6`, and `8` cells per axis,
crosses `x=1` at symmetric distance `0.04 h`, and fixes its gates before
execution: dense-solve relative residual at most `1e-10`, adjacent growth at
most `1.10`, global order at least `0.20`, and finest operator, residual, and
solved-state differences at most `0.60`, `0.65`, and `0.20`. Job `41275455`
first exposed that the operator and solved state passed while the residual
coordinate did not. The isolated immutable diagnostic job `41276518`
reproduced that result from an unchanged source hash. The raw reduced operator
differences were `0.66837491311399466`, `0.59695269865934153`, and
`0.55147552700110425`; solved-state differences were
`0.090282139412239948`, `0.037126430228979719`, and
`0.022081613219203326`. Their global orders were `0.27736071976332538` and
`2.0315950313006996`, and their maximum adjacent growth factors were
`0.92381779702079458` and `0.59476801521194089`. Every level reported the
actual topology change, no rule was pruned, no backend fallback occurred, the
physical volume jumps were `0.16000000000000503`,
`0.10666666666667846`, and `0.07999999999999341`, and the maximum exact-solve
relative residual was `1.7670895540162012e-14`.

The failing raw reduced residual differences were `0.54743877416024689`,
`0.6612192115855795`, and `0.67583560435098922`, giving global order
`-0.30397473455497698`, maximum adjacent growth `1.2078413930395246`, and a
finest value above `0.65`. Their absolute difference norms were instead
`0.20290148255193544`, `0.20399229067654656`, and
`0.16846335821940436`, while the corresponding reference norms decreased from
`0.37063776284968469` to `0.24926677010627693`. This distinction identifies a
coordinate-comparison defect rather than evidence that the assembled physical
residual work itself jumps.

If the closed aggregate constraints are written as `u_full = C u_free`, the
assembled reduced residual and operator are `r_free = C^T r_full` and
`A_free = C^T A_full C`. When the aggregate slave set changes, the same master
coordinate vector selects a different physical test function because its
column of `C` absorbs different slaves. Entrywise or Euclidean comparison of
raw reduced residual coordinates therefore compares different bases. The
corrected acceptance metric evaluates both residuals and operators on the same
physical affine P1 probe space: `{1,x,y,z}` for each of three velocity
components and pressure, for 16 probes total. For probe coefficient vectors
`v` and `w`, it compares `v^T r_free` and `v^T A_free w`. Every master-bearing
aggregate line must independently reproduce all four scalar affine modes to a
row-scaled roundoff bound. Raw reduced-coordinate differences remain recorded
as diagnostics, and none of the predeclared response thresholds is changed.

The isolated immutable execution in job `41279463` passed this corrected
response test from source SHA-256
`5e93afc9f6678d1328c94d46ec5595ebd6dc649856cc612297f45843756c40d2`.
For `4`, `6`, and `8` cells per axis, the physical affine-probe operator
differences were `0.046786767783159551`, `0.031454651029532064`, and
`0.023686113325394276`; the residual-action differences were
`0.042420097718013743`, `0.027705212494174375`, and
`0.020719120017658289`; and the common-physical-DOF solved-state differences
were `0.090282139412239948`, `0.037126430228979719`, and
`0.022081613219203326`. Their global orders were respectively
`0.98205908006808296`, `1.0337852140421999`, and
`2.0315950313006996`, with maximum adjacent growth factors
`0.75302419674423393`, `0.74784194569900186`, and
`0.59476801521194089`. The maximum affine-constraint reproduction error was
`1.5913919487742234e-15`, below its corresponding row-scaled roundoff bound
`8.3370347662518439e-13`; the maximum dense-solve relative residual remained
`1.7670895540162012e-14`. All three physical crossings reported their
topology change. The raw reduced-coordinate diagnostics reproduced the prior
values and remain nonacceptance telemetry. The result XML SHA-256 is
`201da315b0bae857051d87a5420f29851c7155730a89ce2f2c3a812dfe0d835f`.
This closes the finite node-crossing response slice only; the broader WP-7
exits below remain open.

The versioned promotion registry
`free_surface_wp7_cut_stability_qualification_revision_v4.json` binds the
topology-policy and physical node-crossing rows to implementation commit
`79f05c22bf57712c16a9132ac9115a0b5efe0dce` and exact source bytes. Its
runtime contains 16 executable tests and 67 quantitative gates while retaining
five prospective manufactured-error and simulation rows. It accepts only the
combined topology-and-node-crossing prerequisite claim and rejects FSR-07,
WP-7, and Q1 closure requests before execution. This registry must still run
from immutable source before it supplies promoted matrix evidence.

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
