# Conservative phase-indicator transport contract

## Selected representation

The conservative unknown is a bounded liquid indicator `q`, not the signed
distance field. `q=1` denotes liquid and `q=0` denotes the exterior phase. For
positive lumped nodal control volumes `m_i`, the declared discrete liquid
measure is

\[
M_q = \sum_i m_i q_i.
\]

The selected AD-4 direction is a coupled representation: continuous P1 `q`
provides conservative phase transport, while the signed-distance field
provides interface geometry. Geometry reconciliation must preserve the
accepted `q` measure and must remain inside the maintenance transaction.

The reusable correction kernel is implemented in
`LevelSetConservativePhaseTransport`. The first geometry-aware P1 operator is
implemented in `LevelSetConservativePhaseOperator`. It constructs lumped nodal
control volumes and the full algebraic gradient graph from the actual physical
mapping and quadrature. The production application projects `q` from the
authoritative retained cut geometry, advances it on accepted time-step
candidates, reconciles the signed-distance geometry to its nodal liquid
moments, and commits the two fields together. Graph assembly, transport, and
maintenance support the replicated global numbering used by the current MPI
level-set state. The release-scale multi-resolution transport matrix and
explicit moving-crown morphology extraction remain qualification work; fixed
region observers and opt-in resolved-satellite classification are available as
described below.
When conservative phase transport is disabled, the independent signed-distance
transport remains nonconservative and must continue to report that limitation.

## P1 finite-element flux construction

For the assembled gradient matrix

\[
C_{ij}=\int_\Omega N_i\nabla N_j\,dx,
\]

the strong conservative CG contribution is

\[
-\sum_j C_{ij}\cdot(q_j u_j).
\]

Every unordered edge `i<j` is stored once. The central mass-transfer rate into
`i` is

\[
T_{ij}=C_{ji}\cdot(q_i u_i)-C_{ij}\cdot(q_j u_j),
\qquad T_{ji}=-T_{ij}.
\]

The part not represented by pair transfers is the physical-boundary remainder

\[
B_i=-q_i u_i\cdot b_i,
\qquad b_i=\sum_j C_{ji}=\int_{\partial\Omega}N_i n\,ds.
\]

Writing advective indicator transport in conservative form adds the explicit
discrete-divergence source

\[
S_i=q_i\sum_j C_{ij}\cdot u_j.
\]

Consequently `sum_edges(T_ij) + B_i + S_i` equals the lumped strong-CG
advective operator node by node. For constant `q`, the boundary, edge, and
divergence terms cancel exactly even for a nodal velocity field with nonzero
discrete divergence.

The symmetric low-order graph-viscosity coefficient is

\[
d_{ij}=\max\left(\left|C_{ij}\cdot u_j\right|,
                  \left|C_{ji}\cdot u_i\right|\right).
\]

It makes both directed off-diagonal update coefficients nonnegative. The
forward-Euler stage is accepted only when

\[
\max_i \frac{\Delta t}{m_i}
\sum_{j\ne i}\left(d_{ij}-C_{ij}\cdot u_j\right)\le 1.
\]

The low-order edge transfer adds `d_ij(q_j-q_i)`. The raw antidiffusive
transfer removes exactly that addition, so the unlimited target is the
lumped central operator and the accepted correction remains pairwise
conservative.

The graph builder validates partition of unity, the zero gradient sum,
positive mapped quadrature and control volumes, the global row-sum identity,
and closure of the summed lumped volumes against mapped physical measure.
It records geometry, topology, ownership, numbering, and degree-of-freedom
layout revisions. On multiple ranks, only owned cells contribute. Nodal
coefficients are reduced on the field communicator, sparse edge fragments are
merged in canonical order, and each edge is assigned one logical owner from
its endpoint ownership. Endpoint ownership is compared across ranks before the
graph is accepted. The resulting sparse graph is replicated because the
current level-set state uses globally indexed replicated vectors; every rank
therefore obtains the same single-edge ledger without counting ghost cells.
Rank-local preflight and assembly failures are synchronized before later
collectives so all ranks return the same failing-rank diagnostic.

The two-rank operator test checks mapped mass, cancellation of the artificial
partition-boundary remainder, unique owner coverage, identical graph and
limited state on both ranks, bounded transport across the shared interface,
constant preservation, and a fault injected on only one owned cell. Replacing
the replicated sparse merge with owner-to-owner exchange is a scalability
optimization that must preserve these public graph and ledger invariants. A
separate rank-local invalid-option test guards the collective failure path.
The replicated-stage test also changes the previous indicator, lower and upper
bounds, velocity, time step, tolerance, and stage options on one rank at a
time, and requires every rank to reject before entering an asymmetric stage.

## Accepted-step maintenance transaction

The conservative liquid indicator is held at the previous accepted endpoint
during the nonlinear solve. Before initialization and again before each
covered candidate or accepted-step request loop, production compares an exact
canonical representation of the ordered maintenance requests. The comparison
includes cadence and action bits, all numerical configuration and mutable
accounting state, and the replicated shape, invariant, and FE-layout contract
of any conservative-phase graph. The graph's geometry, topology, ownership,
and numbering revisions remain rank-local cache stamps and are deliberately
excluded. It is a direct length-and-word comparison, not a hash. A count,
order, cadence, or velocity-source difference therefore fails collectively
before a per-request stage callback can run.

After a candidate step converges, production performs the following sequence
inside a rollback-capable geometry transaction:

1. advance and limit `q` with the conservative algebraic-edge operator;
2. rebuild the raw transported signed-distance geometry;
3. apply the configured wall-aware signed-distance repair when due;
4. reconcile geometry locally against the transported nodal liquid moments;
5. validate global measure, every nodal moment, cut-context provenance, and
   all transport invariants; and
6. compare the complete trial ledger, current and all-history content
   revisions, geometry-transaction presence, the final live FE layout state,
   canonical authoritative snapshot revision identities (including their
   communicator-replicated geometry/topology/ownership/numbering mesh
   revisions), and the post-stage request state before selecting the same
   commit or reject decision on every rank. Raw operator mesh event counters,
   including geometry, topology, ownership, numbering, label,
   reference-configuration, field-layout, and active-configuration epochs,
   are deliberately excluded because they can advance a different number of
   times on equally valid partitions.

The distributed graph builder applies the same distinction. It requires an
identical global field size, FE layout revision, dimension, and graph options,
but retains each rank's local mesh event counters for its own staleness check.
The production currentness helper reduces whether any rank has a stale local
stamp before deciding whether to rebuild. If one partition is stale, every
rank enters the collective graph builder and refreshes its own local stamps.
The selected two-rank disjoint-wall fixture proves that a real conservative
graph builds and passes the production request preflight while those local
cache stamps differ. It then advances the geometry stamp on one rank only and
requires the production helper to rebuild successfully on both ranks.

A failed stage, nonconverged repair, displacement/topology guard, stale graph,
failed geometry invariant, or consensus mismatch rejects the covered
prepublication candidate. Geometry, lifecycle, mesh-field, refresh-cache,
request, current-state, and every history checkpoint are restored
component-by-component, and diagnostics do not report restoration while any
component is unresolved. Specifically, a geometry rollback interrupted by a
one-shot cut-context callback failure remains active and succeeds on retry.
Publication enters an explicit `Publishing` state before the first
irreversible geometry or ledger action. Committed diagnostics and transaction
reset occur only after both actions succeed. If publication or its logging
fails after that boundary, candidate discard is fail-stop: it does not restore
the algebraic state against published geometry, reject the ledger, or claim a
successful rollback. The enclosing time-loop attempt guard makes the same
distinction: a successful discard restores provisional rate/workspace state,
while a discard callback that refuses recovery disarms that guard before
rethrow so the candidate rates and one-shot workspace state are retained.
History-scatter and the other cross-resource fault-injection cases remain part
of the unqualified full-transaction campaign. No global level-set shift is
part of this sequence. The retained maintenance ledger distinguishes raw
post-transport, post-limit, post-reinitialization, post-correction, and
retained assembly measures, together with nodal mismatch and
interface/contact displacement measures.

Application tests exercise the production sequence as distinct transport-only,
local-reconciliation-only, and reinitialization-plus-reconciliation modes. A
separate negative case verifies rollback after a nonconverged repair, and a
one-shot cut-context callback failure verifies that rollback remains active and
succeeds on retry. Exact two- and four-rank consensus fixtures cover identical
commit, last-rank invariant rejection, missing active transactions, differing
row counts, functional content, algebraic/current/history content, snapshot,
mesh, cut, extension-map, geometry-presence, and final geometry-revision
content drift. A selected two-rank partition fixture also compares the real
production live-geometry serialization, builds a distributed conservative
graph with unequal rank-local mesh cache stamps, passes the production request
preflight, injects snapshot-revision drift, and finally invalidates one rank's
local graph stamp to verify an all-rank rebuild through the production helper.
Paired time-loop regressions inject a commit-ready failure and verify that
successful discard restores the provisional rate state while a fail-stop
discard refusal makes unwind retain the candidate rate state. These are
low-level collective prerequisite gates. They do not prove atomicity across
the later `TimeLoop` acceptance operations or across multiple due artifacts;
that cross-resource fault-injection campaign remains explicitly unqualified.
They also do not replace the required independent release-scale space/time,
maintenance, extension, or four-or-more-rank partition sweeps.

The transport velocity extension is swept over 1, 2, 4, and 16 graph layers
on a regular curved interface while checking row, partition-of-unity, and
wet-to-dry amplification bounds. A deliberately skewed triangle makes the
tangential regression extrapolative and verifies that its rejected row is
replaced by the positive bounded fallback.

The CI transport benchmark now keeps the spatial and temporal studies
independent. Its fixed-step spatial sequence uses `N=16,32,64`; its fixed
`N=48` temporal sequence uses 32, 64, and 128 steps and compares each result
with a separately computed 256-step reference. The test emits full-precision
string properties for interface error, centroid displacement, raw phase drift,
accounted balance residual, Courant number, and observed order. Coupled
`h`/`dt` results remain under separate property names so they cannot be
mistaken for an independent refinement study. Zalesak rotation, reversible
deformation, wall-film advection, and separated-drop component tests likewise
emit their raw transport metrics. These small two-dimensional runs are
regressions, not substitutes for the frozen release matrix.

## Fully discrete edge update

Each interior algebraic edge is stored once in canonical orientation `i<j`.
A positive transfer `F_ij` is received by node `i` and removed from node `j`,
so `F_ji=-F_ij` by construction. The low-order stage is

\[
m_i q_i^L = m_i q_i^n + B_i + S_i + \sum_j L_{ij},
\]

where `B_i` is the integrated physical-boundary transfer, `S_i` is the
explicit discrete-divergence source, and `L_ij=-L_ji` is a bounded low-order
interior transfer. `S_i` is zero for a discretely divergence-compatible
advecting field.

For raw antidiffusive transfers `F_ij=-F_ji`, define

\[
P_i^+ = \sum_j \max(0,F_{ij}), \qquad
P_i^- = \sum_j \min(0,F_{ij}),
\]

and the admissible mass increments

\[
Q_i^+ = m_i(q_i^{max}-q_i^L), \qquad
Q_i^- = m_i(q_i^{min}-q_i^L).
\]

The nodal factors are

\[
R_i^+ = \min(1,Q_i^+/P_i^+), \qquad
R_i^- = \min(1,Q_i^-/P_i^-),
\]

with a factor of one when the corresponding denominator is zero. The
symmetric edge factor is

\[
\alpha_{ij} =
\begin{cases}
\min(R_i^+,R_j^-), & F_{ij}>0,\\
\min(R_i^-,R_j^+), & F_{ij}<0,\\
1, & F_{ij}=0.
\end{cases}
\]

The accepted update is

\[
m_i q_i^{n+1} = m_i q_i^L + \sum_j \alpha_{ij}F_{ij}.
\]

Because the same `alpha_ij` multiplies both orientations, every interior
correction cancels pairwise. The limiter changes neither the external physical
flux nor the declared divergence source.

## Ledger and fail-closed invariants

The stage result retains, for every node, the previous, low-order, raw-target,
and limited indicator; boundary, divergence, low-order, raw, and limited mass
contributions; both nodal factors; and a local balance residual for each
stage. Every edge record retains its raw and limited transfers, shared factor,
and pair-cancellation residuals.

The stage is rejected when any of these conditions fails:

- control-volume masses are not finite and positive;
- bounds are not ordered inside `[0,1]` or exclude the previous indicator;
- an edge is duplicated, reversed, self-connected, or out of range;
- the low-order predictor violates its local invariant domain;
- the limited state violates its local bounds;
- interior cancellation or any nodal/global balance does not close to the
  scaled invariant tolerance; or
- a constant input is not preserved by the low-order and accepted stages when
  constant preservation is required. Boundary data that intentionally change
  a constant state require an explicit stage-level waiver, which is retained
  in the result.

No global phase shift appears in this algorithm.

## Connected-component ledgers and artifacts

The correction constructs deterministic connected components from active
phase-support nodes and algebraic edges. A component identifier is its smallest
canonical node index, so it is invariant under rank ownership. Every resolved
component records previous, low-order, raw-target, and limited liquid measure;
physical-boundary and discrete-divergence transfer; low-order, raw, and limited
interior transfer; and the three balance residuals. Nonzero activity below the
declared component threshold is retained in a separate subthreshold bucket and
is included in every component-closure invariant. It is never discarded to
make a balance close.

Machine-readable accepted-step artifacts are opt-in. Set
`Conservative_phase_write_flux_artifacts=true` and select a positive cadence
with `Conservative_phase_flux_artifact_cadence_steps`. On due accepted steps,
the output rank writes
`conservative_phase_flux/conservative_phase_flux_<field>_step_<step>.json`
under the configured results directory. All ranks complete preflight before
publication. The writer closes a temporary sibling and atomically publishes a
no-replacement final link; an existing final or temporary path is a hard
failure rather than an overwrite. This is per-file publication atomicity only:
several due artifacts are still published sequentially, so the low-level
qualification does not claim a single atomic multi-artifact transaction.

Artifact schema version 2 contains accepted step/time, the output rank's local
graph mesh cache stamps, and the communicator-replicated graph FE-layout
revision; all stage and limiter invariant flags and residuals; every nodal
control-volume state, Courant number, source, transfer, factor, and balance;
every canonical edge and pair-cancellation residual; every resolved and
subthreshold component ledger; and the complete reinitialization,
reconciliation, mismatch, phase-measure, geometry-measure, and displacement
history for that accepted transaction.
Serial tests cover schema content, stale-file refusal, malformed-ledger
rejection, and cadence ordering. The two-rank application test requires a
rank-local preflight fault to fail collectively before publication and a valid
stage to produce exactly one output-rank artifact. Artifact policy and region
membership fingerprints must also agree across ranks.

Film, sheet, and rim accounting uses explicit fixed Eulerian control-volume
regions rather than an undocumented automatic shape guess. Configure
`Conservative_phase_fixed_flux_regions` as semicolon-separated entries of the
form
`name|kind|xmin|xmax|ymin|ymax|zmin|zmax`; `*` is an unbounded coordinate and
the supported kinds are `wall_film`, `sheet`, `rim`, `resolved_satellite`, and
`observer`. Membership is evaluated from canonical phase-node coordinates and
held fixed over a transport stage. For each region, the ledger lists every
member control volume and algebraic edge crossing its boundary, reconstructs
the low-order/raw/limited crossing flux, and verifies all three local balances.
Overlapping observers are allowed because each is an independent diagnostic;
names and binary membership are fail-closed.

`Conservative_phase_classify_nonprimary_components_as_satellites=true` is a
separate explicit policy. It labels every resolved component except the
largest limited-measure component as an individual satellite; an equal-measure
tie selects the smallest deterministic component identifier as primary. This
policy is disabled by default because two deliberately initialized peer drops
must not be silently reinterpreted as a primary drop and a satellite.

The fixed-region and opt-in satellite ledgers close the numerical
control-volume observability gap. They do not automatically extract moving
crown morphology or establish physical resolution. Qualification still needs
predeclared geometry classifiers or observer boxes and the declared
translation, rotation, wall, Zalesak, Enright, jet, and filament refinement
matrices.

## Frozen transport release matrix

The transport-only release matrix is versioned in
`tests/cases/fluid/level_set_phase_transport_release_matrix.json`. It contains
the Cartesian product of:

- translating drops at `D/dx=16,32,64` and graph CFL
  `0.5,0.25,0.125`; and
- three-dimensional reversible Enright deformation on approximately
  `32^3,64^3,128^3` cells at the same three CFL values.

The fixed-CFL spatial and fixed-resolution temporal subsets are declared in
the registry. The temporal order is computed from the weighted differences
between all three final control-volume states, while the spatial study uses
the exact return/translation error. The registry fixes point gates, observed
order gates, and per-resolution wall-time, memory, and output envelopes. Its
`FROZEN_BEFORE_COMPLETE_MATRIX` status records that small feasibility points
were run before freezing, but the complete 18-point matrix has not yet been
claimed or used to check WP-6.

`run_level_set_phase_transport_release.py list` lists the 18 immutable points.
Its `run` action accepts exactly one registered case, resolution, and CFL. It
requires a clean tracked source tree, removes inherited release variables,
uses one thread and one rank, samples resident memory, enforces resource
limits, refuses an existing output directory, and records source, binary,
build, library, machine, registry, gate, and resource provenance. Each point
contains full-step phase history and final per-control-volume, algebraic-edge,
and connected-component flux ledgers. Every artifact is checksummed. A point
can pass its local gates but retains the release disposition
`INCONCLUSIVE_RESOLUTION`.

The `summarize` action verifies all point checksums, requires a single source
commit and executable hash, rejects missing/duplicate/unregistered points,
and evaluates both independent convergence studies. It is the only harness
action that can issue a release `PASS`, and it can do so only when all 18
points and both convergence gates pass. A missing point is
`INFRASTRUCTURE_FAILURE`; a completed but nonconvergent matrix is
`FAIL_METHOD`.

The first `32^3` graph feasibility run exposed ordinary-precision drift from
accumulating millions of mapped quadrature weights. Local physical measure is
now accumulated in extended precision before any communicator reduction. A
default `32^3` graph regression closes mapped and lumped unit-cube measure to
`2e-14`; it prevents a resolution-dependent false rejection before the
scheduled transport work begins.

## Primary method references

- D. Kuzmin and M. Quezada de Luna, “Algebraic entropy fixes and convex
  limiting for continuous finite element discretizations of scalar hyperbolic
  conservation laws,” 2020, <https://arxiv.org/abs/2003.12007>.
- D. Kuzmin, M. Quezada de Luna, D. Ketcheson, and J. Grüll,
  “Bound-preserving flux limiting for high-order explicit Runge–Kutta time
  discretizations of hyperbolic conservation laws,” 2020,
  <https://arxiv.org/abs/2009.01133>.
