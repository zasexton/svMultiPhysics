# WP-8 geometry-coupling and energy architecture record

Status: selected implementation direction; qualification incomplete.

Scope: one-phase unfitted free-surface geometry coupling, nonlinear
convergence, and the work/energy account required by FSR-09.

## Decision

AD-5 selects the partitioned generated-state outer iteration, not a complete
fixed-topology shape tangent.

The generated state is denoted by

\[
G(u) =
\{\text{cut snapshot},\text{curvature projection},
  \text{state-dependent constraints},\text{extension map}\}.
\]

For outer iteration \(k\), the implementation regenerates \(G_k=G(u_k)\),
projects the algebraic state through the regenerated constraints, regenerates
the remaining state from that projected value, and solves

\[
R(u_{k+1};G_k)=0
\]

with \(G_k\) frozen inside the inner Newton solve. The next outer pass restores
the immutable entry history and rate state, regenerates \(G\) from the new
algebraic iterate, and repeats. Acceptance requires a fresh outer pass for
which the inner problem already satisfies every configured absolute residual
criterion and therefore takes zero Newton updates.

This zero-update certificate establishes a same-state algebraic fixed point
for the regenerated residual in the tested finite-dimensional fixtures. It
does not by itself establish contraction, uniqueness, or an energy-stability
theorem.

The production default is the generated-state outer fixed point with a
bounded outer-iteration count. Refreshed within-solve and per-step-only modes
remain comparison controls; they are not the selected WP-8 qualification
path.

## Production transient-scheme prerequisite

The new solver now accepts an optional general parameter:

```xml
<Transient_time_integration_scheme>BackwardEuler</Transient_time_integration_scheme>
```

The exact supported values are `GeneralizedAlpha` and `BackwardEuler`;
aliases, case changes, and all other values are rejected before the time
history is repacked or a transient solve begins. Omitting the parameter
preserves the `GeneralizedAlpha` default. The spectral-radius parameter is
optional with its existing default of `0.5` and is validated only for
generalized-alpha.

Backward Euler has no spectral-radius parameter. Production reports it as
inapplicable, disables generalized-alpha PDE-rate initialization, uses
`alpha_f=1` for generated free-surface/contact state, and leaves
bound-preserving maintenance independent of a time-integration rate factor.
The external-state fixed point therefore observes the accepted endpoint time
and backward-difference stencil directly. The one-step regression holds the
candidate endpoint uncommitted through the preaccept and commit-ready
callbacks, then verifies that history advances exactly once.

This is a production method-selection and endpoint-transaction prerequisite.
It does not establish the backward-Euler constant-surface-tension energy
balance, a refinement threshold, or any WP-8 simulation exit.

## Why the complete-tangent alternative is not selected

`RefreshedFrozenQuadrature` is a local geometry-tangent policy name, not a
statement that the complete AD-5 tangent exists. Current support includes
selected point-location, measure, normal, cut-volume, and surface-functional
first variations where they are explicitly installed. It does not cover the
complete derivative of regenerated high-order rules, curvature projection,
wet-wall/contact construction, extension-map changes, or topology
transitions.

The surface-stress registration path rejects the advertised differentiated
quadrature mode where projector, normal, point-location, measure, or curvature
coverage is incomplete. The selected outer iteration is therefore the only
honest system-level AD-5 direction in the present implementation.

The fixed-topology first-variation fixture in
`FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference`
is retained as a capillary-functional prerequisite. It is not evidence for a
complete monolithic Jacobian.

## Outer-iteration contract

The selected path currently enforces these low-level properties:

1. The constraint-construction callback runs before the projected-state
   callback, so all other generated fields observe the constrained iterate.
2. Every inner Newton problem holds its generated state fixed.
3. Relative residual tolerances are disabled for the inner problems because a
   refresh changes the residual reference.
4. A freshly regenerated residual, not convergence of the preceding frozen
   problem, determines outer convergence.
5. A failed outer pass restores solution, history, rate, auxiliary, bordered,
   and regenerated callback state.
6. A constraint-topology change invalidates the relevant algebraic caches and
   can reallocate the Jacobian sparsity.

The current report contains the outer count, accumulated inner iterations,
last outer algebraic state-change norm, and refreshed residual. It does not
record a norm for each component of \(G\), a contraction ratio, or a
predeclared contraction gate. Consequently the required \(h\), \(dt\), cut
shift, and MPI contraction study remains open.

## Required complete energy account

For an accepted step, the target stored energy is

\[
\mathcal E =
K + E_g + \gamma A_{lg,h}
- \sum_w \gamma\cos(\theta_{e,w})A_{sl,h,w}
+ E_{\mathrm{gas/compressibility}},
\]

where the final term is present only for a model that contains it. The
physical dissipation is

\[
\mathcal D =
D_{\mathrm{viscous}} + D_{\mathrm{Navier}}
+ D_{\mathrm{line}}.
\]

With external work \(W_{\mathrm{ext}}\) and explicitly classified numerical
work \(W_{\mathrm{num}}\), the accepted-step residual must use one declared
time stage and sign convention:

\[
\mathcal R_E =
\mathcal E^{n+1}-\mathcal E^n
+ \Delta t\,\mathcal D
- W_{\mathrm{ext}}-W_{\mathrm{num}}.
\]

The numerical-work ledger must separately expose VMS/PSPG, cut stabilization
or aggregation, extension, pruning, limiting, redistancing, local
reconciliation, and global correction. A closed dissipative case with no
external work must not hide positive unexplained growth behind a later
maintenance operation. Every transport and maintenance substage therefore
needs its own before/after state and work row.

Rejected and rolled-back attempts do not change the accepted state. Their
trial ledgers must be kept separate and their contribution to the accepted
balance must be exactly zero.

## Current energy evidence and its boundary

The authoritative geometry snapshot currently supports the discrete
functional

\[
\gamma A_{lg,h}
- \sum_w \gamma\cos(\theta_{e,w})A_{sl,h,w}
+ \lambda V_h
\]

and its fixed-topology first variation. Accepted records bind that functional
to the state and snapshot revisions. Accepted dynamic-contact records also
contain sharp contact measure, line-friction dissipation, sharp wetted-wall
measure, and wall-slip dissipation. Global volume correction reports the
surface, Young-wall, and volume-potential change caused by the correction.

The level-set maintenance ledger now records the authoritative
surface/wall/volume potential before and after each implemented transport,
limiting, reinitialization, geometry-reconciliation, and global-correction
substage. Rows form one continuous algebraic and functional-state chain.
When an active-volume energy owner is declared, each row also evaluates
kinetic energy, gravitational energy, and endpoint gravitational-potential
power from that row's exact FE-ordered candidate on the same retained
snapshot. Across any declared interfaces, exactly one declaration may own the
active-volume channels; the row publishes a separate modeled-energy change
\(\Delta(K+E_g+E_{lg}+E_{sl})\), excluding the diagnostic volume-constraint
potential, by adding that one bulk contribution to every interface's
surface/wall contribution. The legacy potential-change field remains
separately labeled.
The same candidate-specific row enrichment carries endpoint surface-plus-wall
potential power, volume-constraint power, constant-viscosity bulk
dissipation, and literal exterior-pressure power whenever their production
declarations exist. This prevents a maintenance operation from silently
changing the quadrature domain used by those endpoint channels.
Snapshot, mesh-topology, communicator-wide cut-topology, and extension-map
revisions remain distinct. Accepted rows are published only after their
physical geometry transaction commits. Rejected rows and explicit zero-row
attempts remain diagnostic records with exactly zero accepted contribution.
Before staging a nonempty row, the ledger rejects missing snapshot or
cut-topology revisions, negative measures or surface energy, and any total
potential that is not the sum of its surface, Young-wall, and
volume-constraint contributions. Mesh-topology epoch zero remains valid as
the initial epoch and is included in the nonzero set fingerprint.
The complete FE-ordered algebraic fingerprint and authoritative functional
values are required to agree across the FE communicator.
The modeled-energy maintenance extension has compiled in the application
targets. Frozen serial functional/maintenance fixtures and the frozen
two-rank maintenance fixture exercise its row enrichment and aggregation, but
they do not supply a complete balance or physical qualification result.

These are necessary maintenance channels, not a complete balance. The present
account does not yet combine the stored-energy endpoints and maintenance
changes with bulk viscous, complete Navier-slip and line-friction
dissipation, exterior work, VMS/PSPG, ghost-penalty or aggregation, extension,
and pruning terms into one accepted-step residual. The system declaration
boundary now rejects a second active-volume energy or dissipation owner before
mutation while still permitting additional surface-only interface
declarations. This is the current one-phase ownership rule, not a two-fluid
energy partition. The extension-map revision is provenance, not extension
numerical work. No production backward-Euler
constant-surface-tension energy identity or refinement gate has been
demonstrated. Generalized-alpha stage consistency is therefore also
unqualified.

The current accepted functional history evaluates its stored-energy and
endpoint-rate channels after accepted-step maintenance, while the dynamic
contact stage and maintenance transaction begin from the physical
pre-maintenance endpoint. The pre-maintenance algebraic revision is retained,
but a complete enriched functional record at that revision is not yet
published. A production balance must not pair the post-maintenance stored
state with pre-maintenance rate terms without also inserting the exact
maintenance work, because that would hide or double-count the maintenance
energy change.

### Fail-closed balance-record prerequisite

`Application/Core/FreeSurfaceEnergyLedger.{h,cpp}` now defines the prospective
backward-Euler accepted-step record independently of any channel producer. A
record must explicitly provide finite values, with nonnegative kinetic and
liquid--gas surface energy, for kinetic, gravitational, liquid--gas surface,
solid--liquid wall, and conditional
gas/compressibility stored energy; nonnegative bulk-viscous, Navier-slip, and
line-friction endpoint dissipation rates; external pressure and body-force
work, imposed-traction work, and open-boundary energy flux; and distinct
time-discretization, kinetic-domain-transport,
gravitational-transport-coupling, convection, pressure/continuity,
surface-transport-coupling, weak-boundary, VMS/PSPG, cut-stabilization,
ghost-penalty, aggregation, extension, pruning, limiting, redistancing,
local-reconciliation, and global-correction numerical work. An inapplicable
gas channel must be declared and set to exact zero, so an omitted contribution
cannot silently look like a physical zero. Numerical-work fields are signed
step integrals: positive values add modeled energy and negative values remove
it. Every channel also requires one nonempty owner name and an explicit
`Produced` or `NotApplicable` decision. `NotApplicable` requires an exact-zero
value; `Produced` remains distinct even when its evaluated value happens to
be zero. Gas/compressibility applicability and its channel-source decision
must agree.

Each record now contains three separately revisioned stored-energy states:
the preceding post-maintenance accepted endpoint, the physical
pre-maintenance endpoint on which physical rates and work are evaluated, and
the resulting post-maintenance accepted endpoint. It reports the exact
initial-to-middle physical and middle-to-final maintenance stored-energy
changes separately. A rejected attempt preserves both diagnostic changes but
publishes exact zero for both accepted contributions. Consecutive accepted
records must form one continuous step, exact endpoint time, algebraic-state,
snapshot, mesh-topology, cut-topology, extension-map, and componentwise exact
stored-energy chain from the final state of one record to the initial state of
the next. Fixed-topology acceptance also requires the mesh and cut topology to
agree at all three states. Channel ownership and applicability cannot change
within one ledger history. Published attempt sequencing is also fail closed:
acceptance advances
to attempt one of the next step, while rejection permits only the next attempt
on the identical accepted starting endpoint. A step-controller retry may
change its duration and prospective endpoint, but cannot silently replace its
starting revision or stored-energy state, change its channel ownership, or
reuse an attempt number.

The record evaluates

\[
(\mathcal E^{n+1}-\mathcal E^n)
+ \Delta t\,\mathcal D^{n+1}
- W_{\mathrm{ext}}-W_{\mathrm{num}}
\]

on one verified endpoint interval. Publication is transactional: a rejected
trial retains its diagnostic residual but contributes exactly zero to the
accepted stored-energy change, physical dissipation, external work, numerical
work, and residual. Transaction identifiers, accepted-step indices, and
attempt indices must all be nonzero. Publication grows only the selected
accepted or rejected history before mutating the trial record; it does not
reserve both histories
for every attempt. Every rejection records an explicit outcome reason. The
ledger also accepts an explicit unstaged rejection for nonlinear,
step-controller, preaccept, topology, maintenance-rollback, or publication
failure before a complete balance exists. Such a record keeps unavailable
diagnostic balance values as nonnumbers rather than inventing zeros, while
all accepted contributions remain exact zero. Its physical and
post-maintenance endpoint revisions may remain zero to mean unavailable;
those revisions are mandatory before any balance can be staged. An unstaged
record does not erase the latest established channel-ownership contract. The
three extension-map revision slots must either all be present and nonzero or
all be absent before a balance can be staged; an unstaged rejection may retain
partial extension provenance as unavailable input. The fixed-topology
prerequisite refuses commit when the mesh- or cut-topology
revision changes; such a trial can only be recorded with a topology-change
rejection until a topology-jump classification and acceptance policy exist.

This class is a validation and publication contract, not an assembler.
Production does not yet populate this complete record or connect it to the
time-step transaction. Kinetic and gravitational values have a separate
accepted-history producer described below, including a zero-duration initial
endpoint baseline. Production also stages the complete seven-row declared
residual-work set at a converged backward-Euler operator endpoint, but that
partial record is not a complete balance.
The ledger's focused unit fixtures built and passed all ten frozen serial
filters. They validate this fail-closed prerequisite contract; they do not
supply physical qualification evidence.
Therefore the complete accepted-state ledger, topology-event policy, energy
identity, threshold, simulations, and WP-8 closure all remain open.

### Production channel ownership

The complete record must consume one value from one named owner. The current
ownership boundary is:

| Record channel | Intended production owner | Current status |
|---|---|---|
| Kinetic and gravitational stored energy | Active-liquid state functional evaluated on the authoritative retained volume rules at both accepted endpoints | The Navier--Stokes declaration binds density, velocity, constant conservative body acceleration, and a zero reference point to `evaluateFreeSurfaceActiveVolumeEnergy`. The same endpoint evaluation records the gravitational-potential power \(-\rho\int g\cdot u\), needed to separate conservative body work from the changing-domain remainder. After initial constraints, cut refresh, and extension-state refresh, production records a zero-duration initial functional baseline; each accepted endpoint records the same channels, so consecutive history entries provide the stored-energy endpoints. Complete-ledger connection remains open. |
| Liquid--gas surface and solid--liquid wall stored energy | `FreeSurfaceGeometrySnapshot` discrete functional | Implemented for accepted and maintenance states. |
| Gas/compressibility stored energy | Gas model, when present | Explicitly inapplicable for the selected one-phase envelope; no gas model exists. |
| Bulk-viscous dissipation | Active-volume Galerkin viscous form tested by the endpoint velocity | For constant Newtonian viscosity, `evaluateFreeSurfaceActiveVolumeDissipation` evaluates the production density \(2\mu\,\mathrm{sym}(\nabla u):\mathrm{sym}(\nabla u)\) on retained endpoint rules. The application path reduces it exactly once and attaches it to accepted functional history. Variable-viscosity declarations leave this channel absent rather than substituting a literal coefficient. The frozen serial accepted-functional fixture executes this source path; complete-ledger connection remains open. |
| Navier-slip and line-friction dissipation | Accepted sharp wall/contact state at the declared endpoint | Sharp stage records exist; complete step integration and all-wall coverage remain unconnected. |
| External pressure and body-force work | Separately installed production pressure and forcing forms | Literal exterior pressure now has a retained-interface endpoint-power producer with the production sign \(-p_{\mathrm{ext}}\int u\cdot n_l\); it records outward liquid flux, area, and power in accepted history. Constant conservative body acceleration remains owned by gravitational stored energy, whose endpoint potential-power producer records \(-\rho\int g\cdot u\), and must not be counted twice as external work. Space/time-dependent forcing, prescribed-field forcing, and rotating-frame terms are isolated from the constant conservative acceleration in a named accepted-stage residual operator. Its signed work is staged, but mapping that value into the complete ledger's external body-work channel remains open. |
| Imposed-traction work and open-boundary energy flux | Generated-wet or full-physical boundary form that owns the corresponding condition | Boundary factories assemble the residual, but do not publish distinct work scalars. |
| Time-discretization work | Difference between endpoint-tested backward-Euler inertia and the kinetic-energy increment on the identical declared domains | `evaluateFreeSurfaceBackwardEulerKineticWork` evaluates both velocities on the retained endpoint rules. The backward-Euler application path reduces its components exactly once, recomputes the global identity residual, checks the endpoint kinetic energy against the accepted stored energy, and attaches it to accepted history. Generalized-alpha and steady records leave the channel absent. The frozen serial accepted-functional fixture executes this source path; complete-ledger connection remains open. |
| Kinetic-domain and gravitational transport coupling | Difference between the preceding kinetic energy on the endpoint and preceding retained domains; and gravitational stored-energy change minus endpoint gravitational-potential power integrated over the step | The endpoint-domain kinetic evaluator and accepted stored-energy history expose the kinetic remainder without equating different quadrature domains. Accepted active-volume history records gravitational stored energy and endpoint potential power. Production now pairs the immediately preceding accepted gravitational energy with the maintenance account's post-Transport endpoint and publishes their signed difference minus the endpoint potential power integrated over the step. Frozen serial fixtures exercise the endpoint-domain and gravitational maintenance accounting; the kinetic-domain remainder and complete-ledger connection remain open. |
| Convection and pressure/continuity work | Separately assembled Galerkin momentum and continuity components, including the declared boundary flux convention | The flow module now installs exact additive Galerkin convection/moving-control-volume and pressure/continuity operators. Each channel has one setup-time owner and explicit produced/inapplicable status. For backward Euler with temporal order one, production pairs the constrained operator residual with the converged candidate state and stages \(-\Delta t R_i(u)[u]\). Complete-ledger connection and boundary-flux qualification remain open. |
| Surface-transport coupling work | Difference between endpoint-tested surface/wall first variation and the accepted functional increment | Discrete-energy `SurfaceStress` declarations now explicitly enable an endpoint functional-power producer. It evaluates the existing surface/wall/volume first variation with the accepted velocity and physical gradient on the authoritative snapshot, reduces it once, and stores it beside the accepted functional value. Production now pairs the immediately preceding accepted surface-plus-wall energy with the maintenance account's post-Transport endpoint and publishes their signed difference minus the endpoint first-variation power integrated over the step when that power has complete coverage. Curvature-traction and synthetic functional-only declarations leave the coupling channel absent. The endpoint path executes in the frozen serial fixture; complete-ledger connection and a production MPI path remain open. |
| Weak-boundary work | Nitsche consistency/penalty and other weak-boundary residual components | The flow module now retains exact, separate momentum and continuity forms for fitted free-surface kinematic penalty/Nitsche enforcement and generic weak velocity Nitsche conditions. Generated-active-boundary trace certification is installed for both the production and named work operators. The converged-stage pairing is published; complete-ledger connection remains open. |
| VMS/PSPG work | Full momentum and continuity stabilization residual components | The flow module now installs the exact difference between the complete active VMS/PSPG momentum/continuity forms and their Galerkin bases, including configured PSPG boundary variants and the separately installed direct pressure-gradient formulation. The accepted-stage pairing uses the same setup-time owner/applicability contract as convection and pressure/continuity. Complete-ledger connection remains open. |
| Cut stabilization and ghost-penalty work | Named stabilization residual components | The retired velocity ghost penalty leaves the distinct cut-stabilization channel explicitly inapplicable; aggregation has its own later energy owner. When active, the retained pressure ghost-penalty form is installed under the named ghost-penalty work operator and paired at the converged stage. Complete-ledger connection remains open. |
| Aggregation work | Difference induced by the accepted aggregate constraint projection, with the unconstrained reference defined before execution | Constraint provenance exists; no energy-work comparison is defined. |
| Extension work | Accepted interface-transport change attributable to the bounded extension map | Map revision and amplification are provenance only, not work. |
| Pruning work | Difference between declared unpruned and retained authoritative functionals | Pruning telemetry exists; no paired functional work is published. |
| Limiting, redistancing, local reconciliation, and global correction | Transactional maintenance rows | Surface/wall/volume-potential changes are implemented. With exactly one active-volume owner across all declarations, the same exact before/after candidates now also publish kinetic-plus-gravitational-plus-every-interface surface-plus-wall modeled-energy changes, excluding the constraint potential. Accepted attempt records preserve separate row counts and modeled-energy sums for transport, limiting, reinitialization, geometry reconciliation, and global correction, plus a numerical-maintenance total that deliberately excludes physical transport. All pre-commit and post-accept transactions for one accepted step are then combined by a fail-closed metadata-consistent aggregate, preserving unavailable channels and logging one step-level maintenance account. Rejected attempts preserve the diagnostic changes but publish exact-zero accepted contributions. A second bulk owner is rejected before declaration mutation. Frozen serial and two-rank fixtures execute the modeled row and aggregation path; complete-record connection remains open. |

The residual-work path is transactional as well. Setup accepts either no
declarations or the complete seven-row set, requires communicator-identical
owner, applicability, and operator metadata, and refuses partial coverage.
For each converged backward-Euler temporal-order-one candidate it assembles
each produced operator at the candidate endpoint, suppresses constraint
inhomogeneity and auxiliary coupling, zeros constrained residual rows, pairs
the result with that same candidate state, and stages signed work
\(-\Delta t R_i(u)[u]\) with an algebraic-state revision. Candidate discard
logs exact-zero accepted contributions and clears the stage. Acceptance
requires matching step, attempt, time, duration, and maintenance-chain-start
state revision before adding all seven values to the partial physical-channel
record. Convection, pressure/continuity, nonconservative body force,
weak-boundary, VMS/PSPG, cut-stabilization, and ghost-penalty each have one
explicit produced-or-inapplicable decision. Other energy channels are still
unconnected, so the complete-record flag remains false.

The named work operators are measurement-only formulations. They retain
their formulation records, installed kernels, operator tags, and any exact
generated-boundary trace binding needed for assembly provenance, but they do
not publish normalized physical-problem contributions and are omitted from
the problem-analysis context. Consequently an auxiliary work operator cannot
introduce a nullspace gauge or other constraint into the physical system it
measures. The production equations remain fully visible to the same analysis
and constraint path.

This measurement-isolation increment is frozen by
[`free_surface_wp8_residual_work_20260830_41312481/record.json`](qualification_logs/free_surface_wp8_residual_work_20260830_41312481/record.json).
On source parent `3ee5556efe90d0dc2c10557660c67b802f93824b`, ledgered
`amarsden` job `41312481` completed with exit code `0:0` in `00:34:12` and
a batch peak resident set of 2,458,248 KiB. The hash-bound campaign passed 3
focused Application tests, 6 focused Physics tests, all 96
`ApplicationDriverLevelSetWorkflows` tests, and all 202
`MovingDomainPhysics` tests, with zero failures, errors, or disabled tests.
Its checksum manifest has SHA-256
`720aeb6870d03b0966f2bcf203457c214be2fe72a2e9a6cc4333e02f0f3ac524`.
This accepts only the converged-stage residual-work prerequisite. It does not
connect the complete energy record, close FSR-09 or WP-8, or qualify a
physical simulation exit.

The `Transport` maintenance-row label describes the physical conservative
phase-transport stage. Its modeled stored-energy change is now published under
a separate physical-transport breakdown entry and is excluded from the
attempt's numerical-maintenance subtotal. `Limiting`, `Reinitialization`,
`GeometryReconciliation`, and `GlobalCorrection` map respectively to the
complete ledger's limiting, redistancing, local-reconciliation, and
global-correction channels. This separation prevents the physical transport
change from being silently counted as maintenance work, but it does not yet
decompose that physical change among kinetic-domain, gravitational, surface,
convection, and pressure/continuity coupling terms. A zero-row transaction
leaves modeled-energy availability absent rather than inventing an evaluated
zero. The accepted-step aggregate requires increasing transaction identifiers,
identical step/attempt/time-step metadata, accepted status, complete row
coverage, internally consistent per-substage availability, and one exact
algebraic, functional, geometry, topology, and extension-map row chain before
it publishes a combined maintenance account. It exposes the chain start, the
state after all leading physical Transport rows, and the final maintenance
endpoint with component stored-energy values, gravitational and
surface/wall endpoint power, volume-constraint power, bulk-viscous
dissipation, exterior-pressure power, and revision provenance whenever their
complete production coverage is available.
Transport after any numerical-maintenance row is rejected. The aggregate
also requires the accepted Transport sum to telescope from the chain start to
the post-Transport endpoint and the numerical-maintenance sum to telescope
from there to the final endpoint whenever modeled-energy coverage is
available; both residuals are logged separately. The aggregate remains a
maintenance input to a future complete balance record, not a complete energy
identity.

Production retains that validated account through the remainder of the
accepted-step callback and checks its step, attempt, time, and duration
against the accepted time-step transaction. Before appending the current
accepted functional record, it reads exactly the latest declaration-sized
functional-history group and requires its declaration order and content,
accepted step and time, and nonzero algebraic revision to match the preceding
endpoint of the same time interval. It sums every declared preceding
surface-plus-wall energy and accepts at most one active-volume gravitational
owner. Those values are paired with the post-Transport endpoint to retain
surface/wall and gravitational energy changes, their available endpoint-power
coupling work, endpoint bulk-viscous dissipation, and step-integrated exterior
pressure work. Missing endpoint producers remain unavailable, a zero-row
account does not synthesize an endpoint, and this partial channel record is
logged with the complete-record connection explicitly false. This production
path now retains a communicator-consistent topology-only fingerprint for each
accepted functional's authoritative snapshot and reconstructs the preceding
snapshot, mesh-topology, and cut-topology set fingerprints. Accepted history
also retains the accepted extension-map revision when one exists. This
supplies the revision provenance for the preceding, physical post-Transport,
and post-maintenance stored-energy endpoints, but the still-missing work
channels prevent a complete balance record. Frozen serial accepted-functional
and two-rank maintenance fixtures exercise this partial record, but do not
connect it to the complete balance.

After curvature and velocity-extension refresh, it compares the final
FE-ordered algebraic revision with the maintenance-chain endpoint. After the
current accepted functional group is published, it also reconstructs its
snapshot-set, mesh-topology-set, and complete authoritative owned-rule
cut-topology-set fingerprints,
reads the current extension-map revision, and reports componentwise kinetic,
gravitational, liquid--gas surface, solid--liquid wall, and total modeled
stored-energy changes from the maintenance endpoint. A stored-energy endpoint
match is available only when the bulk kinetic and gravitational producer is
present and requires every one of those revisions and component energies to
agree exactly. Missing bulk coverage or any mismatch is explicitly labeled as
requiring unclassified post-maintenance work; it is not assigned a zero
extension or refresh contribution. The cut-topology fingerprint is reduced
through a fixed-size, partition-independent multiset digest rather than
replicating all owned-rule identities on every rank. An earlier frozen
two-rank active-cut/maintenance fixture executed the then-frozen fingerprint
and endpoint-match path. The structural fingerprint contract has changed in
the current source revision, so that archive does not qualify the revised
contract. The complete energy record remains unconnected until that work and
the other open channels are supplied.

For backward Euler on one unchanged liquid domain, endpoint testing of the
inertial term gives

\[
\Delta t\left(
  \rho\frac{u^{n+1}-u^n}{\Delta t},u^{n+1}
\right)
= K^{n+1}-K^n
+ \frac{\rho}{2}\lVert u^{n+1}-u^n\rVert^2.
\]

The final nonnegative term is time-discretization energy loss and must be
reported explicitly. When the retained liquid domain changes, this identity
does not authorize reusing coefficients from different quadrature domains;
the domain-change remainder belongs to the declared transport/geometry
coupling account.

The low-level endpoint-domain evaluator records
\(K^n\vert_{\Omega_h^{n+1}}\), \(K^{n+1}\), the step-integrated inertia
work, the nonnegative loss, and their algebraic identity residual on rank-owned
retained rules. It also binds both velocity arguments to explicit nonzero
velocity-field revisions. Comparing
\(K^n\vert_{\Omega_h^{n+1}}\) with the preceding accepted stored energy
exposes, rather than hides, the kinetic part of the domain-change remainder.
The source fixture also checks that ghost-only rules contribute exact zero.
The application source now reduces the component integrals over the FE
communicator, recomputes rather than sums the global identity residual,
checks endpoint-domain volume and kinetic energy against the accepted
stored-energy state, and stores the result in the same accepted history
transaction. It hashes only the declared velocity-field slice at each
endpoint, so representation-only level-set maintenance cannot change the
identity of the kinetic-work input. Before a new step is published, its
previous-velocity revision must equal the endpoint-velocity revision in the
latest accepted record for the same declaration. The zero-duration
backward-Euler baseline uses the initial endpoint and velocity revision as
both arguments. After an accepted backward-Euler step advances history, the
preceding accepted velocity is read from the second history slot and the
current accepted endpoint from the current slot.
These additions compiled and executed in the frozen serial accepted-functional
fixture. An equivalent production MPI backward-Euler fixture remains pending,
so this is prerequisite evidence rather than qualification evidence.

The production stored-energy declaration treats the constant
Navier--Stokes `body_force` component as a conservative acceleration with
potential density \(-\rho\,b\cdot x\). Space/time-dependent or prescribed-field
body-force additions are not folded into gravitational energy; their work
still requires the external-work producer. The retained-volume evaluation
also records the endpoint gravitational-potential power
\(-\rho\int b\cdot u\), so a later complete ledger can expose the difference
between the gravitational-energy increment and its endpoint material-domain
power instead of hiding it inside external work. Accepted active-volume values are
evaluated from the FE-ordered accepted solution on the same retained physical
volume rules as the surface functional, reduced exactly once over the FE
communicator, checked against the functional's liquid measure and snapshot
revision, and stored in the accepted functional history. The initial
zero-duration record is created in the first pre-solve callback, after the
time-loop applies the initial strong constraints and production refreshes the
cut geometry and derived extension state. It uses the same algebraic revision
for both endpoint provenance fields and, when dynamic contact is declared,
records the current state with `alpha_f=1`. This supplies consecutive
before/after stored-energy states. For backward Euler only, it also supplies
the zero-duration kinetic-work identity baseline described above; it does not
decompose the remaining residual or feed the complete transactional ledger.
When the Navier--Stokes viscosity is a literal constant, the same endpoint
transaction also records the rank-reduced bulk-viscous dissipation rate on the
identical retained liquid rules. Constitutive viscosity models remain outside
that producer until their pointwise coefficient can be evaluated at the
declared endpoint.

For the discrete-energy `SurfaceStress` path, the accepted transaction also
tests the surface/wall/volume first variation with that endpoint velocity.
This supplies the common-stage virtual-power ingredient needed to compare
\(\Delta E_{\mathrm{surface+wall}}\) with
\(\Delta t\,\delta E_{\mathrm{surface+wall}}[u^{n+1}]\). The signed difference
has not yet been published into the complete numerical-work ledger. The
endpoint path executes in the frozen serial fixture; complete-ledger
connection and a production MPI path remain open.

## Stage and topology policy still required

The nonlinear synchronization callback receives the solve time and time
integration context, and accepted dynamic-contact state has explicit stage
time and stage fraction. That provenance does not yet prove that geometry,
surface/wall/contact work, transport, and every maintenance contribution use
one common stage.

The current source contains a generic reversible final-candidate gate in
`TimeLoop`.  It runs after any required endpoint reconstruction and projected
generated-state synchronization, but before accepted auxiliary events,
adaptive-controller acceptance, commit-ready publication, and system/history
commit.  A typed `CutTopologyChanged` rejection follows the bounded adaptive
retry path, while a fixed-step rejection restores the accepted state and fails
before commit.

The transient Application path now connects a bounded active-cut tracker to
that gate when every active-cut request selects `LinearCorner` geometry.  The
accepted baseline is seeded by the first authoritative
`before_physics_solve` refresh after the time loop has imposed the initial
strong constraints.  Every later cache rebuild and cache hit is observed; a
mismatch or missing key taints the whole attempt monotonically.  Dynamic
contact reconstruction, when configured, precedes a uniform final refresh of
the endpoint solution, and the gate decides only after that refresh.  A
rejected attempt invalidates the cut-refresh cache, curvature cache, and
frozen algebraic extension maps before restoration/retry.  Restored-state
synchronization bypasses the ordinary per-step/frozen refresh cadence so the
accepted cut context is rebuilt before accepted constraints are reapplied.

The topology key is deliberately a bounded probabilistic structural
fingerprint, not an exact set-equality or collision-free certificate.  It
combines the owned generated-rule multiset using global parent/boundary IDs,
request policy, rule category, and epoch-free parent-corner/incidence
descriptors.  The reduction preserves multiplicity while being independent of
rank ownership, traversal order, communicator size, coordinates, raw
level-set values, and source/mesh epochs.  Transient non-`LinearCorner`
requests fail closed because the current descriptor is not a sound arbitrary
high-order connectivity certificate.

Serial source fixtures cover the tracker/cache contract, linear-cut
structural transitions and ordinary-motion invariance, projected
generalized-alpha retry, and fixed-step nonpublication.  Two-rank source
fixtures cover the TimeLoop decision consensus and asymmetric fingerprint
preparation failure.  These revised source fixtures are unbuilt and
unexecuted in the current evidence set and are not qualification evidence.

The postaccept maintenance source path also treats the already committed
physical step and its nested maintenance transaction separately.  Before
maintenance it forces one authoritative accepted-endpoint refresh and captures
that nonzero topology fingerprint before functional evaluation, checkpointing,
or geometry staging.  Only the final maintenance validator supplies candidate
topology evidence.  A complete communicator-consistent mismatch may select the
typed maintenance-only rejection only after every non-topology transaction
invariant has passed exact consensus.  Recovery rolls back the staged
geometry/cache state, restores owned current and stored-history coefficients
through collectively fenced backend phases, refreshes ghosts, and verifies the
accepted topology fingerprint before publishing rejected maintenance-work
rows.  Its terminal diagnostic states that the physical step is retained and
the full free-surface energy attempt is not rejected.  Missing, malformed, or
rank-divergent evidence remains a generic invariant failure.  The current
serial/MPI tests exercise the evidence classification and collective readiness
helpers only; full backend restoration and `on_step_accepted` failure injection
remain unexecuted.

This is a fail-closed fixed-linear-topology prerequisite, not the complete
WP-8 topology-event policy.  The following remain open:

- a canonical connectivity descriptor and qualification for high-order,
  contact-association, and active-boundary topology;
- end-to-end Application evidence for both precommit retry and postaccept
  nested rollback, including backend current/history restoration and injected
  failures inside contact, generated-geometry, and recovery collective phases;
- an accepted-versus-trial topology-event ledger and energy-jump
  classification;
- a minimum resolved-feature policy; and
- executed serial/MPI, rotation, numbering, and partition equivalence of the
  selected action.

## Frozen prerequisite evidence

The low-level matrix is
`tests/cases/fluid/free_surface_wp8_energy_qualification_matrix.json`, with
`tests/cases/fluid/run_free_surface_wp8_energy_qualification.py` as its strict
wrapper. Its exact frozen SHA-256 is
`3fc4094579f1d209525972aa9b1fe449e03f4b82df050d062804d732b86114f1`.
Any byte drift is rejected before structural validation or test discovery. It
also requires every listed test to have exactly one definition in its frozen
source-file mapping before execution discovery. It freezes only:

- exact production parsing and selection of generalized-alpha or backward
  Euler, including backward-Euler spectral-radius inapplicability;
- the backward-Euler one-step endpoint fixed-point and history transaction;
- the fixed-topology surface/wall/volume first-variation fixture and exact
  ghost exclusion;
- regenerated-residual outer convergence in a scalar fixture;
- transactional outer rollback and constraint-topology refresh;
- the selected frozen-inner Jacobian provenance;
- authoritative accepted functional history;
- global-correction functional-work reporting;
- continuous maintenance row chains with separate transport, limiting,
  reinitialization, geometry-reconciliation, and global-correction substages;
- accepted/rejected transaction publication, including explicit zero-row
  outcomes and zero accepted contribution for rejected attempts;
- the prospective complete-channel ledger's fail-closed completeness,
  ownership, endpoint-chain, attempt-sequencing, fixed-topology, gas
  applicability, and accepted-versus-rejected publication contracts;
- separate snapshot, mesh-topology, cut-topology, algebraic-state, and
  extension-map revision provenance;
- bounded extension refresh prerequisites; and
- two-rank deterministic maintenance rows, complete-vector algebraic
  provenance, and communicator-wide geometry refresh.

Executed evidence remains deliberately partial. Wrapper validation returned
`PASS_PREREQUISITE_NONCLOSURE` for 41 frozen tests, 10 method exits, and 6
simulation exits; all 18 wrapper unit tests passed. The application binaries
passed 10 focused ledger filters and 19 other frozen serial filters, while 3
frozen MPI filters passed on 2 ranks. This executes 32 of the matrix's 41
tests. The remaining 2 geometry and 7 systems tests were not part of this
execution evidence. None of these prerequisite results supplies a physical
campaign or complete energy balance.

The only accepted claim is `low_level_prerequisite`. The wrapper rejects every
claim ending in `_closure`, explicitly rejects FSR-09, WP-8, Q3, Q4, Q5, and
complete-energy requests, and does so before build, binary discovery, test
execution, or artifact creation.

## Unqualified exits

The registry machine-records all open method and simulation exits. The method
exits cover the partitioned energy argument, backward-Euler balance, complete
accepted energy/dissipation/external/numerical-work identity, common-stage
contract, same-state full-package refresh neutrality, geometry-dependent
residual directional derivatives, contraction refinement, topology policy,
generalized-alpha consistency, and a prospectively frozen energy-residual
threshold. The maintenance-only accepted/rejected separation and five
implemented substage rows are no longer listed as missing, but they do not
close any of those broader exits.

Production availability of backward Euler removes only the method-selection
prerequisite. The backward-Euler balance exit remains
`REQUIRED_NOT_CLAIMED`; no result in this matrix supplies a free-surface
energy residual.

The six required simulations remain entirely unclaimed:

- static cap;
- capillary relaxation;
- linear capillary wave;
- droplet oscillation;
- sloshing; and
- wetting relaxation.

Each simulation must independently refine the declared spatial and temporal
scales, include cut shifts and representative MPI partitions, publish the
complete ledger, and demonstrate convergence of the unexplained energy
residual. Until those records pass, FSR-09 and WP-8 remain open.

## Source evidence map

- Outer fixed point and transactional restoration:
  `Code/Source/solver/FE/TimeStepping/NewtonSolver.{h,cpp}`.
- Generated-state selection and synchronization:
  `Code/Source/solver/Application/Core/ApplicationDriver.cpp`.
- Transient-scheme XML defaults and parsing:
  `Code/Source/solver/Parameters.{h,cpp}`.
- Backward-Euler endpoint transaction:
  `Code/Source/solver/FE/TimeStepping/TimeLoop.cpp` and
  `Code/Source/solver/FE/Tests/Unit/TimeStepping/test_TimeLoopConvergence.cpp`.
- Maintenance ledger contract:
  `Code/Source/solver/Application/Core/ApplicationDriver.{h,cpp}`.
- Prospective complete-channel balance record:
  `Code/Source/solver/Application/Core/FreeSurfaceEnergyLedger.{h,cpp}`.
- Balance-record transaction and fail-closed fixtures:
  `Code/Source/solver/Application/Tests/Unit/test_FreeSurfaceEnergyLedger.cpp`.
- Maintenance transaction and communicator regressions:
  `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows{,MPI}.cpp`.
- Geometry policy limits:
  `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp`.
- Surface/wall/volume functional, first variation, rank-owned active-volume
  stored-energy and constant-viscosity dissipation primitives, and
  endpoint-domain backward-Euler kinetic identity:
  `Code/Source/solver/FE/Interfaces/FreeSurfaceGeometrySnapshot.{h,cpp}`.
- Accepted functional/contact records:
  `Code/Source/solver/FE/Systems/FESystem.{h,cpp}`.
- Surface-stress tangent containment:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`.
