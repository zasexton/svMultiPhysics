# WP-9 fitted-ALE free-surface architecture and qualification boundary

Status: low-level policy operators, boundary-local normal ownership, exact
consumer binding, bounded fitted surface/contact capability provenance, and
transactional in-memory accepted-policy provenance are present in source. A
bounded fitted-normal operator-stage measurement path now records full,
normal, and tangential mesh-velocity moments. Hash-bound serial and two-rank
telemetry passed, and the frozen 32-test prerequisite matrix passed from a
clean Release source revision. The method exits, all three physical
fitted-ALE campaigns, WP-9, FSR-10, FSR-11, and Q4 remain open.

Audit basis: projection telemetry implementation commit
`afb0f65788215e38e70a170aa108d037527c4248`, clean-matrix runner commit
`6213ef09988a1e364a62ab20d5cd174c57da49f6`, and the two 2026-08-30
campaigns recorded below. Earlier dated diagnostic sections are retained as
historical development evidence and are superseded where this checkpoint
reports a later execution result.

This record updates the WP-9 boundary from
`Documentation/free_surface_boundary_unfitted_audit_20260720.md`. It does
not claim FSR-10, FSR-11, WP-9, or Q4 closure.

## Verdict

The current schema-2 fitted source contract is narrow and fail closed. A
fitted free surface passes preflight only when all of the following are true:

- ALE is enabled;
- mesh velocity comes from a coupled displacement unknown;
- the normal policy is `MatchFluidNormalVelocity`;
- normal enforcement is explicitly `Penalty` or `Nitsche`; and
- the tangential policy is one of `Free`, `SmoothingOnly`, or `Prescribed`.

The three policies have distinct source-level states. `Free` intentionally
adds no tangential boundary row, `SmoothingOnly` adds a tangential
surface-gradient functional, and `Prescribed` adds the projected
mesh-velocity penalty. Prescribed mesh-velocity data, a missing normal
relation, and explicit normal enforcement `None` are rejected at the same
pre-mutation boundary.

The typed programmatic boundary is also fail closed before mutation. It
rejects unknown implementation, active-domain, active-domain-method,
kinematic-enforcement, normal-policy, tangential-policy,
pressure-stabilization, and surface-tension-form enum values, as well as an
unknown ALE mesh-velocity source. Such values cannot be relabeled as a fitted,
unfitted, prescribed-data, or `Unknown` effective configuration.

Schema 1 retains older configurations for regression and migration only.
It requires explicit legacy opt-in and its effective configuration is labeled
`unqualified_explicit_legacy`. A schema-1 pass is not evidence for the
schema-2 supported capability.

This is enough to freeze a truthful source prerequisite harness. It is not
enough to close WP-9 or Q4. The boundary-local normal ownership slice has a
clean Release prerequisite-matrix pass. Measured target error and work,
scaling, restart continuity, and fault-complete accepted-history arguments
remain open. The executed operator-stage telemetry records full, normal, and
tangential mesh-velocity moments plus their projection identity, but not the
complete policy-specific boundary-kinematics/work contract. None of the
required physical ALE campaigns has run.

## Schema and policy contract

| Input mode | Tangential policy | Registration outcome | Qualification |
|---|---|---|---|
| schema 2, coupled displacement, explicit normal enforcement | `Prescribed` | central owner plus projected weak tangential operator | `supported_configuration_envelope` |
| schema 2, coupled displacement, explicit normal enforcement | `Free` | central owner plus explicit natural tangential state | `supported_configuration_envelope` |
| schema 2, coupled displacement, explicit normal enforcement | `SmoothingOnly` | central owner plus tangential surface-smoothing operator | `supported_configuration_envelope` |
| schema 2, prescribed mesh-velocity data | any fitted policy | rejected before system mutation | unsupported |
| schema 2, normal policy or enforcement `None` | any fitted policy | rejected before system mutation | unsupported |
| schema 2, fitted dynamic contact angle | `Prescribed` | rejected before system mutation | unsupported |
| schema 1 with explicit legacy opt-in | `Free`, `SmoothingOnly`, or `Prescribed` | retained only according to the legacy path actually present | `unqualified_explicit_legacy` |

The XML translator does not infer normal enforcement from a penalty value in
schema 2. `Kinematic_penalty` requires explicit
`Kinematic_enforcement=Penalty`; explicit `None` is never promoted. The
schema-1 migration path may preserve the historical inference only when
enforcement was absent, never when `None` was explicit.

## Implemented fitted operators

For coupled displacement \(d_m\), mesh velocity is \(\dot d_m\). The normal
relation is

\[
(\dot d_m-u)\cdot n=0.
\]

The selected penalty or Nitsche form is added to the fluid residual. The
coupled displacement receives the corresponding mesh-row normal relation.
The schema-2 preflight makes the relation mandatory for every accepted
fitted surface.

For `Prescribed`, the tangential mesh residual is equivalent to

\[
\int_{\Gamma_f}\beta_t
\left(I-n\mathbin{\otimes}n\right)
\left(\dot d_m-w_t^\star\right)\cdot\psi_m\,dS .
\]

The projection uses the current boundary normal. Low-level tests exercise
the original and a rotated tetrahedral face: a normal target is projected
out and an in-plane target produces a residual.

For `Free`, the fitted kinematic operator contains only the required normal
relation. Its tangential boundary state is natural: no tangential constraint
or tangential regularization row is appended.

For `SmoothingOnly`, let \(P=I-n\mathbin{\otimes}n\). The boundary
functional is

\[
\frac{\beta_s}{2}\int_{\Gamma_f}
  \left(P\nabla d_m P\right):
  \left(P\nabla d_m P\right)\,dS ,
\]

with residual

\[
\int_{\Gamma_f}\beta_s
  \left(P\nabla d_m P\right):
  \left(P\nabla\psi_m P\right)\,dS .
\]

The same current boundary normal and surface measure define both
projectors and the integral. A source fixture distinguishes this row from
the zero tangential row of `Free` on original and rotated faces. That
fixture is not execution evidence until it is built and run.

## Ownership and truthful provenance

`FESystem::declareMeshTangentialBoundaryPolicy` is the central ownership
registry. A second declaration for the same displacement field and boundary
marker is rejected. The registry also rejects an unknown policy value, a
blank owner, an unbound field, and any target that is not the
mesh-dimensional displacement unknown. The declaration is ownership
metadata; it is not by itself evidence that an operator consumed the policy.

The effective configuration therefore derives provenance from system state:

- `tangential_mesh_owner` is emitted only when the matching central
  declaration exists with the exact selected policy;
- `policy_consumed=true` requires a published central consumer binding and
  exactly one matching policy-specific tangential descriptor for the
  displacement field and boundary marker;
- `operator_tag` comes from that binding and is emitted only while its
  operator remains registered;
- `operator_source` is shared exactly by the binding and its unique matching
  descriptor; and
- `policy_qualification` distinguishes the supported schema-2 envelope from
  explicit unqualified legacy operation.

An unconsumed path reports `policy_consumed=false` and null operator tag and
source. The owner is not fabricated from a module-name constant.

The schema-2 source contract now emits distinct exact sources for the
natural `Free` state, the `SmoothingOnly` surface functional, and the
`Prescribed` projected velocity penalty. This is a source-contract claim,
not qualification evidence.

Accepted tangential history is canonicalized by mesh-displacement field and
boundary marker, independent of module registration order. Each successful
publication builds one complete row group for every current declaration.
Replay validates every row in the latest group against both its shared
accepted provenance and the canonical declarations; an incomplete or
mismatched group fails closed instead of being mistaken for an idempotent
replay. The fitted Navier--Stokes path binds each declaration to exactly one
matching tangential boundary descriptor after that descriptor is installed.
The accepted row retains whether a consumer was bound and its exact operator
tag and descriptor source. A mismatched policy, unknown operator, blank
provenance, missing descriptor, or duplicate matching descriptor is rejected
before the binding is published. Idempotent rebinding and accepted-history
publication revalidate the unique descriptor, so a descriptor added after
initial binding cannot silently preserve a consumed claim. Generic mesh-motion
declarations that have not established this descriptor binding remain
explicitly unbound in history rather than being labeled as consumed.

`FESystem::declareMeshNormalBoundaryConstraint` now provides a separate
normal-ownership registry keyed exactly by mesh-displacement field and boundary
marker. Any second declaration for that key is rejected, including a repeated
owner, while distinct markers remain independent. Fluid weak-velocity rows do
not participate in this ownership key, although their separate boundary-marker
compatibility rules still apply. Declarations distinguish displacement-trace
prescribed targets, time-scaled prescribed velocity targets, and the fitted
fluid-normal-velocity relation. The central registry requires the bound
mesh-dimensional displacement unknown, a scalar target, consistent quantity
and target kinds, and a
`WeakPenalty` declaration label. Prescribed targets reject FE-state and
variational-geometry dependence. The fitted relation additionally requires an
exact related mesh-dimensional fluid velocity unknown, and its target is
structurally required to be exactly that related state field dotted with the
current normal. The declaration carries no concrete penalty or time-scale
coefficient, so its `WeakPenalty` label alone does not validate the installed
row's scaling.

Harmonic, pseudo-elastic, and fitted-fluid registration preflight existing
same-key owners before adding the second module's fields or operators. The
harmonic and pseudo-elastic paths additionally require a finite positive literal
penalty and, for velocity targets, a finite positive literal time scale; they
lower the latter target to time scale times velocity, install the concrete
weak-penalty displacement row, and only then publish the declaration. The fitted
path likewise installs its mesh row before publication. Under fitted `Penalty`,
that mesh row uses `kinematic_penalty`; under fitted `Nitsche`, it remains a
pure `WeakPenalty` row with weight `kinematic_nitsche_gamma / hNormal()`. The
separate fluid row is `WeakPenalty` for fitted `Penalty`. For fitted `Nitsche`,
the fluid row is `WeakNitsche`, with penalty weight based on
`kinematic_nitsche_gamma * viscosity / hNormal()` and the optional squared
polynomial-order scale, plus its consistency and adjoint terms. This registry
and its consumer binding are still symbolic declaration/operator provenance,
not an evaluated boundary-kinematics history or a stability certificate.

## Accepted boundary provenance transaction

Every normal declaration must bind to one registered boundary operator and
exactly one operator-associated mesh `NormalComponent` descriptor before it
can enter accepted history. A fitted fluid-normal relation additionally binds
one reciprocal fluid descriptor and retains whether that descriptor is
`WeakPenalty` or `WeakNitsche`. Binding checks the exact displacement field,
marker, trace, enforcement, descriptor source, operator tag, and related
velocity field. The operator must contain an installed boundary formulation;
a metadata-only operator is insufficient. Publication revalidates descriptor
uniqueness, so later descriptor drift fails before a new accepted group is
committed.

The combined normal-and-tangential publisher owns the accepted transaction
from its first nonempty group. It compares canonical declaration and binding
tokens, accepted step, time, time step, state fingerprint, and existing
history shape across the active FE communicator. It stages both row families,
coordinates local validation failures, commits both or neither, and emits
accepted diagnostics only after collective success. A globally empty
declaration set is a no-op and does not freeze later declarations. After the
first nonempty commit, neither declaration family nor either consumer binding
may change. The individual normal and tangential recorders are explicitly
serial/rank-local APIs and reject distributed use outside the combined
transaction.

Steady and successful transient post-accept paths call this combined publisher.
The accepted-state stamp is a fixed-width fingerprint, not collision-free
state identity. It mixes exact scalar bits with public global row numbers over
backend-certified owned rows. A bounded 65,536-row occupancy reduction first
proves that every global row has exactly one owner; the fingerprint path uses
about 512 KiB of fixed occupancy storage per rank plus the backend's
O(local-owned-row) list, rather than gathering the global vector on every
rank. PETSc uses its primary `Vec` ownership range, Trilinos its cached
one-to-one primary map, FSILS its owned-row layout and inverse FE permutation,
and block vectors offset each certified child layout. PETSc and Trilinos were
not enabled in the focused diagnostic described below and therefore remain
unexecuted source paths.

On the default transient success path, accepted maintenance refreshes the
constraint set, the Application reimposes master-bearing constraints, records
mesh provenance, and only then publishes extension-map artifacts, endpoint
functionals, diagnostics, and VTK output. TimeLoop's later reimposition is
value-idempotent. `SVMP_NO_MPC_STATE_DISTRIBUTE` deliberately disables both
reimpositions for legacy comparison. Fallible accepted-step work that occurs
before this point can still leave an irreversibly committed step without a
mesh-history group; fault-complete publication or recovery remains open.

The combined accepted-boundary transaction described above remains symbolic:
its normal target is retained as an expression, not an evaluated target value.
A separate uncompiled operator-stage history described below now evaluates a
small raw normal-kinematics subset. It is not atomic with the later symbolic
endpoint transaction and does not provide tangential target error, boundary
work, immutable artifact serialization, or restart continuity. Rank-local mesh
revisions remain local provenance rather than claimed communicator-invariant
stamps.

## XML boundary

The top-level XML boundary allowlist now admits the names already consumed by
the downstream modules:

- mesh-motion `Policy` and `Tangential_policy`;
- `Quantity` and `Constraint_quantity`;
- `Target`;
- `Penalty` and `Penalty_scale`;
- `Velocity_time_scale` and `Time_scale`; and
- all four fitted tangential-penalty spellings:
  `Tangential_mesh_penalty`, `TangentialMeshPenalty`,
  `Prescribed_tangential_mesh_penalty`, and
  `PrescribedTangentialMeshPenalty`.

The value is the prescribed velocity-penalty weight for `Prescribed` and
the surface-gradient weight for `SmoothingOnly`. Supplying any of these
aliases with `Free` fails before system mutation because that policy has no
tangential boundary row.

Application-level regressions parse complete `Add_equation` XML, build the
module, and inspect the central declarations or effective free-surface
artifact. They also prove that explicit `None` plus a kinematic penalty fails
without mutating the system. A complete fitted `DynamicContactAngle` input is
translated to the typed `DynamicRenE` model and rejected by module preflight
before fields, forms, boundary descriptors, mesh policies, or operators are
installed.

## Boundary-local enforcement and conflicts

Each fitted boundary owns its normal enforcement, penalty or Nitsche
parameters, prescribed tangential target, and tangential penalty. Generic
weak velocity conditions retain their separate module-level policy.

The low-level suite covers boundary-local fitted parameters, registration
order, projection under rotation, multiple markers, and conflicts between a
mesh-motion owner and the free-surface owner. The current source prerequisite
also covers both same-marker registration orders, harmonic and pseudo-elastic
owners, distinct-marker coexistence in both orders, multiple fitted surfaces,
unrelated-marker weak fluid rows, exact velocity-target lowering, and the fitted
Penalty/Nitsche descriptor split. The frozen qualification matrix still marks
the complete normal-compatibility exit `REQUIRED_NOT_CLAIMED`. The dirty-tree
source now adds symbolic accepted normal history and a two-rank transactional
rollback regression, but neither is part of the frozen matrix or an integrated
Release run.

## Capability exclusions

The supported fitted slice continues to exclude:

- fitted `SurfaceStress`;
- fitted prescribed contact angle;
- fitted dynamic contact angle;
- prescribed mesh-velocity data;
- schema-2 normal policy or enforcement `None`; and
- unfitted active-domain and cut-stabilization controls on a fitted surface.

For every successfully registered `FittedALE` surface, module artifact schema
3 now emits the narrowly scoped `fitted_surface_contact_capability` object. It
records `Automatic` and `CurvatureTraction` as supported surface-tension-form
requests, `None` and `Pinned` as supported contact-line models, and the common
`fail_closed_before_system_mutation` disposition for these exact exclusions:

- `SurfaceStress`, reason
  `fitted_surface_stress_current_frame_gradient_unqualified`;
- `PrescribedAngle`, reason
  `fitted_contact_line_codimension_two_unavailable`; and
- `DynamicRenE`, reason
  `dynamic_contact_requires_sharp_unfitted_level_set`.

Pinned fitted contact lines remain contained to coupled ALE. An excluded
request still produces neither an effective artifact nor a normal/tangential
accepted-history row: the artifact describes the bounded capability of a
successfully validated configuration, while rejected-configuration telemetry
is a separate concern. The source and exact snapshot now carry this narrow
capability contract, but the frozen matrix has not yet been revised and these
new source/test edits have not been compiled or executed.

## Frozen prerequisite evidence

The matrix
`tests/cases/fluid/free_surface_wp9_fitted_ale_qualification_matrix.json`
is byte-frozen at SHA-256
`46225d9c90e71c90881725d4e871466f7a0e19e04a4309627b82e376583715d9`.

The wrapper
`tests/cases/fluid/run_free_surface_wp9_fitted_ale_qualification.py`
checks that digest before parsing the matrix. It partitions the frozen tests
between the Application and Physics binaries and permits only
`low_level_prerequisite`. It rejects FSR-10, FSR-11, WP-9, Q4, and general
fitted-ALE qualification claims before execution arguments are processed or
an artifact directory is created.

Current-tree wrapper validation returns `PASS_PREREQUISITE_NONCLOSURE` for
32 frozen tests, 9 unqualified method exits, and 3 unqualified simulation
exits; all 17 wrapper unit tests pass. This verifies only the frozen matrix,
source-definition discovery, and fail-closed claim boundary. No Application
or Physics binary was supplied to that validation, so none of the frozen
source-contract regressions or physical fitted-ALE campaigns is counted as
executed evidence.

The frozen evidence includes these source-contract regressions:

- `EquationTranslatorMeshMotion.XmlAliasesReachTangentialPolicyModuleRegistration`;
- `EquationTranslatorFreeSurface.XmlTangentialPenaltyAliasesReachTruthfulFittedModule`;
- `EquationTranslatorFreeSurface.XmlExplicitNoneCannotBePromotedByKinematicPenalty`;
- `EquationTranslatorFreeSurface.XmlFittedDynamicContactFailsClosedBeforeSystemMutation`;
- `MovingDomainPhysics.FittedFreeSurfaceQualifiedContractRejectsBeforeMutation`;
- `MovingDomainPhysics.FittedFreeSurfaceTangentialPoliciesRegisterCoupledMeshOwnership`;
- `MovingDomainPhysics.MeshTangentialPolicyConsumerBindingRequiresUniqueExactDescriptor`;
- `MovingDomainPhysics.FittedFreeSurfaceFreeAndSmoothingPoliciesProduceDistinctMeshRows`;
- `MovingDomainPhysics.FittedFreeSurfaceLegacyPrescribedDataReportsUnconsumedPolicy`;
- `MovingDomainPhysics.NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults`;
- current prescribed projection and coupled setup tests;
- explicit schema-1 legacy operator regressions;
- shared mesh-motion consumer and ownership-conflict tests; and
- fitted contact and surface-stress capability-boundary tests.

A passing matrix result is reported only as a prerequisite/non-closure
result. It cannot change any closure disposition.

### 2026-08-17 focused dirty-tree diagnostic

The boundary-local normal-ownership slice was compiled one translation unit at
a time with optimization disabled and a 1 GiB process address-space limit.
The diagnostic binary linked those current objects with existing application
archives, so this is mixed-optimization prerequisite evidence rather than a
Release build or qualification artifact. Eleven selected tests passed:

- structural scalar-target shape validation;
- all three coupled tangential-policy registrations and their fitted normal
  descriptors;
- fitted Nitsche mesh/fluid descriptor separation;
- harmonic and pseudo-elastic same-marker conflict rejection in both module
  orders, plus reverse-order distinct-marker coexistence;
- direct normal-registry validation, duplicate rejection, and displacement
  rebinding protection;
- exact equality of a velocity target scaled as `2 * 0.25` and a displacement
  target of `0.5` at assembled rows;
- invalid penalty, time-scale, and target-shape rejection before mutation;
- the distinct-marker coupled fitted-fluid/harmonic setup; and
- two fitted boundary-local policies with canonical normal declarations and
  one mesh plus one fluid normal descriptor per marker;
- distinct assembled mesh rows for the fitted Free and SmoothingOnly
  tangential policies; and
- fitted Nitsche boundary-locality and registration-order invariance, including
  the assembled Jacobian comparison.

Available memory stayed above the 10 GiB start gate, every compiler, linker,
and test process was capped at 1 GiB, and the 83,392 KiB scratch directory was
deleted after the run. This evidence does not update the frozen matrix digest,
establish immutable source correspondence, or close any method or physical
exit.

### 2026-08-17 accepted boundary-provenance diagnostic

A later current-source diagnostic compiled the changed Eigen, block, and
FSILS vector implementations, `FESystem.cpp`, `ApplicationDriver.cpp`, and the
two focused Physics test translation units one at a time with optimization
disabled and a 1 GiB process address-space limit. The largest compile was
`FESystem.cpp` at 909,452 KiB maximum resident memory; the mixed diagnostic
link used 390,048 KiB. It reused existing archives for unchanged dependencies,
so it is not a clean Release build or an immutable qualification artifact.

Eight selected serial history tests passed. They cover direct normal-registry
and exact-consumer validation, exact tangential-consumer validation, rollback
of a partially staged normal group before accepted diagnostics, an empty
combined call that does not seal later declarations, canonical tangential
replay, pseudo-elastic normal-history freeze, and the mixed fitted-fluid plus
harmonic normal/tangential history and replay fixture. The serial run used
104,772 KiB maximum resident memory.

One new two-rank test passed. It proves that the public individual normal
recorder rejects distributed use, rank-divergent accepted metadata rejects on
every rank without publication, and one rank's duplicate exact descriptor
causes the other rank's already staged normal and tangential groups to roll
back collectively. It classifies one local descriptor error and one remote
coordinated error, retains rollback logging only as telemetry, and leaves both
histories empty. The launcher used 95,720 KiB maximum resident memory.

Three backend ownership tests also passed: Eigen global rows, block-vector
offset rows, and the FSILS shared owned-row mapping. The current configuration
did not enable PETSc or Trilinos, so those conditional implementations were
not compiled or executed. `ApplicationDriver.cpp` compiled,
but no Application integration test invoked its chunked ownership proof,
accepted-state fingerprint, or post-maintenance publication ordering. These
remain source-reviewed rather than executed Application evidence.

Available memory was checked before every compile, link, and test and remained
above the 10 GiB start gate; storage remained far above the 4 GiB gate. Every
compiler, linker, launcher, and test was capped at 1 GiB. The 108,376 KiB
temporary diagnostic directory was deleted after the run. This evidence does
not alter the frozen matrix, close a method exit, or support any audit checkbox.

### 2026-08-17 uncompiled operator-stage measurement slice

The current dirty source adds two bounded foundations for a future evaluated
accepted-boundary history. First, `TimeLoop` can expose an immutable copy of a
converged temporal-order-one operator stage before endpoint reconstruction,
geometry replacement, or history rotation. Backward Euler, the explicit DG0
route, and first-order generalized-alpha retain the exact operator
first-derivative stencil, the distinct generalized-alpha state/rate times, and
the fields for which that derivative is meaningful. Local snapshot preparation
is completed before communicator-wide ghost refresh. The snapshot also retains
all eight rank-local mesh revision domains before `on_nonlinear_done`, allowing
an evaluator to reject a later geometry change. Rank-local observer
failures are coordinated before the nonthrowing discard hook; all other
transaction callbacks and adaptive-controller decisions remain subject to the
documented communicator-symmetric callback contract. Serial source fixtures
cover the endpoint and generalized-alpha values and rejection transaction, and
a new FSILS MPI source fixture injects a rank-zero-only observer failure and
checks all-rank discard without commit.

Second, `BoundaryReductionService` now has a sparse backend-read path that
refreshes owner-to-ghost values and evaluates only faces whose parent cell is
owned locally. PETSc, Trilinos, FSILS, and recursive block-vector read views
resolve public global rows from their owned/overlap layouts and reject a valid
row outside that overlap instead of silently supplying zero. Backend layout and
collective-refresh shape are preflighted before refresh; rank-local mesh revision
epochs remain local cache keys, while cache-hit and cache-publication consensus
are entered in the same order even when revision tracking is unavailable on
some ranks. The new source tests include a deliberately huge reported global
vector backed by only locally relevant rows, divergent rank-local revision
epochs and cache availability, zero-local-ghost participation, and recursive
block reads.

The current dirty source now builds the first deliberately smaller consumer of
those foundations. Each consumer-bound fitted fluid-normal relation registers
three sum reductions with mesh displacement as the primary field and fluid
velocity as a current-only secondary field. A sealed `{0,1}` first-derivative
alias maps `dt(d_mesh)` to the copied exact rate vector. The raw history retains
the boundary measure `A`, mesh-normal integral `Wn`, fluid-normal integral
`Un`, and squared normal-gap integral `gap_sq`, plus exact stage times,
communicator-certified state/rate fingerprints, derivative fields, and all
eight rank-local mesh revision stamps. These are discrete operator-stage
moments; they are explicitly not work, power, penalty energy, dissipation, or
accepted endpoint values.

Registration fails closed when the related fluid polynomial order exceeds the
displacement-primary order supported by the current boundary quadrature. The
Application installs the observer only for temporal-order-one Backward Euler,
the explicit DG0 route, or first-order generalized-alpha. Candidate rejection
clears pending moments, commit-ready does not publish them, and the first
accepted callback action publishes only after system and history acceptance.
A registered measurement on any unsupported scheme now fails collectively
before the transient TimeLoop starts instead of silently omitting numeric
history. The later post-maintenance symbolic publisher is unchanged, so the two
histories are truthful but not one atomic transaction.

New source fixtures cover the exact backend state/rate alias, rank-divergent
stencil rejection before ghost refresh, all-eight-epoch stage capture before a
later nonlinear-done mutation, three-policy measurement registration, the
quadrature preflight, serial raw-moment staging, accepted commit, replay, and
discard. A new two-rank source fixture also stages the four global moments,
retains distinct rank-local eight-epoch stamps, exercises commit/replay/discard,
and injects rank-only state, rate, and declaration drift. None of these latest
measurement edits has been compiled or executed, and no end-to-end Application
measurement-history integration test has run.

This partial raw-normal history still omits vector mesh velocity, normal and
tangential projection norms, prescribed target error, fitted normal error
reporting at the accepted endpoint, exact mesh/fluid/net surface work,
serialization/restart, fault-complete post-accept recovery, and cross-history
atomicity. It therefore does not satisfy method exit 4, change the frozen
matrix, or support WP-9, Q4, or any audit checkbox.

## Open method exits

The matrix keeps the following method work open:

1. Expand the clean matrix beyond the present 32-test source-contract slice
   to the complete operator-stage history, accepted transaction, and backend
   ownership contract.
2. Derive and freeze dimensional, mesh, time-step, and polynomial-order
   scaling for the prescribed tangential penalty.
3. Derive the dimensional and mesh-order scaling of the surface-smoothing
   weight and establish consistency, stability, and surface-work arguments
   for the coupled fluid and mesh rows.
4. Extend the executed full, normal, and tangential mesh-velocity moments with
   target error, normal-kinematic error, and surface work at accepted steps.
5. Serialize those histories and prove restart continuity.
6. Extend the existing two-rank moment/revision regression to rotation,
   numbering, and representative MPI partition equivalence.
7. Expand the frozen capability exclusions beyond the executed surface-stress
   and fitted contact-model cases to every supported registration order and
   boundary combination.
8. Promote the current normal-ownership prerequisite into the frozen matrix,
   freeze the current consumer-bound accepted provenance tests, execute the
   complete compatibility outcomes in an integrated build, exercise the
   Application fingerprint on all enabled distributed backends, and close the
   pre-publication failure/recovery gap.
9. Freeze geometric-conservation, phase-volume, surface-work, and mesh-quality
   metrics before physical campaigns.

## Open physical ALE campaigns

All required physical campaigns remain unclaimed:

- flat translating ALE interface;
- prescribed tangential shear; and
- fitted sloshing.

They must report policy-specific boundary velocity, normal kinematic error,
geometric-conservation error, phase-volume error, surface work, and mesh
quality. Translation and shear require rotation plus spatial and temporal
refinement. Sloshing requires at least three spatial and temporal levels,
both registration orders, and representative MPI partitions.

## Closure rule

No low-level test count, schema-1 regression, fitted example, or policy
declaration can close FSR-10, FSR-11, WP-9, or Q4. Closure requires the open
method exits and all physical campaigns to pass at one immutable source
revision with complete provenance.

## 2026-08-30 clean prerequisite and projection checkpoint

Commit `afb0f65788215e38e70a170aa108d037527c4248` extends the accepted
operator-stage record from a scalar normal-gap view to the full squared
mesh-velocity moment and its orthogonal normal and tangential components. The
published record includes all three RMS values and the residual of

\[
\int_{\Gamma_f}|w_m|^2\,dS
-\int_{\Gamma_f}|w_m\mathbin{\cdot}n|^2\,dS
-\int_{\Gamma_f}|(I-n\mathbin{\otimes}n)w_m|^2\,dS=0.
\]

Hash-bound `amarsden` job `41330972` completed with exit `0:0` in
`00:00:27`, using one node, two tasks, 6 GB, and a batch peak resident set of
353,780 KiB. It passed two focused serial tests, all 204
`MovingDomainPhysics` tests, and the two-rank accepted-history regression. The
serial projection residual was `1.0408340855860843e-17`; the two-rank result
was `1.734723475976807e-18`. Both records retained the expected complete
global moments, while the distributed case retained distinct rank-local
geometry revisions.

Commit `6213ef09988a1e364a62ab20d5cd174c57da49f6` freezes launcher-isolated
single-rank execution for the prerequisite wrapper. Clean-source Release job
`41333535` completed with exit `0:0` in `00:30:20`, using one node, two CPUs,
6 GB, and a batch peak resident set of 3,372,188 KiB. The Application group
passed 4/4 tests and the Physics group passed 28/28 tests with no failures,
errors, disabled tests, unexpected tests, or missing tests. The matrix stayed
byte-identical at SHA-256
`46225d9c90e71c90881725d4e871466f7a0e19e04a4309627b82e376583715d9`,
and every result checksum verified.

The immutable non-closure record is
[`free_surface_wp9_fitted_ale_prerequisite_20260830_41333535/record.json`](qualification_logs/free_surface_wp9_fitted_ale_prerequisite_20260830_41333535/record.json).
This checkpoint establishes only the stated prerequisite and projection
telemetry. Penalty/smoothing scaling, target and kinematic error, surface work,
restart continuity, broader partition equivalence, fault-complete publication,
and the flat-translation, prescribed-shear, and sloshing campaigns remain
open. Therefore WP-9, FSR-10, FSR-11, and Q4 remain unchecked.

## Source evidence map

- XML boundary allowlist:
  `Code/Source/solver/Parameters.cpp`.
- Input translation:
  `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp`.
- Fitted preflight, operators, and effective provenance:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`.
- Central ownership, consumer binding, and accepted transaction:
  `Code/Source/solver/FE/Systems/FESystem.{h,cpp}`.
- Backend-owned accepted-state rows:
  `Code/Source/solver/FE/Backends/`.
- Steady/transient accepted-state fingerprint and publication ordering:
  `Code/Source/solver/Application/Core/ApplicationDriver.cpp`.
- Mesh-motion consumers:
  `Code/Source/solver/Physics/Formulations/MeshMotion/`.
- XML regressions:
  `Code/Source/solver/Application/Tests/Unit/test_EquationTranslator.cpp`.
- Physics regressions:
  `Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp` and
  `Code/Source/solver/Physics/Tests/Unit/test_NavierStokesLegacyBCs.cpp`.
- Two-rank transaction regression:
  `Code/Source/solver/Physics/Tests/Unit/test_SharpBoundaryOperatorMPI.cpp`.
