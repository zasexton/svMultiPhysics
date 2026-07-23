# WP-9 fitted-ALE free-surface architecture and qualification boundary

Status: low-level prerequisite implemented; WP-9, Q4, and physical
fitted-ALE qualification remain open.

Audit basis: repository HEAD
`963cb4c256a2e7db7dede519ea8dafe5a52aacd0`, plus the dirty tracked
source state reviewed on 2026-07-23. A result produced from a dirty source
tree does not establish source-to-binary correspondence.

This record updates the WP-9 boundary from
`Documentation/free_surface_boundary_unfitted_audit_20260720.md`. It does
not claim FSR-10, FSR-11, WP-9, or Q4 closure.

## Verdict

The current schema-2 fitted capability is narrow and fail closed. A fitted
free surface is accepted as supported only when all of the following are
true:

- ALE is enabled;
- mesh velocity comes from a coupled displacement unknown;
- the normal policy is `MatchFluidNormalVelocity`;
- normal enforcement is explicitly `Penalty` or `Nitsche`; and
- the tangential policy is `Prescribed`.

`Free` and `SmoothingOnly` do not have a consumed fitted mesh-motion
operator in the supported schema-2 capability. Both are rejected before
fields, forms, constraints, or tangential-owner declarations are added.
Prescribed mesh-velocity data, a missing normal relation, and explicit
normal enforcement `None` are rejected at the same pre-mutation boundary.

Schema 1 retains older configurations for regression and migration only.
It requires explicit legacy opt-in and its effective configuration is labeled
`unqualified_explicit_legacy`. A schema-1 pass is not evidence for the
schema-2 supported capability.

This is enough to freeze a truthful prerequisite harness. It is not enough
to close WP-9 or Q4. `Free` and `SmoothingOnly` remain unresolved, method
exits remain open, and none of the required physical ALE campaigns has run.

## Schema and policy contract

| Input mode | Tangential policy | Registration outcome | Qualification |
|---|---|---|---|
| schema 2, coupled displacement, explicit normal enforcement | `Prescribed` | central owner plus projected weak tangential operator | `supported_configuration_envelope` |
| schema 2 | `Free` | rejected before system mutation | unsupported |
| schema 2 | `SmoothingOnly` | rejected before system mutation | unsupported |
| schema 2, prescribed mesh-velocity data | any fitted policy | rejected before system mutation | unsupported |
| schema 2, normal policy or enforcement `None` | any fitted policy | rejected before system mutation | unsupported |
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

`Free` and `SmoothingOnly` still exist in the input vocabulary and in
schema-1 regression coverage. They are not schema-2 aliases for a supported
operator. Supporting either in schema 2 requires a real consumed operator
contract or an explicit policy consolidation.

## Ownership and truthful provenance

`FESystem::declareMeshTangentialBoundaryPolicy` is the central ownership
registry. A second declaration for the same displacement field and boundary
marker is rejected. The declaration is ownership metadata; it is not by
itself evidence that an operator consumed the policy.

The effective configuration therefore derives provenance from system state:

- `tangential_mesh_owner` is emitted only when the matching central
  declaration exists;
- `policy_consumed=true` requires the matching prescribed tangential weak
  descriptor for the displacement field and boundary marker;
- `operator_tag` is emitted only for that consumed path;
- `operator_source` comes from the exact matching descriptor; and
- `policy_qualification` distinguishes the supported schema-2 envelope from
  explicit unqualified legacy operation.

An unconsumed path reports `policy_consumed=false` and null operator tag and
source. The owner is not fabricated from a module-name constant.

The current schema-2 `Prescribed` path is the only path for which this record
claims a consumed supported tangential operator.

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

Application-level regressions parse complete `Add_equation` XML, build the
module, and inspect the central declarations or effective free-surface
artifact. They also prove that explicit `None` plus a kinematic penalty fails
without mutating the system.

## Boundary-local enforcement and conflicts

Each fitted boundary owns its normal enforcement, penalty or Nitsche
parameters, prescribed tangential target, and tangential penalty. Generic
weak velocity conditions retain their separate module-level policy.

The low-level suite covers boundary-local fitted parameters, registration
order, projection under rotation, multiple markers, and conflicts between a
mesh-motion owner and the free-surface owner. The compatibility matrix for
separate mesh-motion normal constraints and the fluid normal relation is not
yet frozen.

## Capability exclusions

The supported fitted slice continues to exclude:

- fitted `SurfaceStress`;
- fitted prescribed contact angle;
- fitted dynamic contact angle;
- prescribed mesh-velocity data;
- schema-2 `Free` and `SmoothingOnly`;
- schema-2 normal policy or enforcement `None`; and
- unfitted active-domain and cut-stabilization controls on a fitted surface.

Pinned fitted contact lines remain contained to coupled ALE. A dedicated
pre-mutation fitted dynamic-contact regression and complete capability
provenance remain open method work.

## Frozen prerequisite evidence

The matrix
`tests/cases/fluid/free_surface_wp9_fitted_ale_qualification_matrix.json`
is byte-frozen at SHA-256
`64c67f09b3f21a856efa213eb8f0e9ffe0680aa20dde0274594f493807344965`.

The wrapper
`tests/cases/fluid/run_free_surface_wp9_fitted_ale_qualification.py`
checks that digest before parsing the matrix. It partitions the frozen tests
between the Application and Physics binaries and permits only
`low_level_prerequisite`. It rejects FSR-10, FSR-11, WP-9, Q4, and general
fitted-ALE qualification claims before execution arguments are processed or
an artifact directory is created.

The frozen evidence includes these source-contract regressions:

- `EquationTranslatorMeshMotion.XmlAliasesReachTangentialPolicyModuleRegistration`;
- `EquationTranslatorFreeSurface.XmlTangentialPenaltyAliasesReachTruthfulFittedModule`;
- `EquationTranslatorFreeSurface.XmlExplicitNoneCannotBePromotedByKinematicPenalty`;
- `MovingDomainPhysics.FittedFreeSurfaceQualifiedContractRejectsBeforeMutation`;
- `MovingDomainPhysics.FittedFreeSurfaceTangentialPoliciesRegisterCoupledMeshOwnership`;
- `MovingDomainPhysics.FittedFreeSurfaceLegacyPrescribedDataReportsUnconsumedPolicy`;
- `MovingDomainPhysics.NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults`;
- current prescribed projection and coupled setup tests;
- explicit schema-1 legacy operator regressions;
- shared mesh-motion consumer and ownership-conflict tests; and
- fitted contact and surface-stress capability-boundary tests.

A passing matrix result is reported only as a prerequisite/non-closure
result. It cannot change any closure disposition.

## Open method exits

The matrix keeps the following method work open:

1. Implement consumed schema-2 operators for `Free` and `SmoothingOnly`, or
   consolidate the policies.
2. Derive and freeze dimensional, mesh, time-step, and polynomial-order
   scaling for the prescribed tangential penalty.
3. Establish consistency, stability, and surface-work arguments for the
   coupled fluid and mesh rows.
4. Record measured boundary mesh velocity, normal and tangential projection,
   target error, normal-kinematic error, and surface work at accepted steps.
5. Serialize those histories and prove restart continuity.
6. Establish rotation, numbering, and representative MPI partition
   equivalence.
7. Add an explicit fitted dynamic-contact pre-mutation rejection and complete
   capability provenance.
8. Freeze compatibility outcomes for mesh-motion normal constraints combined
   with fluid kinematic policies.
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

## Source evidence map

- XML boundary allowlist:
  `Code/Source/solver/Parameters.cpp`.
- Input translation:
  `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp`.
- Fitted preflight, operators, and effective provenance:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`.
- Central ownership:
  `Code/Source/solver/FE/Systems/FESystem.{h,cpp}`.
- Mesh-motion consumers:
  `Code/Source/solver/Physics/Formulations/MeshMotion/`.
- XML regressions:
  `Code/Source/solver/Application/Tests/Unit/test_EquationTranslator.cpp`.
- Physics regressions:
  `Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp` and
  `Code/Source/solver/Physics/Tests/Unit/test_NavierStokesLegacyBCs.cpp`.
