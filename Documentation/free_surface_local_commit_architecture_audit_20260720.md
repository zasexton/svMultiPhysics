# Free-surface local-commit architecture audit

Date: 2026-07-20

Branch: `issue-449-modern-mesh-core`

Audit base: `0ec8a3017958aa35e39270c7e0a93bc6c0740c68` (`origin/issue-449-modern-mesh-core`)

Audit head: `a1d3842954b9d9d6818663c62fd4b775a5767d1e`

## Executive assessment

The concern motivating this review is valid. The local commit range contains useful generic cut-geometry work and Navier–Stokes free-surface formulations, but their ownership boundaries are currently blurred in both directions:

- The 4,538-line `IncompressibleNavierStokesVMSModule.cpp` owns configuration overrides, pressure initialization, cut-domain discovery, geometry differentiation, contact-line handling, stabilization, diagnostics, and system installation in addition to the governing flow equations.
- Generic FE assembly and Newton code knows about pressure, velocity, PSPG, and Navier–Stokes component tags. Some of that code changes assembled matrices and solve-time systems; it is not merely diagnostic.
- New FE geometry types are named `FreeSurface...` and carry `Contact`/wetting terminology even though their algorithms are useful for any implicit interface.
- The new “authoritative” geometry snapshot is not yet authoritative in production. The builder is called by FE tests, while production construction can continue through older domain-by-domain paths.
- Newly added level-set controls are exposed through input/options without a committed production lifecycle that executes them. In particular, the algebraic extension kernel is installed without any production code that builds its required frozen map.
- The nine commits mix free-surface work with a very large basis implementation rewrite, JIT history and runtime-policy support, Coriolis forcing, solver diagnostics, validation assets, and other independent changes.

The first cleanup should therefore not be a mechanical split of one long physics file. The priority is to remove hidden physics policy from generic assembly/solver layers, establish one generic implicit-geometry authority, and only then decompose the Navier–Stokes formulation around stable interfaces.

No cleanup is performed by this audit.

## Scope and method

The audited range is:

```text
origin/issue-449-modern-mesh-core..HEAD
```

It consists of nine local commits:

1. `87928980` — changes to unfitted mesh support with level set methods
2. `7aaadf5a` — improving the level-set methods for unfitted meshes and correcting errors for MMS verification
3. `2846445f` — Establish authoritative sharp free-surface geometry
4. `7a33bb5a` — Map generated contact rules by codimension
5. `0b8b4724` — Contain free-surface Nitsche policy per boundary
6. `487b0be0` — Certify unfitted cut stability across spectral backends
7. `ac8231eb` — Migrate supported free-surface inputs to one-way extension
8. `30f79c52` — Harden level-set transport and extension setup
9. `a1d38429` — Validate level-set input before registration

Range size:

- 166 files changed
- 49,681 insertions and 5,869 deletions
- 110 FE files, 11 Physics files, 7 Application files, 2 parameter files, 1 documentation file, and 35 validation-case files
- The first commit alone changes 120 files with 31,330 insertions and 5,592 deletions.

The worktree contains extensive uncommitted and untracked work. To avoid attributing that work to these commits, this audit inspected committed Git objects and the range diff, not the current on-disk versions of modified source files. Symbol names and line references below refer to audit head `ac8231eb`. Current uncommitted changes are deliberately out of scope.

This is an architecture and maintainability audit. It identifies a few behavior risks where they arise directly from misplaced responsibilities, but it is not a replacement for numerical-formulation verification.

Priority labels used below:

- **P0**: resolve before treating this range as merge-ready.
- **P1**: high-value structural cleanup needed for a reusable design.
- **P2**: follow-on simplification after the ownership boundaries are fixed.

## Intended ownership boundary

| Responsibility | Intended owner | Must not know about |
|---|---|---|
| Bulk incompressible momentum, continuity, VMS choices, physical free-surface/contact laws | `Physics/Formulations/NavierStokes` | marker hashing, root reconstruction, solver matrix surgery |
| Implicit-field lifecycle, transport, reinitialization, extension, revisioning | generic `FE/LevelSet` or `FE/ImplicitGeometry` | pressure, liquid, contact angle, Navier–Stokes |
| Cut reconstruction, interface/boundary intersection, quadrature, mapping, ownership | generic FE geometry/quadrature/interfaces | free-surface semantics and flow field names |
| Insertion of local residual/Jacobian contributions | `FE/Assembly` | pressure, velocity, PSPG, component-name conventions |
| Newton iteration and linear solve orchestration | `FE/TimeStepping` | active pressure support and free-surface topology |
| Input translation and runtime options | `Application` and Physics registrars | mathematical implementation details |
| Instrumentation | observers, diagnostic utilities, tests, or tools | permission to silently modify production operators |

The desired dependency direction is:

```text
Application/configuration
    -> Physics/NavierStokes formulation and boundary models
    -> FE implicit-geometry lifecycle

Physics/NavierStokes
    -> generic FE forms, measures, fields, and geometry handles

FE assembly and Newton solver
    -> generic typed metadata/callbacks only
    -X-> no dependency on pressure/PSPG/free-surface names or policies
```

## Findings

### ARC-01 — Generic assembly performs pressure/PSPG-specific matrix transformations (P0)

Files:

- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- `Code/Source/solver/FE/Systems/FormsInstaller.*`

Evidence:

- `StandardAssembler.cpp` gains roughly 3,000 lines in this range.
- `DirectPspgTopologyPolicyStats`, `applyDirectPspgLocalSchurCompletion`, `applyDirectPspgLocalEdgeBalance`, and `applyCutVolumeDirectPspgTopologyPolicy` are implemented in the generic assembler.
- The policy selects contributions by string comparisons such as `test_field_name == "pressure"`, `trial_field_name == "pressure"`, operator tags, and the source tag `navier_stokes_vms_pspg_pressure_gradient`.
- The selected paths mutate local matrices during cut-volume assembly. These are production behavior changes controlled by environment variables, not passive logging.
- `AssemblyDiagnosticContext` is described and used for both diagnostics and policies. The name therefore hides a mutating contract.
- `SystemAssembly.cpp` propagates the context only along selected cut-volume paths, so it is not a complete, general provenance mechanism.
- At audit-head lines 2198–2212, the `insertion_group.size() == 1u` branches are identical. The context is also copied into both `FusedCellTerm` and a parallel context vector.

Why this matters:

The assembler should assemble the forms it is given. Recognizing a particular PDE block and inventing edges or Schur-like completion inside it makes the result depend on field spelling, installation tags, environment state, and whether a fused path happens to preserve the metadata. Other mixed formulations could accidentally trigger the behavior, while equivalent Navier–Stokes formulations with different names could miss it.

Future cleanup:

1. Remove Navier–Stokes/pressure selection and topology mutation from `StandardAssembler`.
2. If the completion is mathematically required, represent it as an explicit Physics-installed stabilization form or a typed `LocalContributionTransform` registered by the formulation.
3. Separate immutable contribution metadata from behavior:
   - typed field/operator/component identifiers, not string names;
   - a read-only `AssemblyObserver` for instrumentation;
   - an explicitly registered transform interface for the rare case where a local matrix transformation is justified.
4. Make metadata propagation uniform across fused, unfused, cut, and full-cell paths, or do not claim it is general provenance.
5. Delete the duplicate branch and parallel storage after ownership is settled.

Acceptance condition: a repository search in `FE/Assembly` should find no pressure, PSPG, Navier–Stokes, or free-surface policy, and disabling diagnostics should never change an assembled operator.

### ARC-02 — The generic Newton solver edits active-pressure systems (P0)

Files:

- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.h`

Evidence:

- `NewtonSolver.cpp` changes by +7,782/-1,474 lines and is 13,629 lines at the audit head.
- It contains pressure-specific data structures and scans beginning around `ActivePressureSupportRankRow` and `ActivePressureUpdateSupportSummary`.
- `applyActivePressureSupportRankClamp` and `applyActivePressureGraphCompletion` modify the matrix/RHS before the linear solve.
- The logic discovers fields by pressure/velocity names and is configured by process-global environment variables.
- The solver also contains large blocks for matrix support sampling, row provenance, update graph construction, bordered-system reduction, Jacobian checking, and backend-specific diagnostics.

Why this matters:

A nonlinear solver may expose generic null-space handling, constraints, or pre-solve transforms, but it must not infer the physical meaning of a field. Pressure support is a property of the discretization chosen by Physics and the active-domain constraint strategy. Encoding it in Newton makes solver behavior non-composable and prevents reuse for other saddle-point systems.

Future cleanup:

1. Remove pressure-named clamping and graph completion from the generic Newton implementation.
2. First determine whether each transformation is:
   - a required mathematical stabilization, in which case Physics must install it explicitly;
   - a generic constraint/null-space operation, in which case it needs a typed solver API and no field-name lookup; or
   - an experiment/diagnostic, in which case move it to a test/tool and keep it out of production solve behavior.
3. Split `NewtonSolver.cpp` into a small iteration state machine plus focused generic components such as `NewtonDiagnostics`, `JacobianChecker`, `LinearSolveDiagnostics`, and `BorderedSystemReduction`.
4. Require explicit options on the system/solver object rather than reading environment variables in the solve path.

Acceptance condition: `NewtonSolver` operates on algebraic systems and declared constraints/null spaces only; it contains no knowledge of pressure, velocity, PSPG, cut cells, or free surfaces.

### ARC-03 — The “authoritative” geometry snapshot is physics-named and not production-authoritative (P0)

Files:

- `Code/Source/solver/FE/Interfaces/FreeSurfaceGeometrySnapshot.*`
- `Code/Source/solver/FE/Assembly/CutIntegrationContext.h`
- `Code/Source/solver/FE/Tests/Unit/Geometry/test_GeneratedActiveBoundaryDomain.cpp`

Evidence:

- FE types are named `FreeSurfaceGeometrySnapshot`, `FreeSurfaceGeometryRevision`, `FreeSurfaceGeometrySnapshotPolicy`, and `FreeSurfaceGeometrySnapshotCache`.
- Generic rule roles include the physics term `Contact`; ledgers and accessors similarly use contact/free-surface language.
- Outside the implementation/header, `buildFreeSurfaceGeometrySnapshot` is called only by FE tests at this audit head. No production path builds it.
- `CutIntegrationContext::addFreeSurfaceGeometrySnapshot` copies the snapshot’s independently built interface, intersection, active-boundary, and volume rules into existing mutable collections. The context therefore retains two representations rather than exposing immutable views of one authority.
- Legacy `addGeneratedInterfaceDomain` and related calls remain available and bypass snapshot binding.
- `assertFreeSurfaceGeometrySnapshotCurrentForMarker` returns quietly for unbound markers and compares copied rule provenance only against `source_value_revision`. It does not enforce the complete revision tuple represented by `snapshot_revision_key` (mesh geometry/topology, ownership, numbering, quadrature policy, source layout/value, and snapshot identity).

Why this matters:

The new type cannot guarantee a single geometry if production does not use it, if older insertion paths bypass it, or if currentness checks validate only one component of the revision. Its FE-level name also prevents clean reuse for immersed solids, phase-change fronts, embedded sources, material interfaces, or other implicit-boundary problems.

Future cleanup:

1. Generalize the concept to `ImplicitInterfaceGeometrySnapshot` or `CutGeometrySnapshot`.
2. Use generic roles such as negative/positive volume, embedded interface, negative/positive exterior trace, and interface–boundary intersection.
3. Have one generic builder atomically construct all related domains from one implicit source, evaluator, backend, policy, and complete revision.
4. Make `CutIntegrationContext` hold immutable snapshot handles/views instead of copying the rule arrays into a second authority.
5. Either retire direct domain-by-domain production insertion or mark it explicitly unverified and keep it out of the free-surface path.
6. Validate complete revision identity. A marker with no bound snapshot should fail when a snapshot-backed path is required.
7. Move snapshot tests to a dedicated generic test file rather than appending them to active-boundary tests.

Acceptance condition: production free-surface setup creates exactly one generic snapshot per source revision, and every volume/interface/exterior/intersection consumer obtains geometry from that snapshot.

### ARC-04 — Level-set options are advertised without a complete production lifecycle (P0)

Files:

- `Code/Source/solver/FE/LevelSet/LevelSetTransport.*`
- `Code/Source/solver/FE/LevelSet/LevelSetVelocityExtensionConstraint.*`
- `Code/Source/solver/FE/LevelSet/LevelSetOptions.h`
- `Code/Source/solver/FE/LevelSet/LevelSetReinitialization.cpp`
- `Code/Source/solver/Application/Translators/LevelSetEquationTranslator.cpp`
- `Code/Source/solver/Application/Core/ApplicationDriver.cpp`
- `Code/Source/solver/Physics/Core/PhysicsModule.*`

Evidence at the audit head:

- When `algebraic_extension_source_field_name` is set, `installLevelSetTransport` installs a `LevelSetVelocityExtensionConstraintKernel` whose `assemble` method throws unless a frozen map has been supplied.
- `setFrozenRows` and `invalidateFrozenMap` are called only by `test_LevelSetTransport.cpp`. There is no production map builder, refresh call, or lifecycle hook in the committed tree. The Application translator nevertheless promotes supported wet-extension input to this coupled algebraic path, making an uninitialized kernel reachable from production input.
- `evaluateLevelSetTransportSafety` and `applyLevelSetBoundPreservingLimiter` have no production callers. Their options are parsed and serialized, but enabling the option does not place the check/projection into accepted-step processing.
- New guards such as `max_zero_set_displacement`, `minimum_relative_volume_error`, `maximum_interface_displacement_fraction`, and `maximum_cumulative_interface_displacement_fraction` are validated and included in the effective-configuration JSON, but the committed reinitialization and volume-correction calls do not consume them.
- `LevelSetOptions.h` now says zero/negative `preserve_band_width` selects a topology-preserving algorithm, while `LevelSetReinitialization.cpp::resolvePreserveBandWidth` still treats zero as no preserved band and negative as the old automatic band. Validation rejects positive values even though the implementation still accepts them.
- `make_level_set_effective_configuration` claims a maintenance ordering and a fixed bounded map policy, but `EffectiveConfigurationArtifact` has no production consumer outside the adapter; only tests retrieve it.
- The artifact and generic level-set translator use `wet`, `one_phase`, `physical_momentum`, `Free_surface_*`, and contact-line terminology. Those describe a particular application of level-set transport, not the generic transport capability.

Why this matters:

Configuration provenance is only trustworthy when every reported control is actually active. An installer that registers a kernel requiring an externally supplied map must also register or require the lifecycle that supplies it. Otherwise a configuration can validate and produce a reassuring artifact yet fail at assembly or silently omit requested safety behavior.

Future cleanup:

1. Introduce one explicit `LevelSetEvolutionController` (or generic implicit-field lifecycle) that owns the sequence: build/freeze extension map, nonlinear solve, transport safety check, candidate limiter, reinitialization, volume correction, and geometry-snapshot refresh.
2. Make installation fail before mutating `FESystem` if a requested feature has no registered lifecycle implementation.
3. Generalize the algebraic relation to a source/extension field map with negative/positive or source/target-side terminology. A Navier–Stokes translator may call the source field velocity and the active phase wet.
4. Carry one map/snapshot revision through the constraint kernel and require the expected level-set/source revision at assembly, not just a caller-provided integer.
5. Either implement every parsed guard in the maintenance transaction or remove the input until it is supported. Align option documentation, validation, and implementation in the same change.
6. Produce structured configuration/provenance from the effective runtime objects and have Application persist it. Avoid a 140-line hand-built JSON string and hard-coded application capability labels in the generic translator.
7. Add an end-to-end test that starts from supported input, builds the map through production lifecycle code, completes assembly/one accepted step, and verifies the recorded effective policy. Tests should not populate production rows manually.

Acceptance condition: every accepted input has a complete production execution path, every effective-configuration field reflects behavior actually used, and extension/maintenance revisions cannot be stale or absent at assembly.

### ARC-05 — The Navier–Stokes formulation owns too many independent subsystems (P1)

Files:

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h`
- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp`

Evidence:

`IncompressibleNavierStokesVMSModule.cpp` is 4,538 lines. Its approximate responsibility bands are:

| Audit-head area | Responsibility |
|---|---|
| lines 72–363 | environment parsing and diagnostic overrides |
| lines 383–1,362 | field compatibility, initial conditions, active pressure support, hydrostatic initialization, constraint validation |
| lines 1,363–3,376 | free-surface validation, cut stabilization, domain construction, velocity extension, shape tangents, curvature, contact lines, dynamic and kinematic forms |
| lines 3,378–4,533 | field creation, bulk flow/VMS construction, BC installation, diagnostics, and formulation installation |

Specific signs of mixed ownership:

- `registerOn` reads many environment-controlled PSPG variants and scales before it even creates fields.
- The final section installs twelve separate `equations_diagnostic_ns_*` operators.
- The direct PSPG pressure-gradient contribution is installed separately with a string component tag so generic assembly policies can recognize it.
- Shape-tangent construction manually combines level-set lookup, normal/tangent variation, surface-measure variation, and cut-volume variation inside the flow formulation.
- `applyFreeSurfaceVelocityExtension` adds a Laplacian of the physical velocity over the inactive domain, including embedded ramp width `1e-6` and cut-cell floor `1e-3` constants. If this is ghost/fictitious-domain stabilization, the name is misleading; if it is intended as true field extension, it should not alter the physical momentum equation under that abstraction.
- The 439-line options header nests free-surface geometry configuration, contact laws, cut stabilization, extension, all conventional BCs, ALE, outlet models, pressure initialization, and rotating-frame options in one aggregate.

Recommended decomposition:

- Keep the main module focused on field declaration, bulk momentum/continuity residuals, the VMS definition, and high-level installation orchestration.
- Extract Physics-owned pieces into focused modules, for example:
  - `NavierStokesBulkForms.*`
  - `FreeSurface/FreeSurfaceOptions.h`
  - `FreeSurface/FreeSurfaceBoundaryValidator.*`
  - `FreeSurface/FreeSurfaceBoundaryForms.*`
  - `FreeSurface/FreeSurfaceContactLineForms.*`
  - `FreeSurface/FreeSurfaceCutStabilization.*`
  - `NavierStokesPressureInitialization.*`
- Move generic geometry/measure differentiation out of Physics. Physics should request the tangent of a generic geometry-backed form, not implement point location and geometry derivatives itself.
- Keep surface tension, external pressure, contact-angle laws, Nitsche choices, and the coefficient/form of Navier–Stokes ghost stabilization under Physics; only their geometry and measures are generic.
- Separate `FreeSurfaceVelocityExtension` into one unambiguous concept:
  - a generic auxiliary-field extension service under FE/field transfer; or
  - a clearly named Navier–Stokes fictitious-domain stabilization form under Physics.
- Split the options into bulk, ALE, free-surface, conventional BC, outlet, and initialization option groups. The parser may compose them into the final module configuration.
- Keep diagnostics out of the governing-equation file. Expose explicit component decomposition or observers for tests instead of installing diagnostic PDE operators through environment switches.

Target outcome: `registerOn` should read like a short orchestration method—declare fields, build the bulk residual, ask boundary-model objects for their residuals/constraints, and install the result.

### ARC-06 — Generated interface infrastructure is only partially physics-agnostic (P1)

Files:

- `Code/Source/solver/FE/Interfaces/GeneratedActiveBoundaryDomain.*`
- `Code/Source/solver/FE/Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.*`
- `Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h`
- `Code/Source/solver/FE/Quadrature/ImplicitBoundaryIntersectionQuadrature.*`

Positive direction:

- Interface–boundary intersection quadrature, active exterior traces, codimension-aware mapping, stable ownership, and source revisioning belong in generic FE infrastructure.
- The class name `GeneratedInterfaceBoundaryIntersectionDomain` and the quadrature location are appropriately physics-neutral.

Remaining leakage and cleanup:

- Internal APIs and summaries still use `contact`, `active_contact_fragment_count`, `orphan_contact_fragment_count`, wet/dry wall language, and `source_contact_stable_ids`. Rename these to intersection/side-neutral terms. Physics may wrap an intersection as a contact line.
- Builders require `LevelSetInterfaceDomain` and `LevelSetInterfaceSource`. For maximum reuse, define an `ImplicitInterfaceDomain`/`ImplicitFieldSource` contract and make level set one adapter. Remaining level-set-specific algorithms can stay under `FE/LevelSet`.
- The scalar-field overload of `buildGeneratedInterfaceBoundaryIntersectionDomain` accepts `GeneratedInterfaceBoundaryIntersectionScalarField` and immediately discards it. Remove the misleading overload/type until it participates in the contract.
- Active-boundary clipping reconstructs roots independently from nodal scalar values, then compares them with an intersection domain. It should use the same reconstruction backend/evaluator that produced the retained interface or be constructed atomically by the snapshot builder.
- `LinearCorner`-only support is a valid initial capability, but it should be represented by a typed backend capability rather than repeated string checks.
- Request validation and revision fields are duplicated across interface, intersection, and active-boundary requests. Consolidate them into one source/revision/policy object.

Acceptance condition: generic headers can describe an embedded interface and its exterior traces without mentioning free surfaces, liquids, wet/dry regions, contact physics, pressure, or Navier–Stokes.

### ARC-07 — Geometry construction, topology, and mapping have multiple sources of truth (P1)

Files:

- `Code/Source/solver/FE/Geometry/CutQuadrature.*`
- `Code/Source/solver/FE/Geometry/CutQuadratureMapping.*`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/LevelSet/LevelSetVolume.cpp`
- the three generated-domain/snapshot implementations under `FE/Interfaces`
- `Code/Source/solver/FE/Quadrature/ImplicitBoundaryIntersectionQuadrature.cpp`

Evidence:

- Face-corner topology switches, small-vector operations, root interpolation, and orientation logic are repeated in active-boundary, intersection, snapshot, quadrature, and level-set-volume code.
- The new `CutQuadratureMapping` service is consumed by the snapshot/codimension work, but `StandardAssembler.cpp` still contains private `referenceToPhysicalVector`, `inverseTransposeReferenceNormal`, and `remapCutInterfaceGeometry` implementations.
- `LevelSetVolume.cpp` has another tetrahedral decomposition and `clipped_volume` reconstruction fallback rather than consistently measuring the generated cut rules.
- Builders assemble related geometry in stages, which allows surface, volume, boundary trace, and intersection representations to drift.

Future cleanup:

1. Create one reference-element subentity/topology service for corner/edge/face connectivity and orientation.
2. Make `CutQuadratureMapping` the sole reference-to-physical mapping implementation used by assemblers and geometry builders.
3. Make volume diagnostics and conservation correction consume the same generated rules/snapshot used for assembly.
4. Put root reconstruction behind a backend interface that can produce volume, interface, trace, and intersection views consistently.
5. Retain specialized fast paths only behind parity tests against the common reference implementation.

### ARC-08 — Core numerical behavior is configured through hidden process-global state (P1)

Across committed source alone, the range introduces or references 174 distinct `SVMP_*` tokens. They cover diagnostics, VMS term selection/scaling, PSPG boundary terms, topology completion, active-pressure clamps and graph completion, cut-volume pruning, time stepping, solver behavior, and application guards.

The most direct generic-layer example is `CutIntegrationContext::minGeneratedCutVolumeFraction()`, which reads `SVMP_MIN_GENERATED_CUT_VOLUME_FRACTION` into a function-local static. The snapshot separately owns a minimum-retained-volume policy and rejects disagreement with that hidden global value.

Why this matters:

- Inputs are not captured by the simulation configuration or result provenance.
- Two systems in one process cannot choose different policies.
- Tests can contaminate one another through process state/static caching.
- Diagnostics and mathematical behavior are difficult to distinguish.
- Reproducing a result requires knowing an undocumented external environment.

Future cleanup:

- Parse supported runtime choices once in Application/registrars and pass typed immutable options down.
- Put cut-rule retention in an explicit `CutRuleRetentionPolicy` recorded in snapshot provenance.
- Restrict environment variables to a thin developer-diagnostics adapter. An environment switch named “diagnostic” must not alter residuals, Jacobians, constraints, or matrices.
- Remove legacy aliases after the explicit configuration path is established.

### ARC-09 — ApplicationDriver contains physics-specific accepted-step policy (P1)

File: `Code/Source/solver/Application/Core/ApplicationDriver.cpp`

The added `logAcceptedPressureUpdateDiagnostic` block is roughly 265 lines and understands pressure fields, active vertices, level-set/cut state, thresholds, rejection/failure policy, and parallel reduction. It also obtains `MeshComm::world()` directly rather than using the communicator owned by the system/run.

Future cleanup:

- Move this into an optional accepted-step observer supplied by the Navier–Stokes/free-surface configuration.
- Give observers a generic system state view and the actual system communicator.
- Keep ApplicationDriver responsible for lifecycle and callback invocation, not pressure semantics.
- Separate unrelated start-time, last-step, and nonlinear-tolerance changes from the free-surface series.

### ARC-10 — Level-set code is in the right broad layer but should expose reusable services (P1)

Files under `Code/Source/solver/FE/LevelSet` are substantially closer to the desired ownership than the free-surface-named interface code. The added global IDs/ownership metadata, lifecycle revisioning, transport, reinitialization, and diagnostics are generally useful across physics.

Cleanup opportunities:

- Split the now 1,685-line `LevelSetTransport.cpp` into transport-form installation, field preflight, safety evaluation, bound-preserving projection, sampling, and parallel-reduction components. Its 2,978-line unit test should be split along the same boundaries.
- Keep level-set infrastructure free of pressure/free-surface aliases. For example, active-side constraints should accept field IDs and generic labels; Physics/Application can provide diagnostic names.
- Extract the nodal-DOF coordinate traversal used by reinitialization into a generic FE interpolation/field-transfer service rather than maintaining native-mesh special paths locally.
- Replace the independent `LevelSetVolume` clipping fallback with measurement of the authoritative generated cut geometry.
- Generalize one-way extension as an arbitrary source-field to target-field/component extension service under `FE/LevelSet` or `FE/FieldTransfer`. Application should only configure and schedule it.
- Keep level-set-specific source/evaluation code as an adapter to the broader implicit-interface geometry contract.

This separation would allow the same machinery to support thermal phase fronts, Darcy interfaces, immersed solids, reaction fronts, and other problems without importing Navier–Stokes semantics.

### ARC-11 — Cut-quadrature records permit incoherent states (P2)

`CutQuadraturePoint` now carries volume, interface-normal, boundary-normal, tangent, and codimension-two data in one broad record. `CutQuadratureRule` also combines `kind`, `side`, and a `geometric_dimension` that may use legacy inference. Provenance contains several raw strings for geometry mode/backend/fallback plus overlapping revision fields.

Future cleanup:

- Prefer tagged geometry payloads or distinct `VolumePoint`, `InterfacePoint`, and `IntersectionPoint` types.
- If a common record is retained, central factories and validators must make kind/dimension/frame combinations impossible to disagree.
- Replace backend/mode/fallback strings with enums or stable typed IDs.
- Store one complete `ImplicitGeometryRevision` rather than copying selected revision fields into each request and rule.

### ARC-12 — Generated marker identity is decentralized and collision-prone (P1)

The range uses independent hard-coded marker spaces for generated interfaces, intersections, and active boundaries (bases around 1,000,000, 2,000,000, and 3,000,000 with modulo ranges). Stable keys serialize `Real isovalue` using `std::to_string`, whose limited decimal representation can alias distinct floating-point values. Hashing into a fixed range can collide, and collision handling is inconsistent across domain kinds.

Future cleanup:

- Create one generic generated-domain identity/registry service.
- Use opaque typed IDs rather than exposing marker arithmetic to Physics.
- Hash a canonical bit representation of floating values, or better, key domains by the source/revision object plus a typed isovalue.
- Detect collisions at allocation and persist the identity in the snapshot rather than independently re-deriving it in every consumer.

### ARC-13 — The test layout mirrors the production coupling (P1)

`test_FreeSurfaceCutStability.cpp` is 4,023 lines but declares only a small number of large tests. It combines:

- generic small-cut aggregation and affine constraints;
- cut geometry and assembly behavior;
- Navier–Stokes pressure-control algebra;
- Young–Laplace/contact physics;
- spectral backend comparisons; and
- MPI/partition-invariance concerns.

Some tests enable environment variables that cause the production formulation to install diagnostic operators. That makes test instrumentation part of production mathematical setup.

Future split:

- FE constraints: aggregation and active-side constraints.
- FE cut integration: geometry, quadrature, mapping, and generic small-cut stability harness.
- Physics/Navier–Stokes: capillary balance, contact laws, pressure gauge, and formulation-specific stabilization.
- MPI/backend suites: partition and spectral-backend invariance.
- A shared `CutStabilityTestHarness` for mesh/rule construction and metrics.
- Explicit diagnostic/component APIs rather than environment-triggered production operators.

Also split snapshot tests out of `test_GeneratedActiveBoundaryDomain.cpp`; one test file should correspond to one production abstraction.

### ARC-14 — The commit series is not reviewable as one free-surface feature (P0)

Large independent changes in this range include:

- FE Basis: +11,374/-3,570 across 26 files; `LagrangeBasis.cpp` alone changes by +6,914/-788.
- FE TimeStepping: +7,965/-1,532 across 5 files.
- FE LevelSet: +2,333/-72 across 11 files, including a 3,447-line transport/extension commit whose lifecycle is incomplete at this head.
- Generic JIT history gradients/Jacobians (`FormKernels` and `LLVMGen`).
- Physics-wide JIT fallback/configuration-artifact policy mixed into the level-set input-validation commit.
- Dense linear algebra spectral backends.
- rotating-frame Coriolis forcing and parser support.
- component-selective velocity Dirichlet behavior.
- generic time-loop/start-time/tolerance behavior.
- solver/backend diagnostics and pressure guards.
- 35 validation-case files, including binary VTK assets, plus an 841-line documentation change and a large mesh-generation script update.

These may be individually valuable, but they obscure the dependency review and make regression attribution difficult.

Future cleanup of history, before or alongside code movement:

1. Split the basis performance rewrite into its own review. Keep public dispatch small and place topology/order specializations in generated or topology-specific translation units with reference/parity tests.
2. Split generic JIT history support and require interpreter/JIT parity tests.
3. Split spectral linear algebra and stability certification from the free-surface formulation.
4. Split Coriolis, component-selective Dirichlet, and time-loop behavior into independent changes.
5. Separate source changes from reproducible validation assets/results; record generator inputs and hashes for generated meshes.
6. Build a narrow commit sequence for generic implicit geometry first, then Physics free-surface use, then validation.

No history rewriting should be attempted until the current dirty worktree is safely accounted for.

### ARC-15 — Small concrete cleanup items (P2)

These are not the main architectural problem, but they are useful low-risk follow-ups once behavior is frozen:

- Remove the identical `SystemAssembly.cpp` if/else branches around diagnostic-context construction.
- Remove the ignored scalar-field intersection-builder overload.
- Replace `std::to_string(isovalue)` in stable keys.
- Replace `MeshComm::world()` in the accepted-pressure observer with the run/system communicator.
- Move nontrivial inline snapshot import/currentness implementation out of the already large `CutIntegrationContext.h` and split lifecycle, registry, and stabilization metadata responsibilities.
- Consolidate repeated vector math, face topology, and root interpolation utilities.
- Give cache ownership, eviction, and concurrency an explicit lifecycle contract.
- Remove raw backend strings and repeated validation blocks in generated-domain requests.

## What should remain physics-specific

Physics-agnostic does not mean moving all free-surface code into FE. The following belong under a focused Navier–Stokes free-surface boundary module:

- dynamic stress balance and its sign convention;
- external pressure and surface-tension coefficients;
- curvature law selection as used by the physical residual;
- normal/tangential kinematic boundary law;
- contact angle, mobility, wall slip, and contact-line constraints;
- Navier–Stokes-specific Nitsche parameters;
- selection and coefficients of velocity/pressure ghost penalties or other CutFEM stabilization;
- pressure initialization/gauge rules specific to the fluid formulation.

FE should provide only the generic fields, geometry, normals/tangents, measures, intersections, revisioning, ownership, quadrature, mapping, constraints, and explicit extension/transfer primitives needed to express those laws.

## What is already pointed in the right direction

- `ImplicitBoundaryIntersectionQuadrature` is placed in a generic quadrature namespace and can become a reusable codimension-two primitive after terminology/topology cleanup.
- Codimension-aware reference/physical mapping belongs in generic geometry; the new mapping service should become the single implementation.
- Global cell IDs, owner ranks, and source revisions are appropriate generic provenance inputs.
- Per-boundary Nitsche options are better than a global free-surface switch; they should remain with the Physics boundary model.
- Level-set transport/reinitialization/lifecycle code is broadly physics-neutral and should be preserved as the level-set adapter to a generic implicit-geometry contract.
- Generic history-gradient/JIT capability can remain in FE once reviewed independently from the free-surface feature.

## Proposed target source layout

The exact names are illustrative; the important part is the dependency boundary.

```text
Physics/Formulations/NavierStokes/
  IncompressibleNavierStokesVMSModule.{h,cpp}   # thin orchestration
  NavierStokesBulkForms.{h,cpp}
  NavierStokesPressureInitialization.{h,cpp}
  FreeSurface/
    FreeSurfaceOptions.h
    FreeSurfaceBoundaryValidator.{h,cpp}
    FreeSurfaceBoundaryForms.{h,cpp}
    FreeSurfaceContactLineForms.{h,cpp}
    FreeSurfaceCutStabilization.{h,cpp}

FE/ImplicitGeometry/ (or equivalently generic FE subdirectories)
  ImplicitFieldSource.h
  ImplicitGeometryRevision.h
  ImplicitInterfaceGeometrySnapshot.{h,cpp}
  ImplicitInterfaceDomain.{h,cpp}
  InterfaceBoundaryIntersectionDomain.{h,cpp}
  ActiveExteriorBoundaryDomain.{h,cpp}
  ImplicitGeometryLifecycle.{h,cpp}

FE/Geometry/
  ReferenceElementSubentities.{h,cpp}
  CutQuadratureMapping.{h,cpp}

FE/Assembly/
  AssemblyContributionMetadata.h
  AssemblyObserver.h
  LocalContributionTransform.h              # only if a generic hook is justified

FE/FieldTransfer/ or FE/LevelSet/
  FieldExtension.{h,cpp}
  LevelSetEvolutionController.{h,cpp}          # lifecycle, revisions, accepted-step transaction
```

## Staged cleanup plan

Do not combine these stages into one rewrite.

1. **Freeze behavior and separate review scope.** Preserve focused regression tests for governing residuals, capillary/contact laws, cut geometry, pressure gauge, MPI ownership, and mapping. Split unrelated basis/JIT/time-loop/Coriolis changes from the feature series.
2. **Complete or withdraw advertised lifecycle features.** Production-wire extension-map refresh, safety/limiter execution, maintenance guards, and configuration persistence before enabling their inputs. Align the documented, validated, and executed policies.
3. **Remove hidden generic-layer physics.** Classify assembler/Newton matrix transformations as required formulation, generic declared algebraic operation, or experiment. Move them accordingly and replace environment behavior with explicit typed options.
4. **Establish one generic geometry authority.** Generalize and production-wire the snapshot, carry a complete revision, and eliminate/bound legacy bypasses.
5. **Centralize reconstruction/topology/mapping.** Make interface, cut volume, exterior trace, boundary intersection, volume diagnostics, and assembly consume the same backend and mapping utilities.
6. **Decompose the Physics module.** Move boundary validation/forms, contact laws, pressure initialization, and CutFEM policies into focused Physics files while keeping the main module as orchestration.
7. **Extract diagnostics and observers.** Move logging, probes, row scans, accepted-step guards, and backend introspection out of governing/assembly/solver files. Verify that enabling instrumentation cannot change the operator.
8. **Split and relocate tests.** Put generic geometry/stability tests under FE and physical balance/contact tests under Physics; share small reusable fixtures.
9. **Qualification.** Re-run interpreter/JIT, fused/unfused, serial/MPI, backend, fitted/unfitted, small-cut, moving-interface, and validation-case matrices after each behavior-preserving stage.

## Suggested review gates

Before cleanup is declared complete:

- `FE/Assembly` and `FE/TimeStepping` contain no pressure/PSPG/Navier–Stokes/free-surface policy.
- Generic implicit-geometry headers contain no contact-angle, wet/dry, liquid, or free-surface terminology.
- One production-created snapshot supplies all related cut measures for a source revision.
- Every relevant rule is tied to the complete geometry revision, not only a scalar value revision.
- Every enabled extension/limiter/maintenance option has a production lifecycle caller, a current revision, and an end-to-end input test.
- Effective-configuration artifacts are persisted and describe executed behavior rather than merely parsed values.
- Numerical policy is explicit in input/options and recorded in provenance; diagnostics cannot mutate results.
- The main Navier–Stokes VMS implementation is primarily governing equations and orchestration.
- A true field extension never silently adds an equation for the physical velocity; any fictitious-domain stabilization is named and documented as such.
- Mapping/root/topology algorithms have a single reference implementation with parity tests for optimized paths.
- Tests are organized by the abstraction they verify, and the free-surface feature no longer depends on production diagnostic operators.
- Independent basis/JIT/time-loop/Coriolis/validation changes are reviewable separately.

## Audit disposition

Recommended disposition: **request architectural cleanup before merge, performed as staged, behavior-preserving changes after the current worktree is secured**.

The highest-risk items are ARC-01 through ARC-04: physics-specific operator mutation in generic assembly/Newton code, a geometry snapshot that is neither generic nor yet authoritative in production, and newly exposed level-set controls without a complete committed lifecycle. Decomposing the long Physics file is important, but doing it first would mostly redistribute code while leaving the more consequential coupling in place.

No source code was modified, no tests were run, and no commits were created as part of this audit.
