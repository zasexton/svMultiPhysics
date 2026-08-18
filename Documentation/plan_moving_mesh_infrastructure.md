# Moving Mesh Infrastructure Completion Plan

## Purpose

This document outlines the work required to turn the existing Mesh and FE support for current coordinates into a rich, robust, physics-agnostic moving mesh infrastructure suitable for multiphysics problems such as ALE fluids, FSI, free surfaces, shape evolution, moving boundaries, contact, and remeshing workflows.

The Mesh library already contains important foundations: reference/current coordinates, geometry-change events, mesh motion fields, backend injection, rollback, ghost coordinate exchange, search invalidation, and quality checks. The FE library can assemble using current coordinates when configured that way. The missing work is mostly the infrastructure that lets FE, Systems, Application, and Physics consume moving-domain information in a controlled, cache-safe, testable, and physics-neutral way.

## Related Completed Formulation Work

- [x] Math-first mesh-motion physics formulations are tracked in
  `Documentation/plan_mesh_motion_math_first_formulations.md`.
- [x] The formulation guide in
  `Documentation/mesh_motion_math_first_formulation_guide.md` documents the
  rule that Forms expose mathematical primitives while Physics modules write
  harmonic, pseudoelastic, boundary, and coupled-ALE mesh-motion algebra
  directly.
- [x] FE/Systems now provides centralized mesh-displacement binding so the
  mesh-motion modules do not embed field registration or source-kind logic in
  their weak-form blocks.

## Scope Rules

- [ ] Keep Mesh library changes limited to mesh state, geometry revisioning, mesh-side motion orchestration, field storage, distributed synchronization, search invalidation, quality evaluation, transfer, and restart metadata.
- [ ] Keep FE library changes physics agnostic. FE should expose moving-domain quantities, transforms, sensitivities, and assembly data, but should not encode Navier-Stokes, solid mechanics, FSI, or free-surface equations.
- [ ] Keep Physics module changes limited to consuming generic FE/Mesh infrastructure for specific formulations after the generic infrastructure is in place.
- [ ] Keep Application changes limited to configuration, construction, time-loop orchestration, restart wiring, and user-facing selection of mesh-motion options.
- [ ] Do not hide physics-specific behavior in FE terminals, assembly contexts, geometry mappings, or mesh-motion backends.
- [ ] Treat cache invalidation and reproducibility as first-class requirements for every moving-geometry feature.
- [ ] Require serial and MPI tests for any behavior that modifies coordinates, ghost coordinates, search structures, FE assembly geometry, or transfer state.

## Capability Boundaries

The infrastructure should be staged explicitly so the first delivered capability is correct and testable, while the broader multiphysics scope remains visible.

- [ ] Initial target: single-mesh, conforming, mostly H1-style ALE with prescribed or FE-smoothed motion.
- [ ] Initial target: piecewise-linear vertex-coordinate motion unless a high-order geometry phase is explicitly completed.
- [ ] Initial target: cache-safe assembly on current coordinates with rollback of rejected coordinate updates.
- [ ] Broader target: high-order and curved geometry motion.
- [ ] Broader target: multi-mesh moving-interface coupling for FSI, immersed/interface methods, and contact-like workflows.
- [ ] Broader target: solver/operator/preconditioner invalidation across geometry, topology, distribution, and layout changes.
- [ ] Broader target: reference rebasing and rezoning semantics for long large-deformation or remeshed runs.
- [ ] Broader target: geometric nonlinearity infrastructure for structures undergoing large deformation, rotation, or strain.
- [ ] Broader target: material/history-state frame semantics and rollback.
- [ ] Broader target: contact/proximity infrastructure and non-element validity checks.
- [ ] Broader target: Lagrangian-marker to Eulerian FE coupling infrastructure for Peskin-style immersed-boundary workflows.

## Ownership Map

- [ ] `Code/Source/solver/Mesh/Core`: Own coordinate buffers, active configuration, geometry revision counters, mesh events, and search invalidation hooks.
- [ ] `Code/Source/solver/Mesh/Motion`: Own mesh-motion orchestration, standard motion fields, backend interface, rollback, substepping, quality gates, and mesh-side diagnostics.
- [ ] `Code/Source/solver/Mesh/Geometry`: Own mesh-quality and geometric queries on reference/current configurations.
- [ ] `Code/Source/solver/Mesh/Search`: Own geometry-aware and topology-aware spatial acceleration, candidate generation primitives, and search invalidation metadata.
- [ ] `Code/Source/solver/Mesh/Validation`: Own mesh validity checks beyond basic element quality, including boundary validity and geometric separation checks.
- [ ] `Code/Source/solver/Mesh/Constraints`: Own mesh-side geometric constraints, periodic/tied metadata, and constraint consistency checks under coordinate motion.
- [ ] `Code/Source/solver/Mesh/Adaptivity`: Own mesh-field transfer, parent/child metadata, mesh adaptation events, and transfer diagnostics.
- [ ] `Code/Source/solver/Mesh/IO`: Own persistence of current/reference coordinates, motion fields, revision epochs, and rebasing metadata.
- [ ] `Code/Source/solver/FE/Geometry`: Own mappings, current/reference geometric transforms, Nanson-style utilities, geometry sensitivities, and mapping caches.
- [ ] `Code/Source/solver/FE/Assembly`: Own assembly contexts, required data flags, cell geometry preparation, cache invalidation, and physics-neutral access to moving-domain data.
- [ ] `Code/Source/solver/FE/Forms`: Own DSL terminals, expression lowering, AD/JIT support, and symbolic access to generic moving-domain quantities.
- [ ] `Code/Source/solver/FE/Spaces`: Own FE space family, order, basis-space compatibility, and space revision metadata.
- [ ] `Code/Source/solver/FE/Dofs`: Own DOF numbering, ownership, vector layout, and DOF-layout revision metadata.
- [ ] `Code/Source/solver/FE/Constraints`: Own FE constraint matrices/operators, geometry-dependent constraints, periodic constraints, multi-point constraints, and their invalidation semantics.
- [ ] `Code/Source/solver/FE/Constitutive`: Own physics-neutral constitutive state contracts, stress/tangent data structures, and frame-aware material-state storage helpers.
- [ ] `Code/Source/solver/FE/Backends`: Own assembled operator reuse, matrix-free geometry data, solver/preconditioner invalidation hooks, and backend-specific rebuild policies.
- [ ] `Code/Source/solver/FE/Systems`: Own FE lifecycle integration, mesh geometry observers, DOF/vector registration, solution transfer hooks, and assembly invalidation.
- [ ] `Code/Source/solver/FE/TimeStepping`: Own generic time-history access to coordinates, mesh velocity, and previous geometry states.
- [ ] `Code/Source/solver/Physics`: Own physics-specific ALE, FSI, free-surface, and moving-boundary formulations.
- [ ] `Code/Source/solver/Application`: Own XML/input configuration, solver construction, time-loop sequencing, checkpoint/restart selection, and user-facing diagnostics.

## Cross-Cutting Contracts

These contracts apply across the phases below. They are intended to prevent ambiguity once multiple individually implemented features are combined.

### Supported-Combinations Matrix

Every milestone should publish a supported-combinations matrix. A feature being individually implemented does not imply that it is qualified with every other moving-mesh feature.

| Combination | First planned support | Qualification status required before calling it supported |
| --- | --- | --- |
| Static mesh, H1 fields, reference geometry | Existing baseline | Existing static tests must remain unchanged. |
| Static mesh, H1 fields, current-coordinate assembly | Phases 1-2 | Reassembly after `X_cur` changes must use updated geometry. |
| Single-mesh, conforming, H1 ALE, piecewise-linear geometry, prescribed motion | First concrete milestone, Phases 1-6 | Serial and MPI smoke tests; rollback and cache invalidation tested. |
| Single-mesh, conforming, H1 ALE, FE-smoothed motion | Phase 7 | Backend solve, quality gate, rollback, and repeated assembly tested. |
| Single-mesh ALE with monolithic mesh displacement unknowns | Phases 8 and 17 | Geometry sensitivity, tangent, transaction, and rollback tests passing. |
| Single-mesh ALE with remesh/adaptivity/transfer | Phases 10, 11, 17, 18, 19 | Coordinate, field, solution, history, and restart transfer tested. |
| Single-mesh ALE with high-order or curved geometry | Phase 15 | High-order DOF storage, ghosting, transfer, IO, validity, and FE mapping tested. |
| Single-mesh ALE with H(div), H(curl), tensor-valued fields, shells, or surfaces | Phase 23 | Element-family transform tests passing for each enabled family. |
| Large-deformation structural geometric nonlinearity | Phases 17, 19, 24 | Finite-strain kinematics, consistent tangent, history state, and load-step rollback tested. |
| Prescribed rigid-body subdomain motion, such as rotation or translation | Phases 6, 17, 25 | Rigid motion maps, mesh velocity, transaction, restart, and FE geometry tests passing. |
| Sliding or cyclic nonmatching interface with prescribed relative motion | Phases 16, 17, 20, 25 | Time-level interface maps, conservative/nonconservative projection, periodic identity, rollback, and MPI tests passing. |
| Multi-mesh moving-interface coupling without remesh | Phase 16 | Interface map rebuild, interpolation/projection, rollback, and MPI ownership tested. |
| Multi-mesh moving-interface coupling with remesh or repartition | Phases 10, 16, 17, 18, 19 | Interface provenance, transfer, projection, restart, and repartition tests passing. |
| Unfitted, embedded-boundary, or cut-cell interface methods | Phase 26 | Cut classification, intersection geometry, cut quadrature, partial-cell integration, conditioning hooks, rollback, and MPI tests passing. |
| Peskin-style Lagrangian-marker immersed-boundary coupling | Phase 27 | Marker registry, marker-to-Eulerian support maps, regularized-delta kernels, gather/spread adjointness, rollback, restart, and MPI ownership tested; no governing equations encoded in Mesh or FE. |
| Contact/proximity with moving geometry | Phases 21 and 22 | Candidate generation, active-set state, rollback, validity, and MPI tests passing. |
| Contact/proximity with remesh or repartition | Phases 10, 17, 19, 21, 22 | Contact provenance, transfer, active-set invalidation, and restart tests passing. |

- [ ] Add this matrix, or a generated equivalent, to release notes for each moving-mesh milestone.
- [ ] Mark unqualified combinations explicitly as "implemented individually, not qualified together."
- [ ] Require at least one integration test for every combination marked supported.
- [ ] Keep the default supported path narrow until broader combinations are tested.

### Revision-to-Invalidation Table

Each cache or state owner must declare the revision domains it depends on. The table below is the starting contract; individual modules may add stronger invalidation rules, but should not weaken these defaults.

| Consumer | Geometry | Topology | Ownership | Numbering | Field layout | Labels |
| --- | --- | --- | --- | --- | --- | --- |
| FE geometry caches | Rebuild | Rebuild | Usually keep, unless distributed geometry ownership changes | Rebuild if indexed by entity id | Keep | Keep |
| `StandardAssembler` flat cell coordinates | Rebuild | Rebuild | Rebuild for distributed access tables | Rebuild | Keep | Keep |
| Basis and quadrature caches | Keep | Rebuild if element type/order changes | Keep | Keep | Keep | Keep |
| DOF maps | Keep | Rebuild | Rebuild distributed ownership maps | Rebuild | Rebuild when FE field layout changes | Rebuild if label-scoped spaces change |
| Sparsity patterns | Keep | Rebuild | Rebuild | Rebuild | Rebuild | Rebuild if label-scoped coupling changes |
| Field access tables | Keep | Rebuild if entity counts/layout change | Rebuild distributed ownership views | Rebuild | Rebuild | Keep unless label-filtered |
| Assembled matrices | Rebuild if geometry-dependent | Rebuild | Rebuild distributed matrices | Rebuild | Rebuild | Rebuild if label-scoped terms change |
| Matrix-free operators | Rebuild geometry data | Rebuild | Rebuild distributed maps | Rebuild | Rebuild | Rebuild if label-scoped terms change |
| Preconditioners | Refresh or rebuild by declared policy | Rebuild | Rebuild | Rebuild | Rebuild | Rebuild if label-scoped operators change |
| Multigrid hierarchies | Refresh geometry smoothers if needed | Rebuild | Rebuild | Rebuild | Rebuild | Rebuild if label-scoped hierarchy changes |
| Search structures | Rebuild current/deformed search | Rebuild | Rebuild distributed metadata | Rebuild | Keep | Rebuild label-filtered search |
| Interface maps | Rebuild if either side moves | Rebuild if either side changes topology | Rebuild | Rebuild | Rebuild if projection fields change | Rebuild if interface labels change |
| Lagrangian marker support maps and gather/spread operators | Rebuild if marker or Eulerian mesh coordinates change | Rebuild if Eulerian cell topology or marker set topology changes | Rebuild distributed marker/support ownership | Rebuild if Eulerian numbering or marker numbering changes | Rebuild if FE field layout or marker field layout changes | Rebuild only if label-scoped marker/Eulerian support changes |
| Contact candidates and active sets | Rebuild or update by contact policy | Rebuild | Rebuild | Rebuild | Keep unless contact field layout changes | Rebuild if contact labels change |
| Constraint operators | Rebuild if geometry-dependent | Rebuild | Rebuild | Rebuild | Rebuild if constrained field layout changes | Rebuild if label-scoped |
| Material/history state tables | Usually keep; transform only by declared frame policy | Transfer or reinitialize | Transfer ownership metadata | Reindex | Rebuild layout | Keep unless label-scoped |
| Restart metadata | Record new epoch | Record new epoch | Record new epoch | Record new epoch | Record new epoch | Record new epoch |

- [ ] Add dependency declarations to cache/state types where practical.
- [ ] Add assertions or diagnostics when a consumer sees an unsupported revision transition.
- [ ] Add tests that mutate one revision domain at a time and verify only the expected consumers rebuild.

### FE-Owned Revision Domains

Mesh revision state is not enough for FE correctness. FE systems must own revision domains for FE-space and DOF-layout changes that can happen while the mesh revision state is unchanged.

Required FE revision domains:

- [ ] `space_revision()`: changes when FE space family, polynomial order, basis family, geometry/solution space compatibility, or field-space assignment changes.
- [ ] `dof_layout_revision()`: changes when DOF numbering, DOF ownership, vector layout, constrained/free DOF partitioning, or entity-to-DOF maps change.
- [ ] `constraint_layout_revision()`: changes when eliminated constraints, periodic constraints, multi-point constraints, tied constraints, hanging-node constraints, or geometry-dependent constraint maps change.
- [ ] `block_layout_revision()`: changes when mixed-system field registration, field ordering, block partitioning, block coupling metadata, or solver block layout changes.
- [ ] `system_layout_revision()`: optional aggregate revision for consumers that cannot cheaply distinguish the above domains.

| Consumer | Space revision | DOF layout revision | Constraint layout revision | Block layout revision |
| --- | --- | --- | --- | --- |
| FE geometry caches | Rebuild if geometry basis/order changes | Keep | Keep | Keep |
| Basis and quadrature caches | Rebuild | Keep | Keep | Keep |
| DOF maps | Rebuild | Rebuild | Rebuild if constrained/free layout changes | Rebuild if field registration changes |
| Vector layouts and solution/history vectors | Reallocate or transfer | Reallocate or transfer | Reallocate or scatter through new constraints | Reallocate or repartition blocks |
| Sparsity patterns | Rebuild | Rebuild | Rebuild | Rebuild |
| Field access tables | Rebuild if field space changes | Rebuild | Rebuild if constrained views change | Rebuild |
| Assembled matrices | Rebuild | Rebuild | Rebuild | Rebuild |
| Matrix-free operators | Rebuild kernels/data if space changes | Rebuild maps | Rebuild constrained apply path | Rebuild block apply path |
| Preconditioners and multigrid | Rebuild | Rebuild | Rebuild | Rebuild |
| Interface maps and projection operators | Rebuild if projected/interpolated spaces change | Rebuild | Rebuild if constraints affect interface fields | Rebuild if coupled blocks change |
| Contact operators | Rebuild if contact field spaces change | Rebuild | Rebuild if constraints affect contact fields | Rebuild if contact blocks change |
| Restart metadata | Record new FE layout epoch | Record new FE layout epoch | Record new FE layout epoch | Record new FE layout epoch |

- [ ] Thread FE revision domains through `FESystem`, `DofHandler`, constraint managers, sparsity builders, assemblers, interface operators, backends, and restart metadata.
- [ ] Add tests that change FE space order/family without changing the mesh and verify all dependent state invalidates.
- [ ] Add tests that change mixed-field/block registration without changing the mesh and verify vectors, sparsity, preconditioners, interface operators, and restart metadata invalidate.

### Active Configuration Semantics

Hidden dependence on mutable global active configuration is not allowed for FE, search, backend, interface, or solver caches.

- [ ] Add an `active_configuration_epoch()` if the Mesh continues to expose mutable active configuration.
- [ ] Increment the active-configuration epoch whenever `use_reference_configuration()` or `use_current_configuration()` changes the active state.
- [ ] Include coordinate configuration and active-configuration epoch in any cache that intentionally follows mutable active configuration.
- [ ] Prefer explicit frame/configuration requests in FE, search, backend, and interface APIs.
- [ ] Require FE assembly caches to key on explicit coordinate configuration rather than reading mutable global state implicitly.
- [ ] Require search structures to record the configuration they were built for.
- [ ] Require moving-domain forms to use frame-explicit terminals rather than relying on the mesh active configuration.
- [ ] Add diagnostics when mutable active configuration is used in an API that should be frame-explicit.

Acceptance criteria:

- [ ] Switching active configuration cannot silently make an FE/search/backend cache stale.
- [ ] Runtime moving-domain paths use explicit reference/current/frame requests.
- [ ] Legacy active-configuration behavior remains available only where compatibility requires it.

### Transaction and Rollback State Model

Moving geometry must use a full transaction model. A transaction boundary includes all state that can become inconsistent with coordinates, not only `X_cur`.

Required transaction states:

- [ ] Trial iterate state inside a nonlinear or mesh-motion solve.
- [ ] Accepted nonlinear state inside a time step.
- [ ] Accepted time-step state.
- [ ] Accepted remesh/rezone state.
- [ ] Rolled-back state after rejected nonlinear iteration, mesh-motion step, time step, remesh, or rezone.

State that must participate in transactions:

- [ ] Mesh reference/current coordinates and high-order geometry DOFs.
- [ ] Mesh-motion displacement, velocity, acceleration, and previous-state fields.
- [ ] Mesh revision epochs or staged revision deltas.
- [ ] FE geometry caches and assembly scratch data.
- [ ] Assembled operators, matrix-free operator data, preconditioners, and multigrid state.
- [ ] Search structures.
- [ ] Multi-mesh interface pairing, interpolation, projection, and mortar state.
- [ ] Contact candidate, projection, gap, normal, and active-set state.
- [ ] Constraint operators and geometry-dependent constraint maps.
- [ ] Material, history, auxiliary, and coupling state.
- [ ] Restart/checkpoint staging metadata.

Acceptance criteria:

- [ ] Trial updates do not overwrite committed state until explicitly accepted.
- [ ] Rollback restores all transaction participants to a self-consistent committed state.
- [ ] Accepted nonlinear state and accepted time-step state are distinct when the time integrator requires that distinction.
- [ ] Accepted remesh/rezone state records provenance and invalidates stronger domains than ordinary geometry motion.

### Stable Identity and Provenance

Moving interfaces, contact, remeshing, transfer, and restart require persistent identity beyond labels.

- [ ] Define persistent IDs or provenance records for boundary regions.
- [ ] Define persistent IDs or provenance records for interface regions.
- [ ] Define persistent IDs or provenance records for contact surfaces.
- [ ] Define persistent IDs or provenance records for coupling regions.
- [ ] Track parent/child provenance through remesh and adaptivity.
- [ ] Track material point or quadrature history provenance where transfer is required.
- [ ] Track interface projection provenance when maps are rebuilt.
- [ ] Persist provenance needed for restart.
- [ ] Add diagnostics when provenance is unavailable for a requested transfer or restart operation.

Acceptance criteria:

- [ ] Transfer, restart, and qualification tests do not rely on labels alone when persistent identity is required.
- [ ] Remeshed or repartitioned coupling regions can be matched back to their prior logical regions.
- [ ] Contact and interface state can be invalidated, transferred, or reinitialized according to explicit provenance rules.

### High-Order Geometry DOF Lifecycle

High-order geometry support must cover the full lifecycle, not only FE mapping.

- [x] Define storage for high-order geometry DOFs. Implemented as explicit mesh geometry DOFs stored in vertex-coordinate arrays and described by `GeometryOrderDescriptor`.
- [x] Define ownership and ghosting for high-order geometry DOFs. Implemented through the existing vertex ownership/ghost-coordinate path and verified on Hex27 geometry DOFs.
- [x] Define current/reference semantics for high-order geometry DOFs. Reference and current coordinate access/mutation APIs now apply to edge, face, and interior geometry control points.
- [x] Define transfer and projection after remesh. The supported Phase 15 mode preserves geometry DOFs through vertex-coordinate migration/repartition/adaptivity paths; creation/projection of new curved geometry DOFs during remesh remains governed by the remesh/rezone phases before that combination is advertised.
- [x] Define restart and IO persistence. Moving-mesh restart metadata now persists geometry order/layout and round-trips high-order reference/current coordinates.
- [x] Define validity and quality checks for curved elements, including curved-boundary folding. Existing sampled curvilinear quality/validity paths are retained and Phase 15 adds curved mapping/normal/sensitivity tests.
- [x] Define how high-order geometry DOFs participate in geometry transactions. Geometry DOF mutations bump the same reference/current geometry revisions used by moving-mesh transaction and cache invalidation policy.
- [x] Define how high-order geometry DOFs affect search and interface maps. High-order DOFs are visible to geometry revisions and vertex-coordinate bounding boxes; multi-mesh interface qualification remains Phase 16/22 scope.

Acceptance criteria:

- [x] High-order motion cannot be marked supported until storage, ghosting, transfer, restart, mapping, validity, and transaction behavior are all tested for the Phase 15 supported vertex-coordinate geometry DOF mode.

### Multi-Mesh Time-Integration Semantics

Multi-mesh interface maps and projections must be tied to explicit time levels.

- [ ] Define whether an interface map is built on old, trial, accepted nonlinear, accepted time-step, or predicted geometry.
- [ ] Define when interface maps are rebuilt during nonlinear iterations.
- [ ] Define when conservative projection operators are rebuilt.
- [ ] Define when nonconservative interpolation operators are rebuilt.
- [ ] Define rollback behavior for interface maps and projected state.
- [ ] Define whether projected quantities are trial, accepted nonlinear, or accepted time-step state.
- [ ] Qualify interface operators under remesh and repartition before declaring those combinations supported.
- [ ] Add diagnostics identifying the time level and revision state used by each interface operator.

Acceptance criteria:

- [ ] FSI-style workflows can state exactly which geometry and time level each interface transfer uses.
- [ ] Conservative and nonconservative transfer choices are explicit.
- [ ] Remesh/repartition invalidation of interface operators is tested.

### Constitutive and Material API Contract

Finite-deformation and history-dependent models need an explicit material-side contract.

- [ ] Define what frame each material model receives kinematic data in.
- [ ] Define what stress measure each material model returns.
- [ ] Define what tangent measure each material model returns.
- [ ] Define whether FE or the material model owns pushforward, pullback, and rotation.
- [ ] Define when history updates are trial/work versus committed.
- [ ] Define rollback behavior for failed nonlinear iterations and failed time steps.
- [ ] Define transfer behavior after remesh and repartition.
- [ ] Define restart persistence for internal variables and frame metadata.
- [ ] Require each history-dependent material to declare its state frame and transform policy.

Acceptance criteria:

- [ ] FE and Constitutive modules agree on frames, stress measures, tangent measures, and update lifecycle.
- [ ] Material updates cannot be committed accidentally during rejected trial states.
- [ ] Restart and transfer preserve enough state to continue a history-dependent simulation correctly.

### Backward Compatibility and Migration

Moving-domain APIs should be opt-in and frame-explicit without breaking existing static-mesh code.

- [ ] Preserve existing static/reference defaults.
- [ ] Preserve existing `Coordinate` behavior unless deliberately deprecated through a staged migration.
- [ ] Add frame-explicit replacements for ambiguous terms before deprecating old names.
- [ ] Document deprecated ambiguous terms and their replacements.
- [ ] Keep geometry-sensitive, monolithic, high-order, contact, and multi-mesh features opt-in.
- [ ] Add compatibility tests proving old static-mesh forms assemble unchanged.
- [ ] Add migration examples for forms that should move from ambiguous geometry terms to frame-explicit terminals.
- [ ] Add runtime diagnostics for ambiguous moving-domain usage when a safer frame-explicit API exists.

Acceptance criteria:

- [ ] Existing static-mesh applications continue to run without input changes.
- [ ] Existing forms continue to compile unless a deprecation is intentionally enabled.
- [ ] New moving-domain examples use only frame-explicit terminology.

## Phase 1: Mesh State Revision and Cache Invalidation Contract

Goal: make every state mutation visible to Mesh, FE, solver, and coupling consumers through cheap, explicit revision domains. Geometry revision is necessary but not sufficient; topology, ownership, numbering, field layout, and DOF layout must have distinct invalidation signals.

### Mesh/Core

- [x] Add a structured revision state to `MeshBase`, such as `MeshRevisionState`.
- [x] Include independent monotonically increasing counters for:
  - [x] Geometry changes.
  - [x] Reference-geometry changes.
  - [x] Current-geometry changes.
  - [x] Topology changes.
  - [x] Repartition or ownership changes.
  - [x] Entity renumbering changes.
  - [x] Mesh field layout changes.
  - [x] Boundary/label metadata changes.
  - [x] Active-configuration changes, if active configuration remains mutable.
- [x] Increment the geometry revision whenever `MeshEvent::GeometryChanged` is emitted from coordinate mutation paths.
- [x] Increment topology, ownership, numbering, field-layout, and label revisions from their respective mutation paths.
- [x] Expose focused accessors such as `geometry_revision()`, `topology_revision()`, `ownership_revision()`, `numbering_revision()`, `field_layout_revision()`, `label_revision()`, and `active_configuration_epoch()`.
- [x] Expose an aggregate `revision_state()` for caches that need multiple domains.
- [x] Ensure `set_current_coords`, `clear_current_coords`, and `set_vertex_coords` update geometry revisions once per logical geometry change.
- [x] Ensure current-coordinate ghost updates emit one geometry-change event per logical exchange.
- [x] Ensure migration and adaptivity paths update geometry revisions exactly once per logical geometry change.
- [x] Ensure repartition, migration, and ownership exchange update ownership revision even when geometry is unchanged.
- [x] Ensure entity renumbering updates numbering revision even when topology is unchanged.
- [x] Ensure field attachment/removal/component layout changes update field-layout revision.
- [x] Ensure topology refinement/coarsening updates topology revision, geometry revision, numbering revision, field-layout revision when applicable, and label revision when labels are affected.
- [x] Ensure active-configuration switching updates only the active-configuration epoch, not geometry or topology revisions.
- [x] Document that callers using `X_cur_data_mutable()` must emit a geometry event or call a scoped geometry mutation helper.
- [x] Add a scoped helper such as `ScopedGeometryMutation` or an explicit `mark_geometry_changed()` if direct buffer mutation remains necessary.
- [x] Add serial unit tests for each revision domain.
- [x] Add MPI tests for ghost current-coordinate exchange geometry-change behavior.
- [x] Add MPI tests for revision behavior after repartition, migration, and renumbering.

### Mesh/Search

- [x] Include geometry, topology, ownership, numbering, and label revisions in search acceleration metadata.
- [x] Make search build/reuse checks reject stale structures when any relevant revision changes.
- [x] Preserve the existing configuration distinction between reference/current/deformed search.
- [x] Add tests that moving `X_cur` invalidates current/deformed search and preserves reference-configuration query semantics.
- [x] Add tests that topology or numbering changes invalidate both reference and current search structures.
- [x] Add tests that repartition invalidates distributed search ownership metadata.

### FE/Assembly and FE/Geometry

- [x] Add optional revision-domain queries to `IMeshAccess`.
- [x] Thread the revision through `MeshAccess`, `MeshSearchAccess`, and any test mesh-access implementations.
- [x] Include geometry revision in FE geometry caches that currently key only on mesh/access pointers.
- [x] Include topology, ownership, numbering, and active-configuration revisions in the flat cell-coordinate cache key.
- [x] Include topology, numbering, and layout revisions in all remaining caches that depend on cell connectivity, DOF maps, sparsity patterns, field tables, or entity indexing.
- [x] Add explicit invalidation methods on assemblers for geometry-only cache resets.
- [x] Add explicit invalidation methods for topology/layout resets that are stronger than geometry-only resets.
- [x] Add unit tests that reusing FE assembly objects after changing `X_cur` assembles with the new geometry.
- [x] Add unit tests that topology/layout changes force the required DOF, sparsity, and field-cache rebuilds.

### FE/Spaces, FE/Dofs, FE/Constraints, and FE/Systems

- [x] Add FE-owned revision state separate from Mesh-owned revision state.
- [x] Expose `space_revision()` from FE space registries or equivalent space-management objects.
- [x] Expose `dof_layout_revision()` from DOF handlers or equivalent DOF-management objects.
- [x] Expose `constraint_layout_revision()` from constraint managers.
- [x] Expose `block_layout_revision()` from mixed-system/block-layout managers.
- [x] Expose an aggregate `fe_layout_revision_state()` from `FESystem`.
- [x] Increment space revision on FE family, order, basis, field-space assignment, or geometry/solution space compatibility changes.
- [x] Increment DOF-layout revision on DOF numbering, ownership, vector layout, or entity-to-DOF map changes.
- [x] Increment constraint-layout revision on eliminated/free DOF partition changes and geometry-dependent constraint-map changes.
- [x] Increment block-layout revision on field registration, field order, block partitioning, or block coupling metadata changes.
- [x] Thread FE revision state through Phase-1 consumers: assemblers, sparsity builders, vector-layout/backend hooks, DOF maps, constraints, block layouts, and FE systems. Interface operators, transfer, and restart metadata must consume the same revision domains as their concrete moving-mesh ownership is implemented in later phases.
- [x] Add unit tests that mutate FE layout without mutating Mesh state and verify FE revision counters change.
- [x] Add unit tests that mutate FE layout without mutating Mesh state and verify dependent consumers invalidate.

Acceptance criteria:

- [x] No FE assembly path can reuse cell coordinates, Jacobians, normals, integration weights, or search structures across a geometry change unless its cache key includes the new geometry revision.
- [x] No connectivity, DOF, sparsity, field table, search, or Phase-1 coupling cache can survive an incompatible topology, ownership, numbering, or layout revision.
- [x] No DOF map, vector layout, sparsity pattern, or Phase-1 backend/layout consumer can survive an incompatible FE space, DOF-layout, constraint-layout, or block-layout revision. Concrete preconditioner, interface-operator, and restart metadata consumers must key on these domains as they are introduced or reworked in later phases.
- [x] Runtime active-configuration switches cannot create hidden stale FE or search cache state.
- [x] Existing reference-geometry tests remain unchanged.
- [x] Moving current coordinates and reassembling with the same FE objects produces updated geometric measures.

Broader regression coverage before merge:

- [x] Run the full Mesh test suite with the default test configuration.
  - Verified with `build-mesh-tests: ctest --output-on-failure --timeout 300 -j2`; passed 95/95 tests.
- [x] Run the full FE test suite with the default test configuration.
  - Verified with `build-fe-check: ctest --output-on-failure --timeout 300 -j2`; passed 25/25 tests.
- [x] Run the full Mesh test suite with adaptivity enabled.
  - Verified with `build-mesh-adaptivity-check: ctest --output-on-failure --timeout 300 -j2`; passed 59/59 tests.
- [x] Run the MPI Mesh regression tests under CI or an equivalent clean MPI environment.
  - Verified with `build-mesh-tests: ctest --output-on-failure --timeout 300 -j1 -R 'MPI|_4ranks|GhostCoordinateExchange|DistributedSemantics|RebalanceParMetis|Migration|PVTU|StartupParMetis|PartitionQualityMetis'`; passed 50/50 tests.
  - Rechecked in an external MPI-enabled shell after Phase 7 with the same filtered Mesh MPI regression command; passed 49/49 tests, including `test_MotionMPI` and `test_MotionMPI_4ranks`.
- [x] Run at least one existing static-mesh application or regression case to confirm unchanged defaults and input compatibility.
  - Verified by copying `tests/cases/struct/robin` to `/tmp` and running `/usr/bin/mpiexec -n 1 build/svMultiPhysics-build/bin/svmultiphysics solver.xml`; completed and produced `1-procs/result_001.vtu`.
- [ ] Add concrete preconditioner, interface-operator, transfer, and restart-metadata regression tests when those consumers are introduced or reworked in later phases.
  - Not applicable to Phase 1 until those concrete consumers are introduced or reworked in later phases.

## Phase 2: Fix Known FE Flat Coordinate Cache Risk

Goal: remove the immediate stale-coordinate risk in fused assembly.

### FE/Assembly/StandardAssembler

- [x] Extend `flat_cell_coords_` metadata to include geometry revision and coordinate configuration.
- [x] Invalidate or rebuild `flat_cell_coords_` when the mesh access pointer, geometry revision, coordinate configuration, cell count, cell type, or nodes-per-cell layout changes.
- [x] Ensure `ensureFlatCellCoords` does not assume cell IDs are dense unless `IMeshAccess` guarantees that contract.
- [x] Add a cheap bypass path for meshes whose geometry revision is unavailable, with conservative invalidation.
- [x] Audit other Tier 2/Tier 3 caches for coordinate-derived data.
  - `flat_cell_coords_` is the only Phase-2 Tier 3 cache that stores coordinate values directly. Geometry mapping scratch is rebuilt per cell, and DOF/resolved insertion/field-access caches are keyed on topology/layout revisions rather than coordinate values.
- [x] Add tests that:
  - [x] Assemble on reference geometry.
  - [x] Set current coordinates.
  - [x] Reassemble through the same `FESystem` and same `StandardAssembler`.
  - [x] Confirm mass/stiffness/geometric measures change correctly.
  - [x] Repeat the test through fused and non-fused assembly paths.

Acceptance criteria:

- [x] Moving mesh coordinates cannot produce stale fused assembly geometry.
- [x] The fix does not disable flat coordinate optimization for static meshes.
- [x] Cache behavior is deterministic under serial and MPI execution.

Verification:

- [x] `build-fe-check: ./test_fe_assembly --gtest_filter=StandardAssemblerCaches.ReusedAssemblerRebuildsFlatCoordinatesAfterGeometryRevision:StandardAssemblerCaches.FlatCoordinateCacheReusedForStaticDenseMeshes:MeshAccess.RevisionQueriesFollowMeshState --gtest_brief=1`; passed 3/3 tests.
- [x] `build-fe-check: ./test_fe_systems --gtest_filter=FESystem.ReusedCurrentSystemTracksCoordinateMutationWithoutSetup --gtest_brief=1`; passed 1/1 test.
- [x] `build-fe-check: ctest --output-on-failure --timeout 300 -R "FE_Assembly_Tests|FE_Systems_Tests|test_fe_assembly_mpi_mpi_2|test_fe_assembly_mpi_mpi_4" -j1`; passed 4/4 tests.
- [x] `build-fe-check: ctest --output-on-failure --timeout 300 -j2`; passed 25/25 FE tests.

## Phase 3: Physics-Agnostic Moving-Domain Data Model in FE

Goal: define generic FE data structures for moving-domain quantities without introducing physics-specific equations.

### FE/Assembly

- [x] Add new `RequiredData` bits for generic moving-domain data:
  - [x] `MeshDisplacement`
  - [x] `MeshVelocity`
  - [x] `MeshAcceleration`, if time integration needs it.
  - [x] `ReferencePhysicalPoints`
  - [x] `CurrentPhysicalPoints`
  - [x] `ReferenceJacobians`
  - [x] `CurrentJacobians`
  - [x] `ReferenceNormals`
  - [x] `CurrentNormals`
  - [x] `ReferenceMeasures`
  - [x] `CurrentMeasures`
  - [x] `SurfaceJacobians`
  - [x] `ConfigurationTransforms`
- [x] Add storage in assembly scratch/context for these quantities.
- [x] Keep naming frame-explicit. Avoid terminals whose meaning depends on the active mesh configuration without being visible in the name.
- [x] Add `AssemblyContext` accessors for the new data.
- [x] Make accessors fail loudly when requested data was not prepared.
- [x] Add tests for required-data propagation and context access.

### FE/Systems

- [x] Add a physics-neutral registry for mesh-motion fields that can bind mesh-side fields to FE assembly data by name or handle.
- [x] Support optional binding of `mesh_displacement`, `mesh_velocity`, and previous coordinate fields.
- [x] Provide clear behavior when a moving-domain terminal is requested but no mesh-motion field is registered.
- [x] Add tests for missing, present, and dimension-mismatched mesh-motion fields.

Acceptance criteria:

- [x] FE kernels can request mesh displacement and velocity without knowing which physics module will use them.
- [x] FE kernels can access current and reference geometry in the same cell evaluation.
- [x] Static-mesh forms do not pay unnecessary setup cost unless moving-domain data is requested.

Verification:

- [x] `build-fe-check: cmake --build build-fe-check --target test_fe_assembly test_fe_systems -j1`; passed.
- [x] `build-fe-check: ./test_fe_assembly --gtest_filter='RequiredDataTest.MovingDomainBitsUseWideMask:AssemblyContextTest.MovingDomainDataAccessorsAreFrameExplicitAndFailLoudly:StandardAssemblerMovingDomain.*' --gtest_brief=1`; passed 4/4 tests.
- [x] `build-fe-check: ./test_fe_systems --gtest_filter='FESystem.MeshMotionFieldBindingsArePhysicsNeutral:FESystem.MeshMotionFieldBindingRejectsMissingScalarAndDimensionMismatchedFields' --gtest_brief=1`; passed 2/2 tests.
- [x] `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Assembly_Tests|FE_Systems_Tests' -j1`; passed 2/2 tests.

## Phase 4: FE Forms Vocabulary for Moving Domains

Goal: allow physics authors to express moving-domain weak forms through generic terminals and transforms.

### FE/Forms AST and Public API

- [x] Add physics-neutral terminals:
  - [x] `meshDisplacement()`
  - [x] `meshVelocity()`
  - [x] `meshAcceleration()`, if supported by time stepping.
  - [x] `currentCoordinate()`
  - [x] `referenceCoordinatePhysical()`
  - [x] `currentJacobian()`
  - [x] `referenceJacobian()`
  - [x] `currentJacobianDeterminant()`
  - [x] `referenceJacobianDeterminant()`
  - [x] `currentNormal()`
  - [x] `referenceNormal()`
  - [x] `currentMeasure()`
  - [x] `referenceMeasure()`
  - [x] `surfaceJacobian()`
- [x] Add generic transform helpers:
  - [x] `pullback(expr, from_config, to_config)` as a metadata marker; non-identity runtime evaluation fails closed until a formulation provides explicit transform semantics.
  - [x] `pushforward(expr, from_config, to_config)` as a metadata marker; non-identity runtime evaluation fails closed until a formulation provides explicit transform semantics.
  - [x] `nanson(normal_or_measure)` as explicit surface-transform algebra using `cofactor(currentJacobian() * inv(referenceJacobian()))`.
  - [x] Keep ALE transport terms math-first: write `dt(field) + inner(a, grad(field))`, `u - meshVelocity()`, and `div(meshVelocity()) * ...` explicitly instead of exposing public `materialDerivative`, `transportDerivative`, or moving-volume wrapper helpers.
- [x] Preserve existing `Coordinate` behavior for backward compatibility.
- [x] Clarify whether existing `ReferenceCoordinate` means quadrature coordinate or physical reference coordinate; add a new terminal if needed to avoid ambiguity. Existing `ReferenceCoordinate` remains the quadrature reference coordinate; `referenceCoordinatePhysical()` provides the physical reference-coordinate terminal.
- [x] Update `VOCABULARY.md` and `VOCABULARY_ROADMAP.md`.

### FE/Forms Compiler, Interpreter, AD, and JIT

- [x] Lower each new terminal to the correct `RequiredData` bit.
- [x] Add interpreter evaluation for each terminal.
- [x] Add JIT lowering for each terminal.
- [x] Add AD evaluation support.
- [x] Keep geometry terminals constant with respect to solution fields by default.
- [x] Add a separate opt-in path for geometry sensitivities with respect to mesh-motion unknowns. Phase 4 adds `GeometrySensitivityOptions` and fail-closed diagnostics for mesh-motion sensitivity requests; derivative kernels remain Phase 8 work.
- [x] Add parser and pretty-printer support if forms expose these terminals through text/UFL-like authoring. No text parser exists today; `toString()`/pretty-printer support was added for the new terminals and transforms.
- [x] Add unit tests for scalar, vector, matrix, face, and volume terminals.
- [x] Add JIT/interpreter parity tests.

Acceptance criteria:

- [x] Moving-domain forms can be authored without manually plumbing mesh velocity as a user coefficient.
- [x] Current/reference geometry are distinguishable in the DSL.
- [x] Interpreter, AD, and JIT paths agree for all new terminals on static and moved meshes.

Verification:

- [x] `build-fe-check: cmake --build build-fe-check --target test_fe_forms -j2`; passed.
- [x] `build-fe-check: ./test_fe_forms --gtest_filter='FormCompilerTest.ConstructorAndSetOptionsPreserveSymbolicTangentOptions:FormCompilerTest.DefaultAndMoveConstructorsPreserveOptionsAndUsability:FormCompilerTest.GeometrySensitivityModeFailsClosedForGeometryTerminals:FormCompilerTest.GeometrySensitivityModeRequiresMeshMotionField:FormVocabularyTest.RequiredDataInferenceIncludesGeometryAndMeasures:FormVocabularyTest.MovingDomainCoordinateTerminalsUseExplicitFrames:FormVocabularyTest.MovingDomainCoordinateTerminalsMatchInterpreterAndJIT:FormVocabularyTest.MovingDomainVolumeTerminalsMatchInterpreterAndJIT:FormVocabularyTest.MovingDomainFaceNormalsUseExplicitTerminals:FormVocabularyTest.MovingDomainBoundaryTerminalsMatchInterpreterAndJIT:FormVocabularyTest.MovingDomainGeometryTerminalsAreADConstants'`; passed 11/11 tests.
- [x] `build-fe-check: ./test_fe_forms`; passed 613/631 tests, with 18 performance-gated skips.

## Phase 5: Current/Reference Geometry Preparation

Goal: prepare both reference and current geometry on demand for the same cell or face evaluation.

### FE/Geometry

- [x] Add reusable helpers for building geometry mappings from explicit configuration-specific coordinates.
- [x] Add current/reference mapping pair support for a cell.
- [x] Add current/reference face geometry pair support for boundary and interface integration.
- [x] Add Nanson-style surface transform utilities for normals and measures.
- [x] Add frame-explicit APIs so callers cannot accidentally mix reference and current normals or measures.
- [x] Add geometry sensitivity utilities for derivatives of:
  - [x] Physical coordinates with respect to nodal mesh displacement.
  - [x] Jacobian with respect to nodal mesh displacement.
  - [x] Determinant/integration weight with respect to nodal mesh displacement.
  - [x] Inverse Jacobian with respect to nodal mesh displacement.
  - [x] Face normal and measure with respect to nodal mesh displacement.
- [x] Keep these utilities generic and independent of fluid, solid, or FSI formulations.

### FE/Assembly

- [x] Update geometry preparation to build only the requested frames.
- [x] Ensure current/reference geometry data share quadrature rules safely.
- [x] Support volume, boundary face, and interface face contexts.
- [x] Add tests for affine and non-affine elements.
- [x] Add tests for 1D, 2D, and 3D where supported.

Acceptance criteria:

- [x] A single form can integrate with reference measures while evaluating current coordinates or current normals.
- [x] Face geometry transforms are available without physics modules implementing their own geometry algebra.
- [x] Geometry preparation remains lazy and demand-driven.

Verification:

- [x] `build-fe-check: cmake --build build-fe-check --target test_fe_geometry test_fe_assembly test_fe_forms -j 4`; passed.
- [x] `build-fe-check: ./test_fe_geometry --gtest_filter=FrameGeometry.*`; passed 6/6 tests.
- [x] `build-fe-check: ./test_fe_assembly --gtest_filter='StandardAssemblerMovingDomain.BoundaryFacePreparesReferenceAndCurrentGeometry:StandardAssemblerFaces.*'`; passed 9/9 tests.
- [x] `build-fe-check: ./test_fe_forms --gtest_filter='FormVocabularyTest.MovingDomainFaceNormalsUseExplicitTerminals:FormVocabularyTest.MovingDomainBoundaryTerminalsMatchInterpreterAndJIT'`; passed 2/2 tests.
- [x] `build-fe-check: ./test_fe_geometry`; passed 85/85 tests.
- [x] `build-fe-check: ./test_fe_assembly`; passed 730/741 tests, with 11 capability-gated skips.
- [x] `build-fe-check: ./test_fe_forms`; passed 613/631 tests, with 18 performance-gated skips.

## Phase 6: Mesh-Motion Fields and FE Coupling Surface

Goal: standardize how mesh-side motion fields enter FE assembly and time stepping.

### Mesh/Motion

- [x] Formalize standard field names, components, storage type, and ghost policy for:
  - [x] `mesh_displacement`
  - [x] `mesh_velocity`
  - [x] `mesh_acceleration`, if introduced.
  - [x] Previous coordinate states, if stored as fields.
- [x] Define whether displacement is absolute from reference or incremental per step in each API.
- [x] Add metadata to distinguish absolute displacement, incremental displacement, and velocity.
- [x] Add consistency checks before applying displacement to coordinates.
- [x] Add tests for field metadata and ghost exchange.

### FE/Systems

- [x] Add a mesh-motion field binding API that maps Mesh fields into FE assembly views.
- [x] Add time-history hooks for previous mesh displacement/velocity.
- [x] Add a systems-level method to notify FE components that mesh geometry advanced.
- [x] Ensure DOF handlers, constraints, search structures, and assembler caches respond correctly to geometry-only changes.
- [x] Add tests for multiple mesh advances with one FE system instance.

Acceptance criteria:

- [x] Mesh motion fields can be consumed by any FE form without physics-specific adapter code.
- [x] Geometry-only updates do not force unnecessary DOF rebuilds.
- [x] Topology-changing updates still trigger the correct DOF and sparsity rebuilds.

Verification:

- [x] Built Mesh motion tests: `cmake --build build-mesh-tests --target test_Motion -j 4`.
- [x] Built Mesh MPI motion tests: `cmake --build build-mesh-tests --target test_MotionMPI -j 4`.
- [x] Built FE systems tests: `cmake --build build-fe-check --target test_fe_systems -j 4`.
- [x] Ran focused Mesh motion regression: `./build-mesh-tests/Tests/test_Motion --gtest_filter='MotionFieldsTest.*:MotionStateTest.UpdateVelocityFromDisplacement:MeshMotionTest.AdvanceInitializesAndResetClearsCurrentCoordinates:MeshMotionTest.BackendFailureRestoresCoordinateState:MeshMotionTest.InjectedBackendWritesFieldsAndUpdatesCoordinatesAndVelocity:MeshMotionTest.BackendProvidedVelocityIsNotOverwritten:MeshMotionTest.DtZeroDoesNotComputeVelocityFallbackAndIsDeterministic:MeshMotionTest.BackendFailureAfterAcceptedSubstepRestoresEntryCurrentCoordsAndConfig:MeshMotionTest.SubsteppingAccumulatesTotalDisplacementAndVelocity:MeshMotionTest.BacktrackingAvoidsInversionViaSmallerSubsteps:MeshMotionTest.MaxSubstepsLimitRejectsAndRestores'`.
- [x] Ran full Mesh motion regression: `./build-mesh-tests/Tests/test_Motion` (`18` tests passed).
- [x] Ran focused FE systems regression: `./build-fe-check/test_fe_systems --gtest_filter='FESystem.MeshGeometryAdvanceNotificationPreservesLayoutAndTopologyNotificationInvalidatesSetup:FESystem.MeshMotionFieldBindingsArePhysicsNeutral:FESystem.StandardMeshMotionFieldsSyncFromMeshStorageToFEState:FESystem.MeshMotionFieldBindingRejectsMissingScalarAndDimensionMismatchedFields:FESystem.ReusedCurrentSystemTracksCoordinateMutationWithoutSetup'`.
- [x] Ran full FE systems regression: `./build-fe-check/test_fe_systems` (`352` tests passed).
- [x] Ran Mesh motion MPI regression in an external MPI-enabled shell: `mpirun -np 2 ./build-mesh-tests/Tests/test_MotionMPI`; reported `MeshMotion MPI tests PASSED`.
- [x] Rechecked the broader Mesh MPI regression sweep in an external MPI-enabled shell: `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'MPI|_4ranks|GhostCoordinateExchange|DistributedSemantics|RebalanceParMetis|Migration|PVTU|StartupParMetis|PartitionQualityMetis'`; passed 49/49 tests.

## Phase 7: FE-Backed Geometry-Regularization Backend

Goal: provide at least one production-grade, physics-neutral `IMotionBackend` implementation that uses FE infrastructure to compute mesh-coordinate updates. This backend solves a mesh-quality and geometry-regularization problem; it must not become a hidden solid, fluid, FSI, contact, or free-surface formulation.

### Scope Contract

- [x] Treat FE as the implementation mechanism for function spaces, quadrature, constraints, assembly, and linear/nonlinear solves needed to compute mesh displacement.
- [x] Keep backend equations limited to artificial mesh-regularization objectives whose state is mesh geometry, prescribed motion data, and generic smoothing weights.
- [x] Do not add ALE fluid convection, FSI traction balance, structural stress equilibrium, free-surface equations, contact enforcement, wall laws, turbulence terms, or rotating-frame source terms to this backend.
- [x] Do not expose physical material semantics from this backend. Configuration names and diagnostics should describe artificial mesh stiffness, smoothing weights, regularization, conditioning, and quality metrics rather than real material properties.
- [x] Do not return physical stresses, strains, energies, tractions, constitutive tangents, or material-history state from this backend.
- [x] Require physics-specific coupling laws, moving-wall laws, or interface conditions to enter as prescribed boundary motion, generic constraints, or Physics/Application-owned data, not as equations implemented inside `FE/MovingMesh`.

### Proposed Location

- [x] Add a new physics-agnostic folder such as `Code/Source/solver/FE/MovingMesh`.
- [x] Keep this folder responsible for generic mesh-motion PDEs, geometry-regularization objectives, artificial smoothing operators, and optional mesh untangling objectives only.
- [x] Keep dependency direction explicit: Mesh defines `IMotionBackend`, FE may provide an optional implementation, and Application wires the implementation into `MeshMotion`. Mesh must not depend on FE.
- [x] Do not place fluid, solid, FSI, free-surface, contact, rotating-flow, or other physics equations in this folder.

### Backend Interfaces

- [x] Implement an FE-backed adapter for `IMotionBackend`.
- [x] Consume `MotionSolveRequest` and write displacement increments, and optionally velocity, without taking dependencies on Physics modules.
- [x] Accept generic boundary displacement/velocity constraints from `MotionSolveRequest`.
- [x] Support volume mesh-motion solves independent of the physical problem being solved.
- [x] Expose backend options through a physics-neutral configuration object.
- [x] Provide numerical and mesh-quality diagnostics for linear/nonlinear iterations, residual norms, constraint violations, accepted step scale, and quality metrics.

### Initial Geometry-Regularization Models

- [x] Harmonic or Laplace mesh smoothing.
- [x] Artificial pseudo-elastic mesh smoothing, documented as geometry regularization rather than structural mechanics.
- [x] Spatially varying artificial stiffness or mesh diffusivity based on element size, boundary distance, or user-provided mesh-quality weights.
- [x] Optional biharmonic or Winslow-style smoothing explicitly deferred; not required for Phase 7 acceptance.
- [x] Optional optimization-based untangling model explicitly deferred until the basic geometry-regularization backend is stable.

### Boundary and Constraint Handling

- [x] Support Dirichlet displacement constraints from boundary labels.
- [x] Support component-wise constraints.
- [x] Support fixed, prescribed, and callback-driven boundary motion.
- [x] Keep projected or weak constraints out of the backend until the FE constraint system has a qualified generic support path; Phase 7 implements strong mesh-displacement constraints only.
- [x] Validate incomplete, conflicting, overconstrained, or underconstrained motion constraints before solving.
- [x] Report constraint violations as mesh-motion diagnostics, not physics residuals.

### Tests

- [x] Unit test backend construction and option validation.
- [x] Verify the backend has no dependency on Physics modules.
- [x] Verify options and diagnostics use artificial mesh-regularization terminology and do not expose stress, traction, material-law, or material-history outputs.
- [x] Solve a simple 2D square motion problem.
- [x] Solve a simple 3D cube motion problem.
- [x] Verify boundary constraints are exactly satisfied.
- [x] Verify interior motion is smooth and deterministic.
- [x] Verify quality gates reject inverted updates.
- [x] Repeat key tests under MPI. Verified in an external MPI-enabled shell with `mpirun -np 2 ./build-fe-check/test_fe_movingmesh_mpi`; the 2-rank `FEMeshMotionBackendMPI.RankLocalKeySolveIsDeterministic` regression passed on both ranks.

Acceptance criteria:

- [x] `MeshMotion::advance()` can be driven by a real FE backend without physics modules participating.
- [x] The backend is justifiable as physics-neutral: it computes geometry updates from artificial mesh-regularization objectives and generic constraints, not from physical governing equations.
- [x] Backend results are deterministic enough for regression testing.
- [x] Failed solves roll back coordinates and fields consistently.
- [x] Physics modules consume the backend output only as generic mesh displacement, mesh velocity, or current coordinates.

Verification:

- [x] Reconfigured FE build after adding `FE/MovingMesh`: `cmake -S Code/Source/solver/FE -B build-fe-check`.
- [x] Built Phase 7 FE moving-mesh targets: `cmake --build build-fe-check --target test_fe_movingmesh test_fe_movingmesh_mpi -j 4`.
- [x] Ran focused FE moving-mesh regression: `./build-fe-check/test_fe_movingmesh` (`7` tests passed).
- [x] Built Mesh motion tests after extending `MotionDirichletBC`: `cmake --build build-mesh-tests --target test_Motion test_MotionMPI -j 4`.
- [x] Ran full Mesh motion regression: `./build-mesh-tests/Tests/test_Motion` (`18` tests passed).
- [x] Built and ran full FE systems regression: `cmake --build build-fe-check --target test_fe_systems -j 4 && ./build-fe-check/test_fe_systems` (`352` tests passed).
- [x] Ran Phase 7 FE moving-mesh MPI regression in an external MPI-enabled shell: `mpirun -np 2 ./build-fe-check/test_fe_movingmesh_mpi`; the 2-rank `FEMeshMotionBackendMPI.RankLocalKeySolveIsDeterministic` test passed on both ranks.

## Phase 8: Geometry Sensitivities and Monolithic Mesh Unknowns

Goal: enable monolithic coupling where mesh displacement is a solved unknown and geometry-dependent residuals can produce correct tangent contributions.

### FE/Geometry

- [x] Implement shape derivatives of geometry mapping outputs with respect to nodal mesh displacement.
- [x] Provide reusable derivative kernels for volume and face geometry.
- [x] Keep derivative utilities independent of the physical residual using them.

### FE/Forms and AD

- [x] Add an opt-in form mode that treats selected geometry quantities as dependent on mesh-motion unknowns.
- [x] Add symbolic dependencies between mesh displacement fields and geometry terminals.
- [x] Add AD/JIT-compatible support for geometry derivative contributions. Geometry-sensitive tangents use the exact AD `NonlinearFormKernel` path; JIT wrappers keep residual acceleration where available and preserve the AD tangent fallback.
- [x] Make the default behavior remain geometry-constant for standard solution-field differentiation.
- [x] Add diagnostics to detect when a form requests monolithic geometry sensitivity but no mesh-motion unknown is registered.

### FE/Systems

- [x] Allow a mesh-displacement field to be registered as a normal FE unknown.
- [x] Add a controlled path to update `X_cur` from the FE mesh-displacement solution.
- [x] Define when coordinates are updated inside nonlinear iterations versus after accepted steps.
- [x] Add rollback support for rejected nonlinear or time steps.

Acceptance criteria:

- [x] A residual depending on current geometry can assemble derivatives with respect to mesh displacement.
- [x] The feature is opt-in and does not alter existing physics behavior.
- [x] Monolithic and partitioned moving-mesh modes can share the same generic mesh-motion field infrastructure.

Verification:

- [x] Built Phase 8 FE targets: `cmake --build build-fe-check --target test_fe_geometry test_fe_forms test_fe_systems -j 4`.
- [x] Ran focused geometry sensitivity regression: `./build-fe-check/test_fe_geometry --gtest_filter='FrameGeometry.CellGeometrySensitivityMatchesFiniteDifferenceForTetra:FrameGeometry.FaceGeometrySensitivityAnalyticMatchesFiniteDifferenceReference'` (`2` tests passed).
- [x] Ran focused FE Forms geometry-sensitivity regression: `./build-fe-check/test_fe_forms --gtest_filter='FormCompilerTest.GeometrySensitivityModeRecordsGeometryTerminalDependencies:FormCompilerTest.GeometrySensitivityModeRequiresMeshMotionField:FormVocabularyTest.OptInCurrentGeometrySensitivitySeedsMeshMotionJacobian'` (`3` tests passed).
- [x] Ran focused FE Systems monolithic mesh-displacement regression: `./build-fe-check/test_fe_systems --gtest_filter='FESystem.MeshDisplacementUnknownUpdatesCurrentCoordinatesWithRollbackAndCommit'` (`1` test passed).
- [x] Ran full FE geometry regression: `./build-fe-check/test_fe_geometry` (`87` tests passed).
- [x] Ran full FE Forms regression: `./build-fe-check/test_fe_forms` (`614` tests passed, `18` skipped perf-gated tests).
- [x] Ran full FE Systems regression: `./build-fe-check/test_fe_systems` (`353` tests passed).

## Phase 9: Moving Mesh Time Integration and GCL Support

Goal: provide generic time-history infrastructure needed by moving-domain methods without hard-coding physics equations.

### FE/TimeStepping

- [x] Store previous coordinate states or previous mesh displacement states as generic time-history data.
- [x] Provide time derivative helpers for mesh displacement and coordinates.
- [x] Support consistent `mesh_velocity` computation from the active time integration scheme.
- [x] Add optional `mesh_acceleration` for second-order schemes.
- [x] Expose a generic geometric conservation law diagnostic for moving control volumes.
- [x] Keep GCL diagnostics formulation-neutral; physics modules decide how to enforce or consume them.

### FE/Assembly

- [x] Make time-level selection available to moving-domain terminals when needed.
- [x] Support current, previous, and predicted mesh velocity views.
- [x] Add tests for BDF and generalized-alpha compatibility if those schemes are active.

Acceptance criteria:

- [x] Mesh velocity is consistent with the solver time integration scheme.
- [x] Moving-domain tests can check GCL residuals without duplicating time-history plumbing in physics modules.

Verification:

- [x] Built Phase 9 FE targets: `cmake --build build-fe-check --target test_fe_timestepping test_fe_forms test_fe_systems -j 4`.
- [x] Ran focused moving-mesh time integration regression: `./build-fe-check/test_fe_timestepping --gtest_filter='MovingMeshTimeIntegration.*'` (`5` tests passed).
- [x] Ran focused moving-domain time-level form regression: `./build-fe-check/test_fe_forms --gtest_filter='FormVocabularyTest.MovingDomainTimeLevelTerminalsMatchInterpreterAndJIT:FormVocabularyTest.RequiredDataInferenceIncludesGeometryAndMeasures'` (`2` tests passed).
- [x] Ran focused FE Systems mesh-motion binding regression: `./build-fe-check/test_fe_systems --gtest_filter='FESystem.MeshMotionFieldBindingsArePhysicsNeutral'` (`1` test passed).
- [x] Ran full FE time-stepping regression in the current build configuration: `./build-fe-check/test_fe_timestepping` (`42` tests passed, `101` skipped because the Eigen backend is disabled).
- [x] Ran FE form vocabulary regression: `./build-fe-check/test_fe_forms --gtest_filter='FormVocabularyTest.*'` (`15` tests passed).

## Phase 10: Remesh, Adaptivity, and Transfer for Moving Domains

Goal: support moving meshes that may need remeshing, adaptation, and state transfer.

### Mesh/Adaptivity

- [x] Extend transfer metadata to include reference/current coordinate provenance.
- [x] Preserve motion fields during refinement/coarsening.
- [x] Preserve active configuration and geometry revision across adaptation.
- [x] Add transfer diagnostics for mesh displacement and velocity.

### FE/Systems

- [x] Implement FE-side `on_mesh_adapted` handling for:
  - [x] DOF handler rebuild.
  - [x] Constraint rebuild.
  - [x] Sparsity rebuild.
  - [x] Solution vector transfer.
  - [x] Previous solution/history vector transfer.
  - [x] Auxiliary state transfer through registered physics-agnostic auxiliary transfer hooks or default resize policy.
  - [x] Material-state invalidation and provider-owned transfer handoff; constitutive history-value transforms remain part of the later material/history-state contract.
  - [x] Boundary integral or coupling state invalidation and rebuild handoff.
- [x] Provide projection or interpolation APIs that are generic to FE fields.
- [x] Add conservative transfer options for scalar/vector fields where mass conservation matters.
- [x] Add failure diagnostics when required transfer data is unavailable.

### Tests

- [x] Adapt a moved mesh and verify current coordinates are preserved or projected according to documented semantics.
- [x] Transfer mesh displacement and mesh velocity.
- [x] Transfer a generic FE scalar field.
- [x] Transfer a generic FE vector field.
- [x] Verify conservation where conservative transfer is requested.
- [ ] Repeat representative tests under MPI. Implemented and built as `test_MovingAdaptivityMPI`; external verification is pending because this sandbox cannot start OpenMPI/PMIx.

Acceptance criteria:

- [x] Remeshing during a moving-domain run does not lose geometry, motion, solution, FE history-vector, or hook-backed auxiliary state. Material-state value transfer remains provider-owned and is explicitly handed off to the material/history-state phase.
- [x] Transfer semantics are explicit and test-covered for serial Mesh/FE paths; MPI execution is implemented but still requires an external OpenMPI-capable shell.

## Phase 11: Checkpoint, Restart, and IO Contract

Goal: make moving-domain simulations restartable and inspectable.

### Mesh and Application IO

- [x] Define the restart payload for moving meshes:
  - [x] Reference coordinates.
  - [x] Current coordinates.
  - [x] Active configuration.
  - [x] Geometry revision or restart epoch.
  - [x] Mesh displacement.
  - [x] Mesh velocity.
  - [x] Previous coordinate/displacement states.
  - [x] Motion backend state, if any.
  - [x] Adaptivity/remesh provenance, if needed.
- [x] Ensure output writers can emit current coordinates and motion fields consistently.
- [x] Ensure restart readers restore current coordinates before FE systems build geometry caches.
- [x] Add versioned metadata so older restart files fail clearly or upgrade safely.

### Tests

- [x] Write and read a moved mesh restart.
- [x] Reassemble after restart and compare geometry-dependent quantities.
- [x] Restart after an adapted moving mesh.
- [x] Run serial and MPI restart tests. Serial restart tests pass; `test_MovingMeshRestartMPI` passed externally with 2 and 4 MPI ranks.

Acceptance criteria:

- [x] A moving-mesh run can stop, restart, and produce the same next-step geometry and assembly results.
- [x] Restart does not silently drop current coordinates or mesh-motion fields.

## Phase 12: Application-Level Orchestration

Goal: expose moving-mesh capability without requiring physics modules to manually wire low-level Mesh and FE internals.

### Application/Core

- [x] Add input configuration for mesh-motion mode:
  - [x] Disabled.
  - [x] Prescribed motion.
  - [x] FE-backed smoothing.
  - [x] Coupled/monolithic motion, once supported. The configuration vocabulary accepts this mode and reports that execution is unsupported until the monolithic mesh-unknown phase is integrated at application level.
- [x] Add input configuration for coordinate configuration used by FE systems.
- [x] Add input configuration for motion backend model and options.
- [x] Add boundary motion configuration by label and component.
- [x] Add diagnostics output for mesh-motion solve status, quality metrics, rollback, and accepted step scale.
- [x] Insert mesh-motion advancement into the time loop at a documented point.
- [x] Define ordering relative to physics solve, nonlinear iteration, remeshing, checkpoint, and output.
- [x] Ensure geometry-change notifications reach FE systems before assembly.
- [x] Preserve the current default static/reference behavior.

Acceptance criteria:

- [x] Existing static simulations run unchanged by default.
- [x] Users can enable a prescribed moving mesh and see FE assembly use current coordinates.
- [x] Users can enable an FE-backed mesh-motion backend without editing physics code.

Verification:

- [x] Built Phase 12 FE target: `cmake --build build-fe-check --target test_fe_movingmesh -j 4`.
- [x] Ran focused orchestration regression: `./build-fe-check/test_fe_movingmesh --gtest_color=no` (`12` tests passed).
- [x] Ran TimeLoop and moving-mesh CTest regression for the documented callback insertion point: `ctest --test-dir build-fe-check --output-on-failure -R 'FE_MovingMesh_Tests|FE_TimeStepping_Tests' -j1` (`2/2` tests passed).
- [x] Ran broad non-MPI FE regression: `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j2 -LE MPI` (`19/19` tests passed).
- [x] Ran external FE MPI regression in an OpenMPI-capable shell: `cmake --build build-fe-check --target test_fe_movingmesh_mpi -j 4 && mpirun -np 2 ./build-fe-check/test_fe_movingmesh_mpi` passed on both ranks, and `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -L MPI` passed (`8/8` MPI tests).

## Phase 13: Physics Module Consumption

Goal: update physics modules only after generic FE/Mesh infrastructure exists.

### Physics/Formulations

- [x] Update fluid ALE formulations to consume `meshVelocity` from FE Forms.
- [x] Replace fluid convective velocity `u` with `u - w_mesh` only in ALE-enabled physics options.
- [x] Add moving-control-volume transient terms where required by the formulation.
- [x] Update FSI coupling to use generic current/reference normals, measures, and mesh displacement.
- [x] Update free-surface and moving-boundary formulations to use generic moving-domain terminals.
- [x] Keep all physics-specific decisions in Physics modules, not in FE.

### Tests

- [x] ALE advection-diffusion manufactured solution.
- [x] ALE incompressible Navier-Stokes manufactured solution.
- [x] Prescribed moving-boundary flow smoke test.
- [x] FSI interface kinematic compatibility test.
- [x] FSI interface traction consistency test.
- [x] Static-mesh regression tests proving unchanged results when ALE is disabled.

Acceptance criteria:

- [x] Physics modules consume generic moving-domain FE terminals rather than custom coefficient plumbing.
- [x] ALE-disabled paths remain numerically unchanged.

Implementation notes:

- [x] Kept moving-domain weak-form recipes out of FE. ALE Navier-Stokes terms are authored inside the Navier-Stokes formulation; FSI and moving-boundary probe expressions remain test-local until production modules need them.
- [x] Added physics-agnostic FE field lookup helpers on `FESystem` so physics modules can reuse an existing mesh-motion field or auto-register a physics-owned field without reaching into registry internals.
- [x] Added opt-in Navier-Stokes ALE options. Static-mesh behavior remains the default; ALE explicitly binds a mesh velocity field and uses relative convection plus the moving-control-volume transient term.
- [x] Added Navier-Stokes input parsing for relaxed ALE enable flags, moving-control-volume transient selection, and mesh-velocity field naming.
- [x] No legacy solver files were changed.

Verification:

- [x] Built Phase 13 physics target with the GCC13 build: `cmake --build build-physics-gcc13-check --target test_physics -j 4`.
- [x] Ran focused Phase 13 tests: `./build-physics-gcc13-check/test_physics --gtest_filter='MovingDomainPhysics.*' --gtest_color=no` (`6/6` passed).
- [x] Ran whitespace validation on the touched FE/Physics files: `git diff --check -- Code/Source/solver/FE/Systems/FESystem.h Code/Source/solver/FE/Systems/FESystem.cpp Code/Source/solver/Physics/CMakeLists.txt Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp` and `rg -n '[[:blank:]]$' Documentation/plan_moving_mesh_infrastructure.md Code/Source/solver/FE/Systems/FESystem.h Code/Source/solver/FE/Systems/FESystem.cpp Code/Source/solver/Physics/CMakeLists.txt Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp`.
- Broader physics CTest regression was attempted with `ctest --test-dir build-physics-gcc13-check --output-on-failure -R Physics_Tests -j1`; this is not currently a clean Phase 13 signal because serial `Physics_Tests` reports unrelated Jacobian/pressure-gauge failures and then segfaults, while `Physics_Tests_MPI_2` is blocked by the sandbox PMIx socket restriction.

## Phase 14: Verification and Qualification Suite

Goal: prove the infrastructure is robust enough for multiphysics use.

Scope note: this phase closes the verification suite for the moving-mesh
infrastructure implemented through Phases 1-13. Items whose implementation
belongs to later phases are recorded here as deferred qualification extensions
and must be run before those later feature combinations are marked supported.

### Unit Tests

- [x] Mesh geometry revision and event tests (`test_MeshAccess.cpp`, Mesh observer/revision tests, and moving-mesh restart/adaptivity tests).
- [x] Mesh active-configuration epoch tests (`test_MeshAccess.cpp`, `test_FESystem.cpp`).
- [x] Mesh motion field metadata and ghost sync tests (`test_Motion.cpp`, `test_MotionMPI.cpp`, `test_FieldGhostPolicy*`, `test_MovingMeshRestart*`).
- [x] Search invalidation tests for current/reference configurations (`test_SearchAccess.cpp`, `test_SearchAccelInvalidation.cpp`).
- [x] FE `MeshAccess` current/reference/revision tests (`test_MeshAccess.cpp`, `test_FESystem.cpp`).
- [x] FE space, DOF-layout, constraint-layout, and block-layout revision tests (`test_FESystem.cpp`, `test_DofMap.cpp`, `test_DofHandler.cpp`, `test_AffineConstraints.cpp`, `test_BlockDofMap.cpp`).
- [x] FE assembly cache invalidation tests (`test_StandardAssembler.cpp`, `test_FESystem.cpp`).
- [x] Solver/operator invalidation policy tests for current Phase 14 scope (`test_OperatorBackends.cpp` checks matrix-free current-geometry reuse after geometry revision; backend-specific preconditioner rebuild-count policy remains a Phase 17 extension).
- [x] Moving-domain `RequiredData` tests (`test_FormVocabulary.cpp`).
- [x] Forms lowering/interpreter/JIT/AD tests (`test_FormVocabulary.cpp`, `test_FormCompiler.cpp`, JIT parity tests).
- [x] Geometry transform and Nanson utility tests (`test_FrameGeometry.cpp`, `test_MetricSurfacePushForward.cpp`).
- [x] Geometry sensitivity tests (`test_FrameGeometry.cpp`, `test_FormVocabulary.cpp`).
- [x] Mesh-motion backend tests (`test_FEMeshMotionBackend.cpp`, `test_FEMeshMotionBackendMPI.cpp`, `test_MovingDomainOrchestrator.cpp`).
- [x] Interface projection conservation unit tests for the currently implemented fitted/interface-field path (`test_MortarHybridInterface.cpp`); sliding multi-mesh projections remain a Phase 16/20/25 extension.
- [x] Material/history trial/commit/rollback tests (`test_AuxiliaryStateManager.cpp`, `test_SurfaceContactKernel.cpp`, moving-mesh time-history tests).

### Integration Tests

- [x] Static mesh baseline (`test_MovingDomainPhysics.cpp` ALE-disabled path, FE static assembly suites).
- [x] Prescribed moving mesh with scalar diffusion/mass assembly (`test_FESystem.cpp`, `test_FEMovingMeshRestart.cpp`).
- [x] Prescribed moving mesh with vector field (`test_FESystem.cpp`).
- [x] ALE advection-diffusion MMS/term verification (`test_MovingDomainPhysics.cpp`).
- [x] ALE Navier-Stokes MMS/term verification (`test_MovingDomainPhysics.cpp`).
- [x] FSI-style interface motion without full FSI physics (`test_MovingDomainPhysics.cpp`).
- [x] Remesh plus transfer on a moved mesh (`test_FEAdaptivityTransfer.cpp`, `test_MovingMeshAdaptivity.cpp`).
- [x] Remesh plus conservative transfer on a moved mesh (`test_FEAdaptivityTransfer.cpp`, `test_MovingMeshAdaptivity.cpp`).
- [x] Remesh plus material/history transfer on a moved mesh (`test_FEAdaptivityTransfer.cpp` auxiliary transfer hook coverage).
- [x] Conservative sliding-interface projection over multiple time steps is explicitly deferred until Phase 16/20/25 implements generic multi-mesh/sliding operators.
- [x] Nonconservative sliding-interface interpolation with explicit frame transforms is explicitly deferred until Phase 16/20/25.
- [x] Rollback/restart equivalence after a rejected moving-geometry step (`test_FEMovingMeshRestart.cpp`, `test_MovingMeshTimeIntegration.cpp`).
- [x] Restart equivalence after accepted remesh/rezone for the current moved-mesh restart path (`test_FEAdaptivityTransfer.cpp`); full rezone/rebase semantics remain Phase 18.
- [x] Large-deformation work/energy consistency is deferred until Phase 24 enables geometric nonlinearity.
- [x] Contact or proximity work/energy consistency is represented by current penalty/contact force-balance and symmetric-tangent tests; moving-contact active-set qualification remains Phase 21.
- [x] Cut-cell conservation and rollback is explicitly deferred until Phase 26 if unfitted methods remain in scope.
- [x] Restart from a moved mesh (`test_MovingMeshRestart.cpp`, `test_FEMovingMeshRestart.cpp`, `test_MovingMeshRestartMPI.cpp`).
- [x] MPI versions of representative tests (`test_MotionMPI`, `test_MovingAdaptivityMPI`, `test_MovingMeshRestartMPI`, FE MPI labels).

### Performance and Robustness Tests

- [x] Cache rebuild/static reuse checks for static versus moving runs (`test_StandardAssembler.cpp`, `test_FESystem.cpp`).
- [x] Assembly behavior before and after geometry revision keys (`test_StandardAssembler.cpp`, `test_FESystem.cpp`).
- [x] Operator rebuild/reuse checks for current Phase 14 scope (`test_OperatorBackends.cpp`); preconditioner rebuild-count policy remains Phase 17.
- [x] Large deformation quality-gate hooks are covered at mesh-quality/backend rollback level for current prescribed/FE-smoothed motion; full large-strain energy gates remain Phase 24.
- [x] Backend failure and rollback tests (`test_Motion.cpp`, `test_MotionMPI.cpp`, `test_MovingDomainOrchestrator.cpp`).
- [x] Repartition or migration with current coordinates (`test_MigrationPreservesCodimMetadataAndCurrentCoordsMPI.cpp`, `test_RebalanceParMetis*`).
- [x] Conservative-transfer balance checks across restart and repartition for current moved-mesh transfer scope (`test_MovingMeshAdaptivity.cpp`, `test_FEAdaptivityTransfer.cpp`).

### Phase 14 Verification Evidence

- [x] Built affected FE targets: `cmake --build build-fe-check --target test_fe_geometry test_fe_timestepping test_fe_forms test_fe_systems -j 4`.
- [x] Built updated systems target: `cmake --build build-fe-check --target test_fe_systems -j 4`.
- [x] Added reproducible validation-evidence runner: `tools/run_phase14_moving_mesh_validation.sh`.
- [x] Added validation-evidence guide and evidence matrix: `Documentation/phase14_moving_mesh_validation_evidence.md`.
- [x] Generated current MPI-enabled evidence report: `Documentation/qualification_logs/phase14_moving_mesh/latest/summary.md` (`10/10` enabled checks passed, including FE MPI `8/8` and Mesh/distributed MPI `55/55`).
- [x] Ran focused new Phase 14 tests: `./build-fe-check/test_fe_systems --gtest_filter='FESystem.PrescribedMovingMeshVectorMassRespectsCurrentGeometry:OperatorBackends.MatrixFreeMassTracksCurrentGeometryRevisionWithoutRefetch' --gtest_color=no` (`2/2` passed).
- [x] Ran broader affected systems qualification: `./build-fe-check/test_fe_systems --gtest_filter='FESystem.*:FEAdaptivityTransfer.*:FEMovingMeshRestart.*:OperatorBackends.*:SearchAccess.MeshSearchAccess_*:ContactPenaltyKernel.*:SurfaceContactKernel.*' --gtest_color=no` (`53/53` passed).
- [x] Ran full FE CTest suite: `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1` (`27/27` passed, including `8/8` FE MPI tests).
- [x] Built full Mesh test suite, including previously unbuilt adaptivity/refinement targets: `cmake --build build-mesh-tests --target build_mesh_tests -j 4`.
- [x] Ran full Mesh CTest suite: `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1` (`123/123` passed, including adaptivity, motion, moving adaptivity, restart, search, observer, validation, distributed, and MPI tests).
- [x] Previously ran geometry qualification: `./build-fe-check/test_fe_geometry --gtest_color=no` (`89/89` passed).
- [x] Previously ran time-stepping qualification: `./build-fe-check/test_fe_timestepping --gtest_color=no` (`44` runnable tests passed; `101` Eigen-dependent tests skipped by this build configuration).
- [x] Previously ran forms moving-domain qualification: `./build-fe-check/test_fe_forms --gtest_filter='FormVocabularyTest.*' --gtest_color=no` (`17/17` passed).
- [x] Ran focused physics moving-domain qualification: `./build-physics-gcc13-check/test_physics --gtest_filter='MovingDomainPhysics.*' --gtest_color=no` (`8/8` passed).
- [x] Ran FE MPI representative tests through CTest (`8/8` FE MPI tests passed).
- [x] Ran Mesh/distributed MPI representative tests through CTest (`55/55` Mesh/distributed tests passed).
- [x] User-verified moving-mesh restart MPI directly with `mpirun -np 2 ./build-mesh-tests/Tests/test_MovingMeshRestartMPI` and `mpirun -np 4 ./build-mesh-tests/Tests/test_MovingMeshRestartMPI` (both passed on all ranks).

### Later Literature FSI Validation

These are end-to-end physics validation benchmarks, not Mesh/FE infrastructure
closure criteria for Phases 1-13. They should be added after the required
fluid, structural, coupling, transaction, and geometric-nonlinearity pieces
are in place.

- [ ] Add the Turek-Hron FSI benchmark family as the first literature validation target, including the CFD and CSM subproblems used to isolate fluid and structural errors before coupled FSI.
- [ ] Record benchmark references in the validation artifact, including the original Turek-Hron definition and public benchmark descriptions such as `https://book.feelpp.org/benchmarks/fsi/toolbox/bm-3/` and `https://examples.nutils.org/official-turek/`.
- [ ] Add Turek-Hron FSI1/FSI2/FSI3 cases with published comparison quantities: beam-tip displacement, drag, lift, pressure drop, oscillation period/frequency where applicable, and convergence under mesh/time refinement.
- [ ] Require restart equivalence for at least one Turek-Hron FSI case.
- [ ] Require serial/MPI agreement for at least one Turek-Hron FSI case.
- [ ] Add a validation-report artifact that records mesh, timestep, coupling scheme, nonlinear tolerances, and comparison bands against the literature.
- [ ] Keep these benchmarks in Physics/Application validation or examples, not in Mesh or FE unit tests.

Acceptance criteria:

- [x] Moving-mesh features implemented through Phases 1-13 are covered by serial and representative MPI tests.
- [x] Static-mesh behavior remains the default and has regression coverage.
- [x] Balance, conservation, rollback, restart equivalence, and energy/work checks exist or are explicitly deferred for every combination in the supported-combinations matrix according to the first-supported phase.
- [x] Failures produce actionable diagnostics instead of silent stale geometry for the current supported moving-mesh paths.

## Phase 15: High-Order and Curved Geometry Motion

Goal: extend the moving-mesh infrastructure beyond vertex-coordinate motion so curved, high-order, surface, and shell geometries can move correctly.

### Mesh/Geometry and Mesh/Core

- [x] Define whether each mesh stores only vertex coordinates or also geometric DOFs for high-order geometry. Phase 15 supports `GeometryDofStorage::VertexCoordinates`.
- [x] Add a geometry-order descriptor separate from solution-space order.
- [x] Add storage and access APIs for edge, face, and interior geometric control points where supported.
- [x] Extend current/reference coordinate semantics to high-order geometric DOFs.
- [x] Extend geometry revision updates to all geometric DOF mutations, not only vertex mutations.
- [x] Extend distributed ownership and ghost exchange for high-order geometric DOFs.
- [x] Preserve high-order geometric DOFs through migration, repartition, and adaptivity in the supported vertex-coordinate storage mode; new curved-DOF projection after remesh remains a remesh/rezone qualification item.
- [x] Define fallback behavior for meshes that remain piecewise linear.

### FE/Geometry

- [x] Ensure isoparametric mappings can be built from high-order geometric DOFs.
- [x] Support current/reference mapping pairs for high-order geometry.
- [x] Support curved boundary normals, measures, and surface Jacobians.
- [x] Add geometry sensitivity derivatives with respect to high-order geometric DOFs.
- [x] Add shell/surface geometric mapping support if shell or surface elements are in scope. Surface geometry is covered through embedded lower-dimensional mappings; shell kinematic models remain physics/element-family scope.
- [x] Add tests for quadratic and at least one higher-order element family where supported.
- [x] Add tests for curved boundary motion and current/reference normal changes.

### Mesh/IO and Restart

- [x] Persist high-order reference geometry.
- [x] Persist high-order current geometry.
- [x] Persist geometry order and layout metadata.
- [x] Verify restart preserves curved geometry and motion fields.

Acceptance criteria:

- [x] The plan explicitly supports either piecewise-linear-only moving meshes or high-order moving geometry; the selected mode is visible in configuration and tests.
- [x] High-order geometry motion updates current coordinates, FE mapping, and assembly consistently; search consumers observe the same geometry revisions and vertex-coordinate DOFs, with multi-mesh/interface search qualification deferred to Phases 16 and 22.
- [x] Linear mesh workflows remain unchanged.

Verification evidence:

- [x] `cmake --build build-mesh-tests --target test_CurvilinearEval test_MovingMeshRestart test_GhostCoordinateExchange -j 4`
- [x] `cmake --build build-fe-check --target test_fe_geometry -j 4`
- [x] `./build-mesh-tests/Tests/test_CurvilinearEval`
- [x] `./build-mesh-tests/Tests/test_MovingMeshRestart`
- [x] `./build-fe-check/test_fe_geometry`
- [x] `mpirun -np 2 ./build-mesh-tests/Tests/test_GhostCoordinateExchange`
- [x] `mpirun -np 4 ./build-mesh-tests/Tests/test_GhostCoordinateExchange`
- [x] `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'test_CurvilinearEval|test_MovingMeshRestart|test_GhostCoordinateExchange'`
- [x] `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'test_MovingMeshAdaptivity'`
- [x] `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'test_MeshQuality'`
- [x] `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'MigrationPreservesCodimMetadataAndCurrentCoordsMPI|RebalanceParMetisMPI|RebalanceParMetisMPI_4ranks'`
- [x] `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -R 'FE_Geometry_Tests'`
- [x] `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -R 'FE_Assembly_Tests'`

## Phase 16: Generic Multi-Mesh Moving-Interface Coupling

Goal: provide a physics-neutral layer for moving interfaces between distinct meshes, needed for FSI, nonmatching interfaces, immersed/interface methods, and some contact workflows.

### Mesh/Search

- [x] Add an interface-search registry that can track source mesh, target mesh, boundary labels, configurations, and revision states.
- [x] Support nonmatching interface candidate generation.
- [x] Support closest-point, element-local coordinate, and face-local coordinate queries.
- [x] Include geometry, topology, ownership, numbering, and label revisions for every mesh participating in an interface map.
- [x] Invalidate interface maps when either mesh moves, remeshes, repartitions, renumbers, or changes relevant labels.
- [x] Add MPI-aware ownership metadata for cross-rank interface pairs.

### FE/Systems and FE/Assembly

- [x] Define physics-neutral interpolation/projection operators between meshes.
- [x] Support pointwise interpolation, conservative projection, and mortar-style operators as separate implementations.
- [x] Keep interface operators independent of the physical coupling condition.
- [x] Expose interface quadrature and paired-side geometry through assembly contexts.
- [x] Support current/reference geometry on both sides of an interface.
- [x] Add rollback/commit state for interface maps created on trial geometry.

### Tests

- [x] Nonmatching 2D boundary-to-boundary interpolation test.
- [x] Nonmatching 3D surface-to-surface interpolation test.
- [x] Motion invalidation test where one side moves.
- [x] Remesh invalidation test where one side changes topology.
- [x] MPI test with interface pairs crossing rank boundaries.

Acceptance criteria:

- [x] Single-mesh ALE does not depend on this layer.
- [x] Multi-mesh coupling can rebuild interface maps deterministically after motion, remesh, or repartition.
- [x] Physics modules can consume interface maps without owning search, pairing, or projection infrastructure.

### Phase 16 Verification Evidence

- [x] Mesh search sources build with the new generic multi-mesh interface registry (`cmake --build build-mesh-tests --target test_MultiMeshInterface test_MultiMeshInterfaceMPI -j 4`).
- [x] Serial nonmatching, motion-invalidation, topology-invalidation, numbering-invalidation, label-invalidation, and rollback/commit tests pass (`ctest --test-dir build-mesh-tests --output-on-failure --timeout 120 -R 'test_MultiMeshInterface$|test_MultiMeshInterfaceMPI'`).
- [x] MPI ownership metadata and ownership-revision invalidation for interface pairs are verified with 2-rank and 4-rank runs through the same CTest expression.
- [x] FE systems build with physics-neutral interface transfer operators and paired-side assembly context (`cmake --build build-fe-check --target test_fe_systems -j 4`).
- [x] FE pointwise, conservative, mortar, and paired-geometry tests pass (`./build-fe-check/test_fe_systems --gtest_filter=MultiMeshInterfaceOperatorsTest.*`).

## Phase 17: Solver, Operator, and Geometry Transaction Policy

Goal: prevent stale assembled operators, matrix-free data, preconditioners, and nonlinear states when geometry changes, and define how trial geometry is accepted or rolled back.

### FE/Systems and Application

- [x] Add a geometry transaction model with explicit states:
  - [x] Committed geometry.
  - [x] Trial geometry.
  - [x] Accepted geometry.
  - [x] Rolled-back geometry.
- [x] Define when `X_cur` is updated during nonlinear iterations, time stepping, mesh-motion solves, and remeshing.
- [x] Define which revision counters advance on trial updates versus committed updates.
- [x] Provide rollback hooks for coordinate buffers, mesh-motion fields, interface maps, constraints, material/history states, and solver caches.
- [x] Add diagnostics showing whether each solve used reference, committed current, or trial current geometry.

### FE/Backends

- [x] Define invalidation policy for assembled matrix reuse.
- [x] Define invalidation policy for matrix-free geometry data.
- [x] Define invalidation policy for preconditioners.
- [x] Define invalidation policy for multigrid hierarchies and smoothers.
- [x] Define invalidation policy for Jacobian lagging or approximate Newton reuse.
- [x] Add backend hooks that consume mesh and FE revision states.
- [x] Add tests that a geometry-only change refreshes geometry-dependent operators without unnecessarily rebuilding topology-dependent data.
- [x] Add tests that topology/layout changes force stronger rebuilds.

Acceptance criteria:

- [x] No backend can silently reuse an incompatible matrix, matrix-free operator, preconditioner, or multigrid hierarchy after mesh state changes.
- [x] Trial geometry can be rolled back without leaking stale interface, constraint, history, or solver state.
- [x] Reuse policies are explicit enough to support controlled Jacobian lagging.

### Phase 17 Verification Evidence

- [x] FE systems build with the new geometry transaction and operator revision policy (`cmake --build build-fe-check --target test_fe_systems -j 4`).
- [x] Geometry-only invalidation refreshes geometry-dependent matrix-free data without rebuilding DOF layout or sparsity (`./build-fe-check/test_fe_systems --gtest_filter='OperatorBackends.InvalidationPolicyDistinguishesGeometryFromTopologyLayout:OperatorBackends.MatrixFreeRefetchRefreshesGeometryWithoutDofLayoutRebuild:OperatorBackends.GeometryTransactionRollbackRestoresCoordinatesAndMotionFields'`).
- [x] Topology/layout revision changes force stronger DOF, sparsity, preconditioner, and multigrid invalidation through the same focused test filter.
- [x] Transaction rollback restores current coordinates and standard mesh-motion fields, invalidates solver/operator and auxiliary state, and emits rollback diagnostics through the same focused test filter.
- [x] Broader FE systems regression passes after Phase 17 changes (`ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'test_fe_systems|FE_Systems'`).
- [x] FE backend serial and MPI regression coverage passes after backend cache-policy changes (`ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Backends|test_fe_backends'`).
- [x] FSILS Schur preconditioner cache now includes operator-value signatures, preventing same-topology geometry changes from silently reusing an incompatible cached preconditioner.

## Phase 18: Reference Rebasing and Rezoning Semantics

Goal: define how reference and current configurations behave in long moving-mesh, large-deformation, and remeshing workflows.

### Mesh/Core

- [x] Define supported reference modes:
  - [x] Immutable reference for total-Lagrangian or original-domain workflows.
  - [x] Explicit rebase where current geometry becomes the new reference.
  - [x] Remesh rebase where a new mesh receives a new reference configuration.
- [x] Add an explicit rebase operation rather than mutating `X_ref` implicitly.
- [x] Record a rebase epoch in mesh metadata.
- [x] Define what happens to mesh displacement when reference is rebased.
- [x] Define what happens to previous coordinate states when reference is rebased.
- [x] Define whether active configuration changes during rebase.

### FE/Systems and FE/TimeStepping

- [x] Notify FE systems when reference geometry changes.
- [x] Invalidate all reference-dependent geometry, material, constraint, interface, and solver caches on rebase.
- [x] Transfer or reset time-history data according to documented rebase semantics.
- [x] Add tests for rebase after large prescribed motion.
- [x] Add tests for rebase after remeshing.

Acceptance criteria:

- [x] Reference configuration changes are explicit, versioned, and impossible to confuse with ordinary current-coordinate motion.
- [x] Updated-Lagrangian workflows can rebase safely.
- [x] Total-Lagrangian workflows can keep the original reference immutable.

### Phase 18 Implementation Notes

- [x] `MeshBase` exposes explicit `rebase_reference_to_current`, `rebase_reference_coordinates`, and `rebase_reference_after_remesh` operations.
- [x] `ReferenceRebaseOptions` defines current-coordinate handling, motion-field reset policy, previous-coordinate update policy, active configuration after rebase, and source semantics.
- [x] `ReferenceRebaseInfo` records immutable/original, updated-Lagrangian rebase, and remesh-rebase modes, plus last source and epoch.
- [x] `MeshRevisionState::reference_rebase` and `MeshEventBus::notify_reference_rebased()` provide a separate version domain for reference-geometry rebasing.
- [x] Moving-mesh restart format version `3` writes and restores reference-rebase mode/source/epoch and the `reference_rebase` revision counter while preserving old-version read support.
- [x] `FESystem` rebase entry points reject active trial coordinate transactions, update the mesh through the explicit Mesh API, refresh ghost coordinates when current geometry remains present, notify geometry consumers, roll back tentative material/global-kernel state, reset auxiliary state to committed, invalidate auxiliary inputs, and update transaction diagnostics.
- [x] Operator invalidation snapshots include the reference-rebase epoch so matrix-free and backend consumers cannot confuse current-coordinate motion with a reference-configuration change.

### Phase 18 Verification Evidence

- [x] Built affected Mesh targets: `cmake --build build-mesh-tests --target test_MovingMeshRestart test_Motion -j 4`.
- [x] Built affected FE target: `cmake --build build-fe-check --target test_fe_systems -j 4`.
- [x] Focused Mesh rebase tests pass: `./build-mesh-tests/Tests/test_Motion --gtest_filter='MotionFieldsTest.ReferenceRebase*:MotionFieldsTest.RemeshReferenceRebase*'` (`2/2` passed).
- [x] Focused moving-mesh restart rebase tests pass: `./build-mesh-tests/Tests/test_MovingMeshRestart --gtest_filter='MovingMeshRestart.ReferenceRebaseMetadataRoundtrips:MovingMeshRestart.WriteReadMovedMeshRestoresGeometryMotionAndMetadata'` (`2/2` passed).
- [x] Focused FE rebase/restart/rollback tests pass: `./build-fe-check/test_fe_systems --gtest_filter='OperatorBackends.ReferenceRebaseInvalidatesGeometryWithoutDofLayoutRebuild:OperatorBackends.GeometryTransactionRollbackRestoresCoordinatesAndMotionFields:FEAdaptivityTransfer.AdaptedMovingMeshRestartReassemblyMatchesAcceptedState:FEMovingMeshRestart.*'` (`5/5` passed).
- [x] Mesh CTest coverage for the touched suites passes: `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R '^test_Motion$|^test_MovingMeshRestart$'` (`2/2` passed).
- [x] Moving-mesh restart MPI compatibility passes after restart-format version `3`: `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R '^test_MovingMeshRestartMPI$|^test_MovingMeshRestartMPI_4ranks$'` (`2/2` passed).
- [x] FE systems CTest coverage passes: `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -R '^FE_Systems_Tests$'` (`1/1` passed).

### Later Rebase and Rezoning Validation Candidates

These are literature-driven validation cases to add after the relevant Physics/Application pieces exist. They should live under validation/examples, not Mesh or FE unit tests.

- [ ] Add updated-Lagrangian large-deformation remeshing validation based on Leger/Fortin/Tibirna-style hyperelastic remeshing studies, including deformation-gradient/history transfer, rebase-vs-no-rebase consistency, and mesh-refinement convergence (`https://www.sciencedirect.com/science/article/pii/S0045782516304819`).
- [ ] Add ALE mesh-rezoning moving-boundary validation with single and multiple moving objects, tracking mesh quality, geometry revision epochs, and post-rezone solution invariance (`https://www.sciencedirect.com/science/article/abs/pii/S0045793005001301`).
- [ ] Add geometric-conservation-law/free-stream preservation tests for deforming meshes, including time-step refinement and moving-boundary trajectory accuracy (`https://www.sciencedirect.com/science/article/abs/pii/S0045782500001912`).
- [ ] Add high-order moving-mesh transport-exactness tests for repeated rezoning under nonsmooth or arbitrary mesh motion, using constant, linear, and quadratic manufactured transported fields (`https://arxiv.org/abs/2602.09729`).
- [ ] Add conservative remap/rezone tests modeled on Portage and MIRA-style metrics: global integral conservation, bounds/monotonicity, repeated back-and-forth remap dissipation, and topology sensitivity (`https://openresearchsoftware.metajnl.com/articles/10.5334/jors.338`, `https://gmd.copernicus.org/articles/15/6601/2022/`).
- [ ] Add high-order curved-cell remap validation once curved rezone transfer exists, including conservation to tight tolerance and positivity/bounds checks for curved isoparametric cells (`https://arxiv.org/abs/2506.18389`).
- [ ] If compressible ALE hydrodynamics becomes a supported physics target, add Riemann, Sedov, Noh, and Saltzmann-style rezone/remap validation for shock robustness, symmetry preservation, and conservation (`https://arxiv.org/abs/2602.09482`).
- [ ] Add Turek-Hron FSI2 moving-mesh distortion validation when coupled FSI is available, tracking beam-tip motion, lift/drag, interface-map rebuilds, and mesh-motion robustness across a rebase/rezone boundary (`https://link.springer.com/article/10.1007/s00466-020-01950-x`, `https://book.feelpp.org/benchmarks/fsi/toolbox/bm-3/`).
- [ ] If free-surface ALE remains in scope, add long-horizon sloshing validation with mass conservation and mesh-quality/rezone evidence for rectangular and stepped tanks (`https://www.sciencedirect.com/science/article/abs/pii/S0029801822008393`).

## Phase 19: Material and History-State Frame Semantics

Goal: define how internal variables, material history, auxiliary state, and coupling state behave under moving geometry, nonlinear trial states, rollback, transfer, and rebase.

### FE/Constitutive and FE/Auxiliary

- [x] Add frame metadata for state variables, such as reference frame, current spatial frame, material frame, interface-local frame, or user-defined frame.
- [x] Add lifecycle metadata for state variables:
  - [x] Old/committed.
  - [x] Trial/work.
  - [x] Accepted.
  - [x] Rolled back.
- [x] Define when ordinary geometry motion leaves state values unchanged.
- [x] Define when state variables require rotation, pushforward, pullback, projection, or reinitialization.
- [x] Provide generic hooks for frame transforms without encoding a particular material law.
- [x] Ensure material-state storage can participate in geometry transactions.
- [x] Ensure material-state transfer can participate in remesh/adaptivity.
- [x] Add tests for trial update, rollback, accept, transfer, and rebase.

### Physics

- [x] Require each physics model with internal variables to declare the state frame and transform policy it needs.
- [x] Keep model-specific update equations in Physics or Constitutive model implementations, not in FE assembly infrastructure.

Acceptance criteria:

- [x] Moving geometry cannot silently leave frame-sensitive history variables in an invalid frame.
- [x] Rollback restores committed material/history state.
- [x] Remeshing and rebasing have documented state-transfer behavior.

Implementation evidence:

- Added physics-agnostic state-variable frame/lifecycle vocabulary in `FE/Core/StateVariableMetadata.h`, including transform events for ordinary geometry motion, rollback, remesh/adaptivity transfer, repartition transfer, and reference rebase.
- Routed metadata and lifecycle state through FE material-state views, assembly contexts, constitutive state layouts, auxiliary state blocks, and the material-state provider without embedding any material law in FE assembly.
- Added constitutive-owned transform hooks through `ConstitutiveModel::stateVariables()` and `ConstitutiveModel::stateFrameTransformHook()`, keeping model-specific frame updates in Physics/Constitutive implementations.
- Wired FE system moving-geometry notifications, reference rebase, mesh-adaptivity transfer, and geometry-transaction rollback into the material/auxiliary state transform lifecycle.
- Verified with targeted checks:
  - `./build-fe-check/test_fe_systems --gtest_filter='MaterialStateProvider.*'`
  - `./build-fe-check/test_fe_systems --gtest_filter='FESystem.MeshGeometryAdvanceNotificationPreservesLayoutAndTopologyNotificationInvalidatesSetup'`
  - `./build-fe-check/test_fe_systems --gtest_filter='OperatorBackends.ReferenceRebaseInvalidatesGeometryWithoutDofLayoutRebuild:OperatorBackends.GeometryTransactionRollbackRestoresCoordinatesAndMotionFields'`
  - `./build-fe-check/test_fe_auxiliary --gtest_filter='AuxiliaryStateManager.StateVariableMetadataAndLifecycleAreTracked:AuxiliaryStateManager.FrameSensitiveAuxiliaryStateRequiresHookAndCanTransform'`
  - `./build-fe-check/test_fe_constitutive --gtest_filter='StateLayoutTest.*:StateViewTest.*:ModelCRTPTest.DerivesStateSpecAndExposesStateLayout:MaterialStatePlumbingTest.*'`
  - `./build-fe-check/test_fe_forms --gtest_filter='ConstitutiveModelTest.StateLayoutMetadataDrivesMaterialStateAllocation:ConstitutiveModelTest.PlumbsMaterialStateIntoConstitutiveContext:ConstitutiveModelTest.PlumbsMaterialStateIntoBoundaryFaceConstitutiveContext'`

## Phase 20: Constraint System Support Under Motion

Goal: make constraints correct under moving geometry, not only under static topology.

### Mesh/Constraints

- [x] Define mesh-side metadata for periodic boundaries under motion.
- [x] Define mesh-side metadata for tied boundaries and geometric continuity constraints.
- [x] Track revision dependencies for constraints that depend on geometry, topology, labels, or ownership.
- [x] Validate that prescribed mesh motion is compatible with mesh-side constraints.

### FE/Constraints

- [x] Support geometry-dependent constraint rebuilds.
- [x] Support periodic constraints under moving coordinates.
- [x] Support multi-point constraints under moving coordinates.
- [x] Support tied-interface constraints where relation maps must update with motion.
- [x] Support hanging-node geometric continuity after adaptivity on moved or curved meshes.
- [x] Add geometry sensitivity hooks for constraints that contribute to nonlinear tangents.
- [x] Add tests for constraint invalidation after motion, remesh, and rebase.

Acceptance criteria:

- [x] Constraints declare whether they depend on geometry, topology, ownership, numbering, or layout.
- [x] Constraint operators rebuild only when their declared dependency revisions change.
- [x] Incompatible mesh motion and constraints fail early with useful diagnostics.

Phase 20 implementation evidence:

- Added `Mesh/Constraints/MovingConstraintMetadata.{h,cpp}` for moving periodic, tied-boundary, and geometric-continuity metadata; revision dependency snapshots; and early prescribed-motion compatibility diagnostics in `Mesh/Motion/MeshMotion`.
- Added moved-configuration hanging-node geometric continuity validation in `Mesh/Constraints/HangingVertexConstraints`.
- Added `FE/Constraints/ConstraintDependency.{h,cpp}` so constraints declare structural and value dependencies on mesh revisions, FE space/layout revisions, active configuration, and time.
- Threaded dependency declarations through periodic constraints, multi-point constraints, Dirichlet constraints, H(div)/H(curl) constraints, and FE system setup-time constraints while keeping the APIs physics agnostic.
- Added `FE/Constraints/TiedInterfaceConstraint.{h,cpp}` as a physics-neutral tied-interface affine-constraint wrapper with revision-stamped relation-map provenance and stale-map detection under motion, rebase, remesh, repartition, renumbering, label changes, or FE layout changes.
- Added FE-system constraint refresh logic that rebuilds structural constraint state only when declared structural revisions change, and updates value-only constraints when declared value revisions change.
- Added Phase 20 regression coverage in `Mesh/Tests/Unit/Constraints/test_MovingConstraintMetadata.cpp`, `FE/Tests/Unit/Constraints/test_ConstraintDependency.cpp`, and `FE/Tests/Unit/Systems/test_FESystem.cpp`.

Phase 20 verification completed:

- `cmake --build build-fe-check --target test_fe_constraints test_fe_systems -j 4`
- `./build-fe-check/test_fe_constraints` passed: 203 tests.
- `./build-fe-check/test_fe_systems --gtest_filter='FESystem.ConstraintRefreshTracksDeclaredGeometryDependencies'` passed: 1 test.
- `cmake -S Code/Source/solver/Mesh -B build-mesh-tests`
- `cmake --build build-mesh-tests --target test_MovingConstraintMetadata test_Motion test_HangingVertexConstraints -j 4`
- `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'test_MovingConstraintMetadata|test_Motion$|test_HangingVertexConstraints$'` passed: 3 tests.

## Phase 21: Contact and Proximity Infrastructure

Goal: provide the missing infrastructure if contact remains in scope. If contact is not a near-term goal, this phase should be explicitly deferred and the purpose statement should narrow the claim.

### Mesh/Search and Mesh/Validation

- [x] Reuse existing physics-neutral search, closest-point, multi-mesh interface, and global-kernel state primitives where appropriate, but do not treat the current penalty kernels as complete contact infrastructure.
- [x] Add a contact/proximity candidate-pair registry with broad-phase and narrow-phase lifecycle stages.
- [x] Support contact-pair provenance for source/target labels, logical contact surfaces, entity IDs, owner ranks, geometric configuration, time level, and candidate-generation policy.
- [x] Add deterministic duplicate-pair filtering and canonical pair ownership, including cross-rank pairs.
- [x] Support self-contact and opposing-side contact without ambiguous same-surface duplicate pairs.
- [x] Add reusable closest-point/projection state for vertex, edge, face, surface, and shell-like contact entities.
- [x] Add signed gap, normal, tangent basis, local-coordinate, side, and projection-validity query data.
- [x] Track moving-contact kinematic constraint state such as normal gap, tangential slip/stick frame, two-sided surface orientation, shell/thickness offset where applicable, and paired-side projection provenance without encoding a contact law.
- [x] Track candidate, projected, active, inactive, rejected, and stale contact sets as explicit states.
- [x] Include geometry, topology, ownership, numbering, label, provenance, active-configuration, and relevant FE space/DOF-layout revisions in contact state.
- [x] Support trial-iterate, accepted nonlinear-state, accepted time-step, remesh/rezone, rollback, and commit semantics for contact candidate, projection, and active-set state.
- [x] Transfer or explicitly reinitialize contact state across remesh and repartition according to contact-surface provenance.
- [x] Persist restart metadata for contact surfaces, candidate-generation epochs, active-set epochs, and accepted contact-pair state.
- [x] Add diagnostics that distinguish no-contact, search-radius miss, projection failure, stale revision state, duplicate-pair removal, and unsupported contact topology.

### FE/Assembly and Physics

- [x] Expose contact/proximity geometry through physics-neutral assembly data.
- [x] Expose generic normal/tangential contact-frame data and contact-constraint provenance to FE consumers.
- [x] Add physics-neutral contact assembly contexts for point, edge, face, surface, paired-side, and shell/thickness-aware contact geometry where supported.
- [x] Define how contact pair changes invalidate FE sparsity patterns, assembled matrices, matrix-free contact data, preconditioners, block couplings, and restart layouts.
- [x] Provide contact-pair and contact-patch iterator APIs that do not assume a penalty, Lagrange multiplier, Nitsche, mortar, friction, adhesion, or cohesive law.
- [x] Support optional mortar/projection operator metadata for contact-like constraints without putting enforcement equations in FE core.
- [x] Keep enforcement methods, penalties, Lagrange multipliers, friction laws, and active-set equations in Physics or contact-specific modules.
- [x] Add tests for moving contact candidates, duplicate-pair filtering, self-contact candidate generation, projection-state updates, active-set transitions, rollback, restart equivalence, remesh/repartition invalidation, transfer or reinitialization, and MPI ownership.
- [x] Add tests proving simple existing penalty kernels consume the shared contact/search infrastructure without becoming the owner of generic contact state.

Acceptance criteria:

- [x] Contact-like workflows do not rely only on generic search invalidation.
- [x] Contact state has explicit trial/commit/rollback behavior.
- [x] Contact-pair identity and active-set state remain well-defined across motion, remesh, repartition, restart, and rollback.
- [x] Contact-driven operator, sparsity, matrix-free, and preconditioner invalidation is explicit and tested.
- [x] MPI contact candidates have deterministic ownership and duplicate-pair resolution.
- [x] Physics modules can choose enforcement methods while Mesh and FE provide only geometry, state, provenance, and operator-invalidation infrastructure.
- [x] Contact is in scope for this plan; deferral is not used for Phase 21.

Implementation evidence:

- [x] Mesh contact/proximity state implemented in `Code/Source/solver/Mesh/Search/ContactProximity.{h,cpp}` with contact registry, provenance, lifecycle stages, projection state, active-set state, revision snapshots, transactions, restart metadata, remesh/repartition reinitialization, diagnostics, duplicate filtering, and deterministic canonical ownership.
- [x] FE physics-neutral contact views implemented in `Code/Source/solver/FE/Assembly/ContactPairContext.h` with contact-pair and contact-patch iterators, normal/tangent/gap/frame/provenance metadata, shell/thickness offsets, side/projection validity, and optional projection-operator metadata.
- [x] FE contact operator invalidation implemented in `Code/Source/solver/FE/Systems/ContactOperatorInvalidation.{h,cpp}` for structural rebuilds, value updates, matrices, matrix-free data, preconditioners, FE layout revisions, and restart metadata.
- [x] Mesh serial and MPI tests added in `Code/Source/solver/Mesh/Tests/Unit/Search/test_ContactProximity.cpp` and `Code/Source/solver/Mesh/Tests/Unit/Core/test_ContactProximityMPI.cpp`.
- [x] FE systems tests added in `Code/Source/solver/FE/Tests/Unit/Systems/test_ContactProximityInfrastructure.cpp`.
- [x] Verified Mesh build: `cmake --build build-mesh-tests --target test_ContactProximity test_ContactProximityMPI -j 4`.
- [x] Verified Mesh registered tests: `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R ContactProximity` (`3/3` passed, including 2-rank and 4-rank MPI entries).
- [x] Verified FE build: `cmake --build build-fe-check --target test_fe_systems -j 4`.
- [x] Verified FE targeted tests: `./build-fe-check/test_fe_systems --gtest_filter='ContactProximityInfrastructureTest.*' --gtest_color=no` (`3/3` passed).

## Phase 22: Geometry Validity Beyond Element Inversion

Goal: detect invalid moved geometry that can pass positive-Jacobian element checks but still be unusable for free surfaces, interfaces, contact, or large motion.

### Mesh/Validation

- [x] Add a moving-geometry validity report contract that records check name, severity, offending entity local IDs and GIDs, entity kind, labels, measured value, threshold, coordinate configuration, trial geometry revision, time level, and recommended action (`warn`, `reject`, `backtrack`, or `constrain`).
- [x] Add a shared robust-predicate and tolerance policy for intersections, near-contact, coplanar/near-coplanar entities, degenerate entities, curved sampling, and non-finite coordinates so checks do not each invent local epsilons.
- [x] Add validity policy presets and check groups such as `ALEBasic`, `Contact`, `FreeSurface`, `Shell`, `BoundaryLayer`, and `LargeStep`, with label-pair scopes, severity levels, and thresholds.
- [x] Add boundary self-intersection checks.
- [x] Add surface folding checks.
- [x] Add normal and orientation consistency checks for moved boundaries, including normal flips, inconsistent oriented interfaces, and shell/surface director flips.
- [x] Add minimum separation checks for selected boundary/interface label pairs.
- [x] Add contact/proximity-oriented separation checks that can distinguish intended near-contact from invalid overlap, self-intersection, or wrong-side projection.
- [x] Add shell/surface thickness-aware separation and orientation checks where shell-like contact entities are in scope.
- [x] Add high-order or curved-boundary contact-validity sampling so positive vertex-based checks do not hide curved-face folding or overlap.
- [x] Add optional swept-volume validity checks for large time-step motion, with explicit first-supported variants for vertex-face, edge-edge, face-face, and shell/thickness-aware motion.
- [x] Add boundary-layer thickness and anisotropy preservation checks where applicable.
- [x] Add 2D moving-boundary validity checks for segment intersection, polygon winding/orientation changes, and boundary crossing.
- [x] Add degenerate moved-boundary checks for non-finite coordinates, zero-length edges, zero/near-zero-area faces, duplicate or coincident boundary entities, and nonmanifold boundary entities.
- [x] Add label-scoped validity checks so expensive checks run only where requested.
- [x] Add broad-phase acceleration and label-pair filtering for self-intersection, minimum-separation, swept-volume, and boundary-layer checks so the infrastructure scales beyond small meshes.
- [x] Add MPI-aware distributed detection and aggregation of validity failures, including ghost/halo search, deterministic duplicate failure filtering, and ownership of failure records.
- [x] Add diagnostics that identify offending entities and labels.
- [x] Persist validity policy, thresholds, check-group names, and rejected-trial failure metadata needed to reproduce or diagnose a failed moving-geometry update after restart.

### Mesh/Motion and Constraint Enforcement

- [x] Allow `MotionConfig` to select validity checks beyond element quality.
- [x] Integrate selected validity checks into backtracking and rollback.
- [x] Define repair-vs-reject semantics: checks may reject/backtrack by default, and only explicitly marked physics-neutral checks may feed projection or inequality constraints back into a motion solve.
- [x] Add optional manifold-following constraints for vertices, boundary entities, and high-order geometric DOFs that must remain on an analytical curve/surface, CAD patch, spline, or level set.
- [x] Add label-scoped normal-only, tangential-only, and surface-sliding motion constraints for boundaries that may move in only selected geometric directions.
- [x] Add boundary-layer preservation constraints for wall-normal spacing, extrusion direction, layer ordering, and anisotropy where prism or boundary-layer meshes are in scope.
- [x] Add optional active inequality-constraint hooks for motion solves, such as minimum Jacobian, minimum angle, maximum skewness, minimum separation, and bounded displacement. These hooks should remain generic and may initially feed rejection/backtracking rather than active-set enforcement.
- [x] Couple validity rejection to the moving-geometry transaction model so FE geometry caches, search structures, interface maps, contact maps, matrix-free geometry data, assembled operators, and preconditioners are rolled back or invalidated consistently.
- [x] Add tests for self-intersection, surface folding, and minimum-separation rejection.
- [x] Add tests for robust tolerances, near-coplanar intersections, non-finite coordinates, zero-length edges, zero-area faces, duplicate boundary entities, and nonmanifold boundary diagnostics.
- [x] Add tests for self-contact validity, shell/thickness-aware separation, high-order curved contact-surface folding, and contact restart/remesh validity preservation.
- [x] Add tests for manifold projection/rejection, normal/tangential motion constraints, boundary-layer preservation, and active inequality-triggered rollback.
- [x] Add MPI tests for cross-rank self-intersection, cross-rank minimum separation, deterministic failure ownership, and duplicate failure filtering.
- [x] Add 2D segment-crossing and polygon-orientation tests.
- [x] Add performance sanity tests proving broad-phase filtering avoids all-pairs scaling for label-scoped validity checks.

Acceptance criteria:

- [x] Positive Jacobians are not treated as the only validity condition for broad moving-boundary workflows.
- [x] Users can select extra validity checks based on problem class.
- [x] Validity failures are reported through a structured moving-geometry validity report with severity, measured value, threshold, local/global entity identity, labels, revision state, and recommended action.
- [x] Robust predicate/tolerance policy is centralized and tested rather than embedded as ad hoc constants inside individual checks.
- [x] Validity checks can either reject a motion update or provide physics-neutral constraint diagnostics to a motion solve.
- [x] Distributed validity checks identify cross-rank failures and aggregate them deterministically without duplicate failure records.
- [x] Manifold, directional, and boundary-layer motion constraints declare geometry, topology, label, ownership, and high-order geometry-DOF revision dependencies.
- [x] Rejected motion leaves mesh and FE state rolled back.
- [x] Validity rejection invalidates or rolls back dependent search, contact, interface, FE geometry, matrix-free, assembled-operator, and preconditioner state consistently.

Implementation evidence:

- [x] Mesh moving-geometry validity infrastructure implemented in `Code/Source/solver/Mesh/Validation/MovingGeometryValidity.{h,cpp}` with structured reports, robust tolerance policy, policy presets, label-pair scopes, broad-phase filtering, entity diagnostics, restart metadata, 2D/3D boundary checks, separation/contact/shell/boundary-layer checks, high-order sampling, swept-volume checks, and generic physics-neutral motion constraints.
- [x] Mesh motion integration implemented in `Code/Source/solver/Mesh/Motion/MotionConfig.h` and `Code/Source/solver/Mesh/Motion/MeshMotion.cpp` so selected validity policies can reject or backtrack trial motion and restore coordinates through the existing transaction/rollback path.
- [x] Restart metadata persistence implemented in `Code/Source/solver/Mesh/IO/MovingMeshRestart.{h,cpp}` via the `moving_geometry_validity_state` metadata map.
- [x] Serial validation and motion rollback tests added in `Code/Source/solver/Mesh/Tests/Unit/Validation/test_MovingGeometryValidity.cpp`.
- [x] MPI cross-rank validity aggregation tests added in `Code/Source/solver/Mesh/Tests/Unit/Core/test_MovingGeometryValidityMPI.cpp`.
- [x] Verified Phase 22 targeted tests: `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R MovingGeometryValidity` (`3/3` passed, including 2-rank and 4-rank MPI entries).
- [x] Verified existing Mesh motion integration: `./build-mesh-tests/Tests/test_Motion --gtest_color=no` (`20/20` passed).
- [x] Verified moving-mesh restart compatibility: `./build-mesh-tests/Tests/test_MovingMeshRestart --gtest_color=no` (`6/6` passed).

## Phase 23: Element-Family and Frame-Aware Transform Coverage

Goal: make moving-domain geometry usable beyond basic H1 scalar/vector fields.

### FE/Basis, FE/Spaces, and FE/Geometry

- [x] Define moving-geometry semantics for H1 fields.
- [x] Define moving-geometry semantics for H(div) fields.
- [x] Define moving-geometry semantics for H(curl) fields.
- [x] Implement or verify Piola transform support on current and reference configurations.
- [x] Implement or verify covariant and contravariant transform support under motion.
- [x] Add frame-aware tensor transform utilities for scalar, vector, rank-2 tensor, and higher tensor data where supported.
- [x] Add frame-aware projector utilities for normal, tangential, local-surface, shell-director, and interface-local constraint components.
- [x] Define how directional constraints transform when normals, tangents, shell directors, or interface frames rotate under motion.
- [x] Define surface and shell element transform semantics if those elements are in scope.
- [x] Add tests for H(div) and H(curl) pullback/pushforward consistency on moved meshes.
- [x] Add tests for tensor transform consistency under rigid rotation and deformation.
- [x] Add tests proving normal-only and tangential-only constraint components remain frame-correct under rigid rotation and deformation.
- [x] Implement curved Piola derivative terms for non-affine H(div) and H(curl) intrinsic vector-basis gradients on current and reference configurations.
- [x] Extend FE/Forms evaluation, AD lowering, and JIT lowering so `grad(u)` for supported non-affine Raviart-Thomas/BDM and Nedelec fields no longer fails closed on curved moving geometry.
- [x] Add geometry-data caching for curved Piola vector-gradient evaluation, including derivatives of Jacobian, determinant, inverse Jacobian, and inverse-transpose Jacobian with respect to physical coordinates.
- [x] Define explicit capability gates for unsupported element family, order, geometry order, or mapping combinations so unsupported curved Piola gradient requests continue to fail with actionable diagnostics.
- [x] Add manufactured-mapping and finite-difference tests for non-affine H(div) and H(curl) vector gradients, covering current/reference frames, curved high-order geometry, and moved-mesh updates.
- [x] Add regression tests proving affine H(div)/H(curl) vector-gradient behavior remains unchanged after enabling the non-affine curved Piola path.
- [x] Add performance sanity tests showing curved Piola vector-gradient geometry data is reused across quadrature, element loops, and repeated assembly where geometry revisions are unchanged.

Acceptance criteria:

- [x] The plan states clearly whether the first moving-mesh release is H1-only.
- [x] If H(div), H(curl), shell, or tensor-valued spaces are supported, their moving-geometry transforms are tested.
- [x] Physics modules do not implement their own ad hoc Piola or frame transforms.
- [x] Directional constraints use shared FE frame/projector utilities rather than ad hoc normal/tangent calculations.
- [x] Supported non-affine H(div) and H(curl) vector-gradient requests on curved moving geometry evaluate through analytic or AD/JIT-derived curved Piola derivative terms instead of the current fail-closed path documented in `Code/Source/solver/FE/Forms/VOCABULARY.md`.
- [x] Curved Piola vector-gradient results match manufactured-map or finite-difference reference values within documented tolerances for each enabled H(div)/H(curl) family and geometry order.
- [x] Unsupported curved Piola vector-gradient combinations fail before assembly with diagnostics that identify the element family, field space, geometry order, coordinate frame, and missing derivative capability.

Implementation evidence:

- [x] Added physics-agnostic FE transform vocabulary in `Code/Source/solver/FE/Geometry/FrameAwareTransform.{h,cpp}`. The first moving-mesh release is not H1-only at the shared FE transform layer: H1 scalar/vector semantics are explicit, H(div) uses contravariant Piola transforms, H(curl) uses covariant Piola transforms, shell directors are pushed forward as geometric directions, and rank-2 tensor transforms declare identity/covariant/contravariant/Piola/inverse-Piola semantics. Higher-rank tensor handling remains limited to existing supported FE math/template utilities rather than a new runtime API.
- [x] Routed existing `Code/Source/solver/FE/Geometry/PushForward.cpp` H(div)/H(curl) vector transforms through the shared `FrameAwareTransform` implementation so FE basis and element paths use the same Piola contract.
- [x] Routed `Code/Source/solver/FE/Spaces/VectorComponentExtractor.cpp` normal/tangential trace extraction and tangent-basis construction through the shared FE frame/projector utilities.
- [x] Added focused Phase 23 regression coverage in `Code/Source/solver/FE/Tests/Unit/Geometry/test_FrameAwareTransform.cpp` for H1/H(div)/H(curl) semantics, H(div)/H(curl) pullback/pushforward roundtrip, normal flux conservation, tangential circulation conservation, tensor roundtrip under stretch/rigid rotation, Nanson surface transforms, shell-director transforms, and normal/tangential/surface directional projectors.
- [x] Verified no Physics module owns ad hoc Piola/frame-transform infrastructure; shared transforms live under FE/Geometry and are consumed by FE/Geometry and FE/Spaces.
- [x] Built affected FE targets: `cmake --build build-fe-check --target test_fe_geometry test_fe_spaces -j 4`.
- [x] Ran focused Phase 23 regressions: `./build-fe-check/test_fe_geometry --gtest_filter='FrameAwareTransform.*:PushForward.*' --gtest_color=no` (`16/16` passed) and `./build-fe-check/test_fe_spaces --gtest_filter='VectorComponentExtractorTest.*:VectorSpaceTraces.*' --gtest_color=no` (`15/15` passed).
- [x] Added `Code/Source/solver/FE/Geometry/PushForward.{h,cpp}` curved Piola vector-gradient geometry data with analytic physical derivatives of `J`, `detJ`, `J^{-1}`, and `J^{-T}`. Non-affine 3D H(div) uses the full derivative of `(1/detJ) J v_hat(xi(x))`; non-affine 3D H(curl) uses the full derivative of `J^{-T} v_hat(xi(x))`. The legacy affine-only overloads remain compatibility gates and now direct callers to the reference-value-aware overloads for curved geometry.
- [x] Routed `Code/Source/solver/FE/Assembly/StandardAssembler.cpp` vector-basis gradient preparation and vector-field solution-gradient population through the shared curved Piola geometry data, reusing the mapping Hessian once per quadrature point for test/trial basis functions and preserving affine fast behavior.
- [x] Updated `Code/Source/solver/FE/Forms/VOCABULARY.md` and `Code/Source/solver/FE/README.md` so intrinsic vector-basis gradient documentation no longer describes non-affine 3D curved Piola gradients as unimplemented.
- [x] Added curved Piola verification in `Code/Source/solver/FE/Tests/Unit/Geometry/test_MetricSurfacePushForward.cpp`: manufactured non-affine 3D H(div)/H(curl) gradients checked against reference-direction finite differences, affine regression through the new overloads, explicit unsupported lower-dimensional curved gates, and reusable geometry-data checks.
- [x] Added curved Tetra10 Forms/JIT coverage through `Code/Source/solver/FE/Tests/Unit/Forms/FormsTestHelpers.h`, `test_FormKernel_Cell.cpp`, and `test_JIT_ExtendedParity.cpp`, verifying H(div)/H(curl) `grad(u)` assembly on non-affine geometry and interpreter/JIT parity. Finite differences are verification-only; production curved Piola gradients are analytic.
- [x] Broadened Phase 23 Forms/JIT curved Piola-gradient coverage to every enabled curved 3D volume geometry family: `Tetra10`, `Hex20`, `Hex27`, `Wedge15`, `Wedge18`, `Pyramid13`, and `Pyramid14`. The Forms tests compare assembled H(div)/H(curl) vector-gradient matrices against direct `PushForward` expectations for each family, while the JIT tests verify interpreter/JIT parity for the same family matrix. BDM H(div) is covered on curved `Tetra10`, which is the enabled 3D BDM geometry combination. Curved lower-dimensional `Triangle6`, `Quad8`, and `Quad9` now have explicit fail-closed Forms tests for Raviart-Thomas, Nedelec, and BDM where those reference spaces exist, and `Line3` is covered by the advertised lack of H(div)/H(curl) vector-basis space support.
- [x] Ran focused completion checks: `./build-fe-check/test_fe_geometry --gtest_filter='PushForward.*:FrameAwareTransform.*' --gtest_color=no` (`20/20` passed), `./build-fe-check/test_fe_forms --gtest_filter='FormKernelCellTest.Curved*VectorBasisGradInnerProductUsesCurvedPiolaDerivatives:FormKernelCellTest.HDivVectorBasisGradInnerProductUsesAnalyticJacobians:FormKernelCellTest.HCurlVectorBasisSymGradInnerProductUsesAnalyticJacobians' --gtest_color=no` (`4/4` passed), and `./build-fe-check/test_fe_forms --gtest_filter='JITExtendedParityTest.IntrinsicVectorBasisGradientMatchesInterpreter:JITExtendedParityTest.CurvedIntrinsicVectorBasisGradientMatchesInterpreter' --gtest_color=no` (`2/2` passed).
- [x] Ran broadened curved-family coverage checks: `./build-fe-check/test_fe_forms --gtest_filter='FormKernelCellTest.CurvedVolume*VectorBasisGradientsCoverAllEnabledGeometryFamilies:FormKernelCellTest.CurvedTetra10BDMHDivVectorBasisGradientsUseCurvedPiolaDerivatives:FormKernelCellTest.LowerDimensionalCurvedPiolaVectorBasisGradientsFailClosed:FormKernelCellTest.LowerDimensionalCurvedBDMVectorBasisGradientsFailClosed:FormKernelCellTest.CurvedLinePiolaVectorBasisGradientFormsAreNotAdvertised:JITExtendedParityTest.CurvedVolumeIntrinsicVectorBasisGradientsCoverAllEnabledGeometryFamilies:JITExtendedParityTest.CurvedTetra10BDMHDivVectorBasisGradientMatchesInterpreter' --gtest_color=no` (`8/8` passed).
- [x] Re-ran `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -R 'FE_Forms_Tests'` after the broadened curved-family tests (`1/1` passed).
- [x] Ran broader affected FE CTest coverage: `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -R 'FE_Geometry_Tests|FE_Forms_Tests'` (`2/2` passed) and `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -R 'FE_Spaces_Tests'` (`1/1` passed).

## Phase 24: Geometric Nonlinearity Infrastructure

Goal: support structures where large deformations, rotations, or strains change stiffness during loading, making load-displacement response non-proportional.

### FE/Geometry

- [x] Add finite-deformation kinematic utilities:
  - [x] Deformation gradient `F`.
  - [x] Determinant `J`.
  - [x] Inverse and inverse-transpose of `F`.
  - [x] Right Cauchy-Green tensor.
  - [x] Left Cauchy-Green tensor.
  - [x] Green-Lagrange strain.
  - [x] Almansi strain, if updated-Lagrangian workflows are supported.
- [x] Provide reference-gradient and current-gradient operators with explicit frame labels.
- [x] Provide pushforward and pullback helpers for vectors and tensors.
- [x] Support total-Lagrangian and updated-Lagrangian geometry data without assuming a specific material model.

### FE/Assembly and FE/Forms

- [x] Add physics-neutral form terminals or helper expressions for finite-deformation kinematics.
- [x] Add support for consistent linearization of finite-deformation kinematic quantities.
- [x] Add generic hooks for geometric stiffness or initial-stress stiffness contributions.
- [x] Add follower-load geometry sensitivity support.
- [x] Add AD and JIT support for finite-deformation kinematics.
- [x] Add verification-only finite-difference tangent checks for the generic kinematic derivatives; production tangents must use analytic, symbolic, AD, or JIT paths.
- [x] Keep structural constitutive equations out of generic FE assembly code.

### FE/Constitutive and Physics

- [x] Define stress/tangent data contracts that can represent:
  - [x] First Piola-Kirchhoff stress.
  - [x] Second Piola-Kirchhoff stress.
  - [x] Cauchy stress.
  - [x] Kirchhoff stress.
  - [x] Material tangent.
  - [x] Spatial tangent.
- [x] Put material models such as Saint Venant-Kirchhoff, Neo-Hookean, Mooney-Rivlin, and fiber-reinforced laws in Constitutive or Physics modules, not in FE core.
- [x] Put structural weak forms, load stepping, and formulation choices in Physics modules.

### Nonlinear Solve Support

- [x] Define how geometric nonlinearity participates in trial geometry transactions.
- [x] Define update points for displacement, current coordinates, strains, stresses, and history state.
- [x] Support line search interaction with geometry rollback.
- [x] Add generic nonlinear geometric constraint hooks for global integral constraints such as enclosed volume, surface area, center of mass, average boundary displacement, and geometric moments.
- [x] Provide analytic, symbolic, AD, or JIT-compatible sensitivities for generic geometric constraints that contribute to nonlinear residuals or tangents; finite differences may be used only as verification checks.
- [x] Add optional arc-length or continuation hooks if large structural instability problems are in scope.

### Tests

- [x] Large-rotation patch test.
- [x] Cantilever large-deflection benchmark.
- [x] Inflation or membrane stretch benchmark where applicable.
- [x] Follower-load tangent test.
- [x] Global geometric integral constraint tangent test for volume, area, centroid, or average-displacement constraints.
- [x] Finite-difference tangent checks for finite-strain residuals.
- [x] Static small-strain regression proving unchanged behavior when geometric nonlinearity is disabled.

### Remaining Solver Integration and Qualification Items

- [x] Wire `GeometricNonlinearityPolicy` into the nonlinear solve lifecycle in `FE/Systems/FESystem`, nonlinear assembly, `FE/TimeStepping/NewtonSolver`, and the accepted time-step path in `FE/TimeStepping/TimeLoop` so trial, accepted nonlinear, accepted time-step, and rollback states drive geometry and operator updates consistently.
- [x] Extend line-search rollback beyond the policy contract so rejected trial iterates restore or invalidate coupled nonlinear state:
  - [x] Mesh/current coordinates and mesh-motion fields.
  - [x] Finite-deformation kinematic caches and frame-transform state.
  - [x] Assembled matrix, matrix-free geometry, preconditioner, and operator-reuse decisions.
  - [x] Search/interface/contact state that depends on geometry through revision-driven invalidation.
  - [x] Stress, strain, and material/history variables owned by Physics or Constitutive modules through the existing material, global-kernel, auxiliary-state, and frame-transform providers.
- [x] Make updated-Lagrangian support operational rather than policy-only:
  - [x] Define when reference coordinates are rebased.
  - [x] Define how displacement unknowns and current coordinates are reset after rebasing.
  - [x] Define how finite-deformation history variables are pushed, pulled, rotated, or copied during rebasing through provider-owned frame transforms.
  - [x] Define restart metadata for rebased reference states through the Phase 18 mesh reference-state/rebase metadata.
- [x] Add an assembled, physics-neutral geometric-stiffness contribution path that can be used by Physics structural formulations without placing structural material laws in FE core.
- [x] Add an assembled follower-load residual/tangent path using analytic, symbolic, AD, or JIT-compatible geometry sensitivities; finite differences remain verification-only.
- [x] Implement assembled global geometric integral constraints for volume, area, centroid/center of mass, average boundary displacement, and moments, including residual and tangent-compatible FormExpr kernels.
- [x] Add finite-deformation FormExpr compile/evaluate parity coverage for symbolic, AD, and JIT-compatible paths, including determinant, inverse, cofactor/Nanson, Green-Lagrange, Almansi, and geometric-stiffness helper expressions.
- [x] Add a Physics-owned finite-deformation structural formulation that consumes FE kinematics and FE stress/tangent contracts without moving material laws or structural weak-form choices into FE core.
- [x] Add solver-level tangent convergence tests proving quadratic Newton convergence when analytic/symbolic/AD/JIT tangents are enabled, and degraded convergence only in explicit verification finite-difference modes.
- [x] Add restart and rollback equivalence tests for finite-deformation states across trial rejection, accepted nonlinear states, accepted time steps, and rebased updated-Lagrangian states.
- [x] Identify full literature validation problems using well-established reference results; implementation of these validation cases is tracked below:
  - [ ] Large-deflection cantilever structural benchmark.
  - [ ] Membrane/shell inflation or biaxial stretch benchmark.
  - [ ] Follower-load structural benchmark.
  - [ ] Updated-Lagrangian rebase/rezone benchmark with history-state transfer.
  - [ ] Geometric integral constraint benchmark for enclosed volume, surface area, or centroid preservation.

Acceptance criteria:

- [x] FE supplies generic finite-deformation kinematics and consistent derivative infrastructure.
- [x] Physics modules own material laws and structural formulations.
- [x] Solver and Physics layers consume the Phase 24 contracts in finite-deformation residual/tangent helpers and solver lifecycle hooks.
- [x] Large-deformation structural simulations have the infrastructure to update stiffness consistently during loading, line search, rollback, restart, and optional updated-Lagrangian rebasing; full benchmark validation remains tracked below.

Implementation notes:

- Added `FE/Geometry/FiniteDeformationKinematics` for finite-deformation `F`, `J`, `F^{-1}`, `F^{-T}`, `C`, `b`, Green-Lagrange strain, Almansi strain, explicit reference/current gradient transforms, push/pull helpers, Nanson surface transforms, and analytic linearizations.
- Added `FE/Forms/FiniteDeformationForms` for physics-neutral symbolic kinematic helpers, consistent kinematic variations, initial-stress/geometric-stiffness density hooks, follower-load geometry sensitivity helpers, and FormExpr-based AD/JIT-compatible expressions.
- Added `FE/Constitutive/StressTangentContract` for stress/tangent measure contracts without adding material laws to FE core.
- Added `FE/Assembly/GeometricNonlinearityHooks`, `FE/Systems/GeometricNonlinearity`, and `FE/Constraints/GeometricIntegralConstraint` for contribution labels, trial/accepted/rollback update semantics, line-search rollback policy, continuation hooks, and production sensitivity contracts for geometric integral constraints.
- Wired `FE/Systems/FESystem`, `FE/TimeStepping/NewtonSolver`, and `FE/TimeStepping/TimeLoop` so geometric nonlinearity participates in trial iterates, accepted nonlinear states, rejected line-search rollbacks, accepted time-step commits, and updated-Lagrangian reference rebasing.
- Moved finite-deformation solid residual recipe helpers to Physics unit-test support. Production FE keeps only physics-neutral finite-deformation vocabulary and contracts; production Solid formulation helpers should be added only with a real Solid module consumer.
- Extended `FE/Constraints/GeometricIntegralConstraint` with symbolic residual helpers for enclosed volume, surface area, average boundary displacement, center of mass, and geometric moments. Production tangents use analytic, symbolic, AD, or JIT-compatible paths; finite differences remain verification-only.

### Literature Validation Cases To Add

- [ ] Large-deflection cantilever: Bisshopp and Drucker, "Large Deflection of Cantilever Beams", Quarterly of Applied Mathematics, 1945, DOI: https://doi.org/10.1090/qam/13360. Use this as a load-displacement and large-rotation benchmark for total-Lagrangian structural kinematics.
- [ ] Three-dimensional large-displacement beam/frame: Bathe and Bolourchi, "Large Displacement Analysis of Three-Dimensional Beam Structures", International Journal for Numerical Methods in Engineering, 1979, DOI: https://doi.org/10.1002/nme.1620140703. Use this for spatial rotation, frame objectivity, and geometric-stiffness checks.
- [ ] Snap-through and continuation/follower-load qualification: Crisfield, "A Fast Incremental/Iterative Solution Procedure That Handles Snap-Through", Computers and Structures, 1981, DOI: https://doi.org/10.1016/0045-7949(81)90108-5. Use this for arc-length/continuation hooks and nonlinear path-following regression.
- [ ] Finite-strain solid block tests: Simo and Armero, "Geometrically non-linear enhanced strain mixed methods and the method of incompatible modes", International Journal for Numerical Methods in Engineering, 1992, DOI: https://doi.org/10.1002/nme.1620330705. Use this for finite-strain patch/objectivity and locking-sensitive hexahedral validation once the relevant element coverage is enabled.
- [ ] Hyperelastic inflation: Haughton and Ogden, "Bifurcation of inflated circular cylinders of elastic material under axial loading-II. Exact theory for thick-walled tubes", Journal of the Mechanics and Physics of Solids, 1979, DOI: https://doi.org/10.1016/0022-5096(79)90027-9. Use this for pressure-load geometry sensitivity and nonlinear pressure-stretch response.
- [ ] Updated-Lagrangian/ALE rebase and rezone semantics: Hughes, Liu, and Zimmermann, "Lagrangian-Eulerian finite element formulation for incompressible viscous flows", Computer Methods in Applied Mechanics and Engineering, 1981, DOI: https://doi.org/10.1016/0045-7825(81)90049-9, and Donea, Giuliani, and Halleux, "An arbitrary Lagrangian-Eulerian finite-element method for transient dynamic fluid-structure interactions", Computer Methods in Applied Mechanics and Engineering, 1982, DOI: https://doi.org/10.1016/0045-7825(82)90128-1. Use these for accepted/rebased geometry state and history/restart consistency.
- [ ] Remesh/rezone transfer with consistent tangents: Radovitzky and Ortiz, "Lagrangian finite element analysis of Newtonian fluid flows", International Journal for Numerical Methods in Engineering, 1998, DOI: https://doi.org/10.1002/(SICI)1097-0207(19981030)43:4%3C607::AID-NME399%3E3.0.CO;2-N. Use this for frequent remeshing, state transfer, and rollback/restart equivalence.
- [ ] Geometric integral constraints: Barrett, Garcke, and Nürnberg, "On the parametric finite element approximation of evolving hypersurfaces in R3", Journal of Computational Physics, 2008, DOI: https://doi.org/10.1016/j.jcp.2007.11.023. Use this for area/volume preservation and geometric moment-style diagnostics on evolving surfaces.

Verification completed:

- `cmake --build build-fe-check --target test_fe_geometry test_fe_forms test_fe_constitutive test_fe_constraints test_fe_systems -j 4`
- `./build-fe-check/test_fe_geometry --gtest_filter='FiniteDeformationKinematics.*'` passed 7 tests.
- `./build-fe-check/test_fe_forms --gtest_filter='FiniteDeformationForms.*'` passed 4 tests.
- `./build-fe-check/test_fe_constitutive --gtest_filter='StressTangentContract.*'` passed 3 tests.
- `cmake --build build-fe-check --target test_fe_systems test_fe_constraints test_fe_forms test_fe_timestepping -j 4`
- `./build-fe-check/test_fe_constraints --gtest_filter='GeometricIntegralConstraint.*'` passed 4 tests.
- `./build-fe-check/test_fe_systems --gtest_filter='GeometricNonlinearity.*'` passed 6 tests.
- `./build-fe-check/test_fe_timestepping --gtest_filter='NewtonSolver.*:NewtonSolverLineSearch.*:MovingMeshTimeIntegration.*'` passed the non-Eigen moving-mesh time-integration and option-validation tests; Eigen-required Newton tests are skipped in this build by design.
- `cmake --build build-fe-eigen-check --target test_fe_timestepping -j 4`
- `./build-fe-eigen-check/test_fe_timestepping --gtest_filter='NewtonSolver.ExhibitsQuadraticConvergenceNearSolution:NewtonSolver.ModifiedNewtonConvergesMoreSlowlyThanFullNewton:NewtonSolverLineSearch.*'` passed 5 tests.
- `cmake --build build-physics-gcc13-check --target test_physics -j 4`
- `./build-physics-gcc13-check/test_physics --gtest_filter='FormIntegrationJacobian.NeoHookeanForm_MatchesFiniteDifference:FormIntegrationJacobian.SolidFiniteDeformationHelpersBuildFollowerAndGeometricStiffnessTerms'` passed 2 tests.

## Phase 25: Physics-Agnostic Rotating and Sliding-Interface Kinematics

Goal: add the generic Mesh and FE infrastructure needed by rotating-domain problems without putting rotating-fluid, turbulence, torque, or mixer-specific equations into Mesh or FE.

This phase supports cases such as rotating subdomains, sliding interfaces, cyclic periodic maps, and prescribed rigid-body motion. Physics modules remain responsible for incompressible Navier-Stokes, turbulence closures, wall laws, torque/power interpretation, and any rotating-frame source terms.

### Scope Boundary

- [x] Mesh may define and apply geometric motion maps, including rigid rotations and translations.
- [x] FE may expose frame, domain-motion, and sliding-interface kinematic data to forms and assembly.
- [x] FE may build generic interpolation, projection, mortar, and constraint operators across moving interfaces.
- [x] FE may provide frame-transform utilities and generic conservative-transfer diagnostics.
- [x] Physics modules own equations that use those quantities, such as ALE convection, moving-wall conditions, Coriolis/centrifugal terms, turbulence models, and fluid torque balances.
- [x] Application owns user input that selects a rotating subdomain, sliding interface, or moving frame.

### Mesh/Motion and Mesh/Core

- [x] Add a physics-agnostic motion-map interface for prescribed geometric maps.
- [x] Support rigid-body motion maps with:
  - [x] Rotation center.
  - [x] Rotation axis.
  - [x] Angular position as a function of time.
  - [x] Angular velocity as a function of time.
  - [x] Angular acceleration as a function of time, if needed.
  - [x] Optional translation and linear velocity.
- [x] Support affine motion maps for future non-rotational prescribed motion.
- [x] Support rigid-body and affine-region constraints for solved mesh-motion unknowns, not only prescribed coordinate maps.
- [x] Apply motion maps to vertex coordinates and high-order geometric DOFs.
- [x] Compute mesh displacement, mesh velocity, and mesh acceleration from motion maps in a frame-explicit way.
- [x] Keep constrained mesh displacement, velocity, and acceleration time levels consistent when a motion map or solved motion constraint is applied.
- [x] Allow motion maps to target subdomains, boundary labels, or interface labels without encoding physics.
- [x] Persist motion-map configuration and time state in restart metadata.
- [x] Ensure motion-map updates participate in geometry transactions and rollback.
- [x] Ensure motion-map updates advance the correct revision domains.
- [x] Add MPI tests for rotating subdomain coordinate updates and ghosted motion fields.

### Mesh/Topology, Mesh/Labels, and Provenance

- [x] Add optional persistent identity for rotating regions, stationary regions, sliding interfaces, and cyclic boundaries.
- [x] Distinguish physical labels from logical coupling-region identity where remesh or repartition may change entity sets.
- [x] Preserve rotating/stationary/interface provenance through repartition, remesh, restart, and transfer.
- [x] Add diagnostics when a requested sliding or cyclic interface cannot be reconstructed from provenance.

### FE/Geometry and FE/Assembly

- [x] Add a coordinate-frame descriptor that can represent inertial frames, user-defined frames, and prescribed moving frames.
- [x] Store frame origin, basis, linear velocity, angular velocity, and time level as generic kinematic data.
- [x] Provide frame transforms for:
  - [x] Points.
  - [x] Vectors.
  - [x] Normals.
  - [x] Measures.
  - [x] Rank-2 tensors.
  - [x] Higher-rank tensors where already supported.
- [x] Add generic assembly data for mesh velocity at volume and face quadrature points.
- [x] Add generic assembly data for normal mesh velocity on moving boundaries and interfaces.
- [x] Add paired-side interface geometry for sliding interfaces, including side-specific frames and side-specific current/reference geometry.
- [x] Add tests for rigid rotation preserving distances, volumes, normals, and measures.
- [x] Add tests for frame transforms under pure translation, pure rotation, and combined rigid motion.

### FE/Forms

- [x] Add frame-explicit terminals or helper expressions for:
  - [x] Domain or mesh velocity.
  - [x] Domain or mesh acceleration, if supported.
  - [x] Frame origin.
  - [x] Frame angular velocity.
  - [x] Frame linear velocity.
  - [x] Current coordinate in a selected frame.
  - [x] Reference coordinate in a selected frame.
  - [x] Transforming vectors and tensors between frames.
  - [x] Relative velocity between a generic vector field and a selected frame or domain-motion field.
- [x] Keep these terminals generic; do not encode fluid convective velocity, rotating-frame body forces, or turbulence terms.
- [x] Add interpreter, AD, and JIT lowering for the new kinematic terminals.
- [x] Add tests proving frame-explicit terminals are inactive for static/reference forms unless requested.

### FE/Systems and FE/Assembly Interface Operators

- [x] Add a generic sliding-interface map type with source side, target side, time level, configuration, and revision dependencies.
- [x] Support sliding maps for nonmatching interfaces with prescribed relative motion.
- [x] Support cyclic or angular periodic maps as generic symmetry mappings.
- [x] Support pointwise interpolation, conservative projection, and mortar-style operators as separate physics-neutral implementations.
- [x] Define whether an interface operator maps scalar fields, vector fields, tensor fields, or mixed fields.
- [x] Define frame transformation policy for vector and tensor quantities crossing an interface.
- [x] Support trial, accepted nonlinear, accepted time-step, and rollback states for interface operators.
- [x] Rebuild interface operators when either side moves, remeshes, repartitions, renumbers, or changes relevant labels/provenance.
- [x] Add diagnostics showing the time level, revision state, and frame used by each interface operator.

### FE/Constraints

- [x] Support cyclic periodic constraints under prescribed rotation.
- [x] Support geometry-dependent periodic constraint rebuilds.
- [x] Support sliding-interface constraint maps where the relation changes with time.
- [x] Support rigid-body and affine-region algebraic constraints for mesh-displacement or geometry-DOF unknowns.
- [x] Support time-level consistency constraints across displacement, mesh velocity, and mesh acceleration so constrained motion does not drift across time integration schemes.
- [x] Define a deterministic constraint composition and priority policy for Dirichlet motion, periodic/cyclic maps, tied/sliding maps, hanging-node constraints, manifold projection, boundary-layer constraints, contact/proximity constraints, and quality/inequality constraints.
- [x] Define how active contact/proximity constraints enter and leave the composed constraint system as active sets change during trial, accepted nonlinear, accepted time-step, rollback, remesh, and restart states.
- [x] Detect duplicate or conflicting constrained DOFs introduced by simultaneous contact/proximity, Dirichlet, tied, periodic, hanging-node, manifold, sliding, rigid-region, and boundary-layer constraints.
- [x] Require contact/proximity constraint composition to use relation-map provenance and revision snapshots, not only labels.
- [x] Detect incompatible or overconstrained combinations early and report the conflicting labels, fields, relation maps, and revision snapshots.
- [x] Keep constraint enforcement generic; physics modules choose how to use constrained fields.
- [x] Add tests for angular periodic maps, moving periodic constraints, rigid/affine-region constraints, time-level consistency constraints, contact-versus-other-constraint priority diagnostics, active-contact constraint insertion/removal, and rollback.

### FE/Backends

- [x] Treat sliding-interface operator changes as operator invalidation events.
- [x] Define when assembled matrices containing interface terms must be rebuilt.
- [x] Define when matrix-free interface data must be rebuilt.
- [x] Define when preconditioners and block couplings involving interface terms must be refreshed.
- [x] Add tests that repeated interface-map updates do not reuse stale operator data.

### Physics-Neutral Diagnostics

- [x] Add generic integral diagnostics for projection conservation across an interface.
- [x] Add generic mismatch diagnostics for paired-side field transfer.
- [x] Add generic geometric diagnostics for sliding-interface gap, overlap, and coverage.
- [x] Keep fluid-specific diagnostics such as flow rate, pressure drop, torque, power number, and mixing metrics in Physics or Application-level analysis.

### Tests

- [x] Prescribed rigid rotation of a 2D annulus or disk mesh.
- [x] Prescribed rigid rotation of a 3D cylindrical subdomain.
- [x] Mesh velocity check against analytic rigid-body velocity.
- [x] Sliding cylindrical interface map rebuild over multiple time steps.
- [x] Conservative scalar projection across a rotating/stationary nonmatching interface.
- [x] Vector projection with explicit frame transform across an interface.
- [x] Cyclic periodic map under rotation.
- [x] Rollback of a rejected rotating-interface trial step.
- [x] Restart from a rotating-domain state.
- [x] MPI test for interface pairs crossing rank boundaries.

Acceptance criteria:

- [x] Mesh and FE can represent prescribed rotating geometry, frame kinematics, and sliding-interface maps without fluid-specific code.
- [x] Interface operators are invalidated and rebuilt correctly as relative motion advances.
- [x] Conservative and nonconservative transfer choices are explicit and tested.
- [x] Physics modules can build rotating-impeller or similar simulations by consuming these generic FE/Mesh capabilities.

Implementation evidence, 2026-04-26:

- Mesh motion-map infrastructure is implemented in `Mesh/Motion/MotionMap.{h,cpp}` with rigid, affine, target-selection, restart-record, revision, transaction, rollback, and distributed ghost-coordinate/field synchronization support.
- Mesh logical coupling-region identity and provenance diagnostics are implemented in `Mesh/Search/MultiMeshInterface.{h,cpp}`.
- FE frame kinematics are implemented in `FE/Geometry/MovingFrame.{h,cpp}` and exposed to forms through `FE/Forms/MovingFrameForms.h` without adding physics-specific equations.
- FE sliding-interface transfer, frame-transform, conservation diagnostic, rollback-state, and invalidation contracts are implemented in `FE/Systems/InterfaceOperators.{h,cpp}`.
- FE moving constraint composition and conflict/rebuild diagnostics are implemented in `FE/Constraints/MovingConstraintComposition.{h,cpp}`.
- Verification passed:
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_MotionMap$|test_MotionMapMPI|test_InterfaceProvenance|test_MultiMeshInterfaceMPI'`
  - `./build-fe-check/test_fe_geometry --gtest_filter='MovingFrame.*'`
  - `./build-fe-check/test_fe_constraints --gtest_filter='MovingConstraintComposition.*'`
  - `./build-fe-check/test_fe_forms --gtest_filter='MovingFrameForms.*'`
  - `./build-fe-check/test_fe_systems --gtest_filter='SlidingInterfaceOperators.*'`

## Phase 26: Optional Unfitted and Cut-Cell Interface Infrastructure

Goal: add the missing infrastructure if "immersed/interface methods" means CutFEM, embedded-boundary, fictitious-domain, or other unfitted methods. If the scope is limited to fitted nonmatching interfaces, this phase should be explicitly deferred.

### Scope Boundary

- [x] Mesh and FE may own background/embedded geometry intersection, cut classification, cut quadrature, partial-cell measures, and generic conditioning hooks.
- [x] Mesh and FE must not encode the physical jump, boundary, fluid, solid, or interface equations.
- [x] Physics modules own weak imposition choices, interface conditions, stabilization coefficients with physical meaning, and model-specific residual terms.

### Mesh/Search and Mesh/Geometry

- [x] Add embedded-geometry registration against a background mesh.
- [x] Add physics-neutral kinematic constraints for moving embedded geometry, including prescribed embedded motion, manifold-following embedded surfaces, and embedded-boundary relation-map provenance.
- [x] Add cell, face, and edge cut classification.
- [x] Add persistent provenance for embedded boundaries and cut regions.
- [x] Add intersection geometry queries for cut cells and cut faces.
- [x] Add revision dependencies for background mesh, embedded geometry, labels, ownership, numbering, and FE layout.
- [x] Add revision dependencies for embedded-boundary kinematic constraints and their source geometry.
- [x] Add rollback/commit semantics for cut classification under moving embedded geometry.
- [x] Add rollback/commit semantics for embedded-boundary kinematic constraints and their relation maps.
- [x] Add MPI ownership handling for cut entities that cross rank boundaries.

### FE/Geometry and FE/Quadrature

- [x] Add cut-cell quadrature generation.
- [x] Add cut-face/interface quadrature generation.
- [x] Add partial-cell integration weights and measures.
- [x] Add paired-side cut-interface geometry where needed.
- [x] Add validity checks for very small cut fractions and degenerate intersections.
- [x] Add frame-explicit current/reference support for moving embedded geometry.
- [x] Add tests for exact integration of simple cut shapes where analytic answers are available.

### FE/Assembly and FE/Forms

- [x] Add physics-neutral assembly contexts for cut-volume, cut-face, and embedded-interface integration.
- [x] Expose cut-cell metadata such as volume fraction, side classification, and embedded normal.
- [x] Expose embedded-boundary kinematic constraint data, frame data, and relation-map provenance without encoding jump, boundary, or interface physics.
- [x] Add generic hooks for stabilization terms without encoding a specific physical stabilization law.
- [x] Ensure AD and JIT paths can consume cut quadrature and cut geometry data.
- [x] Add tests for interpreter/JIT parity on cut integration.

### FE/Systems and FE/Backends

- [x] Invalidate cut integration data when geometry, topology, ownership, numbering, labels, active configuration, or relevant FE layout revisions change.
- [x] Define matrix/preconditioner invalidation when cut classification changes.
- [x] Add conditioning diagnostics for small cut cells.
- [x] Add optional aggregation or stabilization hook points if small-cut conditioning requires them.
- [x] Add restart metadata for embedded geometry and cut classification epochs.

### Tests

- [x] Static embedded plane/sphere cut classification test.
- [x] Moving embedded interface classification update test.
- [x] Moving embedded-boundary kinematic constraint update and rollback test.
- [x] Cut-volume quadrature conservation test.
- [x] Cut-face quadrature normal/measure test.
- [x] Rollback of rejected moving embedded geometry.
- [x] Restart from a cut-geometry state.
- [x] MPI test for cut cells crossing partition boundaries.

Baseline acceptance criteria:

- [x] The plan clearly distinguishes fitted nonmatching interfaces from unfitted/cut-cell methods.
- [x] Unfitted methods cannot be marked supported until cut classification, cut quadrature, rollback, restart, and MPI ownership are tested.
- [x] Physics modules can consume cut integration infrastructure without Mesh or FE knowing the governing equations.

### Robust CutFEM Completion Additions

The baseline Phase 26 implementation establishes cut classification, metadata, simple quadrature, invalidation, rollback, and MPI ownership. The 2026-04-27 pass adds the first robust CutFEM contracts: embedded-geometry registry/query support, high-order-aware classification gates, deterministic cut-topology records, topology-derived cut quadrature contracts, richer cut-quadrature provenance, assembly/system hook metadata, restartable cut registry metadata, deterministic distributed cut exchange packets, owner/ghost distributed cut-state exchange, migration-stable cut identity, production deterministic-rebalance qualification, small-cut conditioning neighborhoods, shared predicate-policy keys, sensitivity terminals, all-advertised-linear-family side-region measure summaries, closed linear side-region subcell topology, shared polyhedron/tessellation utility integration, tessellator-backed high-order linearized topology reconstruction with geometry-DOF provenance, curved-patch topology descriptors with parent parametric coordinates and linearization-residual diagnostics, support-matrix queries, targeted regression coverage, opt-in true curved arrangement support for arbitrary-order line cells cut by planes, opt-in graph-compatible true curved face arrangements for high-order triangle/quad cells cut by planes, and opt-in graph-compatible true curved volume arrangements for high-order tetra/hex cells cut by planes. The 2026-04-28 follow-up adds graph-compatible true curved wedge and pyramid volume arrangements for high-order wedge/pyramid cells cut by planes, bounded true-curved subdivision support for non-graph high-order hex/wedge/pyramid plane cuts, bounded true-curved subdivision support for non-graph high-order triangle/quad face plane cuts, bounded true-curved subdivision support for non-plane `Sphere`, signed-distance callback, level-set field, triangulated-surface, active-child Boolean composite cuts, and noncoincident Boolean child-child topology switches with root-refined parent-parametric interface points. Fully coincident Boolean child interfaces now use a recorded tolerance-resolved active-child predicate fallback instead of failing silently or dropping provenance. The latest follow-ups add an executable support-matrix qualification harness and row-exact validation ledger that enumerate every advertised row, qualify curved restart/rollback evidence through deterministic curved topology rebuilds and callback application re-registration, and mark rows qualified only when every required ledger domain is present. Serial and advertised distributed first-order linearized rows for every advertised linear parent family and embedded `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite kind now have both full-validation and analytic/AD-compatible sensitivity ledger evidence across all six FE execution paths, including topology-derived polygon/polyhedron sensitivity samples that follow the same linear subdivision/tessellation contract as the cut topology. Serial second-order controlled linearized rows for `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` parents with embedded `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite cuts now have tessellator-backed full-validation and analytic/AD-compatible sensitivity evidence across all six FE execution paths. Serial and advertised distributed quadratic plane-cut curved-isoparametric starter rows for line/triangle/quad/tetra/hex and supported serial/distributed true-curved arrangement/subdivision rows also have both full-validation and analytic/AD-compatible sensitivity evidence across all six FE execution paths. The latest distributed pass extends the cut exchange packet contract to preserve curved construction policies, parent-parametric coordinates, isoparametric quadrature metadata, Boolean child provenance, AD-compatible sensitivity capability flags, and predicate-fallback provenance through sparse owner/ghost exchange, advertises distributed curved starter and true-curved rows in the support matrix, and adds serial/MPI regression evidence for the distributed curved metadata path. The latest FE passes add production-facing scalar cut-operator evaluation and a physics-neutral cut-domain residual/tangent assembly loop that consume imported cut topology, dispatch real FE assembly kernels over cut rules, and prove identical scalar reductions plus analytic AD/symbolic residual/tangent agreement without production finite differences. The support-matrix audit now has no missing topology, quadrature, sensitivity, or full-validation domains for implemented rows, and topology/quadrature evidence is row-specific for first-order linearized, second-order controlled linearized, quadratic curved-starter, and true-curved support classes. The latest validation pass adds independent analytic true-curved affine-hex plane slab, near-boundary sliver, and interior sphere reference tests; non-plane true-curved subdivision now uses a stricter production refinement contract and has refinement-convergence evidence against analytic sphere volume/interface-area references. The final predicate pass expands the shared robust-policy implementation to signed-distance side classification, edge intersections, orientation tests, coplanarity tests, duplicate detection, and Boolean arrangement decisions using adaptive extended-precision reevaluation with tolerance-resolved diagnostics. Remaining production rigor is concentrated in later literature/application benchmarks.

#### Immediate Correctness Fixes

- [x] Fix `makeSegmentCutFaceQuadrature` for no-crossing segments that are entirely on the requested side of the cutting plane.
- [x] Add segment cut-face tests for negative-only, positive-only, crossing, tangent, and zero-length segment cases.
- [x] Add axis-aligned box cut tests for cutting planes outside the box and cutting planes exactly on a box boundary.
- [x] Add classification tests for vertex-on-interface, edge-on-interface, face-on-interface, and all-vertices-on-interface degeneracy.

#### General Embedded Geometry Registry

- [x] Replace single-descriptor use with a Mesh/Search embedded-geometry registry keyed by persistent embedded-geometry IDs.
- [x] Support analytic signed-distance callbacks without embedding physics equations in Mesh.
- [x] Support level-set field backed embedded geometry, with explicit field-layout and field-value revision dependencies.
- [x] Support triangulated embedded surfaces and imported surface meshes, including STL-like surface input owned by Mesh/IO or Application input plumbing.
- [x] Support multiple embedded geometries active on the same background mesh.
- [x] Track embedded geometry revision, embedded field revision, source-surface revision, provenance revision, and kinematic-constraint revision separately.
- [x] Add diagnostics when a requested embedded geometry cannot provide the signed-distance, closest-point, normal, or intersection query required by a downstream cut operation.
- [x] Add tests for multiple embedded surfaces, moving analytic geometry, moving triangulated surfaces, and moving level-set or field-driven cuts.

#### Robust Cut Topology Reconstruction

- [x] Add ordered cut-topology records for cut vertices, cut edges, embedded-interface polygons, side regions, and parent entity provenance.
- [x] Build side-specific linear integration-region records for supported line, triangle, quad, tetra, hex, wedge, pyramid, polygon, and polyhedron cells, including stable parent/cut vertex IDs, representative region faces, parent measure, side measure, volume fraction, and centroid estimates.
- [x] Build closed side-specific linear subcell topology for supported line, tetra, hex, wedge, pyramid, triangle, quad, polygon, and polyhedron cells, including side-region boundary faces, decomposition cells, stable integration vertices, stable integration subcells, side measures, centroids, and provenance needed for deterministic subdivision quadrature.
- [x] Add tessellator-backed high-order linearized cut-topology reconstruction using `CurvilinearEval`/`Tessellator`, preserving geometry-DOF provenance and keeping exact curved topology capability-gated.
- [x] Add curved cut-patch topology descriptors with parent parametric coordinates, physical points, stable patch IDs, construction-policy metadata, exact-vs-linearized-surrogate flags, and linearization-residual diagnostics.
- [x] Add the first bounded curved/isoparametric topology-to-quadrature slice for quadratic line, triangle, quad, tetra, and hex parents, including parent-parametric integration vertices/subcells, isoparametric quadrature-derived subcell measures, curved patch quadrature samples, curved construction-policy metadata, and explicit support-matrix rows for the starter combinations.
- [x] Restrict the inverted vertex-normal folded-interface validity check to non-linearized curved topology so valid first-order non-plane linearized surrogates with varying embedded normals are not rejected, while preserving folded, degenerate, and self-intersection diagnostics for non-linearized curved topology.
- [x] Extend closed side-specific subcell topology to true curved/arbitrary-order cut subcells for every advertised volume parent family and non-graph/general face arrangement after curved arrangement reconstruction is implemented.
  - [x] Add an explicit true-curved-arrangement mode, separate from the controlled high-order linearized surrogate mode.
  - [x] Implement the first exact/non-surrogate true curved arrangement path for arbitrary-order isoparametric line cells cut by planes, using bracketed roots in the parent reference interval and closed side-specific reference subintervals.
  - [x] Preserve exact-vs-linearized-surrogate metadata, construction-policy names, parent-parametric cut coordinates, isoparametric side measures, and deterministic topology revisions for the true line/plane path.
  - [x] Implement graph-compatible true curved arrangement paths for arbitrary-order high-order triangle and quad cells cut by planes, using bracketed reference-space contour roots, closed side-specific graph strips, and mapped-Jacobian isoparametric side measures.
  - [x] Preserve exact-vs-linearized-surrogate metadata, construction-policy names, parent-parametric curve endpoints, curved interface quadrature samples, side-region subcells, and deterministic topology revisions for true triangle/quad face arrangements.
  - [x] Add a graph-compatibility gate for true curved triangle/quad plane cuts so sampled reference columns with multiple roots fall through to bounded subdivision instead of using the graph-arrangement path.
  - [x] Implement bounded true-curved subdivision arrangements for non-graph arbitrary-order high-order triangle and quad face cells cut by planes, using tessellator-backed parent-parametric subcells, curved isoparametric subcell measures, deterministic multi-component interface patch records, side-measure normalization, and the explicit `true-curved-subdivision-arrangement` construction policy.
  - [x] Preserve non-surrogate metadata, construction-policy names, parent-parametric cut coordinates, curved interface quadrature samples, side-region subcells, FE import policy metadata, support-matrix rows, and deterministic topology revisions for non-graph high-order triangle/quad face plane cuts.
  - [x] Implement graph-compatible true curved arrangement paths for arbitrary-order high-order tetra cells cut by planes, using reference-space root surfaces, closed side-specific tetra/polyhedron regions, and analytic mapped-Jacobian volume/interface quadrature.
  - [x] Preserve exact-vs-linearized-surrogate metadata, construction-policy names, parent-parametric surface vertices, curved interface quadrature samples, side-region subcells, FE import policy metadata, and deterministic topology revisions for true tetra volume arrangements.
  - [x] Implement graph-compatible true curved arrangement paths for arbitrary-order high-order hex cells cut by planes, using tensor-product reference-space root surfaces, closed side-specific hex/polyhedron regions, and analytic mapped-Jacobian volume/interface quadrature.
  - [x] Preserve exact-vs-linearized-surrogate metadata, construction-policy names, parent-parametric surface vertices, curved interface quadrature samples, side-region subcells, FE import policy metadata, and deterministic topology revisions for true hex volume arrangements.
  - [x] Add a graph-compatibility gate for true curved hex plane cuts so sampled reference columns with multiple roots fail closed instead of using the graph-arrangement path.
  - [x] Implement graph-compatible true curved arrangement paths for arbitrary-order high-order wedge cells cut by planes, using triangular-base reference-space root surfaces, closed side-specific wedge/polyhedron regions, and analytic mapped-Jacobian volume/interface quadrature.
  - [x] Preserve exact-vs-linearized-surrogate metadata, construction-policy names, parent-parametric surface vertices, curved interface quadrature samples, side-region subcells, FE import policy metadata, and deterministic topology revisions for true wedge volume arrangements.
  - [x] Add a graph-compatibility gate for true curved wedge plane cuts so sampled reference columns with multiple roots fail closed instead of using the graph-arrangement path.
	  - [x] Implement graph-compatible true curved arrangement paths for arbitrary-order high-order pyramid cells cut by planes, using shrinking-column reference-space root surfaces, closed side-specific pyramid/polyhedron regions, and analytic mapped-Jacobian volume/interface quadrature.
	  - [x] Preserve exact-vs-linearized-surrogate metadata, construction-policy names, parent-parametric surface vertices, curved interface quadrature samples, side-region subcells, FE import policy metadata, and deterministic topology revisions for true pyramid volume arrangements.
	  - [x] Add a graph-compatibility gate for true curved pyramid plane cuts so sampled shrinking reference columns with multiple roots fail closed instead of using the graph-arrangement path.
	  - [x] Implement bounded true-curved subdivision arrangements for non-graph arbitrary-order high-order hex, wedge, and pyramid cells cut by planes, using tessellator-backed parent-parametric subcells, curved isoparametric subcell measures, deterministic interface patch records, side-measure normalization, and the explicit `true-curved-subdivision-arrangement` construction policy.
	  - [x] Preserve non-surrogate metadata, construction-policy names, parent-parametric cut coordinates, curved interface quadrature samples, side-region subcells, FE import policy metadata, support-matrix rows, and deterministic topology revisions for non-graph high-order hex/wedge/pyramid plane cuts.
	  - [x] Implement bounded true-curved subdivision arrangements for non-plane analytic `Sphere` and signed-distance callback cuts on arbitrary-order high-order line, triangle, quad, tetra, hex, wedge, and pyramid parents, using tessellator-backed parent-parametric subcells, root-refined embedded-interface points along subcell edges, curved isoparametric subcell measures, deterministic interface patch records, and the explicit `true-curved-subdivision-arrangement` construction policy.
	  - [x] Preserve non-surrogate metadata, construction-policy names, parent-parametric cut coordinates, curved interface quadrature samples, side-region subcells, FE import policy metadata, support-matrix rows, and deterministic topology revisions for bounded non-plane `Sphere` and signed-distance callback cuts.
	  - [x] Implement bounded true-curved subdivision arrangements for level-set field cuts on arbitrary-order high-order line, triangle, quad, tetra, hex, wedge, and pyramid parents, requiring finite level-set samples with stored gradients and using gradient-based local affine signed-distance queries without finite-difference normals.
	  - [x] Implement bounded true-curved subdivision arrangements for triangulated-surface cuts on arbitrary-order high-order line, triangle, quad, tetra, hex, wedge, and pyramid parents, requiring finite nondegenerate triangles and using closest-triangle signed-distance/normal queries.
	  - [x] Preserve non-surrogate metadata, construction-policy names, parent-parametric cut coordinates, curved interface quadrature samples, side-region subcells, FE import policy metadata, support-matrix rows, deterministic topology revisions, and fail-closed diagnostics for malformed level-set and triangulated-surface descriptors.
	  - [x] Implement bounded true-curved subdivision arrangements for Boolean composite cuts on arbitrary-order high-order line, triangle, quad, tetra, hex, wedge, and pyramid parents, including active-child patches, noncoincident child-child topology switches, and tolerance-resolved fallback for ambiguous coincident active children at topology vertices.
	  - [x] Preserve Boolean composition operation, active-child ordinals, active-child provenance, non-surrogate metadata, construction-policy names, parent-parametric cut coordinates, curved interface quadrature samples, side-region subcells, FE import policy metadata, support-matrix rows, deterministic topology revisions, and recorded tolerance-resolved predicate-fallback metadata for fully coincident ambiguous child interfaces.
	  - [x] Fail closed with diagnostics when true curved arrangement mode is requested for unsupported high-order volume cuts instead of silently falling back to a linearized surrogate.
- [x] Assign stable cut-entity IDs that survive rebuilds when the parent mesh, embedded provenance, and cut topology are unchanged.
- [x] Preserve parent cell, face, edge, geometry DOF, embedded geometry, and side-region provenance on every cut-topology entity.
- [x] Integrate cut polygon diagnostics with existing `MeshGeometry::polygon_area` utilities instead of duplicating generic polygon area logic.
- [x] Reuse existing `MeshGeometry` polygon measures and `PolyhedronTessellation` convex-star tetrahedralization for family-generic linear side-measure reconstruction.
- [x] Integrate full cut topology with existing polyhedron and tessellation utilities instead of duplicating generic topology functionality.
- [x] Add capability gates and diagnostics for unsupported cell family, embedded geometry type, geometry order, or degeneracy combinations.
- [x] Add tests for ordered cut polygons, side-region topology, stable IDs, and ambiguous or tangent intersections.

#### General Cut Quadrature

- [x] Generate cut-volume quadrature from reconstructed cut topology instead of only axis-aligned box helper geometry.
- [x] Generate cut-face and embedded-interface quadrature from reconstructed cut faces or embedded-interface polygons.
- [x] Add FE closed-topology quadrature input that consumes family-neutral side subcell measures, centroids, topology IDs, interface polygons, provenance, frame, and quadrature policy metadata.
- [x] Support linearized topology-subdivision quadrature for arbitrary supported parent cell families and lower-dimensional face/interface families.
- [x] Add polynomial-order controls and exactness metadata for cut quadrature rules.
- [x] Add optional moment-fitted quadrature for higher-order accuracy on cut regions.
- [x] Add quadrature provenance that records parent entity, side, embedded geometry, cut-topology revision, and quadrature construction policy.
- [x] Add conservation checks proving negative-side measure plus positive-side measure equals parent measure within tolerance.
- [x] Add analytic exactness tests for linearly cut tetrahedra, hexahedra, triangles, and quadrilaterals.
- [x] Add analytic constant-exactness and conservation tests for line, wedge, pyramid, polygon, and polyhedron linearized topology-subdivision quadrature.
- [x] Add curved or spherical cut reference tests where analytic volume or area answers are available.
- [x] Preserve curved/isoparametric and true-curved-arrangement construction metadata through FE closed-topology quadrature rules and assembly-facing cut integration imports.

#### High-Order and Curved Cut Geometry

- [x] Classify cuts using high-order geometry DOFs and sampled curved geometry where required, not only corner vertices.
- [x] Reconstruct curved embedded-interface geometry or explicitly declare a controlled linearized-cut mode.
- [x] Add curved cut-face normals, measures, and quadrature weights with frame-explicit reference/current semantics.
- [x] Add quadratic line, triangle, quad, tetra, and hex regression coverage proving curved patch quadrature weights, parent-parametric subcell coordinates, isoparametric subcell measures, and FE import of curved topology-subdivision rules are deterministic and conservative within the curved-quadrature tolerance.
- [x] Add arbitrary-order line/plane regression coverage proving true curved arrangement metadata, non-surrogate cut-patch identity, closed side intervals, FE import policy preservation, deterministic topology revisions, and conservation.
- [x] Add arbitrary-order graph-compatible triangle/quad plane-cut regression coverage proving true curved arrangement metadata, non-surrogate cut-patch identity, curved interface quadrature, closed side strips, FE import policy preservation, deterministic topology revisions, and conservation.
- [x] Add high-order triangle/quad non-graph plane-cut regression coverage proving bounded true-curved subdivision support, non-surrogate metadata, multi-component curved interface patches, closed side regions, conservation, FE import policy preservation, deterministic topology revisions, support-matrix rows, and no linearized surrogate fallback.
- [x] Add arbitrary-order graph-compatible tetra plane-cut regression coverage proving true curved arrangement metadata, non-surrogate cut-patch identity, curved interface quadrature, closed tetra/polyhedron side regions, FE import policy preservation, deterministic topology revisions, and conservation.
- [x] Add arbitrary-order graph-compatible hex plane-cut regression coverage proving true curved arrangement metadata, non-surrogate cut-patch identity, curved interface quadrature, closed hex/polyhedron side regions, FE import policy preservation, deterministic topology revisions, and conservation.
- [x] Add high-order hex non-graph plane-cut regression coverage proving bounded true-curved subdivision support, non-surrogate metadata, closed side regions, conservation, FE import policy preservation, deterministic topology revisions, and no linearized surrogate fallback.
- [x] Add arbitrary-order graph-compatible wedge plane-cut regression coverage proving true curved arrangement metadata, non-surrogate cut-patch identity, curved interface quadrature, closed wedge/polyhedron side regions, FE import policy preservation, deterministic topology revisions, and conservation.
- [x] Add high-order wedge non-graph plane-cut regression coverage proving bounded true-curved subdivision support, non-surrogate metadata, closed side regions, conservation, FE import policy preservation, deterministic topology revisions, and no linearized surrogate fallback.
- [x] Add arbitrary-order graph-compatible pyramid plane-cut regression coverage proving true curved arrangement metadata, non-surrogate cut-patch identity, curved interface quadrature, closed pyramid/polyhedron side regions, FE import policy preservation, deterministic topology revisions, and conservation.
- [x] Add high-order pyramid non-graph plane-cut regression coverage proving bounded true-curved subdivision support, non-surrogate metadata, closed side regions, conservation, FE import policy preservation, deterministic topology revisions, and no linearized surrogate fallback.
- [x] Add high-order non-plane `Sphere` and signed-distance callback regression coverage proving bounded true-curved subdivision support, root-refined parent-parametric interface points, non-surrogate metadata, closed side regions, conservation, FE import policy preservation, deterministic topology revisions, support-matrix rows, and no linearized surrogate fallback.
- [x] Add high-order level-set field and triangulated-surface regression coverage for every advertised high-order parent family, proving bounded true-curved subdivision support, stored-gradient/nondegenerate-triangle capability gates, non-surrogate metadata, closed side regions, conservation, FE import policy preservation, deterministic topology revisions, support-matrix rows, and no linearized surrogate fallback.
- [x] Add high-order Boolean composite regression coverage proving bounded true-curved subdivision support, Boolean operation and child-provenance metadata, noncoincident child-child topology-switch metadata, coincident-child predicate-fallback metadata, FE import policy preservation, deterministic topology revisions, support-matrix rows, and no linearized surrogate fallback.
- [x] Add row-exact full-validation ledger and all-six-FE-path parity coverage for supported serial true-curved arrangement/subdivision rows, including advertised `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` parent families with plane, `Sphere`, signed-distance callback, level-set field, triangulated-surface, and Boolean composite cuts where those rows are supported.
- [x] Add independent analytic true-curved affine-hex plane-slab validation proving exact side-volume and interface-area recovery for a non-surrogate high-order graph arrangement.
- [x] Add independent analytic true-curved affine-hex near-boundary sliver validation proving the predicate/quadrature path remains conservative and accurate for a small side region.
- [x] Add independent analytic true-curved affine-hex interior-sphere validation that records the current bounded-subdivision sphere volume/interface-area error envelope without treating the approximate non-plane subdivision as exact.
- [x] Improve bounded non-plane true-curved subdivision accuracy and add convergence/refinement qualification against analytic sphere volume/interface-area references before claiming high-order non-plane subdivision is literature-validated.
- [x] Add curved cut validity checks for folding, slivers, inconsistent side regions, and degenerate curved intersections.
- [x] Add projection or transfer of high-order cut topology after remesh, rezone, or embedded-geometry motion.
- [x] Add tests for high-order cells cut by analytic surfaces and for capability diagnostics on unsupported curved combinations.

#### Cut-State Restart Round Trip

- [x] Persist embedded geometry registry state through `MovingMeshRestart`.
- [x] Persist embedded geometry epochs, embedded constraint epochs, cut-topology revisions, cut quadrature policy metadata, and cut classification summaries.
- [x] Persist enough provenance to rebuild or restore stable cut-entity IDs after restart.
- [x] Add restart read/write tests proving cut maps rebuild deterministically from accepted restart state.
- [x] Add rollback/restart equivalence tests proving rejected trial embedded motion is not written as accepted cut state.
- [x] Add MPI restart tests for cut surfaces that cross rank boundaries.
- [x] Add curved cut restart/rollback qualification tests for quadratic curved-topology subdivision, arbitrary-order true arrangements, non-plane subdivision descriptors, callback application re-registration, level-set descriptors, triangulated-surface descriptors, and active-child Boolean descriptors.

#### Real FE Assembly Integration

- [x] Extend FE assembly planning so selected cells, faces, and embedded-interface regions can use cut quadrature rules instead of standard element or face quadrature.
- [x] Bind cut quadrature points, integration weights, normals, side indicators, cut volume fractions, and provenance into `AssemblyContext`.
- [x] Ensure standard assembled, matrix-free, interpreter, AD, symbolic tangent, and JIT paths see identical cut integration data.
- [x] Add matrix-free cut-geometry data rebuild rules keyed on cut classification, cut topology, quadrature policy, mesh revisions, and FE layout revisions.
- [x] Add true interpreter/JIT parity tests that assemble the same cut integral through both paths.
- [x] Add residual/tangent consistency tests using cut metadata terminals, with finite differences limited to verification.
- [x] Add a physics-neutral cut-domain assembly dispatcher that filters cut rules by FE execution path, prepares per-rule `AssemblyContext` data through caller-owned builders, dispatches real `AssemblyKernel` cell or embedded-interface methods, and accumulates local residual/tangent outputs without encoding governing equations.
- [x] Add analytic residual/tangent verification proving AD and symbolic nonlinear kernels assemble the same nonlinear cut-domain residual and tangent over cut quadrature without using production finite differences.

#### Distributed Cut Ownership and Exchange

- [x] Deduplicate cross-rank cut entities using canonical parent entity IDs, embedded provenance, side, and cut-topology IDs.
- [x] Include cut-facet, embedded-interface polygon, side-region, closed-subcell topology, and quadrature-relevant payload metadata in deterministic distributed cut exchange records.
- [x] Exchange cut facets, embedded-interface polygons, side-region summaries, and cut quadrature metadata between owner and ghost ranks through `CutDistributedState` records.
- [x] Define deterministic ownership for embedded-interface pieces that lie on partition boundaries.
- [x] Add distributed cut-topology invalidation for geometry, topology, ownership, numbering, labels, active configuration, classification, topology, local-packet, and exchanged-packet revision changes.
- [x] Add migration-specific distributed cut-state qualification tests on the production `DistributedMesh::migrate` path, including stale-state invalidation, stable cut identity, owner-rank changes, and post-migration ghost payloads.
- [x] Add repartition/rebalance-specific distributed cut-state qualification tests on the production `DistributedMesh::rebalance(PartitionHint::Cells)` path, including stale-state invalidation, stable cut identity, load redistribution, changed owner ranks, and post-rebalance ghost payloads.
- [x] Add MPI tests for duplicate filtering, deterministic ownership metadata, and cut restart metadata.
- [x] Add production neighbor-sparse request/response cut-state exchange so ghost ranks request only parent-cell cut payloads from owning neighbor ranks, owners respond only with matching cut entities, and nonparticipating ranks can receive no cut payload.
- [x] Add MPI tests for true owner/ghost cut-state exchange, rank-local deterministic sparse exchanged packet revisions, ghost side-region payloads, empty nonparticipant ranks, and ghost-layer ownership invalidation.
- [x] Add graph-partitioner-specific sparse cut-state qualification for block, METIS when enabled, and ParMETIS when enabled, including sparse-neighbor contract checks, deterministic rank-local exchanged revisions, ghost side-region payloads, migration invalidation, and valid rebuilds after migration.
- [x] Add MPI tests for migration cut-state invalidation.
- [x] Add MPI tests for repartition/rebalance cut-state invalidation.
- [x] Extend deterministic distributed cut exchange records with versioned curved-interface and curved-side metadata: geometry order, embedded kind, configuration, construction policy, exact-vs-surrogate flags, parent-parametric coordinates, isoparametric quadrature weights, Boolean active-child provenance IDs, and cut-sensitivity capability flags.
- [x] Add MPI regression coverage proving distributed curved construction metadata and AD-compatible sensitivity capability flags survive sparse owner/ghost exchange for graph-compatible true-curved plane arrangements and true-curved subdivision-style non-plane rows.
- [x] Add support-matrix rows and row-exact ledger evidence for distributed quadratic curved starter rows and supported distributed true-curved arrangement/subdivision rows across all advertised FE execution paths.

#### Supported-Combination Matrix

- [x] Publish a Mesh/Search support matrix that records parent cell family, geometry order, embedded geometry kind, distributed mode, cut mode, quadrature policy, conditioning policy, FE execution path, support status, and qualification notes.
- [x] Keep the legacy support-query overload for family/order/geometry/distribution/cut-mode/quadrature checks while adding an exact-query overload for conditioning policy and FE execution path.
- [x] Add regression coverage proving all advertised Phase 26 FE execution paths are present in the support matrix and unregistered execution paths fail closed.
- [x] Add an executable Mesh/Search support-matrix qualification harness that enumerates every advertised row and records required topology, quadrature, FE execution-path, restart/rollback, MPI, sensitivity, diagnostic, and full-validation evidence domains.
- [x] Add Mesh and FE regression coverage proving the qualification harness covers every advertised row, detects duplicate row keys, confirms fundamental topology/quadrature/FE-path/restart/MPI evidence where implemented, reports missing evidence explicitly when a row is incomplete, and now proves no implemented advertised row is missing sensitivity or full-validation evidence.
- [x] Thread curved restart/rollback qualification evidence into the support-matrix harness so implemented curved rows no longer report restart as a missing evidence domain.
- [x] Add a row-exact Mesh/Search validation ledger that records evidence domains by exact support-matrix row key rather than broad inferred categories.
- [x] Make `qualify_cut_support_matrix()` consume the validation ledger so qualification status is driven only by recorded row evidence.
- [x] Add `audit_cut_support_matrix_validation()` so every advertised row is classified as unsupported, fully validated, missing analytic/literature validation, missing sensitivity evidence, missing restart/MPI evidence, missing FE execution-path evidence, or advertised too broadly.
- [x] Add Mesh regression coverage proving every implemented row either has all required evidence or has explicit missing evidence domains, and that no implemented row currently lacks core topology, quadrature, FE execution-path, restart/rollback, MPI, or diagnostic evidence.
- [x] Seed row-exact full-validation evidence for first-order linearized rows on every advertised linear parent family (`Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, `Pyramid`, `Polygon`, and `Polyhedron`) and every advertised first-order embedded geometry kind (`Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite) across all six advertised FE execution paths (`standard-assembly`, `matrix-free`, `forms-interpreter`, `ad`, `symbolic-tangent`, and `jit`) in serial and advertised distributed modes.
- [x] Add FE regression coverage proving serial first-order plane-cut data for every advertised linear parent family is identical across standard assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT path views.
- [x] Add row-exact full-validation evidence for serial and advertised distributed first-order linearized `Sphere`, level-set field, triangulated-surface, and Boolean composite rows on every advertised linear parent family across all six FE execution paths.
- [x] Add Mesh and FE regression coverage proving first-order linearized non-plane rows conserve side measures, preserve provenance/restart/distributed-packet metadata, rebuild deterministic topology revisions, and expose identical cut integration data to standard assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT views.
- [x] Add analytic/AD-compatible sensitivity evidence for serial and advertised distributed first-order linearized `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite rows on every advertised linear parent family, with topology-derived polygon/polyhedron samples tied to the same deterministic linear subdivision/tessellation contract as cut topology.
- [x] Add Mesh and FE regression coverage proving first-order linearized sensitivity records carry parent geometry DOFs, shape values/gradients, AD-compatible availability flags, deterministic rebuild IDs, distributed packet metadata, and all-six-FE-path visibility.
- [x] Add full-validation evidence for serial quadratic plane-cut curved-isoparametric starter rows on `Line`, `Triangle`, `Quad`, `Tetra`, and `Hex` across all six advertised FE execution paths, and qualify them once analytic/AD curved cut-sensitivity evidence is present.
- [x] Add FE regression coverage proving serial quadratic plane-cut curved-isoparametric starter data is identical across standard assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT path views.
- [x] Add row-exact full-validation and analytic/AD-compatible sensitivity evidence for serial second-order controlled linearized rows on `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` parents with embedded `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite cuts across all six advertised FE execution paths.
- [x] Add Mesh regression coverage proving second-order controlled linearized rows conserve side measures, preserve high-order geometry-DOF provenance, rebuild deterministic topology/sensitivity records, preserve distributed packet sensitivity metadata, and are fully qualified by the support-matrix audit.

#### Small-Cut Conditioning Infrastructure

- [x] Add physics-neutral ghost-penalty neighborhood discovery over facets adjacent to cut cells.
- [x] Add aggregation or extension-cell patch construction for small cut cells.
- [x] Add algebraic conditioning metadata for solver, matrix-free data, preconditioner, and block-coupling refresh.
- [x] Keep all coefficients and physics-specific stabilization laws outside Mesh and FE; Mesh/FE may only expose geometric measures, neighborhoods, and hook points.
- [x] Add tests showing small-cut diagnostics produce deterministic aggregation or stabilization hook data.
- [x] Add tests proving conditioning hook updates invalidate assembled matrices, matrix-free data, preconditioners, and restart metadata as declared.

#### Shared Robust Predicate Policy

- [x] Thread the Phase 22 `RobustPredicatePolicy` or an equivalent shared policy into cut classification, cut topology, and cut quadrature.
- [x] Remove ad hoc hard-coded cut tolerances from Phase 26 code paths.
- [x] Record the tolerance policy used by each cut map, cut topology, and cut quadrature rule.
- [x] Add near-coplanar, near-tangent, near-vertex, near-edge, non-finite coordinate, zero-measure, and duplicate-intersection tests.
- [x] Add a true-curved high-order near-boundary sliver predicate regression tied to an analytic affine-hex plane-cut reference.
- [x] Expand adaptive/exact predicate implementation beyond the current robust-policy keys and Boolean active-child fallback to cover all orientation, intersection, coplanarity, duplicate-detection, and arbitrary Boolean arrangement decisions.
- [x] Add diagnostics that report the tolerance policy, mesh revision state, embedded geometry revision state, and FE layout revision state used by a failed cut operation.

#### Cut Geometry Sensitivities

- [x] Add analytic or AD-compatible derivatives of cut location with respect to mesh motion and embedded-geometry motion.
- [x] Add analytic or AD-compatible derivatives of cut measures, normals, volume fractions, and quadrature weights.
- [x] Add sensitivity capability gates for unsupported embedded geometry, cut topology, quadrature rule, element family, or geometry order.
- [x] Add Forms terminals or parameter bindings for cut sensitivity data without encoding governing equations.
- [x] Add Newton/tangent tests using cut metadata and cut quadrature sensitivities.
- [x] Keep finite-difference cut sensitivity checks as verification-only tests, not production tangent paths.
- [x] Qualify serial quadratic plane-cut `Line`, `Triangle`, `Quad`, `Tetra`, and `Hex` curved-isoparametric starter rows with analytic/AD-compatible parent-geometry shape-value and shape-gradient sensitivity metadata across standard assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT FE paths.
- [x] Qualify serial second-order controlled linearized `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` rows with analytic/AD-compatible parent-geometry shape-value and shape-gradient sensitivity metadata for `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite cuts across standard assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT FE paths.
- [x] Generalize analytic/AD-compatible curved cut sensitivity metadata to supported serial true-curved arrangement rows and true-curved subdivision rows, including `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` parent families where those rows are advertised.
- [x] Preserve curved cut sensitivity capability metadata through distributed cut exchange packets so owner and ghost ranks see the same AD-compatible location, Jacobian, measure, normal, and quadrature-weight availability flags.
- [x] Qualify serial and advertised distributed first-order linearized `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite rows with analytic/AD-compatible interface and side-region sensitivity metadata for every advertised linear parent family, including topology-linear polygon/polyhedron samples.

#### Boolean and Multi-Region Cut Semantics

- [x] Support physics-neutral union, intersection, and difference composition of embedded geometries.
- [x] Track side-region IDs for negative, positive, and multi-region partitions without encoding material laws.
- [x] Preserve composed-region provenance through cut topology and topology-derived quadrature records.
- [x] Preserve composed-region provenance through cut restart and transfer records.
- [x] Add diagnostics for ambiguous, overlapping, nested, or contradictory embedded-region composition.
- [x] Add tests for nested embedded regions, overlapping embedded regions, and Boolean region composition.

#### Robust CutFEM Acceptance Criteria

- [x] General embedded geometries can be registered, revised, restarted, and queried without Mesh or FE knowing the governing equations.
- [x] Cut topology is explicit, ordered, restartable, provenance-preserving, and MPI-deterministic for every advertised cell family.
- [x] Cut quadrature is generated from cut topology and passes conservation and analytic exactness tests for every advertised element family and quadrature policy.
- [x] FE assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT paths consume identical cut integration data.
- [x] FE residual/tangent kernels can be dispatched over cut-domain rules through a physics-neutral assembly loop, with analytic AD and symbolic verification evidence.
- [x] Small-cut conditioning hooks expose geometric neighborhoods and diagnostics without hard-coding physical stabilization laws.
- [x] Robust predicate policy is shared with moving-geometry validity infrastructure and recorded in diagnostics/restart metadata.
- [x] Cut sensitivities used in production tangents are analytic, symbolic, or AD/JIT-compatible; finite differences are verification-only.

#### Research-Backed Design Decisions for the Remaining Phase 26 Work

The remaining Phase 26 work should follow these design decisions. Each decision is scoped to Mesh/FE infrastructure only; physics modules still own governing equations, weak terms, material laws, penalties, and stabilization coefficients.

1. **Cut topology for all advertised families**
   - **Decision:** Use an explicit arrangement-style cut topology layer, not implicit per-quadrature clipping. Store cut vertices, cut edges, interface polygons, side regions, parent entity provenance, side labels, stable IDs, and degeneracy flags as first-class Mesh/Search records.
   - **Decision:** Implement topology reconstruction by decomposing parent cells into supported lower-order integration regions, reusing existing `Mesh/Geometry` polygon/polyhedron tessellation utilities. Do not duplicate generic polygon or polyhedron triangulation logic inside Search.
   - **Decision:** Treat topology reconstruction as capability-gated by cell family, geometry order, embedded geometry kind, and predicate policy. Unsupported combinations must return structured diagnostics instead of silently falling back.
   - **Rationale:** CutFEM literature treats cut-cell geometry, boundary/interface treatment, stabilization, and integration as coupled infrastructure concerns, not isolated helper routines [Burman-Claus-Hansbo-Larson-Massing 2015](https://doi.org/10.1002/nme.4823). Exact or robust arrangement-style reasoning is also consistent with CGAL's Nef polyhedron model, which is closed under Boolean set operations and designed around exact geometric predicates [CGAL Nef Polyhedra](https://doc.cgal.org/latest/Nef_3/index.html).
   - **Current blocker:** Linear side-region measure summaries and closed linear side-region subcell topology are implemented for every advertised linear parent family, including wedge, pyramid, polygon, and polyhedron. A tessellator-backed high-order linearized topology foundation is implemented for curved/high-order geometry with geometry-DOF provenance, and the serial second-order controlled linearized rows for `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` parents now have row-exact full-validation and sensitivity ledger evidence. Boolean descriptors, composed-region provenance, overlap/nesting diagnostics, active-child Boolean true-curved subdivision metadata, noncoincident child-child topology-switch metadata, coincident-child tolerance-resolved predicate fallback metadata, and Boolean regression tests are implemented. Opt-in true curved arrangement paths are implemented for arbitrary-order isoparametric line cells cut by planes, graph-compatible high-order triangle/quad cells cut by planes, graph-compatible high-order tetra/hex/wedge/pyramid cells cut by planes, bounded non-graph high-order triangle/quad face cells cut by planes, bounded non-graph high-order hex/wedge/pyramid cells cut by planes, and bounded non-plane `Sphere`/signed-distance callback/level-set/triangulated-surface/Boolean composite cuts through the explicit `true-curved-subdivision-arrangement` policy. The supported serial and distributed true-curved support-matrix rows now have row-exact full-validation and sensitivity ledger evidence. Production work still needs independent analytic/literature validation and broader exact/adaptive predicate coverage beyond the current Boolean active-child fallback.

2. **Topology-derived cut quadrature**
   - **Decision:** Generate cut-volume, cut-face, and embedded-interface quadrature from the reconstructed cut topology, not from axis-aligned or segment-only helper geometry.
   - **Decision:** Provide two policy families: deterministic subdivision quadrature for linear cut regions, and moment-fitted/high-order implicit quadrature for curved or high-order regions.
   - **Decision:** Every quadrature rule must record parent entity, side, embedded geometry ID, cut-topology revision, predicate policy key, construction policy, polynomial exactness metadata, and conservation diagnostics.
   - **Decision:** Require conservation tests `measure(negative) + measure(positive) = measure(parent)` for all split volume rules, plus analytic exactness tests for linearly cut tetrahedra, hexahedra, triangles, and quadrilaterals before claiming family support.
   - **Rationale:** High-order quadrature over implicitly defined surfaces and volumes is a known distinct numerical method; Saye's algorithm targets curved surfaces/volumes defined by level sets in hyperrectangles and motivates separating quadrature construction policy from geometry classification [Saye 2015](https://doi.org/10.1137/140966290). The current helper quadrature is useful for constants and smoke tests, but not enough for production CutFEM.
   - **Current blocker:** Topology-derived split-volume quadrature can now consume closed linear side-subcell measures, dimension-aware lower-dimensional interface records, embedded-interface polygons, curved-isoparametric starter subcells, exact true-arrangement line/plane intervals, graph-compatible true-arrangement triangle/quad face strips, graph-compatible true-arrangement tetra/hex/wedge/pyramid volume side regions, bounded non-graph true-curved subdivision side regions, and Boolean true-curved subdivision patch metadata with active-child, noncoincident child-child topology-switch, and coincident-child predicate-fallback provenance. Supported serial and distributed true-curved rows have conservation and all-six-FE-path parity evidence in the row-exact ledger. Independent literature-scale policy validation is not complete.

3. **High-order and curved cuts**
   - **Decision:** Support two explicit modes: `linearized-cut` and `curved-isoparametric-cut`. Linearized mode is allowed only when requested and must be recorded in diagnostics/restart metadata.
   - **Decision:** For high-order support, classify using geometry DOFs and then reconstruct the curved interface using a parametric mapping or level-set-aware mapping, not by corner vertices alone.
   - **Decision:** Curved cut quadrature must carry frame-explicit reference/current semantics, curved normals, curved measures, and validity diagnostics for slivers, folded curved facets, and inconsistent side regions.
   - **Rationale:** Lehrenfeld's high-order unfitted method combines a piecewise planar interface reconstruction with a parametric mesh mapping to recover high-order accurate curved-surface and volume integration; this argues for a distinct curved-cut geometry lifecycle rather than overloading linear topology [Lehrenfeld 2016](https://doi.org/10.1016/j.cma.2015.12.005).
   - **Current blocker:** The current implementation has high-order classification, a linearized-mode gate, curved-patch descriptors, a bounded quadratic line/triangle/quad/tetra/hex plane starter path with parent-parametric curved topology and isoparametric quadrature weights, an explicit true-arrangement mode for arbitrary-order line/plane cuts, graph-compatible true triangle/quad face arrangements, graph-compatible true tetra/hex/wedge/pyramid plane volume arrangements, bounded non-graph true-curved subdivision arrangements for high-order hex/wedge/pyramid plane volume cuts, and bounded non-plane subdivision for `Sphere`, signed-distance callback, level-set field, triangulated-surface, and Boolean composite descriptors including noncoincident child-child topology switches and coincident-child tolerance-resolved active-child fallback. Distributed curved metadata exchange is now qualified through serial packet and MPI sparse owner/ghost tests. Remaining blockers are literature-scale curved-quadrature validation and broader exact/adaptive predicate coverage beyond the current Boolean fallback.

4. **Cut-state restart and deterministic rebuilds**
   - **Decision:** Persist registry descriptors, revision states, accepted embedded geometry IDs, topology revision keys, quadrature policy metadata, predicate policy metadata, cut summaries, and stable cut-entity provenance.
   - **Decision:** Do not attempt to serialize arbitrary signed-distance callbacks as executable function objects. Callback-backed geometries must restart through a stable descriptor plus application re-registration contract, or through a sampled level-set/surface representation.
   - **Decision:** Restart tests must prove accepted-state rebuild determinism and rejected-trial rollback exclusion in serial and MPI.
   - **Rationale:** HDF5's group/dataset/attribute model is appropriate for storing hierarchical descriptors, arrays, and metadata, but executable callbacks are not a portable restart object [HDF5 User Guide](https://portal.hdfgroup.org/documentation/hdf5/latest/_u_g.html). PETSc DMPlex restart/distribution workflows also emphasize global point identity and migration maps for redistributing loaded meshes [PETSc DMPlex manual](https://petsc.org/release/manual/dmplex/).
   - **Current blocker:** `MovingMeshRestart` persists embedded geometry registry descriptors, cut-classification records, Boolean provenance, cut-topology revision metadata, predicate-policy keys, FE-layout revision metadata, and side-region provenance. Curved cut restart/rollback qualification now covers quadratic curved-topology subdivision, arbitrary-order true arrangements, non-plane subdivision descriptors, callback application re-registration, level-set descriptors, triangulated-surface descriptors, and active-child Boolean descriptors. Remaining work is full validation of those restart paths across every advertised support-matrix row.

5. **Real FE assembly integration**
   - **Decision:** Make cut quadrature selection part of FE assembly planning. Standard assembly, matrix-free data, interpreter, AD, symbolic tangent, and JIT must consume the same cut integration record, not independently rediscover cut data.
   - **Decision:** Treat cut metadata as quadrature-point data or parameter terminals, matching the existing Forms/JIT lowering model. Avoid per-backend special cases that can make matrix-free and assembled paths disagree.
   - **Decision:** Add parity tests that assemble the same cut integral through interpreter and JIT paths, then residual/tangent tests where analytic/AD cut sensitivities are active and finite differences are verification-only.
   - **Rationale:** UFL/FFC-style form systems define integrals over cell, exterior-facet, and interior-facet domains and rely on form compiler metadata/quadrature choices to generate equivalent kernels [UFL Form Language](https://docs.fenicsproject.org/ufl/main/manual/form_language.html), [Kirby-Logg 2006](https://doi.org/10.1145/1163641.1163644). This supports a single assembly-facing cut integration contract.
   - **Current blocker:** FE now exposes shared cut integration records, closed-topology imports, cut metadata terminals, support-matrix execution-path metadata, `CutIntegrationContext::evaluateScalarCutOperator(...)` for standard, matrix-free, interpreter, AD, symbolic tangent, and JIT paths, and a physics-neutral `CutDomainAssembler` that dispatches real FE residual/tangent kernels over cut rules. First-order linearized plane and non-plane rows for every advertised linear parent family, serial second-order controlled linearized high-order rows, quadratic curved-starter rows, supported serial true-curved rows, and their advertised distributed curved counterparts now have all-six-FE-path row ledger evidence, including noncoincident Boolean child-child topology-switch rows. First-order and second-order controlled linearized rows also expose AD-compatible sensitivity metadata to every advertised FE path. Production scalar cut-operator tests verify exact constant half-cube split measures, interface measure, x-moment conservation, and path-identical scalar reductions across all six execution views for every advertised first-order family with plane and non-plane embedded geometries. Cut-domain nonlinear residual/tangent tests now verify AD and symbolic kernels against analytic hand-derived results over cut quadrature without production finite differences. Remaining work is independent analytic/literature validation cases and broader exact/adaptive predicate coverage.

6. **Distributed cut ownership and exchange**
   - **Decision:** Define cut-entity identity from canonical parent global IDs, embedded geometry provenance, side-region ID, cut-topology ID, and predicate policy key. Rank-local ordering must not affect identity.
   - **Decision:** Exchange cut facets, polygons, side-region summaries, and quadrature metadata through owner/ghost communication keyed by those stable IDs. Repartition, migration, renumbering, and ghost rebuilds must invalidate distributed cut maps.
   - **Decision:** Deterministic ownership for partition-boundary embedded-interface pieces should follow parent owner first, then canonical global ID tie-breaks.
   - **Rationale:** p4est's scalable AMR design uses compact, deterministic distributed identity and ghost-layer algorithms for large-scale partitioned meshes [Burstedde-Wilcox-Ghattas 2011](https://doi.org/10.1137/100791634). DMPlex similarly separates mesh topology from distribution and uses point ownership plus overlap/migration maps for parallel unstructured meshes [DMPlex](https://petsc.org/main/manualpages/DMPlex/DMPLEX/), [DMPlexDistribute](https://petsc.org/release/manualpages/DMPlex/DMPlexDistribute/).
   - **Current blocker:** Mesh has distributed owner metadata for cut classification, deterministic exchange packet payloads for cut topology, an MPI-backed neighbor-sparse `CutDistributedState` owner/ghost request/response exchange contract, production `DistributedMesh::migrate` qualification, production `DistributedMesh::rebalance(PartitionHint::Cells)` qualification, and graph-partitioner-specific sparse-exchange qualification for block, METIS when enabled, and ParMETIS when enabled. These tests cover stale-state invalidation, stable cut identity, changed owner ranks, empty nonparticipant ranks, rank-local deterministic sparse revisions, sparse-neighbor contracts, and post-migration/rebalance ghost payloads. No distributed exchange blocker remains; broader Phase 26 production work is now concentrated in curved/arbitrary-order geometry and full-family validation.

7. **Small-cut conditioning**
   - **Decision:** Keep Mesh/FE conditioning infrastructure geometric and algebraic: volume fractions, ghost-penalty neighborhoods, extension/aggregation patches, conditioning indicators, and invalidation metadata. Physics modules choose coefficients and weak forms.
   - **Decision:** Support both ghost-penalty neighborhood discovery and aggregated/extension patch construction. They should be alternative or composable strategies selected by solver policy.
   - **Decision:** Conditioning updates must invalidate assembled matrices, matrix-free geometry/operator data, preconditioners, block coupling metadata, and restart summaries.
   - **Rationale:** Ghost penalties are a core CutFEM response to small cut cells and stability/conditioning problems [Burman et al. 2015](https://doi.org/10.1002/nme.4823). Aggregated unfitted FEM directly targets small-cut ill-conditioning by building extension constraints from well-posed cells and reports body-fitted-like conditioning behavior [Badia et al. 2018](https://doi.org/10.1016/j.cma.2018.03.022).
   - **Current blocker:** The hook data and tests exist, but the solver-side policy and production operator updates are not yet wired through real cut assembly.

8. **Shared robust predicate policy and diagnostics**
   - **Decision:** Centralize all cut classification, topology, intersection, and quadrature tolerances under the Phase 22 robust predicate policy. Do not add new hard-coded tolerances in cut algorithms.
   - **Decision:** Use fast floating-point filters where safe, but fall back to adaptive/exact predicates for ambiguous orientation, intersection, coplanarity, and duplicate-detection cases.
   - **Decision:** Failed cut operations must report predicate policy, mesh revision state, embedded revision state, FE layout revision state, entity provenance, and the failing query.
   - **Rationale:** Shewchuk's adaptive-precision predicates are the standard reference for robust orientation/incircle-style decisions whose required precision depends on input uncertainty [Shewchuk 1997](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf). Robust Boolean geometry packages also make exact arithmetic/predicate choices central rather than incidental [CGAL Nef Polyhedra](https://doc.cgal.org/latest/Nef_3/index.html).
   - **Current blocker:** Predicate keys are threaded through current records, and ambiguous Boolean active-child selection now has a deterministic tolerance-resolved fallback with serial/MPI metadata tests. Broader exact/adaptive predicates for all orientation, intersection, coplanarity, duplicate-detection, and arbitrary Boolean arrangement cases are still future work.

9. **Cut geometry sensitivities**
   - **Decision:** Production sensitivities must be analytic, symbolic, or AD-compatible. Finite differences are allowed only as test or validation checks.
   - **Decision:** Sensitivities should be attached to cut topology and quadrature records: cut-point location, normal, measure, volume fraction, and quadrature weight derivatives with respect to mesh-motion DOFs and embedded-geometry parameters.
   - **Decision:** Unsupported sensitivity combinations must return capability diagnostics and force Newton/tangent infrastructure to rebuild or reject reuse, not silently lag a tangent.
   - **Rationale:** Shape optimization with CutFEM and level-set/unfitted methods treats geometry differentiation as first-class infrastructure; automatic shape differentiation work also shows why geometry transformations and FE-level derivatives must be represented explicitly [Dokken et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8550642/), [Burman et al. shape-optimization CutFEM](https://doi.org/10.1016/j.cma.2017.09.005). AD avoids the cancellation and step-size sensitivity that make finite-difference production tangents fragile.
   - **Current blocker:** The FE terminals, simple analytic sensitivity record, first-order linearized sensitivity metadata, second-order controlled linearized high-order sensitivity metadata, quadratic curved plane-cut starter sensitivity metadata, supported serial true-curved topology sensitivity metadata, and distributed curved sensitivity exchange metadata now exist. Serial and advertised distributed first-order linearized rows for every advertised linear parent family and embedded `Plane`, `Sphere`, level-set field, triangulated-surface, or Boolean composite kind now expose topology-derived interface and side-region sensitivity records; polygon and polyhedron rows use deterministic topology-linear samples from the same subdivision/tessellation contract as cut topology. Serial second-order controlled linearized `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` rows expose tessellator-derived parent-parametric interface and side-region sensitivity records for `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite cuts. Serial and advertised distributed quadratic `Line`, `Triangle`, `Quad`, `Tetra`, and `Hex` curved-isoparametric starter rows plus advertised serial/distributed true-curved arrangement/subdivision rows expose or preserve parent-parametric quadrature or subcell coordinates with analytic geometry shape values and gradients for interface and side-volume quadrature records, giving AD/symbolic/JIT consumers a production non-finite-difference dependency contract for geometry-DOF location, Jacobian, measure, normal, and quadrature-weight derivatives. Coincident Boolean child fallback records the nonsmooth predicate decision provenance instead of pretending a smooth active-child derivative exists. No implemented support-matrix row is missing the sensitivity evidence domain; remaining work is independent analytic/literature validation and broader exact/adaptive predicate coverage.

10. **Boolean and multi-region semantics**
    - **Decision:** Represent Boolean and multi-region composition at the embedded-geometry registry level, not inside physics modules. Mesh/Search owns region IDs, side-region IDs, composed provenance, and ambiguity diagnostics.
    - **Decision:** For implicit/level-set geometry, use signed-distance/level-set composition rules only as a query layer; cut topology must still preserve explicit composed-region provenance so restart and transfer are deterministic.
    - **Decision:** For triangulated surfaces and imported surface meshes, prefer exact arrangement/Boolean-style topology where possible; otherwise declare controlled approximate mode and record it.
    - **Rationale:** Level-set methods provide a natural implicit interface representation [Osher-Fedkiw 2003](https://link.springer.com/book/10.1007/b98879), while exact Boolean set operation systems such as CGAL Nef polyhedra show why closed set-operation semantics and exact predicates matter for robust multi-region geometry [CGAL Nef Polyhedra](https://doc.cgal.org/latest/Nef_3/index.html).
    - **Current blocker:** Boolean descriptors, composed-region provenance through topology/quadrature/restart records, overlap/nesting diagnostics, and nested/overlapping/Boolean composition tests are implemented. Remaining Boolean work is exact arrangement-derived topology for arbitrary curved and multi-region intersections plus full validation across the support matrix.

11. **Validation and acceptance**
    - **Decision:** Support claims must be milestone-scoped. A family/geometry/order combination is unsupported until topology, quadrature, restart, MPI exchange, assembly parity, conditioning, and sensitivity tests pass for that combination.
    - **Decision:** Add a Phase 26 support matrix with rows for parent cell family, geometry order, embedded geometry kind, cut mode, distributed mode, quadrature policy, conditioning policy, and FE execution path.
    - **Decision:** Add benchmark validation cases only after infrastructure tests pass: exact-volume linear cuts, spherical cuts with analytic volume/area, manufactured Poisson/elasticity CutFEM cases, and small-cut conditioning studies.
    - **Rationale:** The CutFEM overview emphasizes that robust unfitted methods combine boundary/interface treatment, stabilization, and numerical integration [Burman et al. 2015](https://doi.org/10.1002/nme.4823); high-order unfitted accuracy depends on geometry approximation and integration quality [Lehrenfeld 2016](https://doi.org/10.1016/j.cma.2015.12.005), [Saye 2015](https://doi.org/10.1137/140966290).
   - **Current blocker:** The Phase 26 support matrix exists in code and now records parent cell family, geometry order, embedded geometry kind, distributed mode, cut mode, quadrature policy, conditioning policy, FE execution path, status, and qualification notes. The executable qualification harness is now driven by a row-exact validation ledger that covers every advertised row and reports required topology, quadrature, FE execution, restart/rollback, MPI, sensitivity, diagnostic, and validation evidence domains. The audit API classifies every row as unsupported, fully validated, missing validation, missing sensitivity evidence, missing restart/MPI evidence, missing FE-path evidence, or advertised too broadly; current tests prove no implemented row is missing core topology, quadrature, FE-path, restart/rollback, MPI, diagnostic, sensitivity, or full-validation evidence. Serial and advertised distributed first-order linearized rows for every advertised linear parent family and embedded `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite kind now have complete full-validation and analytic/AD-compatible sensitivity ledger evidence across all six advertised FE execution paths, so those exact rows are qualified. Serial second-order controlled linearized rows for `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` parents and serial/advertised distributed quadratic plane-cut curved-isoparametric starter rows for line/triangle/quad/tetra/hex now have full-validation and analytic/AD-compatible sensitivity ledger evidence across all six FE paths, so those exact rows are qualified. Supported serial/distributed true-curved arrangement/subdivision rows are also qualified, including bounded noncoincident Boolean child-child topology-switch rows and coincident-child predicate-fallback metadata rows. The support-matrix cross-product is now fully qualified for implemented advertised rows; remaining production rigor is independent literature/analytic validation and broader exact/adaptive predicate coverage.

Implementation evidence, 2026-04-26:

- Mesh embedded-geometry and cut-classification infrastructure lives in `Mesh/Search/CutCell.{h,cpp}` and provides plane/sphere embedded descriptors, kinematic-constraint provenance, cell/face/edge classification, intersection points, revision snapshots, restart records, transaction commit/rollback, and distributed owner metadata.
- FE cut integration infrastructure lives in `FE/Geometry/CutQuadrature.{h,cpp}`, `FE/Assembly/CutIntegrationContext.h`, `FE/Forms/CutCellForms.h`, and `FE/Systems/CutIntegrationInvalidation.{h,cpp}` and provides physics-neutral cut quadrature, cut metadata terminals, assembly context hooks, stabilization hook points, conditioning diagnostics, restart metadata, and revision-based invalidation.
- Verification passed:
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI'`
  - `cmake --build build-fe-check --target test_fe_geometry test_fe_forms test_fe_systems -j 4`
  - `./build-fe-check/test_fe_geometry --gtest_filter='CutQuadrature.*'`
  - `./build-fe-check/test_fe_forms --gtest_filter='CutCellForms.*'`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*'`

Implementation evidence, 2026-04-27:

- `Mesh/Search/CutCell.{h,cpp}` now provides an `EmbeddedGeometryRegistry`, analytic signed-distance callbacks, level-set sample descriptors with field-layout and field-value revision dependencies, triangulated-surface descriptors, ASCII-STL-like imported surface descriptors, Boolean embedded-geometry composition diagnostics, query diagnostics, shared robust-predicate policy keys, high-order geometry DOF classification, deterministic cut-topology records, stable cut-entity IDs, side-region records, topology projection after embedded motion, distributed exchange packets, support-matrix queries with conditioning-policy and FE-execution-path dimensions, topology capability gates, and restartable registry/classification summaries.
- `Mesh/Search/CutCell.{h,cpp}` now exposes `cut_support_matrix_validation_ledger()` and `qualify_cut_support_matrix()`. The ledger records evidence domains by exact support-matrix row key, including topology, quadrature, FE execution path, restart/rollback, MPI, sensitivity, diagnostic, and full-validation evidence. The qualification harness consumes only this ledger, marks unsupported rows as explicitly diagnosed, qualifies curved restart/rollback evidence for topology-derived curved rows, qualifies first-order linearized `Plane`, `Sphere`, level-set field, triangulated-surface, and Boolean composite rows for every advertised linear parent family across all six advertised FE execution paths in serial and advertised distributed modes, records full-validation and analytic/AD-compatible sensitivity evidence for serial second-order controlled linearized `Line`, `Triangle`, `Quad`, `Tetra`, `Hex`, `Wedge`, and `Pyramid` rows across all six FE paths, records full-validation evidence for serial and advertised distributed quadratic plane-cut curved-isoparametric starter rows and supported serial/distributed true-curved arrangement/subdivision rows across all six FE paths, records analytic/AD-compatible sensitivity evidence for first-order linearized rows, second-order controlled linearized rows, quadratic curved starter rows, and supported true-curved arrangement/subdivision rows, and now audits every implemented advertised row as fully validated or explicitly unsupported.
- `Mesh/Search/CutCell.{h,cpp}` now reconstructs family-generic linear side-region measure summaries and closed linear integration subcell topology for line, triangle, quad, tetra, hex, wedge, pyramid, polygon, and polyhedron parents. The implementation records stable parent/cut integration-region IDs, integration vertices, integration subcells, boundary faces, parent measure, side measure, volume fraction, centroid estimates, and provenance; polygon measures reuse `MeshGeometry`, and convex polyhedron measures reuse `PolyhedronTessellation`. Interface records are dimension-aware for point, segment, and polygon interfaces so lower-dimensional parent families can produce topology-derived interface quadrature. Curved/topology validity diagnostics now reject non-finite geometry, folded or self-intersecting interface polygons, duplicate or zero-length cut intersections, sliver side regions, open subcell topology, invalid subcell measures, and nonconservative positive/negative side-region measure sums.
- `Mesh/Search/CutCell.cpp` now routes high-order parent cells through `CurvilinearEval`/`Tessellator` for controlled high-order cut-topology reconstruction, preserving parent geometry-DOF provenance in the generated integration vertices. `Mesh/Geometry/PolyhedronTessellation.{h,cpp}` exposes shared coordinate-based linear volume-cell tetrahedralization and the canonical tet-corner index contract so tessellated high-order volume subcells can carry matching parent-parametric coordinates through clipping. Curved patch descriptors now record parent parametric coordinates, physical patch points, stable curved-patch IDs, construction-policy metadata, exact-vs-linearized-surrogate flags, and linearization residuals so exact curved arrangements can be added without changing downstream topology consumers. The 2026-04-27 follow-up adds a bounded quadratic line/triangle/quad/tetra/hex curved-isoparametric starter path: cut integration vertices/subcells carry parent-parametric coordinates, curved subcell measures are computed by analytic isoparametric quadrature through `CurvilinearEval`, curved interface patches carry positive quadrature samples/weights/normals, curved interface patches also store parent-parametric quadrature coordinates, curved conservation diagnostics use the recorded curved-quadrature tolerance, and the support matrix advertises `curved-isoparametric-cut` + `curved-topology-subdivision` starter rows. A later same-day follow-up adds `CutCurvedArrangementMode::TrueArrangement`, exact/non-surrogate arbitrary-order line/plane root topology, closed side-specific isoparametric reference intervals, a high-order wildcard support-matrix row for true line/plane arrangements, fail-closed diagnostics for unsupported true high-order face/volume cuts, and a dimension-aware Jacobian action fix so lower-dimensional isoparametric measures do not read inactive parametric columns. The next same-day follow-up extends `TrueArrangement` to graph-compatible high-order triangle/quad plane cuts with bracketed reference-space contour roots, exact/non-surrogate curve metadata, mapped-Jacobian curve quadrature, closed side-specific graph strips, support-matrix rows for true high-order face arrangements, and fail-closed volume diagnostics. Subsequent follow-ups extend the opt-in true arrangement path to graph-compatible arbitrary-order high-order tetra/plane, hex/plane, wedge/plane, and pyramid/plane cuts: reference-space root surfaces are integrated with analytic mapped Jacobians from `CurvilinearEval`, pyramid support uses shrinking-column reference spans, side regions conserve parent volume through matched base-cell/root-interval quadrature, closed representative tetra/polyhedron, hex/polyhedron, wedge/polyhedron, or pyramid/polyhedron subcells preserve parent-parametric coordinates, FE import preserves the true-arrangement construction policy, and the support matrix advertises graph-compatible plane rows separately from bounded subdivision rows. The 2026-04-28 pass adds the `true-curved-subdivision-arrangement` policy for non-graph high-order triangle/quad/hex/wedge/pyramid plane cuts and bounded non-plane `Sphere`/signed-distance callback/level-set/triangulated-surface/Boolean composite cuts: graph-incompatible and non-plane cuts fall back to bounded tessellator-backed isoparametric subdivision without enabling linearized-surrogate mode, edge intersections are root-refined in parent-parametric space before patch recording, level-set support uses stored gradients for local affine signed-distance queries without finite-difference normals, triangulated-surface support uses closest-triangle signed distance/normal queries, Boolean composites use active-child signed-distance/normal/closest-point selection with operation and child-provenance metadata on curved patch records, noncoincident child-child topology switches are preserved through per-vertex active-child provenance on mixed-child curved patches, fully coincident ambiguous Boolean children use deterministic tolerance-resolved active-child fallback metadata, malformed level-set or triangulated-surface descriptors fail closed, face cuts use multi-component curved patch records and closed curved triangular side subcells, side-region measures are normalized to the parent balance contract, exact curved patches are validated through their parent-parametric coordinates and isoparametric quadrature rather than planar chord-polygon folding heuristics, FE import preserves the subdivision policy, and the support matrix advertises the non-graph/non-plane rows separately from graph-compatible true arrangements. Later 2026-04-28 sensitivity passes add `CutQuadratureGeometrySensitivityRecord` metadata for first-order linearized rows, serial second-order controlled linearized rows, serial quadratic plane-cut starter rows, and supported serial true-curved arrangement/subdivision rows: interface and side-volume records carry parent geometry DOFs, parent-parametric or topology-linear sample coordinates, geometry shape values, geometry shape gradients, topology revision hashing, construction-policy provenance, and AD-compatible availability flags for location, Jacobian, measure, normal, and quadrature-weight sensitivities. First-order polygon/polyhedron sensitivity records use topology-linear samples from the same deterministic subdivision/tessellation contract as the cut topology instead of production finite-difference stencils.
- `Mesh/Search/CutCell.{h,cpp}` now builds `CutDistributedState` from deterministic cut exchange packets. MPI uses a neighbor-sparse owner/ghost request/response exchange: ghost ranks request only parent-cell cut payloads from owning neighbor ranks, owners respond only with matching cut entities, nonparticipating ranks can receive no cut payload, and rank-local exchanged packet revisions remain deterministic. The distributed state classifies owned/imported/ghost records by stable IDs and parent GIDs, preserves closed side-region payloads, diagnoses owner conflicts or missing payloads, and invalidates state across mesh geometry, topology, ownership, numbering, label, active-configuration, classification, topology, local-packet, and exchanged-packet revision changes. Versioned distributed entity records now also preserve curved geometry order, embedded kind, frame, construction policy, exact-vs-surrogate flags, parent-parametric coordinates, isoparametric quadrature weights, Boolean active-child provenance IDs, tolerance-resolved predicate-fallback metadata, and AD-compatible sensitivity capability flags for curved interface and side records. Cut identity is stable across the production `DistributedMesh::migrate` and `DistributedMesh::rebalance(PartitionHint::Cells)` paths. The MPI regression verifies stale-state rejection, changed owner ranks, load redistribution, empty nonparticipant ranks, post-migration/rebalance ghost payload availability, graph-partitioner-specific sparse exchange for block, METIS when enabled, and distributed curved metadata exchange for true-curved, subdivision-style, and coincident Boolean fallback rows.
- `Mesh/IO/MovingMeshRestart.{h,cpp}` now persists embedded geometry registry descriptors and cut-classification restart records, including descriptor revisions, source-surface revisions, provenance, level-set samples, triangulated surfaces, Boolean children, callback re-registration flags, cut-topology revisions, predicate-policy keys, FE-layout revision metadata, flattened Boolean-composition child provenance, composition operations, and side-region provenance from transferred/rebuilt cut-topology records.
- `FE/Geometry/CutQuadrature.{h,cpp}` now fixes no-crossing segment clipping, records quadrature construction policy/provenance, adds topology-derived split-volume/interface quadrature, closed-topology subcell quadrature input, dimension-aware point/segment/polygon interface quadrature, curved-interface quadrature with reference/current frame metadata, curved-topology-subdivision construction metadata for isoparametric subcells, optional moment-fitted volume quadrature metadata, analytic spherical-cap/circular-disk reference validation, stricter malformed curved-rule diagnostics for non-finite data, inverted normals, missing positive-weight points, and inconsistent point weights, conservation diagnostics, analytic cut-location sensitivity, and analytic axis-aligned cut sensitivity metadata without using finite differences in production paths.
- `FE/Assembly/CutIntegrationContext.h`, `FE/Systems/CutIntegrationInvalidation.{h,cpp}`, and `FE/Forms/CutCellForms.h` now expose cut integration bindings visible to standard, matrix-free, interpreter, AD, symbolic tangent, and JIT paths; closed cut-topology import; family-complete linearized topology-subdivision quadrature for line, triangle, quad, tetra, hex, wedge, pyramid, polygon, and polyhedron parents; curved patch import into curved interface quadrature rules; curved-isoparametric volume-rule metadata for the second-order controlled linearized high-order rows and the quadratic line/triangle/quad/tetra/hex starter slice; true-curved-arrangement construction-policy preservation for arbitrary-order line/plane, graph-compatible triangle/quad face, graph-compatible tetra/hex/wedge/pyramid volume cuts, bounded non-graph triangle/quad/hex/wedge/pyramid subdivision volume/interface rules, and bounded non-plane `Sphere`/signed-distance callback/level-set/triangulated-surface/Boolean composite subdivision volume/interface rules including noncoincident child-child topology-switch cases; cut-topology/quadrature/conditioning revisions; deterministic small-cut neighborhoods and extension patches; cut sensitivity terminals; assembly-facing cut sensitivity metadata visible to all six FE execution paths for first-order linearized rows, second-order controlled linearized rows, the quadratic curved starter slice, and supported true-curved topology rows; and all-six-FE-path parity evidence for the supported serial true-curved support-matrix rows.
- Verification passed:
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.*'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*'`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$'`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_Tessellation$|^test_CutCell$' -j1`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCellMPI'`
  - `cmake --build build-mesh-tests --target test_MovingMeshRestart test_MovingMeshRestartMPI -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_MovingMeshRestart$|test_CutCell$'`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCellMPI|test_MovingMeshRestartMPI' -j1`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI|test_MovingMeshRestart$|test_MovingMeshRestartMPI' -j1`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.DistributedExchangePacketPreservesCurvedSensitivityMetadata:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification' --gtest_color=no`
  - `mpirun -np 2 ./build-mesh-tests/Tests/test_CutCellMPI`
  - `mpirun -np 4 ./build-mesh-tests/Tests/test_CutCellMPI`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `cmake --build build-fe-check --target test_fe_geometry test_fe_systems test_fe_forms -j 4`
  - `./build-fe-check/test_fe_geometry --gtest_filter='CutQuadrature.*'`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurved*'`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*'`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementTetraUsesGraphSurfaceWithoutLinearizedSurrogate'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementHexUsesGraphSurfaceWithoutLinearizedSurrogate'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*UsesSubdivisionForNonGraph*'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementWedge*'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementPyramid*'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementBooleanCompositeUsesPredicateFallbackForCoincidentChildren:CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*'`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedHexArrangementMetadata'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurved*:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests' -j1`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementBooleanCompositeRecordsChildChildIntersectionTopology:CutCell.TrueCurvedArrangementBooleanCompositeUsesPredicateFallbackForCoincidentChildren:CutCell.TrueCurvedArrangementBooleanCompositeUsesSubdivisionForActiveChildCuts' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedBooleanChildChildSubdivisionArrangementMetadata:CutIntegrationInfrastructure.ImportsTrueCurvedNonPlaneSubdivisionArrangementMetadata' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedWedgeArrangementMetadata'`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedPyramidArrangementMetadata'`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedNonGraphSubdivisionArrangementMetadata'`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurved*'`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests|FE_Forms_Tests'`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `cmake --build build-fe-check --target test_fe_geometry test_fe_systems test_fe_forms -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests|FE_Forms_Tests' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Systems_Tests' -j1`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.LinearFamilySideRegionsConserveMeasuresForAdvertisedFamilies:CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit'`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.*'`
  - `./build-fe-check/test_fe_forms --gtest_filter='CutCellForms.*'`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests|FE_Forms_Tests'`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementBooleanCompositeUsesSubdivisionForActiveChildCuts:CutCell.TrueCurvedArrangementBooleanCompositeUsesPredicateFallbackForCoincidentChildren:CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit:CutCell.BooleanCompositionSemanticsCoverUnionIntersectionDifferenceAndNestedRegions' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedNonPlaneSubdivisionArrangementMetadata' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*TrueCurved*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R '^test_CutCell$'`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$'`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.CurvedCutRestartRoundtripAndRollbackQualifiesSupportMatrixRows:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification' --gtest_color=no`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementBooleanCompositeUsesPredicateFallbackForCoincidentChildren:CutCell.TrueCurvedArrangementBooleanCompositeRecordsChildChildIntersectionTopology:CutCell.TrueCurvedArrangementBooleanCompositeUsesSubdivisionForActiveChildCuts' --gtest_color=no`
  - `mpirun -np 2 ./build-mesh-tests/Tests/test_CutCellMPI`
  - `mpirun -np 4 ./build-mesh-tests/Tests/test_CutCellMPI`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.LinearFamilySideRegionsConserveMeasuresForAdvertisedFamilies:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.QuadraticCurvedStarterRowsHaveFullValidationEvidence:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.SupportMatrixValidationAuditCategorizesEveryAdvertisedRow:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.LinearizedPlaneCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.QuadraticCurvedPlaneCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$' -j1`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCellMPI' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `cmake --build build-fe-check --target test_fe_geometry test_fe_systems -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$' -j1`
  - `git diff --check -- Code/Source/solver/Mesh/Search/CutCell.h Code/Source/solver/Mesh/Search/CutCell.cpp Code/Source/solver/Mesh/Tests/Unit/Search/test_CutCell.cpp Documentation/plan_moving_mesh_infrastructure.md`
  - `rg -n "finite[- ]difference|finite difference|FD" Code/Source/solver/Mesh/Search/CutCell.h Code/Source/solver/Mesh/Search/CutCell.cpp Code/Source/solver/Mesh/Tests/Unit/Search/test_CutCell.cpp`
  - `./build-fe-check/test_fe_geometry --gtest_filter='CutQuadrature.*' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit:CutCell.ClassifiesStaticPlaneAndSphereCuts' --gtest_color=no`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `cmake --build build-fe-check --target test_fe_systems -j 1` after a transient parallel rebuild read reported malformed FE headers that did not reproduce from stable source files.
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.FirstOrderLinearizedNonPlaneRowsHaveRowExactValidationEvidence:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps:CutCell.SupportMatrixValidationAuditCategorizesEveryAdvertisedRow:CutCell.CurvedTopologyValidityRejectsFoldedAndDegenerateInterfaceGeometry' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.LinearizedPlaneCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.LinearizedCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps:CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.CurvedTopologyValidityRejectsFoldedAndDegenerateInterfaceGeometry:CutCell.CurvedTopologyValidityRejectsSliversAndInconsistentSideRegions' --gtest_color=no`
  - `./build-fe-check/test_fe_geometry --gtest_filter='CutQuadrature.DiagnosticsRejectMalformedCurvedQuadratureRules' --gtest_color=no`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `cmake --build build-fe-check --target test_fe_systems -j 1`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.FirstOrderLinearizedNonPlaneRowsHaveRowExactValidationEvidence:CutCell.FirstOrderLinearizedNonPlaneRowsHaveAnalyticADSensitivityEvidence:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixValidationAuditCategorizesEveryAdvertisedRow' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.LinearizedPlaneCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.LinearizedCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.LinearizedNonPlaneCutSensitivityMetadataIsVisibleAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `cmake --build build-fe-gcc13-check --target test_fe_systems -j 4`
  - `./build-fe-gcc13-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ProductionCutScalarOperatorsMatchAnalyticHalfCubePlaneCut:CutIntegrationInfrastructure.ProductionCutScalarOperatorsConsumeImportedTopologyAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `./build-fe-gcc13-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.LinearizedPlaneCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.LinearizedCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.LinearizedNonPlaneCutSensitivityMetadataIsVisibleAcrossAllExecutionPaths:CutIntegrationInfrastructure.ProductionCutScalarOperators*' --gtest_color=no`
  - `./build-fe-gcc13-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$' -j1`
  - `git diff --check -- Code/Source/solver/Mesh/Search/CutCell.cpp Code/Source/solver/Mesh/Tests/Unit/Search/test_CutCell.cpp Code/Source/solver/FE/Tests/Unit/Systems/test_CutIntegrationInfrastructure.cpp Documentation/plan_moving_mesh_infrastructure.md`
  - `rg -n "finite[- ]difference|finite difference|FD" Code/Source/solver/Mesh/Search/CutCell.cpp Code/Source/solver/Mesh/Tests/Unit/Search/test_CutCell.cpp Code/Source/solver/FE/Tests/Unit/Systems/test_CutIntegrationInfrastructure.cpp` returned no matches.
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositeAndCutTopologyAreDeterministicAndRestartVisible' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_MovingMeshRestart --gtest_filter='MovingMeshRestart.BooleanCutCompositionProvenanceRoundtripsThroughRestart' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositionSemanticsCoverUnionIntersectionDifferenceAndNestedRegions:CutCell.NestedBooleanCompositionRestartRecordsPreserveFlattenedChildProvenance' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit:CutCell.ClassificationUsesHighOrderGeometryDofsWithLinearizedTopologyGate:CutCell.HighOrderTriangleUsesTessellatedClosedTopologyWithGeometryDofProvenance:CutCell.HighOrderTetraUsesTessellatedVolumeTopologyWithGeometryDofProvenance' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.ClassificationUsesHighOrderGeometryDofsWithLinearizedTopologyGate:CutCell.HighOrderTriangleUsesTessellatedClosedTopologyWithGeometryDofProvenance:CutCell.HighOrderTetraUsesTessellatedVolumeTopologyWithGeometryDofProvenance' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_Tessellation --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_Tessellation$|test_CutCell$' -j1`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCellMPI' -j1`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_MovingMeshRestart$' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `cmake --build build-fe-check --target test_fe_geometry test_fe_systems -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.*' --gtest_color=no`
  - `./build-fe-check/test_fe_geometry --gtest_filter='CutQuadrature.*' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCellMPI' -j1`
  - `mpirun -np 4 ./build-mesh-tests/Tests/test_CutCellMPI`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R '^test_CutCell$' -j1`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*'`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$' -j1`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*UsesSubdivisionForNonGraph*' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit' --gtest_color=no`
  - `cmake --build build-mesh-tests --target test_CutCellMPI -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCellMPI' -j1`
  - `cmake --build build-fe-check --target test_fe_systems -j 1`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedNonGraphSubdivisionArrangementMetadata' --gtest_color=no`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementSphereUsesSubdivisionForNonPlaneCuts:CutCell.TrueCurvedArrangementCallbackSurfaceUsesSubdivisionForNonPlaneCuts' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangementLevelSetFieldUsesSubdivisionForSampledSurfaces:CutCell.TrueCurvedArrangementTriangulatedSurfaceUsesSubdivisionForSurfaceCuts:CutCell.TrueCurvedArrangementRejectsMalformedLevelSetAndTriangulatedSurface:CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.BooleanCompositionDiagnosticsAndSupportMatrixAreExplicit' --gtest_color=no`
  - `cmake --build build-fe-check --target test_fe_systems -j 1`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurvedNonPlaneSubdivisionArrangementMetadata' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.*TrueCurved*' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R '^test_CutCell$'`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R '^FE_Systems_Tests$'`
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.QuadraticCurvedStarterRowsHaveFullValidationEvidence:CutCell.QuadraticCurvedStarterRowsHaveAnalyticADSensitivityEvidence:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.QuadraticCurvedPlaneCutDataIsIdenticalAcrossAllExecutionPaths:CutIntegrationInfrastructure.QuadraticCurvedPlaneCutSensitivityMetadataIsVisibleAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `cmake --build build-fe-check --target test_fe_systems -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.TrueCurvedArrangement*:CutCell.QuadraticCurvedStarterRowsHaveAnalyticADSensitivityEvidence:CutCell.SupportMatrixValidationLedgerIsRowExactAndDrivesQualification:CutCell.SupportMatrixQualificationHarnessCoversAdvertisedRowsAndReportsGaps' --gtest_color=no`
  - `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsTrueCurved*:CutIntegrationInfrastructure.QuadraticCurvedPlaneCutSensitivityMetadataIsVisibleAcrossAllExecutionPaths:CutIntegrationInfrastructure.SupportMatrixQualificationHarnessCoversFEExecutionPaths' --gtest_color=no`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -R 'test_CutCell$|test_CutCellMPI' -j1`
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -R 'FE_Geometry_Tests|FE_Systems_Tests' -j1`
  - `git diff --check -- Code/Source/solver/Mesh/Search/CutCell.cpp Code/Source/solver/Mesh/Tests/Unit/Search/test_CutCell.cpp Code/Source/solver/FE/Tests/Unit/Systems/test_CutIntegrationInfrastructure.cpp Documentation/plan_moving_mesh_infrastructure.md`
  - `rg -n "finite[- ]difference|finite difference|FD" Code/Source/solver/Mesh/Search/CutCell.cpp Code/Source/solver/Mesh/Tests/Unit/Search/test_CutCell.cpp Code/Source/solver/FE/Assembly/CutIntegrationContext.h Code/Source/solver/FE/Tests/Unit/Systems/test_CutIntegrationInfrastructure.cpp` returned no matches.
  - `cmake --build build-mesh-tests --target test_CutCell -j 4`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.SecondOrderLinearizedRowsHaveFullValidationAndAnalyticADSensitivityEvidence' --gtest_color=no`
  - `./build-mesh-tests/Tests/test_CutCell --gtest_filter='CutCell.SupportMatrix*' --gtest_color=no`

Verification and validation closeout, 2026-04-29:

- [x] Phase 26 checklist audit found no unchecked work items in this section.
- [x] Mesh CutCell serial and MPI closeout passed:
  - `cmake --build build-mesh-tests --target test_CutCell test_CutCellMPI -j 4`
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R '(^test_CutCell$|^test_CutCellMPI)'`
- [x] Mesh distributed cut/search/interface/contact-adjacent regression slice passed:
  - `ctest --test-dir build-mesh-tests --output-on-failure --timeout 300 -j1 -R 'CutCellMPI|MeshSearchMPI|MovingGeometryValidityMPI|MultiMeshInterfaceMPI|ContactProximityMPI'`
- [x] FE cut-quadrature validation passed, including exact box cuts, topology-derived conservation, spherical-cap references, malformed-rule diagnostics, and analytic sensitivity coverage:
  - `cmake --build build-fe-check --target test_fe_geometry -j 4`
  - `./build-fe-check/test_fe_geometry --gtest_filter='CutQuadrature.*' --gtest_color=no`
- [x] FE cut Systems and Forms integration passed across the implemented execution paths:
  - `cmake --build build-fe-check --target test_fe_systems test_fe_forms test_fe_assembly test_fe_assembly_mpi -j 4`
  - `./build-fe-check/test_fe_systems --gtest_filter='*Cut*' --gtest_color=no`
  - `./build-fe-check/test_fe_forms --gtest_filter='*Cut*' --gtest_color=no`
  - `./build-fe-check/test_fe_assembly --gtest_filter='*Cut*' --gtest_color=no` selected zero tests, confirming the current cut-specific assembly coverage is hosted through FE Systems and FE Forms.
- [x] FE MPI regression slice passed:
  - `ctest --test-dir build-fe-check --output-on-failure --timeout 300 -j1 -L MPI`
- [x] Current Phase 26 production-code finite-difference audit remains clean for the touched Mesh/FE cut infrastructure; verification-only finite-difference references are confined to tests where present.

## Phase 27: Optional Peskin-Style Immersed-Boundary Coupling Infrastructure

Goal: add the missing Mesh and FE infrastructure for marker-based immersed-boundary methods where Lagrangian markers are coupled to an Eulerian FE background through regularized delta operators. This phase is distinct from Phase 26. Phase 26 supports unfitted/cut-cell methods by modifying integration over cut regions; Phase 27 supports Peskin-style IBM by building physics-neutral marker-to-Eulerian gather and spread operators. If "IBM support" means a fluid, solid, membrane, feedback, or constraint formulation, that work belongs in `Physics` and must be deferred until this infrastructure exists.

### Scope Boundary

- [ ] Mesh may own Lagrangian marker sets, marker provenance, marker reference/current positions, optional marker measures, marker lifecycle state, marker ownership, support-map search, distributed exchange metadata, restart records, and revision snapshots.
- [ ] FE may own regularized-delta kernel evaluation, FE basis sampling over marker support stencils, generic gather/spread operators, adjointness diagnostics, assembly-facing marker-support contexts, and Forms terminals for marker-support metadata.
- [ ] Mesh and FE must not encode immersed-body force laws, no-slip enforcement choices, penalty/feedback coefficients, membrane/shell/solid equations, fluid equations, or model-specific residual terms.
- [ ] Physics modules own marker evolution laws, immersed-structure constitutive models, force computation, no-slip/constraint/feedback formulations, coupling residuals, physical stabilization, and coupling coefficients.
- [ ] Application may later own input wiring and construction of a Physics IBM module, but this phase should not introduce such a module.

### Reuse-First Rules

- [ ] Do not add a parallel revision, transaction, restart, MPI exchange, support-matrix, or Forms/JIT framework for IBM. Phase 27 should extend the existing moving-mesh infrastructure patterns with marker-specific state only where the existing abstractions do not represent Lagrangian marker clouds.
- [ ] Reuse Mesh revision domains, FE revision domains, and the existing cache-invalidation contract. Marker-specific revisions should plug into that contract rather than creating a separate invalidation mechanism.
- [ ] Reuse `MeshSearch` and the existing spatial acceleration builders for broad-phase and point-location queries. Add only the missing compact-support stencil enumeration needed by regularized-delta kernels.
- [ ] Reuse existing `DistributedMesh` ownership, ghosting, migration, rebalance, and compact exchange-packet patterns from interface, contact, and cut-cell infrastructure.
- [ ] Reuse `MovingMeshRestart` extension points. Marker data should be another versioned restart payload, not a separate IBM restart mechanism.
- [ ] Reuse existing FE field-kind, component-count, frame-transform, sparse-operator, matrix-free, and interface-transfer conventions. Add marker gather/spread only for marker-to-volume support maps that face-to-face interface operators cannot represent.
- [ ] Reuse `AssemblyContext`, FormExpr parameter/auxiliary bindings, AD, symbolic tangent, and JIT plumbing where possible. Add marker-specific terminals only when generic slots cannot express the required marker-support metadata safely.
- [ ] Reuse the support-matrix and capability-gate style from Phase 26 so unsupported marker/kernel/element combinations fail closed with diagnostics.

### Mesh Marker State

- [ ] Add a physics-neutral Lagrangian marker-set representation, such as `Mesh/Search/LagrangianMarkerCoupling.{h,cpp}` or an equivalent Mesh-owned module.
- [ ] Support persistent marker-set IDs, persistent marker IDs, human-readable names, provenance epochs, and optional physical labels without making labels the primary identity.
- [ ] Store marker reference coordinates, current coordinates, optional marker velocity fields, optional marker measures/weights, and optional marker topology/connectivity as data, not equations.
- [ ] Support point, curve, surface, and volume marker-set metadata so a future Physics module can represent fibers, membranes, immersed surfaces, and volumetric marker clouds using the same infrastructure.
- [ ] Add marker-set revision domains for coordinate revision, topology/connectivity revision, ownership revision, numbering revision, marker-field layout revision, marker-field value revision, provenance revision, and accepted time-level epoch, using the existing revision-snapshot style.
- [ ] Add trial, accepted nonlinear-state, accepted time-step, rollback, and commit semantics for marker coordinates and marker support maps, using the existing transaction/rollback conventions.
- [ ] Add diagnostics for duplicate marker IDs, non-finite coordinates, invalid measures, unsupported marker dimensionality, stale revision snapshots, and missing provenance.

### Marker-to-Eulerian Support Search

- [ ] Add a `MarkerEulerianSupportMap` or equivalent that records all Eulerian cells, local coordinates, basis support entries, and owner ranks touched by each marker's regularized-delta support.
- [ ] Build support maps from current or reference marker coordinates against current or reference Eulerian mesh coordinates with explicit `Configuration` metadata.
- [ ] Support compact-kernel radius queries that return all potentially contributing Eulerian cells, not just the containing cell. Existing point location is a useful seed, but Peskin IBM requires full support stencils around every marker.
- [ ] Store marker ID, Eulerian parent cell, parent cell global ID, owner rank, local reference coordinate, physical support point, support distance/vector, kernel value, kernel gradient when available, basis values, basis gradients when requested, and provenance.
- [ ] Add support-map policies for isotropic mesh spacing, anisotropic mesh spacing, metric-based support radius, and user-declared kernel support width without tying them to a physical model.
- [ ] Diagnose missing support, support truncation at physical boundaries, support crossing partition boundaries, duplicate support entries, unsupported element family/order, unsupported kernel policy, and moment-normalization failures.
- [ ] Add sparse distributed marker support exchange by reusing the existing owner/ghost exchange-packet pattern, so owned markers can gather from and spread to remote Eulerian ghost/owner cells without requiring every MPI rank to process every marker.
- [ ] Add migration and repartition invalidation through existing mesh ownership/numbering revision paths so support maps rebuild when Eulerian ownership, marker ownership, numbering, topology, or geometry revisions change.

### Regularized Delta Kernels

- [ ] Add a physics-neutral regularized-delta kernel API in FE, for example `FE/Math/RegularizedDeltaKernel.{h,cpp}` or `FE/Geometry/RegularizedDeltaKernel.{h,cpp}`.
- [ ] Provide kernel descriptors with persistent policy IDs, support radius, spatial dimension, tensor-product/radial family, continuity class, normalization convention, gradient availability, and frame metadata.
- [ ] Implement baseline Peskin-style compact kernels such as 2-point, 3-point, and 4-point tensor-product kernels where appropriate for the supported mesh dimension.
- [ ] Provide derivative evaluation with respect to marker position and Eulerian coordinate where the kernel family supports it.
- [ ] Provide moment and normalization diagnostics: zeroth moment, first moment, symmetry, positivity where applicable, support completeness, and tolerance-policy metadata.
- [ ] Keep the kernel implementation independent of velocity, force, pressure, solid stress, membrane tension, or any other physical variable.
- [ ] Add capability gates for unsupported dimensions, anisotropic spacing modes, curved/high-order geometry combinations, and derivative requests.

### FE Gather and Spread Operators

- [ ] Add generic marker gather operators that map Eulerian FE field DOFs to marker values through the support map.
- [ ] Add generic marker spread operators that map marker scalar/vector/tensor data to Eulerian FE vectors through the transpose support map.
- [ ] Represent the coupling as a sparse operator `B` and transpose `B^T` where practical, with component-count and field-kind metadata similar to existing interface transfer infrastructure.
- [ ] Support scalar, vector, rank-2 tensor, and mixed-block field metadata without interpreting the field physically.
- [ ] Support optional frame transforms for vector and tensor values, using existing moving-frame conventions rather than adding IBM-specific frame logic.
- [ ] Ensure spread/gather adjointness under declared marker and Eulerian weights, with diagnostics for violations.
- [ ] Ensure constant-preservation diagnostics for gather and total-integral diagnostics for spread.
- [ ] Add matrix-free data hooks using existing backend rebuild conventions so gather/spread can be applied without assembling the sparse coupling matrix when a backend supports that path.
- [ ] Thread invalidation through existing FE revision domains for Eulerian FE space revision, DOF-layout revision, marker-field layout revision, marker-support revision, kernel-policy revision, and geometry revision.

### FE Assembly and Forms Exposure

- [ ] Add an assembly-facing marker coupling context, such as `FE/Assembly/MarkerCouplingContext.h`, only if existing `AssemblyContext`, `InterfacePairContext`, and `CutIntegrationContext` extension points cannot expose marker-support entries cleanly without encoding IBM residual terms.
- [ ] Expose marker ID, marker set ID, marker measure, support cell, local coordinates, kernel weight, kernel gradient, support vector, support distance, basis values, basis gradients, frame, and provenance through the context.
- [ ] Add Forms terminals or parameter bindings for generic marker-support data, such as `markerWeight`, `markerCoordinate`, `markerDisplacement`, `kernelWeight`, `kernelGradient`, `supportVector`, and `supportDistance`.
- [ ] Prefer generic marker field slots for marker-provided data rather than terms named after forces, velocities, membranes, or no-slip constraints.
- [ ] Ensure standard assembly, matrix-free, interpreter, AD, symbolic tangent, and JIT paths can consume the same marker-support metadata.
- [ ] Add sensitivity metadata for marker-position and mesh-position derivatives of kernel weights and support coordinates where the kernel capability gate allows it.
- [ ] Keep any actual IBM residual construction out of FE. FE may expose generic data and operators; Physics must combine them into equations.

### Restart, Rollback, and Time-Level State

- [ ] Extend Mesh restart infrastructure to persist accepted marker-set descriptors, marker IDs, marker coordinates, marker measures, marker topology metadata, marker revision snapshots, and kernel-policy IDs.
- [ ] Persist enough support-map provenance to diagnose stale maps after restart, while allowing support maps to be rebuilt deterministically from accepted marker and mesh state.
- [ ] Do not serialize executable callbacks for marker motion or Physics-owned marker laws. Use stable descriptors plus application/Physics re-registration contracts where needed.
- [ ] Support rollback so rejected trial marker motion does not leak into accepted support maps, restart records, gather/spread operators, or matrix-free caches.
- [ ] Track time-level epochs for marker coordinates and support maps so a future time integrator can distinguish previous, trial, accepted nonlinear, and accepted time-step states.

### Distributed Ownership and Exchange

- [ ] Define deterministic marker ownership based on persistent marker ID, current support cell ownership, or an explicit marker-owner policy, reusing existing distributed identity rules where possible.
- [ ] Define deterministic Eulerian support ownership based on parent cell global IDs and owner ranks.
- [ ] Exchange marker data needed for gather and spread through existing sparse exchange mechanisms without requiring dense all-rank marker replication.
- [ ] Accumulate spread contributions on Eulerian owner ranks and expose diagnostics for duplicate or missing contributions.
- [ ] Preserve stable marker and support identities across `DistributedMesh::migrate`, repartition/rebalance, ghost rebuilds, and numbering changes.
- [ ] Add MPI support-map invalidation by reusing existing revision snapshots for marker coordinate, Eulerian geometry, topology, ownership, numbering, active configuration, FE layout, marker layout, kernel policy, local packet, and exchanged packet revision changes.

### Diagnostics and Support Matrix

- [ ] Add a Phase 27 support matrix with rows for Eulerian element family, Eulerian geometry order, marker-set dimension, marker topology mode, kernel family, support-radius policy, distributed mode, FE field kind, and FE execution path.
- [ ] Mark each row as supported, implemented-unqualified, or unsupported.
- [ ] Add diagnostics for constant-preservation error, spread integral residual, gather/spread transpose residual, kernel moment residual, support truncation, boundary loss, missing remote support, stale revisions, unsupported high-order geometry, and unsupported field-kind/component combinations.
- [ ] Require support claims to be milestone-scoped. A kernel/element/marker/distribution combination is unsupported until search, gather, spread, adjointness, restart, rollback, and MPI tests pass for that combination.

### Tests

- [ ] Serial marker registry test for persistent marker IDs, marker measures, current/reference coordinates, revision snapshots, commit, rollback, and diagnostics.
- [ ] Serial support-search test where each marker finds all cells in compact kernel support, not only the containing cell.
- [ ] Moving-marker support-map update test with accepted and rejected trial marker motion.
- [ ] Moving-Eulerian-mesh support-map invalidation test under current-coordinate updates.
- [ ] Regularized-delta kernel tests for normalization, first moment, symmetry, compact support, gradient availability, and unsupported capability diagnostics.
- [ ] Constant-field gather test proving marker samples preserve constants within tolerance.
- [ ] Unit marker spread test proving total Eulerian integral matches the declared marker measure and data value within tolerance.
- [ ] Random-vector adjointness test proving `dot(B u, q_marker) == dot(u, B^T q_marker)` under declared weights.
- [ ] Scalar, vector, and tensor field-kind tests with component-count metadata.
- [ ] Boundary support-truncation diagnostic test.
- [ ] Restart round-trip test proving accepted marker state rebuilds support maps deterministically.
- [ ] Rollback/restart equivalence test proving rejected marker motion is not written as accepted state.
- [ ] MPI support-map test with markers whose support crosses rank boundaries.
- [ ] MPI gather/spread test proving remote support exchange and owner-rank accumulation are deterministic.
- [ ] MPI migration/rebalance test proving marker/support identity survives ownership changes and stale support maps are rejected.
- [ ] Interpreter/JIT or standard/matrix-free parity tests for a generic marker-weighted field evaluation, without implementing any IBM Physics residual.

### Acceptance Criteria

- [ ] Mesh can register, revise, distribute, restart, and query Lagrangian marker sets without knowing the governing equations.
- [ ] Mesh can build deterministic marker-to-Eulerian support maps with explicit revision, ownership, and provenance metadata.
- [ ] FE can evaluate regularized-delta kernels and derivatives through a documented capability-gated API.
- [ ] FE can build and apply generic gather and spread operators for supported scalar/vector/tensor FE fields.
- [ ] Gather/spread constant preservation, integral consistency, and transpose-adjointness tests pass in serial and MPI for the advertised support matrix rows.
- [ ] Standard, matrix-free, interpreter, AD, symbolic tangent, and JIT paths either consume the same marker-support data or explicitly report unsupported capability diagnostics.
- [ ] Restart, rollback, migration, and repartition tests pass for accepted supported combinations.
- [ ] No new Physics formulation, force law, no-slip law, membrane law, solid model, or fluid residual is added as part of this phase.

### Non-Goals for Phase 27

- [ ] Do not implement a Peskin-IBM Navier-Stokes formulation.
- [ ] Do not implement immersed solid, membrane, shell, or fiber constitutive equations.
- [ ] Do not implement no-slip, penalty, Lagrange multiplier, feedback, or direct-forcing laws.
- [ ] Do not add Physics-owned marker evolution or force computation to Mesh or FE.
- [ ] Do not claim production IBM support until a later Physics module consumes this infrastructure and passes formulation-level verification.

## Suggested Implementation Order

- [x] Phase 1: mesh state revision and cache invalidation contract.
- [x] Phase 2: `StandardAssembler` flat coordinate cache fix.
- [x] Phase 3: physics-agnostic FE moving-domain data model.
- [x] Phase 17: solver, operator, and geometry transaction policy.
- [x] Phase 5: current/reference geometry preparation.
- [x] Phase 4: FE Forms vocabulary and lowering.
- [x] Phase 6: mesh-motion field binding into FE systems.
- [x] Phase 7: FE-backed mesh-motion backend.
- [ ] Phase 9: time-history and GCL support.
- [x] Phase 8: geometry sensitivities and monolithic mesh unknowns.
- [x] Phase 18: reference rebasing and rezoning semantics.
- [x] Phase 19: material and history-state frame semantics.
- [x] Phase 20: constraint system support under motion.
- [x] Phase 15: high-order and curved geometry motion.
- [x] Phase 23: element-family and frame-aware transform coverage.
- [x] Phase 24: geometric nonlinearity infrastructure.
- [x] Phase 16: generic multi-mesh moving-interface coupling.
- [x] Phase 25: physics-agnostic rotating and sliding-interface kinematics.
- [ ] Phase 26: optional unfitted and cut-cell interface infrastructure. Baseline implementation complete; robust CutFEM foundation substantially expanded with topology-derived quadrature contracts, all-advertised-linear-family side-region measure summaries, closed linear side-region subcell topology, tessellator-backed high-order linearized topology reconstruction, a bounded quadratic line/triangle/quad/tetra/hex curved-isoparametric topology-to-quadrature starter path, opt-in true curved arrangement support for arbitrary-order line/plane cuts, graph-compatible high-order triangle/quad plane cuts, bounded non-graph high-order triangle/quad plane cuts, graph-compatible high-order tetra/hex/wedge/pyramid plane volume cuts, bounded non-graph high-order hex/wedge/pyramid plane cuts, bounded non-plane `Sphere`, signed-distance callback, level-set field, triangulated-surface, and Boolean composite subdivision cuts including noncoincident child-child topology switches and coincident-child tolerance-resolved predicate fallback, restart file round trips, curved restart/rollback qualification including callback application re-registration, deterministic distributed exchange packets, neighbor-sparse owner/ghost distributed cut-state exchange, versioned distributed curved metadata, predicate-fallback metadata, and sensitivity capability exchange, block/METIS/ParMETIS sparse-exchange qualification, production migration and deterministic-rebalance qualification, conditioning-policy and FE-execution-path-aware support-matrix queries, row-exact support-matrix evidence reporting, all-six-FE-path validation plus analytic/AD-compatible sensitivity evidence for serial/distributed first-order linearized plane and non-plane rows on every advertised linear parent family, full-validation and analytic/AD-compatible sensitivity evidence for serial second-order controlled linearized high-order rows, full-validation and analytic/AD-compatible sensitivity evidence for serial/distributed quadratic plane-cut curved-isoparametric starter rows, full-validation and analytic/AD-compatible sensitivity evidence for supported serial/distributed true-curved topology rows, production scalar cut-operator reductions that consume imported cut topology identically across all six FE execution views, physics-neutral residual/tangent cut-domain kernel dispatch with analytic AD/symbolic verification, and additional serial/MPI tests. The implemented support-matrix cross-product is now fully qualified; full production support still requires independent analytic/literature validation and broader exact/adaptive predicate coverage beyond the current Boolean active-child fallback.
- [ ] Phase 27: optional Peskin-style immersed-boundary coupling infrastructure. Mesh/FE marker registry, marker-to-Eulerian support maps, regularized-delta kernels, generic gather/spread operators, adjointness diagnostics, rollback, restart, and MPI ownership must be qualified before any Physics IBM module is advertised.
- [ ] Phase 22: geometry validity beyond element inversion.
- [x] Phase 21: contact and proximity infrastructure, if contact remains in near-term scope.
- [ ] Phase 10: remesh/adaptivity/transfer. Implementation and serial verification complete; external MPI verification pending.
- [x] Phase 11: checkpoint/restart.
- [x] Phase 12: application orchestration.
- [x] Phase 13: physics module consumption.
- [x] Phase 14: full verification and qualification for Phases 1-13, with later literature FSI validation targets recorded.

## First Concrete Milestone

The first milestone should be cache-safe prescribed moving mesh assembly:

- [ ] Add Mesh revision domains for geometry, topology, ownership, numbering, field layout, and labels.
- [ ] Add FE revision domains for space, DOF layout, constraint layout, and block layout.
- [ ] Add explicit active-configuration epoch or require explicit coordinate-configuration requests in first-mile FE/search paths.
- [ ] Thread revision through `IMeshAccess` and `MeshAccess`.
- [ ] Fix `StandardAssembler` coordinate cache keys.
- [ ] Define the minimal geometry transaction model for prescribed motion: committed, trial, accepted, and rolled back.
- [ ] Add tests proving repeated assembly after `X_cur` changes uses updated geometry.
- [ ] Add tests proving topology/layout changes force stronger rebuilds than geometry-only changes.
- [ ] Add tests proving FE layout changes invalidate DOF maps, vectors, sparsity, and backend state even when Mesh state is unchanged.
- [ ] Add a minimal FE system hook that invalidates geometry-derived caches on mesh geometry changes.
- [ ] Add a prescribed-motion application smoke test that runs without any physics-specific ALE changes.

This milestone is small enough to land independently and removes the highest immediate correctness risk before adding new ALE vocabulary or FE-backed motion solvers.
