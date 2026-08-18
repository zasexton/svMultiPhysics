# New OOP Solver Multi-Mesh Support Plan

## Goal

Enable the new OOP solver path to handle more than one `<Add_mesh>` block so
multi-participant FE problems can be represented by named meshes, participant
scoped fields, interface maps, and coupled forms. The immediate target is a
two-mesh FSI problem, but the infrastructure should support `n > 1`
participants without hard-coding fluid/solid assumptions.

## Current Limitations

- Multiple meshes are loaded into `SimulationComponents::meshes`, but
  `SimulationBuilder::createFESystem()` builds the FE system from only the first
  mesh.
- `EquationTranslator::buildInput()` rejects `meshes.size() != 1`.
- Physics modules currently receive one mesh and generally assume fields live
  on that mesh.
- Interface projections are parsed but not translated into new-path interface
  maps.
- New-path output writes against the primary mesh model.

## Implementation Order

1. Build `MeshCollection` and `CompositeMeshAccess`.
2. Add participant-scoped FE fields and DOF distribution.
3. Update equation translation to pass participant context.
4. Translate projections/interfaces.
5. Register and wire FSI through participant-scoped fields and interface forms.
6. Extend sparsity/backend support and output.
7. Add staged unit and integration tests.

## 1. Mesh Collection

### Concrete Changes

- Add `Application/Core/MeshCollection.h`.
- Add `Application/Core/MeshCollection.cpp`.
- Define a participant record with:
  - mesh name
  - optional domain id
  - loaded `svmp::Mesh`
  - face-name to label map
  - source `MeshParameters` metadata needed by translators
- Add lookup helpers:
  - `participantByName(name)`
  - `participantByDomain(domain_id)`
  - `participantOwningFace(face_name)`
  - `participants()`
- Add validation for duplicate names, duplicate domain ids, duplicate global
  face names, missing mesh pointers, and missing domain ids when multi-domain
  equation translation needs them.
- Update `SimulationComponents` to own a `MeshCollection` while retaining
  compatibility accessors for existing single-mesh code.

### Checklist

- [x] Add `MeshParticipant` data structure.
- [x] Add `MeshCollection` container and lookup APIs.
- [x] Convert `SimulationBuilder::loadMeshes()` to populate `MeshCollection`.
- [x] Preserve existing `components_.meshes` behavior until all callers migrate.
- [x] Add validation diagnostics with mesh names and XML element context.
- [x] Add unit tests for duplicate mesh names.
- [x] Add unit tests for duplicate domain ids.
- [x] Add unit tests for duplicate face names across participants.
- [x] Add unit tests for domain-id lookup.

## 2. Composite Mesh Access

### Concrete Changes

- Add `FE/Assembly/CompositeMeshAccess.h`.
- Add `FE/Assembly/CompositeMeshAccess.cpp`.
- Implement `assembly::IMeshAccess` by delegating to participant-local
  `MeshAccess` adapters.
- Assign global offsets for:
  - vertices
  - cells
  - boundary faces
  - interior faces
- Maintain reverse maps:
  - global cell id to participant/local cell id
  - global vertex id to participant/local vertex id
  - global face id to participant/local face id
- Preserve participant identity for domain filters, diagnostics, output, and
  coupling.
- Make boundary markers globally unambiguous. The first implementation can
  allocate globally unique marker ids and keep a map from `(participant,
  local_marker)` to global marker.

### Checklist

- [x] Implement offset tables for all participants.
- [x] Implement `numCells()`, `numOwnedCells()`, `numVertices()`,
      `numOwnedVertices()`, `numBoundaryFaces()`, and `numInteriorFaces()`.
- [x] Implement cell traversal and owned-cell traversal.
- [x] Implement `getCellNodes()` with vertex id remapping.
- [x] Implement `getNodeCoordinates()` with participant-local delegation.
- [x] Implement `getCellType()` and geometry order delegation.
- [x] Implement `getCellDomainId()`.
- [x] Implement boundary-face traversal with global marker ids.
- [x] Implement interior-face traversal for each participant.
- [x] Add participant lookup helpers for global ids.
- [x] Add tests for two disconnected tetrahedral meshes.
- [x] Add tests for boundary label remapping.
- [x] Add tests for coordinate and connectivity consistency.

## 3. FE System Construction

### Concrete Changes

- Add an `FESystem` construction path that accepts a shared
  `assembly::IMeshAccess` plus optional participant metadata.
- In `SimulationBuilder::createFESystem()`, use:
  - existing single-mesh constructor when one mesh is loaded
  - `CompositeMeshAccess` when more than one mesh is loaded
- Add accessors on `FESystem` for participant metadata where needed by modules,
  output, and coupling diagnostics.
- Audit direct `FESystem::mesh()` usage and either keep it single-mesh only or
  replace it with participant-aware access.

### Checklist

- [x] Add participant metadata storage to `FESystem`.
- [x] Add `FESystem` constructor for composite mesh access.
- [x] Update `SimulationBuilder::createFESystem()`.
- [x] Preserve current single-mesh path.
- [x] Audit all `FESystem::mesh()` callers.
- [x] Add clear errors for APIs that remain single-mesh only.
- [x] Add unit tests for composite-system setup with no physics modules.
- [x] Add regression tests for existing single-mesh setup.

## 4. Participant-Scoped Fields

### Concrete Changes

- Extend `FE::systems::FieldSpec` with optional participant scope:
  - `participant_name`
  - or a domain/region selector that resolves to one participant
- Extend field registry and DOF distribution so a field can allocate DOFs only
  on cells/nodes belonging to its participant.
- Ensure field access plans reject cells outside the field scope.
- Make boundary condition translation validate that the target face belongs to
  the same participant as the field being constrained.

### Checklist

- [x] Add participant scope to `FieldSpec`.
- [x] Extend `FieldRegistry` storage for participant scope.
- [x] Update DOF distribution to skip cells outside field scope.
- [x] Update field DOF maps to support sparse participant coverage.
- [x] Update assembler field-access plans for scoped fields.
- [x] Add validation for field/face participant mismatch.
- [x] Add tests for a fluid field on `lumen` only.
- [x] Add tests for a solid field on `wall` only.
- [x] Add tests that no DOFs are created on unrelated participants.
- [x] Add tests for assembly over two uncoupled participants.

## 5. Equation Translation

### Concrete Changes

- Remove the `meshes.size() != 1` rejection in
  `EquationTranslator::buildInput()`.
- Extend `EquationModuleInput` with:
  - mesh collection reference or snapshot
  - participant name for each translated domain
  - domain id to participant lookup
  - face name to participant lookup
- Resolve each `<Domain id="...">` against `<Add_mesh><Domain>`.
- Preserve existing behavior for single-mesh equations with no explicit domain
  mapping.
- Update module factories so they create participant-scoped fields.

### Checklist

- [ ] Extend `EquationModuleInput`.
- [ ] Update `EquationTranslator::buildInput()`.
- [ ] Resolve domain blocks to participant names.
- [ ] Resolve boundary conditions to participant faces.
- [ ] Keep single-mesh behavior unchanged.
- [ ] Update Poisson translation.
- [ ] Update Navier-Stokes translation.
- [ ] Update mesh-motion translation when present in the new path.
- [ ] Add tests for two separate equations on two different meshes.
- [ ] Add tests for invalid domain-to-mesh references.

## 6. Projection And Interface Maps

### Concrete Changes

- Translate `<Add_projection>` blocks into interface-map declarations.
- Resolve source and target faces to `(participant, face_name, marker)`.
- For conforming faces, build an exact vertex/face correspondence from
  coordinates and orientation.
- Store interface map metadata in a solver-visible registry.
- Reject nonmatching surfaces initially with diagnostics that include face names,
  participant names, and mismatch counts.
- Later, add nonconforming projection support through the search/projection
  infrastructure.

### Checklist

- [ ] Add projection translation entry point.
- [ ] Resolve projection source and target participants.
- [ ] Build conforming interface map.
- [ ] Compute and store side orientation.
- [ ] Validate matching vertex coordinates.
- [ ] Validate matching triangular or quadrilateral face partitions.
- [ ] Store map in coupling/interface registry.
- [ ] Add tests for elastic-pipe `lumen_wall` to `wall_inner`.
- [ ] Add tests for mismatched surfaces.
- [ ] Add tests for reversed face orientation.

## 7. FSI New-Path Module

### Concrete Changes

- Register an `FSI` factory in `EquationModuleRegistry`.
- Translate legacy FSI domain blocks into participant roles:
  - fluid participant
  - solid participant
  - interface region
  - mesh-motion participant/field
- Create participant-scoped FE fields:
  - fluid velocity
  - fluid pressure
  - solid displacement
  - mesh displacement
- Register monolithic interface coupling forms through the coupling
  infrastructure.
- Translate fluid and solid boundary conditions onto participant-local faces.
- Keep the legacy solver path unchanged.

### Checklist

- [ ] Add FSI registration file or extend existing FSI module registration.
- [ ] Add XML-to-FSI-options translator.
- [ ] Resolve fluid and solid participants from domain ids.
- [ ] Resolve interface from projection metadata.
- [ ] Register participant-scoped fluid fields.
- [ ] Register participant-scoped solid fields.
- [ ] Register participant-scoped mesh-motion field.
- [ ] Register FSI interface forms.
- [ ] Translate fluid Neumann/Dirichlet BCs.
- [ ] Translate solid constraints.
- [ ] Add tests that FSI module registers fields/forms without solving.
- [ ] Add tests for missing projection/interface diagnostics.

## 8. Sparsity And Backend Support

### Concrete Changes

- Ensure sparsity building includes:
  - intra-participant element couplings
  - interface couplings across participants
  - mesh-motion to fluid geometry sensitivity couplings
- Validate Eigen backend first because it is simpler for mixed layouts.
- Audit FSILS block assumptions:
  - uniform DOF count per node
  - node-interleaved layout
  - participant-local fields with missing DOFs on unrelated nodes
- Add a mixed/block layout path if FSILS cannot represent sparse participant
  field coverage directly.

### Checklist

- [ ] Add sparsity tests for cross-participant interface blocks.
- [ ] Add sparsity tests for fields that do not exist on every participant.
- [ ] Verify Eigen backend assembly and solve for a two-mesh toy problem.
- [ ] Audit FSILS point-block assumptions.
- [ ] Add FSILS diagnostics for unsupported multi-mesh layouts.
- [ ] Add FSILS support for participant-scoped mixed layouts or block wrappers.
- [ ] Qualify serial FSILS.
- [ ] Qualify MPI FSILS after serial is stable.

## 9. Multi-Mesh Output

### Concrete Changes

- Extend output to write one VTU per participant per time step.
- Suggested names:
  - `result_lumen_001.vtu`
  - `result_wall_001.vtu`
- Slice global state vectors into participant-local field arrays.
- Write only fields that are defined on each participant.
- Preserve restart state as one global state with participant metadata.
- Add PVD collection support for multiple participant files per step if needed.

### Checklist

- [ ] Add participant-aware output loop.
- [ ] Add field slicing from global state to participant mesh.
- [ ] Skip fields not defined on a participant.
- [ ] Preserve current single-mesh output naming.
- [ ] Add multi-file PVD output if needed.
- [ ] Add restart metadata for participant-scoped fields.
- [ ] Add output tests for two-mesh field slicing.
- [ ] Add output tests for elastic-pipe lumen/wall results.

## 10. Tests And Qualification

### Concrete Changes

- Add staged tests so multi-mesh support can be qualified before full FSI is
  enabled.
- Use small synthetic meshes for unit tests and the elastic-pipe fixture for
  integration tests.
- Keep existing single-mesh tests unchanged and run them as regression coverage.

### Checklist

- [ ] Unit test `MeshCollection` parsing and validation.
- [ ] Unit test `CompositeMeshAccess` traversal.
- [ ] Unit test participant-scoped DOF allocation.
- [ ] Unit test two uncoupled equations on two meshes.
- [ ] Unit test projection/interface map validation.
- [ ] Integration test coarse elastic-pipe meshes with
      `<Use_new_OOP_solver>true</Use_new_OOP_solver>` through setup.
- [ ] Integration test FSI module field/form registration.
- [ ] Smoke test one serial elastic-pipe FSI time step.
- [ ] Regression test existing single-mesh OOP fluid cases.
- [ ] Regression test existing single-mesh OOP Poisson/Darcy cases.
- [ ] MPI qualification after serial tests are stable.

## First Patch Recommendation

Start with `MeshCollection` and `CompositeMeshAccess` plus focused unit tests.
This removes the primary-mesh-only structural assumption without touching FSI
math or solver behavior. Once the FE system can traverse multiple named meshes
as one disjoint computational domain, participant-scoped fields and interface
coupling can be added incrementally.
