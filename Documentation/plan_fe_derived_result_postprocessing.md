# FE Derived Result Postprocessing Infrastructure

## Goal

Add physics-agnostic FE infrastructure for appending derived result fields to solver output files after a solve step. Physics modules should be able to register derived results using the existing `FE::forms` vocabulary, while the FE library owns expression validation, entity-scoped evaluation, mesh-field attachment, and output integration.

The motivating Darcy example is:

```text
Darcy_flux = -K * grad(Pressure)
```

For an H1 pressure formulation, this derived flux should initially be written as cell data because `grad(Pressure)` is generally discontinuous across element boundaries.

## Non-Goals

- Do not add XML syntax for user-defined postprocessing expressions in the initial implementation.
- Do not add Darcy-specific logic to the FE library.
- Do not couple FE postprocessing to VTK-specific concepts beyond writing through existing mesh fields.
- Do not require postprocessing definitions to be scalar only.
- Do not force all derived fields into nodal output when their natural scope is cell, face, or quadrature based.

## Desired Developer Model

Physics modules define postprocessing in formulation-local headers:

```cpp
// Physics/Formulations/Poisson/PoissonPostProcessing.h

inline void registerDarcyPostProcessing(
    FE::systems::FESystem& system,
    FE::FieldId pressure_id,
    const FE::spaces::FunctionSpace& pressure_space,
    const PoissonOptions& options)
{
    using namespace svmp::FE::forms;
    using namespace svmp::FE::post;

    auto p = StateField(pressure_id, pressure_space, options.field_name);
    auto K = FormExpr::constant(options.diffusion);

    system.addDerivedResult(
        DerivedResultBuilder("Darcy_flux")
            .scope(DerivedResultScope::Cell)
            .shape(FE::systems::FEQuantityShape::vector(pressure_space.spatialDimension()))
            .policy(DerivedResultPolicy::CellAverage)
            .expression(-K * grad(p))
            .build());
}
```

The physics module calls this from `registerOn()` after registering its primary fields. The application output path remains generic:

```cpp
writePrimarySolutionFields(...);
sim.fe_system->appendDerivedResultFields(mesh, state);
mesh.save_parallel(io);
```

## Architecture Overview

The implementation should be layered as follows:

```text
Physics formulation module
  registers FE::post::DerivedResultDefinition using FormExpr

FE::systems::FESystem
  stores definitions in a physics-agnostic registry

FE::post::DerivedResultEvaluator
  evaluates definitions against SystemStateView on requested mesh entities

Mesh field layer
  receives attached fields on Vertex, Edge, Face, or Volume entities

Application output path
  asks FESystem to append derived result fields before mesh.save_parallel()
```

## Scope and Output Semantics

The FE registry should support all intended scopes, even if some output sinks require follow-up writer support.

| Derived scope | Mesh association | Initial output behavior |
| --- | --- | --- |
| `Vertex` | `EntityKind::Vertex` | Write as VTK `PointData` |
| `Cell` | `EntityKind::Volume` | Write as VTK `CellData` |
| `Face` | `EntityKind::Face` | Attach as mesh face field, then extend writer or sidecar output |
| `BoundaryFace` | `EntityKind::Face` subset by marker | Attach as mesh face field for selected boundary faces |
| `Edge` | `EntityKind::Edge` | Attach as mesh edge field, then extend writer or sidecar output |
| `QuadraturePoint` | FE-owned layout, not direct mesh field | Defer direct file output or project/reduce to another scope |

Initial production support should fully complete `Vertex` and `Cell` output because these map directly to existing VTK point and cell data. The registry and evaluator API should still be designed so `Face`, `BoundaryFace`, `Edge`, and `QuadraturePoint` can be added without changing physics-facing APIs.

## Main Implementation Steps

### 1. Add FE Postprocessing Namespace and File Layout

Create a new FE-library area for derived result infrastructure.

Suggested files:

```text
Code/Source/solver/FE/PostProcessing/DerivedResultTypes.h
Code/Source/solver/FE/PostProcessing/DerivedResultBuilder.h
Code/Source/solver/FE/PostProcessing/DerivedResultRegistry.h
Code/Source/solver/FE/PostProcessing/DerivedResultRegistry.cpp
Code/Source/solver/FE/PostProcessing/DerivedResultEvaluator.h
Code/Source/solver/FE/PostProcessing/DerivedResultEvaluator.cpp
Code/Source/solver/FE/PostProcessing/DerivedResultOutput.h
Code/Source/solver/FE/PostProcessing/DerivedResultOutput.cpp
```

Checklist:

- [x] Create `Code/Source/solver/FE/PostProcessing/`.
- [x] Add the new headers and source files to the FE CMake target.
- [x] Use namespace `svmp::FE::post`.
- [x] Keep all types independent of `Physics/`.
- [x] Include only FE, Mesh, and standard-library dependencies.
- [x] Add brief file-level comments describing the physics-agnostic responsibility of each file.
- [x] Confirm the new files build when included but unused.

### 2. Define Derived Result Types

Add the core metadata types in `DerivedResultTypes.h`.

Suggested API:

```cpp
enum class DerivedResultScope {
    Vertex,
    Edge,
    Face,
    BoundaryFace,
    Cell,
    QuadraturePoint
};

enum class DerivedResultPolicy {
    PointValue,
    CellCentroid,
    CellAverage,
    FaceCentroid,
    FaceAverage,
    PatchAverage,
    L2Projection
};

struct DerivedResultDefinition {
    std::string name{};
    DerivedResultScope scope{DerivedResultScope::Cell};
    DerivedResultPolicy policy{DerivedResultPolicy::CellAverage};
    FE::systems::FEQuantityShape shape{};
    forms::FormExpr expression{};
    std::vector<FieldId> referenced_fields{};
    std::optional<int> marker{};
    bool enabled{true};
};
```

Checklist:

- [x] Add `DerivedResultScope`.
- [x] Add `DerivedResultPolicy`.
- [x] Add `DerivedResultDefinition`.
- [x] Reuse `FEQuantityShape` for scalar, vector, and tensor shape metadata.
- [x] Store the derived expression as `forms::FormExpr`.
- [x] Store referenced fields explicitly for validation and efficient evaluator setup.
- [x] Include optional marker metadata for boundary-face and region-restricted output.
- [x] Add helper functions for scope-to-string and policy-to-string for diagnostics.
- [x] Add helper functions for component count and target mesh entity kind where applicable.
- [x] Document that `QuadraturePoint` is not directly a mesh-field scope.

### 3. Add a Developer-Facing Builder

Add `DerivedResultBuilder` so physics modules can register outputs cleanly without hand-populating every field.

Suggested usage:

```cpp
system.addDerivedResult(
    DerivedResultBuilder("Darcy_flux")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::vector(dim))
        .expression(-K * grad(p))
        .referencedField(pressure_id)
        .build());
```

Checklist:

- [x] Add `DerivedResultBuilder`.
- [x] Require a non-empty result name.
- [x] Require a valid expression before `build()`.
- [x] Require explicit shape before `build()`.
- [x] Provide `.scope(...)`.
- [x] Provide `.policy(...)`.
- [x] Provide `.shape(...)`.
- [x] Provide `.expression(...)`.
- [x] Provide `.referencedField(...)` and `.referencedFields(...)`.
- [x] Provide `.marker(...)` for boundary-face and region-restricted future use.
- [x] Default `enabled` to true.
- [x] Emit clear exceptions when required metadata is missing.
- [x] Keep the builder header-only if it remains small and dependency-light.

### 4. Add a Derived Result Registry

Add a small registry that stores definitions and enforces unique names.

Suggested API:

```cpp
class DerivedResultRegistry {
public:
    DerivedResultHandle registerDefinition(DerivedResultDefinition def);

    const DerivedResultDefinition& get(DerivedResultHandle handle) const;
    const DerivedResultDefinition& get(std::string_view name) const;

    bool contains(std::string_view name) const noexcept;
    std::span<const DerivedResultDefinition> all() const noexcept;
};
```

Checklist:

- [x] Add `DerivedResultHandle`.
- [x] Add `DerivedResultRegistry`.
- [x] Enforce unique result names.
- [x] Preserve registration order for deterministic output order.
- [x] Reject invalid definitions with clear diagnostics.
- [x] Validate that result names are suitable mesh-field names.
- [x] Expose lookup by handle.
- [x] Expose lookup by name.
- [x] Expose ordered iteration.
- [x] Add unit tests for duplicate-name rejection.
- [x] Add unit tests for ordered iteration.

### 5. Add FESystem Registration API

Expose the registry through `FESystem`.

Suggested API:

```cpp
post::DerivedResultHandle addDerivedResult(post::DerivedResultDefinition def);

std::span<const post::DerivedResultDefinition> derivedResults() const noexcept;

void appendDerivedResultFields(
    MeshBase& mesh,
    const SystemStateView& state,
    const post::DerivedResultOutputOptions& options = {}) const;
```

Checklist:

- [x] Add a `std::unique_ptr<post::DerivedResultRegistry>` member to `FESystem`.
- [x] Initialize the registry lazily or in the constructor, following existing registry patterns.
- [x] Add `FESystem::addDerivedResult(...)`.
- [x] Add `FESystem::derivedResults()`.
- [x] Add `FESystem::appendDerivedResultFields(...)`.
- [x] Ensure registration invalidates any derived-output setup cache if needed.
- [x] Keep this API independent of `Application/`.
- [x] Keep this API independent of `Physics/`.
- [x] Add unit tests registering definitions through `FESystem`.
- [x] Confirm existing systems with no derived results behave identically.

### 6. Add Expression Validation

Derived result expressions should be checked before evaluation so invalid developer registrations fail early.

Validation rules:

- Expressions must not contain `TestFunction`.
- Expressions must not contain unbound `TrialFunction`.
- Expressions must not contain integration measures such as `dx`, `ds`, or `dS`.
- Expressions must have a shape compatible with the declared result shape.
- Expressions using differential operators on H1 fields may target `Cell`, `Face`, or recovered `Vertex` policies, but not raw vertex point values unless explicitly allowed.
- Expressions may reference `StateField` and `DiscreteField` nodes with concrete `FieldId`s.

Checklist:

- [x] Add a `validateDerivedResultDefinition(...)` function.
- [x] Reject invalid or empty `FormExpr` expressions.
- [x] Reject `TestFunction` nodes.
- [x] Reject unsupported `TrialFunction` nodes.
- [x] Reject measure/integral nodes.
- [x] Reject unsupported auxiliary-output nodes unless runtime values can be supplied.
- [x] Validate declared component count against inferred expression shape where inference is available.
- [x] Validate scope and policy compatibility.
- [x] Validate that `BoundaryFace` scope includes a marker.
- [x] Validate that `CellAverage` targets `Cell`.
- [x] Validate that `FaceAverage` targets `Face` or `BoundaryFace`.
- [x] Validate that `PatchAverage` targets `Vertex`.
- [x] Add diagnostics that include result name, scope, policy, and expression string.
- [x] Add tests for each rejected expression category.

### 7. Add a General Expression Evaluator for Derived Results

Add a reusable evaluator that can evaluate scalar, vector, or tensor `FormExpr` values at quadrature points or sample points using the solved state.

Important distinction:

- Do not use `PointEvaluator` for this feature because it intentionally excludes `grad`, `curl`, `div`, and related FE differential operators.
- Reuse the form-kernel evaluation concepts that already support `StateField`, `DiscreteField`, and differential operators in assembly contexts.

Suggested core API:

```cpp
class DerivedResultEvaluator {
public:
    DerivedResultEvaluator(const systems::FESystem& system,
                           const systems::SystemStateView& state);

    void evaluateToMeshField(MeshBase& mesh,
                             const DerivedResultDefinition& def) const;
};
```

Checklist:

- [x] Add `DerivedResultEvaluator`.
- [x] Bind the current `FESystem`.
- [x] Bind the current `SystemStateView`.
- [x] Create or reuse a global solution view when `state.u_vector` is present.
- [x] Gather field coefficients from the system DOF maps.
- [x] Populate `AssemblyContext` field value and gradient data for referenced fields.
- [x] Support scalar output.
- [x] Support vector output.
- [x] Support tensor output.
- [x] Evaluate `StateField` and `DiscreteField` references by `FieldId`.
- [x] Evaluate `grad(StateField)` for scalar and vector fields.
- [x] Evaluate ordinary algebraic form vocabulary such as `inner`, `asVector`, `asTensor`, `sym`, and scalar operations.
- [x] Provide a clear error for unsupported expression nodes.
- [x] Avoid duplicating low-level form interpreter logic where a reusable internal helper can be extracted.
- [x] Add unit tests evaluating a constant expression.
- [x] Add unit tests evaluating a state-field value.
- [x] Add unit tests evaluating `grad` of a linear H1 scalar field.
- [x] Add unit tests evaluating a vector expression such as `-K * grad(p)`.

### 8. Implement Cell-Scope Evaluation

Cell-scope evaluation is the first required target for Darcy flux.

Policies:

```text
CellCentroid: evaluate expression at one reference centroid per cell.
CellAverage: integrate expression over quadrature points and divide by cell measure.
```

Checklist:

- [x] Implement `CellCentroid`.
- [x] Implement `CellAverage`.
- [x] Select an appropriate quadrature rule for `CellAverage`.
- [x] Compute cell measure from quadrature weights.
- [x] Evaluate each component at each quadrature point.
- [x] Accumulate component-wise weighted averages.
- [x] Attach or update an `EntityKind::Volume` mesh field.
- [x] Write one result tuple per local cell.
- [x] Preserve existing `GlobalCellID` fields.
- [x] Define behavior for ghost cells consistently with existing VTK output.
- [x] Ensure linear H1 pressure with constant `K` produces exact constant cell flux.
- [x] Add serial tests for cell data output.
- [x] Add MPI tests if the current FE test harness supports mesh-field output checks.

### 9. Implement Vertex-Scope Evaluation

Vertex-scope evaluation should support both direct point values and recovered values.

Policies:

```text
PointValue: evaluate expressions that are continuous and well-defined at vertices.
PatchAverage: average adjacent cell evaluations into vertices.
L2Projection: future high-quality projection option.
```

Checklist:

- [x] Implement `PointValue` for expressions that do not require discontinuous derivatives.
- [x] Reject raw `PointValue` for expressions such as `grad(H1 field)` unless explicitly allowed.
- [x] Implement `PatchAverage` for cell-derived quantities recovered to vertices.
- [x] Build cell-to-vertex adjacency or reuse existing mesh topology.
- [x] Accumulate adjacent cell values into each vertex.
- [x] Apply ownership or ghost synchronization rules for distributed meshes.
- [x] Attach or update an `EntityKind::Vertex` mesh field.
- [x] Add tests for nodal state-field output through the derived-result path.
- [x] Add tests for recovered gradient output on a simple mesh.
- [x] Document that recovered vertex gradients are visualization fields, not conservative flux fields.

### 10. Prepare Face, Boundary-Face, Edge, and Quadrature Scopes

The registry should expose these scopes immediately, but implementation can be phased.

Face and boundary-face policies:

```text
FaceCentroid
FaceAverage
```

Edge policies:

```text
PointValue
EdgeAverage
```

Quadrature-point policies:

```text
QuadratureValue
ProjectToCell
ProjectToVertex
```

Checklist:

- [x] Add enum values for all intended scopes.
- [x] Add validation for scope/policy combinations.
- [x] Add placeholder evaluator branches that fail with clear "not yet implemented" messages.
- [x] Add design notes for face-field output in VTK.
- [x] Decide whether face and edge fields are written into the volume output, sidecar files, or both.
- [x] Define boundary-face marker filtering.
- [x] Define behavior for interior faces.
- [x] Define ownership rules for distributed face and edge fields.
- [x] Define a storage plan for quadrature-point derived values.
- [x] Decide whether quadrature output must be projected before ordinary VTK output.
- [x] Add tests confirming unsupported scopes fail clearly until implemented.

### 11. Add Mesh Field Attachment Helpers

Derived result output should attach fields through one consistent helper instead of duplicating mesh-field management in the evaluator.

Suggested helper:

```cpp
FieldHandle ensureDerivedResultField(
    MeshBase& mesh,
    EntityKind kind,
    std::string_view name,
    std::size_t components,
    DerivedResultOverwritePolicy overwrite);
```

Checklist:

- [x] Add `DerivedResultOutputOptions`.
- [x] Add overwrite behavior: replace existing compatible fields, reject incompatible fields, or rename.
- [x] Implement an `ensureDerivedResultField(...)` helper.
- [x] Validate component count.
- [x] Validate scalar type is `Float64`.
- [x] Remove and recreate incompatible fields only when the selected overwrite policy allows it.
- [x] Zero-fill field data before evaluation when needed.
- [x] Return a typed span or pointer for writing component data.
- [x] Add tests for creating a new vertex field.
- [x] Add tests for creating a new cell field.
- [x] Add tests for incompatible pre-existing fields.

### 12. Integrate with Application Output

The application should only know that FE can append derived fields. It should not know individual derived-result names or physics semantics.

Current flow:

```text
evaluate primary fields at vertices
mesh.save_parallel(...)
```

Target flow:

```text
evaluate primary fields at vertices
append FE derived result fields
mesh.save_parallel(...)
```

Checklist:

- [x] Add a call to `sim.fe_system->appendDerivedResultFields(mesh, state)` in `ApplicationDriver::outputResults`.
- [x] Place the call after primary state fields are written.
- [x] Place the call before `mesh.save_parallel(io)`.
- [x] Keep output disabled when `<Save_results_to_VTK_format>` is false.
- [x] Preserve existing primary-field output behavior.
- [x] Add trace logging for derived result evaluation begin/end.
- [x] Add per-result trace logging with name, scope, policy, and component count.
- [x] Ensure systems with no derived results do nothing.
- [x] Ensure derived-output failures include enough context to identify the result.
- [x] Add an application-level test that output includes a registered derived field.

### 13. Add Physics-Side Postprocessing Headers

Physics modules should register postprocessing through formulation-local headers.

Suggested Darcy/Poisson files:

```text
Code/Source/solver/Physics/Formulations/Poisson/PoissonPostProcessing.h
```

Optional future examples:

```text
Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h
Code/Source/solver/Physics/Formulations/Solid/SolidPostProcessing.h
Code/Source/solver/Physics/Formulations/Heat/HeatPostProcessing.h
```

Checklist:

- [x] Add `PoissonPostProcessing.h`.
- [x] Keep the header limited to registration helpers.
- [x] Include FE postprocessing headers and form vocabulary headers.
- [x] Register Darcy flux only when the Poisson module is configured as Darcy.
- [x] Use `StateField(pressure_id, space, field_name)` for the pressure expression.
- [x] Use `-K * grad(p)` for Darcy flux.
- [x] Select `DerivedResultScope::Cell`.
- [x] Select `DerivedResultPolicy::CellAverage`.
- [x] Select vector shape with spatial dimension components.
- [x] Call the registration helper from `PoissonModule::registerOn()`.
- [x] Keep all Darcy naming in Physics, not FE.
- [x] Add tests confirming the Poisson module can register with and without Darcy postprocessing.

### 14. Add Darcy Flux Output

Use the new infrastructure to append Darcy flux to output files.

Expected output:

```text
PointData:
  Pressure

CellData:
  GlobalCellID
  Darcy_flux
```

Checklist:

- [x] Register `Darcy_flux` from the Darcy-through-Poisson path.
- [x] Confirm `Darcy_flux` is not registered for ordinary non-Darcy Poisson unless intentionally requested.
- [x] Confirm `Darcy_flux` has spatial-dimension components.
- [x] Confirm 2D outputs either write two components or use the existing mesh-field component convention.
- [x] Confirm ParaView opens the output and recognizes the cell vector field.
- [x] Confirm linear manufactured pressure produces exact flux.
- [x] Confirm constrained Darcy Square test writes a `Darcy_flux` cell field.
- [x] Confirm the result is cell data, not point data.

### 15. Add Unit Tests for FE Infrastructure

Tests should be FE-focused and physics-agnostic where possible.

Checklist:

- [x] Add tests for `DerivedResultDefinition` validation.
- [x] Add tests for `DerivedResultRegistry`.
- [x] Add tests for `DerivedResultBuilder`.
- [x] Add tests for cell-constant scalar expression output.
- [x] Add tests for cell vector expression output.
- [x] Add tests for `grad` of a linear scalar field.
- [x] Add tests for `CellCentroid`.
- [x] Add tests for `CellAverage`.
- [x] Add tests for vertex `PointValue`.
- [x] Add tests for incompatible scope/policy combinations.
- [x] Add tests for no-op behavior when no derived results are registered.
- [x] Add MPI-safe tests if matching FE output tests already exist.

### 16. Add Physics and Driver Tests

Physics tests should verify that module-level registration produces output through the generic FE path.

Checklist:

- [x] Add a Darcy/Poisson physics test with known linear pressure.
- [x] Register Darcy flux through `PoissonPostProcessing.h`.
- [x] Solve or directly initialize a known pressure state.
- [x] Append derived result fields.
- [x] Verify the mesh has a `Darcy_flux` `EntityKind::Volume` field.
- [x] Verify each cell value matches `-K grad(p)`.
- [x] Add a driver-level Square XML output test if the current test harness supports solver-driver execution.
- [x] Verify the produced `.vtu` contains `Darcy_flux` under `CellData`.
- [x] Verify existing output tests still pass.

### 17. Document Developer Usage

Add concise documentation for physics developers.

Suggested file:

```text
Code/Source/solver/FE/Docs/DerivedResultPostProcessing.md
```

Checklist:

- [x] Explain the FE/Physics ownership split.
- [x] Show the Darcy flux registration example.
- [x] Show a scalar derived result example.
- [x] Show a vector derived result example.
- [x] Explain scope choices.
- [x] Explain policy choices.
- [x] Explain why H1 gradients should usually be cell data or recovered vertex data.
- [x] Explain that XML expression definitions are intentionally unsupported initially.
- [x] Explain how output fields appear in VTK.
- [x] Document currently implemented and future scopes.

### 18. Future XML Expression Support

This is intentionally deferred, but the FE infrastructure should not block it.

Checklist:

- [x] Keep definitions serializable enough for future parser integration.
- [x] Avoid requiring physics-specific C++ types in the FE definitions.
- [x] Keep field references by `FieldId` internally while allowing future field-name resolution.
- [x] Keep expression validation independent of registration source.
- [x] Document that XML support would need a `FormExpr` parser and field-name resolver.
- [x] Add no XML user API in the initial implementation.

## Darcy-Specific Notes

For Darcy through scalar H1 pressure:

```text
q = -K grad(p)
```

The correct initial output association is `Cell` because:

- `p` is continuous in H1.
- `grad(p)` is generally discontinuous between cells.
- Flux recovered directly from H1 pressure is not H(div)-conforming.
- Cell-average flux is honest and stable for visualization and diagnostics.

If users later need conservative normal-continuous fluxes, add a projection/reconstruction policy such as:

```text
DerivedResultPolicy::HDivProjection
```

That should be a new recovery mode, not the default H1 gradient output.

## Completion Criteria

The plan is complete when:

- [x] FE exposes a physics-agnostic registry for derived result definitions.
- [x] Physics modules can register derived results using `FE::forms::FormExpr`.
- [x] The application output path appends registered derived fields without physics-specific logic.
- [x] `Vertex` derived results can be written as point data.
- [x] `Cell` derived results can be written as cell data.
- [x] Darcy registers `Darcy_flux = -K * grad(Pressure)` from a formulation-local postprocessing header.
- [x] Darcy flux appears in the VTK output as cell data.
- [x] Unit tests cover registration, validation, evaluation, and output attachment.
- [x] Documentation explains how future physics modules add their own derived fields.
