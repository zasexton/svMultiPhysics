# H(div) Trace Boundary Condition Infrastructure Plan

## Objective

Add the FE-library infrastructure needed to support:

- prescribed normal flux on `H(div)` fields via strong boundary constraints
- unconstrained or "passive" outflow behavior without introducing Darcy-specific logic
- generic weak boundary relations on normal traces when a physics module needs them

The FE layer must remain physics-agnostic. The target is not "Darcy outflow" as a special case. The target is first-class support for normal-trace boundary data on vector-basis spaces.

## Scope

### In Scope

- strong inhomogeneous normal-trace constraints for `H(div)` fields
- time-dependent normal-trace data
- generic weak BC wrappers that operate on scalar trace quantities such as `u·n`
- analysis metadata and compatibility updates for the new BC types
- serial and MPI test coverage for orientation, ownership, and constraint updates
- documentation and usage guidance

### Out of Scope for This Effort

- Darcy-specific FE logic
- complementarity or inequality constraints such as "allow outflow but forbid inflow"
- one-sided upwind or characteristic outflow policies encoded in the FE core
- introducing a new PDE-level notion of "passive flux out" into FE

## Current State Summary

### What Already Exists

- `H(div)` spaces are first-class and already expose normal continuity and normal-trace semantics in [HDivSpace.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Spaces/HDivSpace.h:16).
- `TraceSpace` already constructs a scalar face trace for `H(div)` normal traces in [TraceSpace.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Spaces/TraceSpace.cpp:612).
- The form language already supports geometry normals through [FormExpr.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/FormExpr.h:485), so weak terms like `dot(u, n)` are expressible.
- The analysis vocabulary already distinguishes `NormalComponent` and `NormalFlux` in [BoundaryConditionDescriptor.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h:45).

### What Is Missing

- The current strong `H(div)` boundary constraint is homogeneous-only in [HDivNormalConstraint.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Constraints/HDivNormalConstraint.h:15).
- Generic strong Dirichlet lowering explicitly rejects `H(div)` vector-basis spaces in [StrongDirichletConstraint.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Constraints/StrongDirichletConstraint.cpp:62).
- The standard BC wrappers are written around H1-style full-field pairings in [StandardBCs.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/StandardBCs.h:81) and [StandardBCs.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/StandardBCs.h:137), not scalar trace operators.
- The current `BoundaryCondition` interface has a clean path for pointwise `StrongDirichlet` declarations and affine constraints, but not an explicit first-class path for non-pointwise strong system constraints.

## Required Capability Model

The FE layer should expose the following generic capabilities:

1. Prescribe a scalar normal trace `u·n = g` on a tagged boundary for an `H(div)` field.
2. Leave the normal trace unconstrained on a tagged boundary.
3. Add weak boundary terms involving scalar traces such as:
   - `g * tau(v)`
   - `alpha * tau(u) * tau(v)`
   - coupled terms built from `tau(u)`, `tau(v)`, and auxiliary inputs
4. Mark those BCs with mathematically correct analysis metadata.

For `H(div)`, the trace operator is the scalar normal component `tau(w) = w·n`.

## Recommended Architecture

## Phase 1: Add a Real Inhomogeneous Normal-Trace Constraint for H(div)

### Why

Prescribed normal flux on an `H(div)` field is an essential boundary condition on the normal trace. It cannot be implemented correctly by the existing pointwise `StrongDirichletConstraint`, because `H(div)` face DOFs are trace moments or orientation-sensitive face DOFs, not nodal point values.

### Recommended Design

Add a new constraint class dedicated to nonzero normal-trace data on `H(div)` spaces.

Recommended class name:

- `constraints::HDivNormalTraceConstraint`

Possible transitional path:

- keep `HDivNormalConstraint` as a homogeneous convenience wrapper around the new class
- or rename the existing class and keep a compatibility alias if needed

### Constraint Contract

The new constraint should:

- accept `FieldId`
- accept `boundary_marker`
- accept scalar boundary data `g(x, t, dt)` as a `FormExpr`
- support homogeneous and inhomogeneous values
- support `updateValues(...)` for time-dependent data
- lower into `AffineConstraints` using actual trace DOFs, not pointwise volume-node coordinates

### Lowering Algorithm

For each owned boundary face on the tagged marker:

1. Build or access a `TraceSpace` for the local face of the volume `H(div)` space.
2. Interpolate or project the scalar boundary data `g` into that face trace space.
3. Map the resulting face coefficients back to the corresponding volume-face DOFs using `TraceSpace::face_dof_indices()`.
4. Apply the resulting coefficient values as Dirichlet inhomogeneities on the associated global DOFs.
5. Cache the affected DOFs and the face-local evaluation/projection metadata needed for `updateValues(...)`.

### Important Design Rules

- Do not sample `g` only at points unless the trace basis is nodal and that path is explicitly correct.
- Prefer a trace-space interpolation or projection routine, because that matches the actual `H(div)` trace DOF meaning.
- Preserve face orientation signs exactly as `TraceSpace` and DOF numbering define them.
- Only the owning rank should insert or update each constrained global DOF.

### Concrete Files to Modify

- `Code/Source/solver/FE/Constraints/HDivNormalConstraint.h`
- `Code/Source/solver/FE/Constraints/HDivNormalConstraint.cpp`
- `Code/Source/solver/FE/Constraints/SystemConstraint.h`
  - interface likely already sufficient; update only if the chosen design needs richer trace-update context
- `Code/Source/solver/FE/Spaces/TraceSpace.h`
- `Code/Source/solver/FE/Spaces/TraceSpace.cpp`
  - only if helper methods are needed for face-local interpolation/projection and coefficient extraction
- `Code/Source/solver/FE/CMakeLists.txt`

### Likely Helper Additions

If the existing `TraceSpace` API is not enough, add one or more of:

- a helper to interpolate a scalar `FormExpr` onto a face trace space
- a helper to return volume-global DOFs for the face in trace-basis order
- a small utility for caching per-face trace-work metadata for time updates

## Phase 2: Formalize Strong BC Lowering Beyond Pointwise StrongDirichlet

### Why

`StrongDirichlet` is currently a pointwise boundary-value declaration model. `H(div)` prescribed normal trace is a different kind of strong boundary condition. It should not be forced through the same declaration path if that path assumes nodal coordinates and component-wise value setting.

### Recommended Design

Add an explicit BC-to-system-constraint hook for strong non-pointwise constraints.

Recommended interface addition to `forms::bc::BoundaryCondition`:

- `installSystemConstraints(systems::FESystem&, FieldId)` or
- `buildSystemConstraints(FieldId) -> std::vector<std::unique_ptr<constraints::ISystemConstraint>>`

### Why This Is Better Than Reusing setup() Ad Hoc

- it makes non-pointwise strong BC lowering an explicit public contract
- it avoids encoding `H(div)` behavior inside `StrongDirichlet`
- it avoids relying on hidden side effects inside `setup()`
- it makes future `H(curl)` tangential essential BCs fit the same model

### BoundaryConditionManager Changes

Update the manager so that `apply()` / `applyAll()`:

1. calls BC setup
2. installs any BC-owned system constraints
3. collects metadata
4. contributes weak terms
5. installs pointwise `StrongDirichlet` constraints
6. preserves affine-constraint lowering for periodic and MPC-style BCs

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/BoundaryCondition.h`
- `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
  - only if direct non-BC installation helpers are also desired
- `Code/Source/solver/FE/Forms/BoundaryConditions.h`
  - only if declaration helpers are added here

## Phase 3: Add Trace-Oriented Boundary Condition Wrappers in FE/Forms

### Why

The existing standard BC wrappers assume full-field H1-style pairings:

- natural load: `inner(flux, v)`
- robin: `alpha * inner(u, v) - inner(rhs, v)`

That is not the right abstraction for `H(div)` trace conditions. The right abstraction is a scalar trace operator `tau(...)`.

### Recommended New BC Types

Add generic trace-based wrappers rather than Darcy-named wrappers.

Recommended BC classes:

- `forms::bc::NormalTraceEssentialBC`
  - strong prescribed `u·n = g` for `H(div)` fields
- `forms::bc::TraceLoadBC`
  - weak term of the form `-∫ g * tau(v) ds`
- `forms::bc::TraceRobinBC`
  - weak term of the form `∫ alpha * tau(u) * tau(v) ds - ∫ rhs * tau(v) ds`

Optional ergonomic helper:

- `forms::normalComponent(expr)` or `forms::trace::normal(expr)`

This helper is not strictly required because `dot(expr, FormExpr::normal())` already exists, but it would make BC authorship and metadata lowering clearer.

### Design Constraints

- these BCs must remain usable outside Darcy
- the BC class should not assume that the conjugate variable is pressure
- the weak trace BCs should accept scalar `FormExpr` arguments and use `tau(u)` / `tau(v)` consistently

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/StandardBCs.h`
- `Code/Source/solver/FE/Forms/BoundaryConditions.h`
- `Code/Source/solver/FE/Forms/FormExpr.h`
- `Code/Source/solver/FE/Forms/FormExpr.cpp`
  - only if adding a dedicated normal-trace helper
- `Code/Source/solver/FE/Forms/Vocabulary.h`
  - if helper constructors are exposed there

## Phase 4: Align Analysis Metadata With the New BC Types

### Why

The analysis layer already understands `NormalComponent` and `NormalFlux`, but the new BC producers must emit the correct descriptor values or the constraint-rank, nullspace, and compatibility reports will be misleading.

### Required Metadata Behavior

- strong prescribed `u·n = g` on an `H(div)` field should report:
  - `trace_kind = TraceKind::NormalComponent`
  - `enforcement_kind = EnforcementKind::Strong`
- a weak normal-trace load should report:
  - `trace_kind = TraceKind::NormalComponent`
  - `enforcement_kind = EnforcementKind::WeakConsistent`
- a weak normal-trace Robin relation should report:
  - `trace_kind = TraceKind::NormalComponent`
  - `enforcement_kind = EnforcementKind::WeakPenalty`

### Concrete Files to Modify

- `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h`
  - likely no enum changes needed
- `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.cpp`
  - only if string or lowering behavior needs updates
- `Code/Source/solver/FE/Analysis/CompatibilityAnalyzer.cpp`
- `Code/Source/solver/FE/Analysis/SpaceCompatibilityAnalyzer.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Analysis/`

## Phase 5: Verification Strategy

### Unit Tests Required

#### Constraint-Level Tests

- homogeneous `H(div)` normal-trace constraint still constrains the same DOFs as today
- nonzero prescribed trace sets correct inhomogeneities for RT/BDM-style boundary DOFs
- time-dependent trace data updates the inhomogeneities through `updateValues(...)`
- face orientation sign is correct on differently oriented faces
- only locally owned DOFs are written on MPI ranks

#### TraceSpace-Level Tests

- scalar face-trace interpolation matches the expected normal component for known fields
- face trace coefficient ordering matches volume face DOF ordering
- `lift`/`restrict`/`face_dof_indices` remain consistent for `H(div)` traces

#### BC-Level Tests

- `NormalTraceEssentialBC` installs the correct system constraint and reports correct metadata
- `TraceLoadBC` generates the expected residual form for `tau(v)`
- `TraceRobinBC` generates the expected residual form for `tau(u) * tau(v)`
- boundary-condition manager handles these BCs without conflicting with existing pointwise strong BC logic

#### Analysis Tests

- new BC descriptors map to the expected trace capabilities
- nullspace anchoring behavior is reported consistently for strong normal-trace BCs

### Suggested Test Files

- `Code/Source/solver/FE/Tests/Unit/Systems/test_VectorBasisConstraints.cpp`
- `Code/Source/solver/FE/Tests/Unit/Systems/test_BoundaryConditionManager.cpp`
- `Code/Source/solver/FE/Tests/Unit/Forms/test_BoundaryConditionHelpers.cpp`
- `Code/Source/solver/FE/Tests/Unit/Spaces/test_TraceSpace.cpp`
- `Code/Source/solver/FE/Tests/Unit/Analysis/test_BoundaryConditionDescriptor.cpp`

Add new dedicated files if the existing suites become too overloaded.

## Phase 6: Documentation and Usage Guidance

### Documentation Goals

- document the distinction between pointwise value BCs and trace BCs
- state explicitly that "passive outflow" is not an FE primitive
- show that physics modules can realize open/outflow behavior by either:
  - leaving `u·n` unconstrained
  - imposing a condition on the conjugate field
  - using a generic weak trace relation

### Concrete Files to Update

- `Code/Source/solver/FE/README.md`
- `Code/Source/solver/FE/Forms/SYSTEMS_INTEGRATION.md`
- `Code/Source/solver/FE/Forms/VOCABULARY.md`
- `Code/Source/solver/FE/Docs/Book/chapters/ch08_assembly_boundary_conditions_and_constraints.tex`
  - optional, if book-level documentation is kept current

## Implementation Decisions That Should Be Locked Before Coding

1. Choose whether the new strong constraint is:
   - `HDiv`-specific in the first pass, or
   - a more general trace-constraint framework that could later cover `H(curl)` tangential essential BCs
2. Choose whether face data is imposed by:
   - interpolation onto the trace basis, or
   - `L2` projection onto the trace basis
3. Decide whether the BC API should expose:
   - explicit trace-specific BC classes only, or
   - a generic scalar trace BC base class plus `NormalTrace...` convenience wrappers
4. Decide whether to:
   - add a new boundary-condition hook for system constraints, or
   - temporarily install such constraints through `setup()` as an implementation detail

Recommendation:

- first pass should add an explicit BC-to-system-constraint hook
- first pass should implement `H(div)` normal-trace support directly
- first pass should keep the weak BC wrappers generic and trace-based

## Completion Checklist

### Design and API

- [x] confirm that FE scope is generic normal-trace support, not Darcy-specific outflow semantics
- [x] decide whether the first-pass strong constraint is `H(div)`-specific or a generalized trace-constraint base
- [x] decide interpolation vs projection for prescribed trace data
- [x] decide the explicit BC hook for non-pointwise strong system constraints
- [x] decide final public class names for the new constraint and BC wrappers

### Constraint Infrastructure

- [x] implement inhomogeneous `H(div)` normal-trace constraint class
- [x] preserve or wrap the existing homogeneous `HDivNormalConstraint` behavior
- [x] cache enough per-face or per-DOF data to support `updateValues(...)`
- [x] verify orientation handling on all supported `H(div)` element families
- [x] verify MPI ownership and duplicate-face safety

### BoundaryCondition and Systems Plumbing

- [x] extend `forms::bc::BoundaryCondition` with a first-class non-pointwise strong-constraint hook
- [x] update `BoundaryConditionManager` to install BC-owned system constraints
- [x] keep existing `StrongDirichlet` installation unchanged for nodal/component-wise value BCs
- [x] keep periodic and MPC-style affine constraint lowering working

### Forms-Level BC Wrappers

- [x] add `NormalTraceEssentialBC`
- [x] add `TraceLoadBC`
- [x] add `TraceRobinBC`
- [x] add any needed helper for `tau(w) = w·n`
- [x] add public helper constructors if the project prefers factory-style BC creation

### Analysis Integration

- [x] ensure new BCs emit `TraceKind::NormalComponent` metadata
- [x] ensure weak trace load and Robin BCs emit the correct enforcement kinds
- [x] add analysis regression tests for the new descriptors

### Tests

- [x] add unit tests for nonzero prescribed normal trace on `H(div)` spaces
- [x] add time-dependent update tests for normal-trace constraints
- [x] add trace-space coefficient-ordering tests if new helpers are introduced
- [x] add BC-manager tests for mixed strong and weak trace BC paths
- [x] add MPI regression coverage if the new constraint touches distributed DOF ownership

### Documentation

- [x] update FE boundary-condition docs to distinguish pointwise and trace essential BCs
- [x] document that "passive outflow" is realized by leaving the trace unconstrained or by imposing a generic conjugate-field relation, not by a Darcy-specific FE feature
- [x] add at least one short usage example for prescribed `u·n = g`

## Definition of Done

This effort is complete when all of the following are true:

- an `H(div)` field can accept nonzero prescribed normal-trace data through a first-class FE boundary-condition path
- the constraint can be updated in time without rebuilding the whole BC declaration layer
- weak trace BCs can be expressed without Darcy-specific logic
- analysis metadata and compatibility reports stay correct
- serial and MPI tests cover the new constraint semantics
- FE documentation explains how to use these features and what remains intentionally outside FE scope
