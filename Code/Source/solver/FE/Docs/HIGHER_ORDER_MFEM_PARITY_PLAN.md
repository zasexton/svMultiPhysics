# Higher-Order MFEM Parity Plan

## Objective

Bring the FE library's higher-order finite element stack to practical parity with
MFEM for the following in-scope areas:

- arbitrary-order scalar `H1` and `L2` spaces on supported cell topologies
- arbitrary-order `H(div)` and `H(curl)` spaces where the basis contract already claims support
- arbitrary-order 2D quadrilateral serendipity
- first-class scalar NURBS spaces
- higher-order trace spaces
- variable-order spaces on a fixed mesh

This plan treats parity as an end-to-end requirement across:

- basis construction
- element wrappers
- space exposure
- DOF numbering and orientation
- trace restriction
- assembly compatibility
- tests and documentation

## First-Pass Scope

### In Scope

- `H1`, `L2`, `H(div)`, and `H(curl)` higher-order infrastructure
- basis-family exposure for `Lagrange`, `Hierarchical`, `Bernstein`, `Spectral`
- arbitrary-order 2D quadrilateral serendipity
- scalar B-spline and NURBS spaces
- higher-order `H1`, `H(div)`, and `H(curl)` trace support
- variable-order spaces on a fixed mesh

### Explicitly Out of Scope for the First Pass

- `C1` / Hermite parity
- 3D serendipity parity
- full `hp` mesh adaptivity
- GPU or matrix-free parity work
- MFEM embedded-manifold families such as `ND_R1D` / `RT_R2D`

## Delivery Principles

- Remove architectural blockers before adding more basis families.
- Make serial and parallel DOF paths evolve together.
- Add executable tests for each claimed capability before updating support docs.
- Prefer first-class space APIs over generic-basis escape hatches when parity is the goal.
- Keep each phase independently verifiable.

## Current Progress

Validated implementation work completed so far:

- arbitrary-order scalar `DofLayoutInfo` for line, triangle, quad, tetra, hex, wedge, and pyramid cells
- serial and parallel scalar face-interior permutation for higher-order triangle and quadrilateral faces
- executable DOF tests for higher-order scalar `H1` / `L2` layout counts and shared-face orientation behavior
- mixed-face serial and parallel conforming `H1` numbering for higher-order wedge and pyramid cells
- MPI regression tests for mixed-face wedge and pyramid scalar `H1` numbering
- basis-aware first-class `H1` / `L2` exposure for `Hierarchical`, `Bernstein`, and `Spectral`
- request-driven first-class scalar `H1` / `L2` exposure for `BSpline` and `NURBS`
- request-driven `SpaceFactory` paths that now preserve scalar basis family instead of collapsing back to Lagrange-only spaces
- scalar tensor-product spline and NURBS trace support on quadrilateral and hexahedral faces
- executable trace tests covering oriented B-spline edge traces and NURBS quadrilateral face traces
- first-class `H(div)` normal-trace FE spaces on higher-order faces
- first-class `H(curl)` tangential-trace FE spaces on higher-order edges and faces
- executable vector trace tests covering higher-order `H(div)` quad-face traces and `H(curl)` edge / triangular-face traces
- additive per-element order storage and query APIs on `FunctionSpace` / `AdaptiveSpace`
- serial mixed-order scalar and vector DOF numbering through the mesh-topology API
- executable variable-order tests covering cell-specific element selection and mixed-order `H1` / `H(curl)` numbering
- compatible-tensor first-class vector spline / NURBS spaces on `Quad4` for `H(div)` and `H(curl)`
- DOF association, edge-orientation, and trace support for the supported vector spline / NURBS spaces
- additive public `evaluate_jacobian()` support on `FunctionSpace` with scalar-only `evaluate_gradient()` semantics preserved
- low-order helper audit and scope tightening for region-based DOF extraction utilities
- full regression qualification across FE basis, element, space, dof, and assembly test surfaces

## Current Validation Matrix

| Area | Status | Validation |
| --- | --- | --- |
| Parity scope and sequencing | complete | this plan file |
| Scalar `H1` high-order DOF layout on line/triangle/quad/tetra/hex | complete | `test_fe_dofs`, `DofHandler.*` |
| Scalar `L2` high-order DOF layout on line/triangle/quad/pyramid/wedge/hex | complete | `test_fe_dofs`, `DofHandler.*` |
| Scalar shared-face orientation on tetra/hex | complete | `test_fe_dofs`, `DofHandler.*` |
| First-class `H1` / `L2` `Hierarchical` / `Bernstein` / `Spectral` space creation | complete | `test_fe_spaces`, `FunctionSpaces.*` |
| First-class scalar spline / NURBS space creation through the request API | complete | `test_fe_spaces`, `FunctionSpaces.*` |
| Existing space regression coverage after scalar-space API changes | complete | `test_fe_spaces` |
| Conforming higher-order wedge / pyramid `H1` numbering | complete | `test_fe_dofs`, `DofHandler.*`; `test_fe_mpi`, `DofHandlerMPI.*` |
| Arbitrary-order 2D serendipity | complete | `test_fe_basis`, `SerendipityBasis.*`; `test_fe_spaces`, `FaceRestrictionTest.*`, `TraceSpace.*`; `test_fe_elements`, `ElementFactoryErrors.*` |
| Higher-order scalar / vector trace parity | complete | `test_fe_spaces`, `TraceSpace.*`; `test_fe_elements`, `InterfaceContinuityTest.*` |
| Variable-order spaces | complete | `test_fe_spaces`, `FunctionSpaces.AdaptiveSpace*`; `test_fe_dofs`, `DofHandler.MixedOrder*`; `test_fe_mpi`, `DofHandlerMPI.DistributedVariableOrder*` |
| Scalar spline / NURBS trace support | complete | `test_fe_spaces`, `FaceRestrictionTest.*`, `TraceSpace.*`; `test_fe_elements` regression |
| Quad-compatible vector spline / NURBS spaces | complete | `test_fe_basis`, `BasisFactory.CreatesCompatibleQuadVectorSplineAndNurbsBases`; `test_fe_elements`, `ElementFactory.CreatesCompatibleVectorSplineAndNurbsElements`; `test_fe_spaces`, `FunctionSpaces.SpaceFactoryRequestCreatesFirstClassVectorNurbsSpaces`, `TraceSpace.CompatibleVectorNurbsQuadEdgeTraceMatchesVolume` |
| Public vector Jacobian evaluation | complete | `test_fe_spaces`, `FunctionSpaceGradients.*`; `test_fe_assembly`, `AssemblyContextMultiField.FieldJacobianForVectorField` |
| Low-order helper audit and scoping | complete | `test_fe_dofs`, `DofTools.ExtractDofsInRegion*` |
| Broader assembly-phase qualification | complete | `test_fe_assembly`; `test_fe_basis`; `test_fe_elements`; `test_fe_spaces`; `test_fe_dofs` |

## Phase 0: Lock Down the Parity Contract

### Goals

- Convert the review findings into an implementation contract.
- Make the current gaps measurable through tests.

### Concrete Steps

1. Create a parity matrix covering:
   - continuity: `H1`, `L2`, `H(div)`, `H(curl)`
   - basis family: `Lagrange`, `Hierarchical`, `Bernstein`, `Spectral`, `Serendipity`, `BSpline`, `NURBS`
   - topology: line, triangle, quad, tetra, hex, wedge, pyramid
   - order buckets: low order, order 4+, variable order, trace
2. Add a design note summarizing:
   - what "MFEM parity" means for this effort
   - what is intentionally deferred
   - which existing FE claims are basis-only versus end-to-end
3. Add failing or skipped tests that represent the current missing capabilities.
4. Record the target validation set for each later phase.

### Primary Files and Modules

- `Code/Source/solver/FE/Docs/`
- `Code/Source/solver/FE/Tests/`
- `Code/Source/solver/FE/Basis/README.md`

### Exit Criteria

- There is one authoritative parity checklist.
- The missing items are represented in tests or documented TODO validation cases.

## Phase 1: Remove the Scalar High-Order Infrastructure Blocker

### Goals

- Make arbitrary-order scalar `H1` and `L2` spaces usable end-to-end.

### Concrete Steps

1. Replace hard-coded `DofLayoutInfo::Lagrange(...)` tables with formula-based entity counts.
2. Split scalar DOF-layout logic into topology-specific helpers for:
   - line
   - triangle
   - quad
   - tetra
   - hex
   - wedge
   - pyramid
3. Remove the `order > 3` failure path in scalar DOF layout.
4. Update both serial and parallel DOF distribution paths to consume the same generalized layout logic.
5. Generalize scalar edge and face ordering logic so higher-order face interiors work beyond the current low-order assumptions.
6. Add regression tests for:
   - scalar `H1` at orders 4, 5, 6 on simplex and tensor-product cells
   - scalar `L2` at orders 4, 5, 6 on simplex and tensor-product cells
   - correct global DOF sharing and orientation behavior across neighboring elements
7. Verify that `H1Space` and `L2Space` can be instantiated and globally numbered at those orders without using `GenericBasisSpace`.

### Primary Files and Modules

- `Code/Source/solver/FE/Dofs/DofHandler.cpp`
- `Code/Source/solver/FE/Dofs/DofHandler.h`
- `Code/Source/solver/FE/Spaces/H1Space.cpp`
- `Code/Source/solver/FE/Spaces/L2Space.cpp`
- `Code/Source/solver/FE/Tests/Unit/`

### Exit Criteria

- Scalar `H1` and `L2` spaces work above order 3.
- Serial and parallel DOF numbering agree for the tested cases.
- Support docs no longer over-claim basis-only capability as end-to-end support.

## Phase 2: Promote Alternative High-Order Scalar Bases to First-Class Spaces

### Goals

- Stop treating non-Lagrange high-order families as mostly generic-basis escape paths.

### Concrete Steps

1. Refactor `SpaceFactory` so basis family is a first-class input for scalar spaces.
2. Decide on the API surface:
   - either configurable `H1Space` / `L2Space`
   - or family-specific space wrappers
3. Add first-class `H1` and `L2` exposure for:
   - `Hierarchical`
   - `Bernstein`
   - `Spectral`
4. Ensure those spaces use standard element wrappers, quadrature defaults, and assembly paths.
5. Add tests comparing:
   - DOF counts
   - interpolation behavior
   - assembled mass/stiffness consistency
   - basis-family selection through the public API
6. Reserve `GenericBasisSpace` for genuinely custom or experimental basis families.

### Primary Files and Modules

- `Code/Source/solver/FE/Spaces/SpaceFactory.cpp`
- `Code/Source/solver/FE/Spaces/SpaceFactory.h`
- `Code/Source/solver/FE/Spaces/GenericBasisSpace.*`
- `Code/Source/solver/FE/Elements/ElementFactory.cpp`
- `Code/Source/solver/FE/Basis/README.md`

### Exit Criteria

- Users can request named high-order scalar families through the normal space API.
- `GenericBasisSpace` is no longer required for common MFEM-like scalar basis choices.

## Phase 3: Close the Serendipity Gap

### Goals

- Reach practical MFEM parity for 2D quadrilateral serendipity.

### Concrete Steps

1. Expand `SerendipityBasis` from the current retained low-order contract to arbitrary-order 2D quadrilateral support, using the MFEM-sized serendipity family with interior modes appearing at order 4 and above.
2. Keep 3D serendipity explicitly deferred unless there is a separate decision to pursue it.
3. Update node ordering and face-node generation for higher-order serendipity quads.
4. Extend `FaceRestriction` so serendipity faces are not hard-limited to quadratic order.
5. Extend `TraceSpace` face prototype inference for higher-order serendipity quadrilateral traces.
6. Add tests for:
   - order 3, 4, 5 quadrilateral serendipity basis size and evaluation
   - face restriction correctness
   - trace restriction correctness
   - assembly of representative bilinear forms
7. Update the basis support matrix only after the higher-order trace and restriction paths work.

### Primary Files and Modules

- `Code/Source/solver/FE/Basis/SerendipityBasis.*`
- `Code/Source/solver/FE/Spaces/FaceRestriction.cpp`
- `Code/Source/solver/FE/Spaces/TraceSpace.cpp`
- `Code/Source/solver/FE/Tests/Unit/`

### Exit Criteria

- Arbitrary-order 2D quadrilateral serendipity is basis-complete and space-complete.
- The implemented quadrilateral family matches the MFEM-sized H1 serendipity DOF count, including interior modes for order 4 and above.
- Trace and restriction utilities work for the supported serendipity orders.

## Phase 4: Add First-Class Scalar Spline and NURBS Spaces

### Goals

- Turn spline and NURBS support into a first-class FE capability instead of a mostly generic-basis feature.

### Concrete Steps

1. Introduce first-class scalar B-spline and NURBS space construction APIs.
2. Define a stable input contract for:
   - per-axis orders
   - knot vectors
   - control-net extents
   - rational weights
3. Ensure the new spaces use standard element wrappers and assembly paths.
4. Add trace support for scalar spline and NURBS spaces on tensor-product faces.
5. Add projection, interpolation, and representative assembly tests on line, quad, and hex.
6. Ensure documentation clearly states fixed-order versus variable-order capabilities for these spaces.

### Primary Files and Modules

- `Code/Source/solver/FE/Basis/BSplineBasis.*`
- `Code/Source/solver/FE/Basis/NURBSTensorBasis.*`
- `Code/Source/solver/FE/Spaces/`
- `Code/Source/solver/FE/Elements/`
- `Code/Source/solver/FE/Tests/Unit/`

### Exit Criteria

- Scalar spline and NURBS spaces are public FE features, not just generic-basis constructions.
- Trace restriction works for supported tensor-product spline and NURBS faces.

## Phase 5: Add Higher-Order Trace Parity

### Goals

- Make higher-order traces first-class for scalar and vector spaces.

### Concrete Steps

1. Generalize `TraceSpace` beyond scalar volume spaces.
2. Add explicit higher-order trace support for:
   - `H1`
   - `H(div)`
   - `H(curl)`
3. Reuse the existing orientation machinery already present in `OrientationManager` and `StandardAssembler`.
4. Define a stable trace FE prototype model for:
   - triangle faces
   - quad faces
   - vector traces where orientation matters
5. Add tests covering:
   - tetra and hex traces
   - wedge and pyramid traces where already supported by the basis family
   - face orientation under permutations
   - trace assembly on internal faces
6. Ensure trace space creation can be driven from the same basis-aware space selection API as the volume space.

### Primary Files and Modules

- `Code/Source/solver/FE/Spaces/TraceSpace.cpp`
- `Code/Source/solver/FE/Spaces/FaceRestriction.cpp`
- `Code/Source/solver/FE/Spaces/OrientationManager.*`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Tests/Unit/Spaces/`

### Exit Criteria

- Higher-order traces are available for scalar and vector spaces in the supported families.
- Orientation-sensitive vector traces are validated by tests.

## Phase 6: Add Variable-Order Space Support

### Goals

- Move from a wrapper-style `AdaptiveSpace` to a real variable-order FE space model.

### Concrete Steps

1. Define per-element polynomial order storage.
2. Add APIs for:
   - setting element order
   - querying element order
   - requesting element and trace FE views at a given order
3. Extend DOF numbering to support mixed orders on a fixed mesh for scalar spaces.
4. Extend DOF numbering to support mixed orders on a fixed mesh for vector spaces.
5. Define trace compatibility rules across order transitions.
6. Add refinement and prolongation scaffolding only after mixed-order numbering is stable.
7. Add tests for:
   - mixed-order scalar spaces
   - mixed-order vector spaces
   - internal faces between unequal orders
   - stable assembly and trace behavior

### Primary Files and Modules

- `Code/Source/solver/FE/Spaces/AdaptiveSpace.*`
- `Code/Source/solver/FE/Dofs/DofHandler.*`
- `Code/Source/solver/FE/Spaces/SpaceFactory.*`
- `Code/Source/solver/FE/Tests/Unit/`

### Exit Criteria

- Variable-order spaces are a real public feature.
- Mixed-order assembly and trace restriction work on a fixed mesh.

### Current Validated Scope

- `AdaptiveSpace` stores per-cell order and can return cell-specific element spaces.
- Mixed-order scalar/vector numbering works on the mesh-topology API in serial.
- Mixed-order trace compatibility is implemented for fixed meshes:
  - scalar spaces use trace-space transfer operators
  - `H(curl)` / `H(div)` use shared lower-order moments plus zeroed higher-order excess modes
- Dense prolongation and restriction operators are available through `SpaceInterpolation`.
- Owner-contiguous MPI mixed-order numbering is validated for scalar `H1` and vector `H(curl)` shared-interface cases.

## Phase 7: Add Vector NURBS and Public API Cleanup

### Goals

- Finish the remaining high-order parity gaps after the scalar, trace, and variable-order foundations are in place.

### Concrete Steps

1. Add first-class compatible-tensor `H(div)` spline / NURBS support on quadrilateral spaces.
2. Add first-class compatible-tensor `H(curl)` spline / NURBS support on quadrilateral spaces.
3. Define DOF association and orientation rules for the supported vector spline and NURBS spaces.
4. Extend trace support to the supported vector spline and NURBS spaces.
5. Implement vector-valued Jacobian evaluation in the public function-space API while preserving scalar-only `evaluate_gradient()` semantics.
6. Audit helper utilities that still assume low-order or vertex-based behavior and upgrade or clearly scope them.
7. Review `C1Space` separately to ensure it is not silently relying on scalar Lagrange numbering assumptions if global numbering is expected to work.
8. Leave a follow-on item for broader MFEM-style vector spline / NURBS coverage beyond the current quadrilateral compatible-tensor path.

### Primary Files and Modules

- `Code/Source/solver/FE/Basis/VectorBasis.*`
- `Code/Source/solver/FE/Spaces/FunctionSpace.cpp`
- `Code/Source/solver/FE/Spaces/`
- `Code/Source/solver/FE/Dofs/DofTools.*`
- `Code/Source/solver/FE/Tests/Unit/`

### Exit Criteria

- Vector spline and NURBS spaces are first-class on the supported quadrilateral compatible-tensor path.
- Public evaluation APIs are no longer missing obvious higher-order vector operations.

## Suggested Sequencing

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 5
6. Phase 6
7. Phase 4
8. Phase 7

Rationale:

- Phase 1 removes the main architectural blocker.
- Phase 2 keeps the public API aligned with the new scalar infrastructure.
- Phase 3 and Phase 5 close the most visible MFEM parity gaps after scalar high-order works.
- Phase 6 should happen before deeper spline and vector extensions so variable-order rules are not retrofitted later.
- Phase 4 and Phase 7 depend on the earlier space and trace work being stable.

## Suggested PR Boundaries

### PR 1

- Phase 0 artifacts
- test matrix
- parity design note

### PR 2

- scalar generalized DOF layout
- serial and parallel numbering updates
- scalar higher-order tests

### PR 3

- basis-aware scalar space API
- first-class `Hierarchical`, `Bernstein`, and `Spectral` space exposure

### PR 4

- arbitrary-order 2D serendipity
- higher-order serendipity face restriction and trace support

### PR 5

- higher-order trace parity for `H1`, `H(div)`, and `H(curl)`

### PR 6

- variable-order space support on fixed meshes

### PR 7

- first-class scalar spline and NURBS spaces

### PR 8

- vector NURBS
- API cleanup
- documentation finalization

## Completion Checklist

- [x] Write the parity scope note and test matrix.
- [x] Add executable tests for the currently missing parity items.
- [x] Replace hard-coded scalar `Lagrange` DOF tables with formula-based layouts.
- [x] Remove the scalar `order > 3` limit in serial DOF numbering.
- [x] Remove the scalar `order > 3` limit in parallel DOF numbering.
- [x] Validate arbitrary-order scalar `H1` on supported topologies.
- [x] Validate arbitrary-order scalar `L2` on supported topologies.
- [x] Refactor `SpaceFactory` to make basis family a first-class scalar-space input.
- [x] Add first-class `Hierarchical` space exposure.
- [x] Add first-class `Bernstein` space exposure.
- [x] Add first-class `Spectral` space exposure.
- [x] Implement arbitrary-order 2D quadrilateral serendipity.
- [x] Extend higher-order serendipity face restriction.
- [x] Extend higher-order serendipity trace support.
- [x] Add first-class scalar spline spaces.
- [x] Add first-class scalar NURBS spaces.
- [x] Add scalar spline and NURBS trace support.
- [x] Add higher-order `H1` trace support.
- [x] Add higher-order `H(div)` trace support.
- [x] Add higher-order `H(curl)` trace support.
- [x] Add per-element order storage and query APIs.
- [x] Add mixed-order scalar DOF numbering.
- [x] Add mixed-order vector DOF numbering.
- [x] Add variable-order prolongation and restriction operators.
- [x] Add mixed-order trace compatibility constraints on fixed meshes.
- [x] Add owner-contiguous MPI mixed-order numbering through the topology API.
- [x] Add first-class compatible-tensor `H(div)` spline / NURBS support on quadrilateral spaces.
- [x] Add first-class compatible-tensor `H(curl)` spline / NURBS support on quadrilateral spaces.
- [x] Implement vector-valued gradient or Jacobian evaluation in the public API.
- [x] Audit helper utilities for low-order-only assumptions.
- [x] Reconcile documentation with the actual completed support surface.
- [x] Run basis, element, space, and assembly validation for each completed phase.
- [ ] Extend vector spline / NURBS support beyond the current quadrilateral compatible-tensor path.

## Definition of Done

This effort is complete when:

- the public FE APIs expose the in-scope higher-order families without requiring generic-basis fallback for common use cases
- scalar and vector higher-order spaces assemble correctly in serial and parallel
- trace spaces work for the supported higher-order scalar and vector families
- variable-order spaces work on fixed meshes
- documentation describes only capabilities that are validated by tests
