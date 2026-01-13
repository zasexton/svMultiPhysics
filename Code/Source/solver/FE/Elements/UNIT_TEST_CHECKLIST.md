# Unit Test Checklist for FE Elements Library

This document lists additional unit tests that should be added to improve coverage and ensure correctness of the Elements subfolder. Tests are organized by priority and component.

**Last Updated:** 2026-01-12
**Status Legend:**
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Coverage Summary

The Elements subfolder contains **24 files** (12 headers, 12 implementations) totaling approximately **2,473 lines** of code. Current test coverage is estimated at **70-75%** with significant gaps in ElementValidator, high-order element topology, and batch processing validation.

### Existing Test Files
| File | Lines | Primary Coverage |
|------|-------|------------------|
| test_ElementFactoryCacheValidator.cpp | 112 | Factory, cache, validator basics |
| test_ReferenceElement.cpp | 148 | ReferenceElement topology |
| test_LagrangeAndDGElements.cpp | 458 | Lagrange/DG assembly |
| test_SpectralElementCollocation.cpp | 184 | Spectral collocation properties |
| test_Order0Elements.cpp | 92 | Constant basis elements |
| test_ElementErrorPaths.cpp | 833 | Error handling across all classes |
| test_InterfaceContinuity.cpp | 651 | H(div)/H(curl) interface continuity |
| test_VectorSpectralAndComposite.cpp | 145 | Vector/spectral/composite basics |
| test_ConvergenceRates.cpp | 230 | Interpolation convergence |

---

## High Priority Tests

### 1. ElementValidator Comprehensive Tests (CRITICAL)

The ElementValidator class has only **1 basic test case** despite being critical for mesh quality assessment. These tests are essential for detecting degenerate elements that cause solver failures.

#### 1.1 `ElementValidator_InvertedQuadrilateralDetection`
- [ ] Not started
- **What it tests**: Validator detects inverted (negative Jacobian) quadrilateral elements
- **Why needed**: Inverted elements are a common mesh error that causes assembly failures. The current test only validates identity-mapped elements. Real meshes frequently contain inverted elements due to mesh generation errors or large deformations.
- **Implementation**: Create quad with vertices in clockwise order (inverted), verify `positive_jacobian == false` and `min_detJ < 0`

#### 1.2 `ElementValidator_InvertedTriangleDetection`
- [ ] Not started
- **What it tests**: Validator detects inverted triangular elements
- **Why needed**: Triangles can be inverted when vertices are ordered incorrectly or during mesh motion in ALE methods
- **Implementation**: Create triangle with clockwise vertex ordering, verify negative Jacobian detection

#### 1.3 `ElementValidator_InvertedTetrahedronDetection`
- [ ] Not started
- **What it tests**: Validator detects inverted tetrahedral elements
- **Why needed**: 3D mesh inversion is common in mesh generation and deforming domain simulations
- **Implementation**: Create tetrahedron with inverted orientation, verify detection

#### 1.4 `ElementValidator_InvertedHexahedronDetection`
- [ ] Not started
- **What it tests**: Validator detects inverted or twisted hexahedral elements
- **Why needed**: Hexahedra can become inverted through node permutation errors or mesh deformation. Twisted hexes (positive at some quadrature points, negative at others) are particularly dangerous.
- **Implementation**: Create hex with inverted corner mapping, verify detection at all quadrature points

#### 1.5 `ElementValidator_DegenerateTriangleDetection`
- [ ] Not started
- **What it tests**: Validator reports very small positive Jacobian for nearly-collapsed triangles
- **Why needed**: Degenerate elements (e.g., triangle with collinear vertices) have detJ approaching zero, causing numerical instability. These should be flagged even though technically not inverted.
- **Implementation**: Create triangle with vertices nearly collinear, verify `min_detJ` is very small (< tolerance)

#### 1.6 `ElementValidator_DegenerateQuadrilateralDetection`
- [ ] Not started
- **What it tests**: Validator detects nearly-collapsed quadrilaterals
- **Why needed**: Quads degenerating to triangles (two vertices coincident) cause singular Jacobians
- **Implementation**: Create quad with two adjacent vertices nearly coincident, verify small detJ

#### 1.7 `ElementValidator_HighAspectRatioConditionNumber`
- [ ] Not started
- **What it tests**: Validator reports high condition number for elongated elements
- **Why needed**: High aspect ratio elements (e.g., 100:1 length ratio) have poor conditioning even when not inverted. The condition number metric in ElementQuality is designed to catch these cases but is currently untested.
- **Implementation**: Create quad with aspect ratio 100:1, verify `max_condition_number > 100`

#### 1.8 `ElementValidator_SkewedElementConditionNumber`
- [ ] Not started
- **What it tests**: Validator reports high condition number for skewed elements
- **Why needed**: Highly skewed elements (small angles approaching 0 or 180 degrees) cause numerical issues in gradient computation
- **Implementation**: Create quad with one angle < 5 degrees, verify high condition number

#### 1.9 `ElementValidator_IdealElementConditionNumber`
- [ ] Not started
- **What it tests**: Validator reports condition number near 1 for ideal elements
- **Why needed**: Establishes baseline - equilateral triangle, unit square, unit cube should have condition numbers very close to 1.0
- **Implementation**: Validate ideal elements, verify `max_condition_number` is within [1.0, 1.1]

#### 1.10 `ElementValidator_WedgeAndPyramidQuality`
- [ ] Not started
- **What it tests**: Validator correctly handles wedge and pyramid element quality
- **Why needed**: Wedge/pyramid elements have mixed topology and non-constant Jacobians; quality metrics must sample correctly
- **Implementation**: Test well-shaped and poorly-shaped wedge/pyramid elements

---

### 2. High-Order Element Topology Tests

High-order elements (Triangle6, Tetra10, Hex27, etc.) are under-represented in current tests. These tests ensure correct DOF counts and edge/face connectivity for high-order variants.

#### 2.1 `ReferenceElement_HighOrderTriangle6EdgeNodes`
- [ ] Not started
- **What it tests**: Triangle6 (quadratic) has correct edge node connectivity
- **Why needed**: Triangle6 has 3 edge midpoint nodes. The current `edge_nodes()` test only validates linear triangles. Incorrect edge connectivity breaks inter-element continuity.
- **Implementation**: Verify `edge_nodes(i)` returns 3 nodes per edge (2 vertices + 1 midpoint) for Triangle6

#### 2.2 `ReferenceElement_HighOrderQuad9EdgeNodes`
- [ ] Not started
- **What it tests**: Quad9 (quadratic) has correct edge node connectivity
- **Why needed**: Quad9 has 4 edge midpoint nodes plus 1 center node. Edge nodes must be correctly identified for assembly.
- **Implementation**: Verify 3 nodes per edge for Quad9

#### 2.3 `ReferenceElement_HighOrderTetra10EdgeNodes`
- [ ] Not started
- **What it tests**: Tetra10 (quadratic) has correct edge node connectivity
- **Why needed**: Tetra10 has 6 edge midpoint nodes. Edge DOFs affect inter-element continuity.
- **Implementation**: Verify 3 nodes per edge (2 vertices + 1 midpoint) for all 6 edges

#### 2.4 `ReferenceElement_HighOrderTetra10FaceNodes`
- [ ] Not started
- **What it tests**: Tetra10 face connectivity includes all 6 nodes per face
- **Why needed**: Each triangular face of Tetra10 has 6 nodes (3 vertices + 3 edge midpoints). Face DOFs are critical for boundary conditions.
- **Implementation**: Verify `face_nodes(i)` returns 6 nodes for each face

#### 2.5 `ReferenceElement_HighOrderHex27EdgeNodes`
- [ ] Not started
- **What it tests**: Hex27 (quadratic) has correct edge node connectivity
- **Why needed**: Hex27 has 12 edges, each with 3 nodes (2 vertices + 1 midpoint = 12 edge nodes total)
- **Implementation**: Verify 3 nodes per edge for all 12 edges

#### 2.6 `ReferenceElement_HighOrderHex27FaceNodes`
- [ ] Not started
- **What it tests**: Hex27 face connectivity includes all 9 nodes per face
- **Why needed**: Each quad face has 9 nodes (4 vertices + 4 edge midpoints + 1 center). Face nodes must be correctly identified for face integrals.
- **Implementation**: Verify 9 nodes per face for all 6 faces

#### 2.7 `ReferenceElement_HighOrderWedge18TopologyConsistency`
- [ ] Not started
- **What it tests**: Wedge18 has consistent edge and face connectivity
- **Why needed**: Wedge18 has mixed topology (triangular top/bottom, quad sides). Node ordering must be consistent.
- **Implementation**: Verify edge nodes and face nodes match expected counts

#### 2.8 `ReferenceElement_HighOrderPyramid14TopologyConsistency`
- [ ] Not started
- **What it tests**: Pyramid14 has correct mixed-topology face connectivity
- **Why needed**: Pyramid14 has 1 quad base (9 nodes) and 4 triangular faces (6 nodes each). Complex topology requires careful validation.
- **Implementation**: Verify face node counts and vertex ordering

#### 2.9 `ReferenceElement_CubicElementNodeCounts`
- [ ] Not started
- **What it tests**: Cubic (order 3) elements have correct total node counts
- **Why needed**: Node count formulas differ by element family: Line4=4, Tri10=10, Quad16=16, Tet20=20, Hex64=64. Incorrect counts cause buffer overflows.
- **Implementation**: Verify `num_nodes()` matches formulas for order-3 elements

#### 2.10 `ReferenceElement_NodeCoordinatesInsideElement`
- [ ] Not started
- **What it tests**: All node coordinates from `NodeOrderingConventions` lie within the reference element
- **Why needed**: Nodes outside the reference element indicate incorrect node placement formulas. This would cause extrapolation instead of interpolation.
- **Implementation**: For all element types and orders 1-4, verify all nodes satisfy element containment constraints

---

### 3. ElementTransform Facet Operation Tests

Facet operations (`facet_vertices`, `facet_to_reference`, `reference_facet_normal`) are only implicitly tested through `compute_facet_frame`. Direct tests are needed to validate these helper functions.

#### 3.1 `ElementTransform_FacetVerticesTriangle`
- [x] Completed
- **What it tests**: `facet_vertices()` returns correct vertices and coordinates for each triangle edge
- **Why needed**: Triangle has 3 edges; each edge should return 2 vertices with correct reference coordinates. This function is used internally by `compute_facet_frame` but never validated directly.
- **Implementation**: For each edge (0, 1, 2), verify returned vertex indices and coordinates match expected values

#### 3.2 `ElementTransform_FacetVerticesQuad`
- [ ] Not started
- **What it tests**: `facet_vertices()` returns correct vertices for each quad edge
- **Why needed**: Quad has 4 edges in [-1,1]^2 reference space. Edge vertex coordinates differ from triangle.
- **Implementation**: Verify vertex indices and coordinates for all 4 edges

#### 3.3 `ElementTransform_FacetVerticesTetrahedron`
- [ ] Not started
- **What it tests**: `facet_vertices()` returns correct vertices for each tetrahedron face
- **Why needed**: Tetrahedron has 4 triangular faces. Face vertex ordering determines outward normal direction.
- **Implementation**: Verify 3 vertices per face with correct reference coordinates

#### 3.4 `ElementTransform_FacetVerticesHexahedron`
- [ ] Not started
- **What it tests**: `facet_vertices()` returns correct vertices for each hexahedron face
- **Why needed**: Hexahedron has 6 quad faces. Face vertex ordering is critical for normal direction consistency.
- **Implementation**: Verify 4 vertices per face with correct reference coordinates

#### 3.5 `ElementTransform_FacetToReferenceTriangleEdges`
- [ ] Not started
- **What it tests**: `facet_to_reference()` correctly maps edge parameter to triangle reference coordinates
- **Why needed**: Given parameter t in [0,1] on an edge, must return correct (x,y) in reference triangle. Used for edge integral evaluation.
- **Implementation**: For each edge, map t=0, 0.5, 1 and verify reference coordinates

#### 3.6 `ElementTransform_FacetToReferenceQuadEdges`
- [ ] Not started
- **What it tests**: `facet_to_reference()` correctly maps edge parameter to quad reference coordinates
- **Why needed**: Quad edges in [-1,1]^2 use different parameterization than triangle edges in [0,1]^2
- **Implementation**: Verify mapping for all 4 edges at multiple parameter values

#### 3.7 `ElementTransform_FacetToReferenceTetrahedronFaces`
- [ ] Not started
- **What it tests**: `facet_to_reference()` maps face barycentric coordinates to tetrahedron reference coordinates
- **Why needed**: Face integration requires mapping 2D face coordinates to 3D element reference space
- **Implementation**: For each face, map barycentric coordinates (1/3, 1/3, 1/3) and verify result

#### 3.8 `ElementTransform_FacetToReferenceHexahedronFaces`
- [ ] Not started
- **What it tests**: `facet_to_reference()` maps face coordinates to hexahedron reference coordinates
- **Why needed**: Hex faces use tensor-product parameterization [-1,1]^2
- **Implementation**: For each face, map (0,0) and corner points, verify 3D reference coordinates

#### 3.9 `ElementTransform_ReferenceFacetNormalTriangle`
- [ ] Not started
- **What it tests**: `reference_facet_normal()` returns correct outward normals for triangle edges
- **Why needed**: Reference normals define the canonical outward direction before mapping to physical space. Incorrect normals cause sign errors in flux calculations.
- **Implementation**: For each edge, verify normal points outward from reference triangle

#### 3.10 `ElementTransform_ReferenceFacetNormalTetrahedron`
- [ ] Not started
- **What it tests**: `reference_facet_normal()` returns correct outward normals for tetrahedron faces
- **Why needed**: 3D face normals are more complex; each face has a unique outward direction
- **Implementation**: For each face, verify normal points outward using dot product with centroid-to-face vector

#### 3.11 `ElementTransform_ReferenceFacetNormalHexahedron`
- [ ] Not started
- **What it tests**: `reference_facet_normal()` returns correct outward normals for hexahedron faces
- **Why needed**: Hex faces have axis-aligned normals in reference space (+/-x, +/-y, +/-z)
- **Implementation**: Verify normals are correct unit vectors for all 6 faces

#### 3.12 `ElementTransform_ReferenceFacetNormalWedgeAndPyramid`
- [ ] Not started
- **What it tests**: `reference_facet_normal()` handles wedge and pyramid mixed topology
- **Why needed**: Wedge has triangular and quad faces; pyramid has triangular faces and quad base
- **Implementation**: Verify all face normals point outward for both element types

---

## Medium Priority Tests

### 4. ElementCache Batch Processing Tests

The `get_batch()` method and `BatchEvaluationHints` are documented but not validated. These tests ensure batch processing works correctly for assembly performance optimization.

#### 4.1 `ElementCache_GetBatchMatchesSingleCalls`
- [ ] Not started
- **What it tests**: `get_batch()` returns identical results to repeated `get()` calls
- **Why needed**: `get_batch()` is designed for performance but must produce identical results to single-element queries. Currently only tested for empty input handling.
- **Implementation**: Create 10 identical elements, compare `get_batch()` results against individual `get()` calls

#### 4.2 `ElementCache_GetBatchMixedElementTypes`
- [ ] Not started
- **What it tests**: `get_batch()` handles batches with different element types
- **Why needed**: Real assemblies may process mixed-element meshes. Batch processing must correctly handle heterogeneous element sets.
- **Implementation**: Batch with Triangle, Quad, Tetra elements; verify each entry matches expected element type

#### 4.3 `ElementCache_GetBatchLargeBatch`
- [ ] Not started
- **What it tests**: `get_batch()` handles large batches (100+ elements) without errors
- **Why needed**: Performance-critical path must handle realistic mesh sizes without memory issues
- **Implementation**: Create batch of 100 identical elements, verify all entries returned correctly

#### 4.4 `ElementCache_BatchHintsPrefetch`
- [ ] Not started
- **What it tests**: Setting `prefetch=true` in BatchEvaluationHints doesn't cause errors
- **Why needed**: Even though prefetch is not yet implemented, the API must accept the hint without crashing
- **Implementation**: Call `get_batch()` with `prefetch=true`, verify results still correct

#### 4.5 `ElementCache_BatchHintsSimdWidth`
- [ ] Not started
- **What it tests**: Different `simd_width` values (1, 4, 8, 16) are accepted without errors
- **Why needed**: Future SIMD optimization will use this hint; API must be stable now
- **Implementation**: Call `get_batch()` with various simd_width values

#### 4.6 `ElementCache_OptimalSimdWidthDetection`
- [ ] Not started
- **What it tests**: `optimal_simd_width()` returns a valid value (1, 4, 8, or 16)
- **Why needed**: CPU feature detection must return a reasonable default for current platform
- **Implementation**: Call `optimal_simd_width()`, verify result is in expected range

#### 4.7 `ElementCache_ThreadSafetyHighContention`
- [ ] Not started
- **What it tests**: Multiple threads (16+) calling `get()` simultaneously don't cause race conditions
- **Why needed**: Assembly is often parallelized; cache must be thread-safe under high contention
- **Implementation**: Spawn 16 threads, each calling `get()` on different elements 100 times

#### 4.8 `ElementCache_ClearDuringAccess`
- [ ] Not started
- **What it tests**: Calling `clear()` while other threads are accessing cache doesn't crash
- **Why needed**: Cache may be cleared during adaptive remeshing while background threads are still assembling
- **Implementation**: One thread calls `clear()` while others call `get()`; verify no crashes

---

### 5. IsogeometricElement Integration Tests

IsogeometricElement is designed to wrap NURBS/B-spline bases but is only tested with Lagrange stand-ins. These tests validate actual IGA workflows.

#### 5.1 `IsogeometricElement_BSplineBasisIntegration`
- [ ] Not started
- **What it tests**: IsogeometricElement correctly wraps a BSplineBasis from the Basis library
- **Why needed**: IGA workflows require B-spline basis functions. The current tests only use Lagrange placeholders, leaving the primary use case untested.
- **Implementation**: Create BSplineBasis, wrap in IsogeometricElement, verify `num_dofs()` and `basis()` work correctly

#### 5.2 `IsogeometricElement_QuadratureCompatibility`
- [ ] Not started
- **What it tests**: IsogeometricElement validates quadrature dimension matches basis dimension
- **Why needed**: Mismatched dimensions cause silent errors in integration. Validation is implemented but not tested.
- **Implementation**: Attempt to create IsogeometricElement with mismatched basis/quadrature dimensions, verify exception

#### 5.3 `IsogeometricElement_FieldTypeValidation`
- [ ] Not started
- **What it tests**: Scalar B-spline basis rejects Vector field type
- **Why needed**: Field type must match basis type; incorrect combinations should throw
- **Implementation**: Create scalar B-spline, attempt to set Vector field type, verify exception

#### 5.4 `IsogeometricElement_HdivContinuityValidation`
- [ ] Not started
- **What it tests**: Vector basis with H(div) continuity is accepted, H1 is rejected
- **Why needed**: IGA vector elements must have correct continuity for mixed methods
- **Implementation**: Test various continuity/basis combinations, verify validation

#### 5.5 `IsogeometricElement_AssemblyIntegration`
- [ ] Not started
- **What it tests**: IsogeometricElement can be used in element-level assembly
- **Why needed**: End-to-end test that IGA elements work with the assembly system
- **Implementation**: Assemble mass matrix using IsogeometricElement with B-spline basis, verify positive-definite result

---

### 6. MixedElement and CompositeElement Assembly Tests

MixedElement and CompositeElement are tested for construction but not for integration with the assembly system. These tests validate block DOF handling.

#### 6.1 `MixedElement_BlockDofLayout`
- [ ] Not started
- **What it tests**: MixedElement `num_dofs()` correctly sums sub-element DOFs
- **Why needed**: Mixed methods (e.g., velocity-pressure) require correct DOF counting for block system setup
- **Implementation**: Create MixedElement with RT1 (velocity) + P0 (pressure), verify total DOF count

#### 6.2 `MixedElement_SubElementAccess`
- [ ] Not started
- **What it tests**: `sub_elements()` returns correct sub-element vector
- **Why needed**: Assembly code must access individual sub-elements for block matrix construction
- **Implementation**: Create MixedElement, verify `sub_elements()` returns expected components

#### 6.3 `MixedElement_FieldIdAssociation`
- [ ] Not started
- **What it tests**: Each sub-element has correct FieldId association
- **Why needed**: FieldId is used to route sub-element contributions to correct matrix blocks
- **Implementation**: Create MixedElement with distinct FieldIds, verify associations preserved

#### 6.4 `MixedElement_AssemblyBlockStructure`
- [ ] Not started
- **What it tests**: MixedElement produces correct block structure in assembled matrix
- **Why needed**: Mixed formulations require saddle-point structure (A, B, B^T, 0 blocks)
- **Implementation**: Assemble divergence-form system, verify block sparsity pattern

#### 6.5 `CompositeElement_EnrichmentCombination`
- [ ] Not started
- **What it tests**: CompositeElement correctly combines base element with bubble enrichment
- **Why needed**: Bubble functions for stabilization require correct DOF counting and basis access
- **Implementation**: Create Lagrange + bubble composite, verify DOF count and component access

#### 6.6 `CompositeElement_XFEMEnrichment`
- [ ] Not started
- **What it tests**: CompositeElement can represent XFEM-style enrichment
- **Why needed**: Extended finite element methods add enrichment functions to capture discontinuities
- **Implementation**: Create base + Heaviside enrichment composite, verify structure

#### 6.7 `CompositeElement_DimensionConsistency`
- [ ] Not started
- **What it tests**: CompositeElement rejects components with mismatched dimensions
- **Why needed**: All components must have same spatial dimension for valid assembly
- **Implementation**: Attempt to combine 2D and 3D elements, verify exception

---

### 7. ElementFactory Extended Coverage Tests

ElementFactory has good coverage but some edge cases are missing.

#### 7.1 `ElementFactory_AllElementTypesAllOrders`
- [ ] Not started
- **What it tests**: Factory creates valid elements for all (ElementType, order) combinations up to order 4
- **Why needed**: Systematic verification that factory handles all supported configurations
- **Implementation**: Loop over all element types and orders 1-4, verify creation succeeds

#### 7.2 `ElementFactory_SpectralElementOrders`
- [ ] Not started
- **What it tests**: Factory creates spectral elements for orders 2-10
- **Why needed**: Spectral methods use high polynomial orders; factory must support them
- **Implementation**: Request spectral elements with orders 2, 5, 10, verify creation

#### 7.3 `ElementFactory_BDMAllOrders`
- [ ] Not started
- **What it tests**: Factory creates BDM elements for orders 1-3 on Triangle, Quad, Tetra, Hex
- **Why needed**: BDM H(div) elements are used in mixed methods; all order/topology combinations should work
- **Implementation**: Create BDM elements with varying orders on all supported element types

#### 7.4 `ElementFactory_NedelecAllOrders`
- [ ] Not started
- **What it tests**: Factory creates Nedelec elements for orders 0-2 on Tetra, Hex
- **Why needed**: Nedelec H(curl) elements are essential for electromagnetics
- **Implementation**: Create Nedelec elements on 3D element types

#### 7.5 `ElementFactory_ErrorMessageQuality`
- [ ] Not started
- **What it tests**: Factory exceptions contain informative error messages
- **Why needed**: Users need clear guidance when requesting invalid combinations
- **Implementation**: Trigger various factory errors, verify message content identifies the problem

---

## Lower Priority Tests

### 8. Convergence and Accuracy Tests

These tests verify numerical accuracy of element operations.

#### 8.1 `LagrangeElement_MassMatrixSymmetry`
- [ ] Not started
- **What it tests**: Assembled mass matrix is symmetric for Lagrange elements
- **Why needed**: Symmetry is a fundamental property; asymmetric mass matrices indicate bugs
- **Implementation**: Assemble mass matrix, verify M == M^T to machine precision

#### 8.2 `LagrangeElement_MassMatrixPositiveDefinite`
- [ ] Not started
- **What it tests**: Assembled mass matrix is positive definite
- **Why needed**: Mass matrices must be positive definite for well-posed problems
- **Implementation**: Compute eigenvalues, verify all positive

#### 8.3 `LagrangeElement_StiffnessMatrixSymmetry`
- [ ] Not started
- **What it tests**: Assembled stiffness matrix is symmetric for Laplacian operator
- **Why needed**: Stiffness matrix symmetry is required for symmetric eigenvalue problems
- **Implementation**: Assemble K, verify K == K^T

#### 8.4 `VectorElement_DivergenceAccuracy`
- [ ] Not started
- **What it tests**: RT/BDM divergence computation matches analytical divergence
- **Why needed**: Divergence accuracy is critical for mass conservation in mixed methods
- **Implementation**: Project polynomial vector field, compute divergence, compare to exact

#### 8.5 `VectorElement_CurlAccuracy`
- [ ] Not started
- **What it tests**: Nedelec curl computation matches analytical curl
- **Why needed**: Curl accuracy is essential for Maxwell equation solvers
- **Implementation**: Project polynomial vector field, compute curl, compare to exact

#### 8.6 `SpectralElement_DiagonalMassMatrixProperty`
- [ ] Not started
- **What it tests**: Spectral element mass matrix is diagonal due to Gauss-Lobatto collocation
- **Why needed**: Diagonal mass matrices enable explicit time-stepping; this is the key spectral element advantage
- **Implementation**: Assemble mass matrix with GLL quadrature, verify diagonal dominance

#### 8.7 `HighOrderElement_ConvergenceRate`
- [ ] Not started
- **What it tests**: Error decreases at rate O(h^{p+1}) for order-p elements
- **Why needed**: Theoretical convergence rate verification for high-order methods
- **Implementation**: Solve Poisson on refined meshes with p=2,3,4, verify convergence slopes

---

### 9. Thread Safety and Performance Tests

#### 9.1 `ElementValidator_ThreadSafety`
- [ ] Not started
- **What it tests**: ElementValidator can be called from multiple threads simultaneously
- **Why needed**: Parallel mesh validation requires thread-safe quality checks
- **Implementation**: Spawn threads validating different elements concurrently

#### 9.2 `ElementTransform_ThreadSafety`
- [ ] Not started
- **What it tests**: ElementTransform static methods are thread-safe
- **Why needed**: Assembly parallelization requires thread-safe gradient transforms
- **Implementation**: Spawn threads calling `gradients_to_physical()` concurrently

#### 9.3 `ReferenceElement_CreationPerformance`
- [ ] Not started
- **What it tests**: `ReferenceElement::create()` is efficient for repeated calls
- **Why needed**: Factory may be called per-element in naive code; should be fast
- **Implementation**: Time 10000 create() calls, verify reasonable performance

---

### 10. Negative and Error Case Tests

#### 10.1 `LagrangeElement_NegativeOrderThrows`
- [ ] Not started
- **What it tests**: LagrangeElement constructor throws for order < 0
- **Why needed**: Clear error for invalid input
- **Implementation**: Attempt construction with order = -1, verify exception

#### 10.2 `VectorElement_InvalidContinuityThrows`
- [ ] Not started
- **What it tests**: VectorElement throws for H1 continuity (should be H_div or H_curl)
- **Why needed**: Vector elements must have appropriate continuity
- **Implementation**: Attempt VectorElement with H1 continuity, verify exception

#### 10.3 `SpectralElement_Order0Throws`
- [ ] Not started
- **What it tests**: SpectralElement throws for order 0 (needs order >= 1 for GLL nodes)
- **Why needed**: Gauss-Lobatto nodes require at least 2 points
- **Implementation**: Attempt SpectralElement with order 0, verify exception

#### 10.4 `MixedElement_NullSubElementThrows`
- [ ] Not started
- **What it tests**: MixedElement throws for null sub-element pointers
- **Why needed**: Null pointers cause crashes during assembly
- **Implementation**: Attempt MixedElement with nullptr, verify exception

#### 10.5 `CompositeElement_EmptyComponentsThrows`
- [ ] Not started
- **What it tests**: CompositeElement throws for empty component vector
- **Why needed**: Empty composite element is meaningless
- **Implementation**: Attempt CompositeElement with empty vector, verify exception

#### 10.6 `ElementFactory_UnknownElementTypeThrows`
- [ ] Not started
- **What it tests**: ElementFactory throws for ElementType::Unknown
- **Why needed**: Unknown type should not silently produce garbage
- **Implementation**: Request element with Unknown type, verify exception with message

#### 10.7 `ReferenceElement_UnknownTypeThrows`
- [ ] Not started
- **What it tests**: `ReferenceElement::create()` throws for Unknown element type
- **Why needed**: Consistent error handling across all factory methods
- **Implementation**: Attempt `create(ElementType::Unknown)`, verify exception

---

## Checklist Summary

### Existing Coverage (Implemented)

| Category | Tests | Status |
|----------|-------|--------|
| ElementFactory basics | 6 | Complete |
| ReferenceElement linear elements | 8 | Complete |
| LagrangeElement assembly | 12 | Complete |
| SpectralElement collocation | 6 | Complete |
| Order-0 elements | 4 | Complete |
| Error paths | 20+ | Complete |
| H(div)/H(curl) interface continuity | 8 | Complete |
| Vector/spectral/composite basics | 6 | Complete |
| Convergence rates | 4 | Complete |
| **Subtotal** | ~74 | |

### Missing Tests (This Checklist)

| Category | Tests | Priority | Rationale |
|----------|-------|----------|-----------|
| **ElementValidator Comprehensive** | 10 | **CRITICAL** | Only 1 existing test; degenerate element detection completely missing |
| **High-Order Element Topology** | 10 | High | Under-tested; breaks assembly for quadratic+ elements |
| **ElementTransform Facet Operations** | 12 | High | Only implicit coverage; direct validation needed |
| **ElementCache Batch Processing** | 8 | Medium | Batch API documented but untested |
| **IsogeometricElement Integration** | 5 | Medium | Primary use case (B-splines) untested |
| **Mixed/Composite Assembly** | 7 | Medium | Block system assembly untested |
| **ElementFactory Extended** | 5 | Medium | Edge cases missing |
| **Convergence/Accuracy** | 7 | Lower | Numerical validation |
| **Thread Safety/Performance** | 3 | Lower | Parallel assembly support |
| **Negative/Error Cases** | 7 | Lower | Robustness |
| **Subtotal** | **74** | | |

### Overall Summary

| Metric | Count |
|--------|-------|
| Existing Tests | ~74 |
| Missing Tests | 74 |
| **Grand Total** | ~148 |
| Current Coverage | ~50% |
| Target Coverage | 90%+ |

### Priority Ranking

1. **CRITICAL**: ElementValidator Comprehensive (10 tests) - Mesh quality detection completely untested
2. **HIGH**: High-Order Element Topology (10 tests) - Required for p-refinement
3. **HIGH**: ElementTransform Facet Operations (12 tests) - Required for boundary conditions
4. **MEDIUM**: ElementCache Batch Processing (8 tests) - Performance API
5. **MEDIUM**: IsogeometricElement Integration (5 tests) - IGA workflows
6. **MEDIUM**: Mixed/Composite Assembly (7 tests) - Mixed methods
7. **MEDIUM**: ElementFactory Extended (5 tests) - Completeness
8. **LOWER**: Convergence/Accuracy (7 tests) - Numerical validation
9. **LOWER**: Thread Safety (3 tests) - Parallel support
10. **LOWER**: Error Cases (7 tests) - Robustness

---

## References

### Core Finite Element References
1. Zienkiewicz, O.C., Taylor, R.L., Zhu, J.Z. "The Finite Element Method: Its Basis and Fundamentals" 7th ed. - Element quality metrics, reference elements
2. Hughes, T.J.R. "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis" - Isoparametric elements, Jacobian conditions
3. Ern, A., Guermond, J.L. "Theory and Practice of Finite Elements" - Mixed methods, inf-sup stability

### Element Quality and Mesh Validation
4. Shewchuk, J.R. "What Is a Good Linear Finite Element?" - Element quality metrics, condition numbers
5. Knupp, P.M. "Algebraic Mesh Quality Metrics" - Jacobian-based quality measures
6. Field, D.A. "Qualitative Measures for Initial Meshes" - Aspect ratio, skewness metrics

### High-Order Elements
7. Szabo, B., Babuska, I. "Finite Element Analysis" - p-FEM, high-order element theory
8. Karniadakis, G., Sherwin, S. "Spectral/hp Element Methods for CFD" 2nd ed. - High-order element implementation

### Isogeometric Analysis
9. Cottrell, J.A., Hughes, T.J.R., Bazilevs, Y. "Isogeometric Analysis: Toward Integration of CAD and FEA" - IGA element formulation
10. Piegl, L., Tiller, W. "The NURBS Book" 2nd ed. - NURBS basis functions

### Mixed Finite Elements
11. Boffi, D., Brezzi, F., Fortin, M. "Mixed Finite Element Methods and Applications" - Mixed element theory, RT/BDM/Nedelec elements
12. Monk, P. "Finite Element Methods for Maxwell's Equations" - H(curl) elements, Nedelec spaces

### VTK Compatibility
13. VTK File Formats Documentation - Node ordering conventions
14. VTK Lagrange Element Ordering - https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/

---

## Implementation Notes

### Testing Degenerate Elements

When implementing ElementValidator tests, create degenerate elements using these patterns:

```cpp
// Inverted quadrilateral (clockwise vertices)
std::vector<math::Vector<Real,3>> inverted_quad = {
    {0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}  // CW instead of CCW
};

// Nearly-collapsed triangle (collinear vertices)
std::vector<math::Vector<Real,3>> degenerate_tri = {
    {0, 0, 0}, {1, 0, 0}, {0.5, 1e-10, 0}  // Third vertex almost on line
};

// High aspect ratio quad
std::vector<math::Vector<Real,3>> elongated_quad = {
    {0, 0, 0}, {100, 0, 0}, {100, 1, 0}, {0, 1, 0}  // 100:1 aspect ratio
};
```

### Testing High-Order Node Ordering

Use `NodeOrderingConventions::get_node_coords()` to obtain expected node positions, then verify `ReferenceElement::edge_nodes()` returns correct subsets:

```cpp
auto coords = basis::NodeOrderingConventions::get_node_coords(ElementType::Triangle6, 2);
auto ref_elem = ReferenceElement::create(ElementType::Triangle6);
auto edge0_nodes = ref_elem.edge_nodes(0);
// Verify edge0_nodes indices map to coords on edge 0
```

### Testing Facet Operations

For `facet_to_reference()` tests, verify round-trip consistency:

```cpp
// Map facet center to reference, then verify it lies on the facet
auto xi = ElementTransform::facet_to_reference(ElementType::Tetra4, face_id, {1.0/3, 1.0/3, 0});
// Verify xi satisfies face equation
```
