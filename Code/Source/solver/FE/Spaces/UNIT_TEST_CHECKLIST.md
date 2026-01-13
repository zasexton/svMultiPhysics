# Unit Test Checklist for FE Spaces Library

This document lists additional unit tests that should be added to improve coverage and ensure correctness of the Spaces subfolder. Tests are organized by priority and component.

**Last Updated:** 2026-01-13
**Status Legend:**
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Coverage Summary

The Spaces subfolder contains **23 header files** and approximately **3,200+ lines** of implementation code. Current test coverage is estimated at **65-70%** with significant gaps in composite space types (Mortar, Enriched, Adaptive, Composite), vector space trace operators, and IsogeometricSpace advanced functionality.

### Existing Test Files
| File | Lines | Primary Coverage |
|------|-------|------------------|
| test_FunctionSpaces.cpp | 339 | H1/L2/HCurl/HDiv/ProductSpace/MixedSpace metadata |
| test_C1Space.cpp | 71 | C1Space cubic interpolation |
| test_TraceSpace.cpp | 279 | TraceSpace face operations |
| test_FaceRestriction.cpp | 438 | FaceRestriction DOF extraction |
| test_OrientationManager.cpp | 400 | Edge/face orientation |
| test_DGOperators.cpp | 268 | DG jump/average operators |
| test_VectorComponentExtractor.cpp | 172 | Vector decomposition |
| test_SpaceInterpolation.cpp | 114 | L2/nodal projection |
| test_SpaceCacheWorkspace.cpp | 124 | Caching utilities |
| test_VectorSpaceTraces.cpp | 88 | Basic trace operators |
| test_VectorSpaceOperators.cpp | 53 | Divergence/curl evaluation |
| test_SpaceCompatibility.cpp | 42 | Inf-sup checks |
| test_MixedSpaceEvaluation.cpp | 76 | MixedSpace component access |
| test_FunctionSpaceGradients.cpp | 47 | Gradient computation |
| test_IsogeometricSpace.cpp | 70 | IGA metadata/constant interp |
| test_SpacesIntegration.cpp | 71 | Integration testing |

**Total Existing Tests:** ~118 test cases in 2,664 lines

---

## High Priority Tests (CRITICAL)

### 1. MortarSpace Comprehensive Tests (CRITICAL - No existing tests)

MortarSpace is a wrapper for interface coupling (mortar methods, Lagrange multipliers) but has **zero dedicated tests**. It only passes through metadata checks in `test_FunctionSpaces.cpp`. All wrapper delegation and interface-specific behavior is untested.

#### 1.1 `MortarSpace_WrapperDelegatesEvaluate`
- [ ] Not started
- **What it tests**: MortarSpace correctly delegates `evaluate()` to the wrapped interface space
- **Why needed**: MortarSpace is a semantic wrapper; its primary job is to forward all operations to the underlying interface space. If delegation is broken, mortar coupling computations will silently produce wrong results. This is the most fundamental operation.
- **Implementation**: Create MortarSpace wrapping H1Space on Triangle3, call evaluate() with known coefficients, verify result matches direct H1Space evaluation

#### 1.2 `MortarSpace_WrapperDelegatesInterpolate`
- [x] Completed
- **What it tests**: MortarSpace correctly delegates `interpolate()` to the wrapped interface space
- **Why needed**: Interpolation is used to project boundary data onto the mortar space. Broken delegation would cause incorrect interface constraints.
- **Implementation**: Create MortarSpace wrapping L2Space, call interpolate() with polynomial function, verify coefficients match direct L2Space interpolation

#### 1.3 `MortarSpace_WrapperDelegatesGradient`
- [ ] Not started
- **What it tests**: MortarSpace correctly delegates `evaluate_gradient()` to the wrapped interface space
- **Why needed**: Gradients on interfaces are needed for flux computations in DG and mortar methods. Incorrect gradient delegation causes wrong interface flux terms.
- **Implementation**: Create MortarSpace wrapping H1Space, call evaluate_gradient(), verify result matches direct call

#### 1.4 `MortarSpace_MetadataMatchesWrappedSpace`
- [ ] Not started
- **What it tests**: All metadata accessors (field_type, continuity, polynomial_order, etc.) match the wrapped space
- **Why needed**: SpaceType should be Mortar while all other metadata should delegate. Mismatched metadata causes assembly errors when the system queries space properties.
- **Implementation**: Create MortarSpace wrapping various space types, verify space_type() == Mortar but all other accessors match wrapped space

#### 1.5 `MortarSpace_InterfaceSpaceAccessor`
- [ ] Not started
- **What it tests**: `interface_space()` and `interface_space_ptr()` return the correct wrapped space
- **Why needed**: External code may need to access the underlying space directly for specialized interface assembly. These accessors must return the actual wrapped space, not a copy or null.
- **Implementation**: Verify interface_space_ptr().get() == original space pointer

#### 1.6 `MortarSpace_NullInterfaceSpaceThrows`
- [ ] Not started
- **What it tests**: Constructor throws for null interface space pointer
- **Why needed**: Null pointer would cause crashes during any operation. Early detection at construction provides clear error message.
- **Implementation**: Attempt MortarSpace(nullptr), verify FEException is thrown

#### 1.7 `MortarSpace_DofCountMatchesInterface`
- [ ] Not started
- **What it tests**: `dofs_per_element()` returns correct count from wrapped space
- **Why needed**: DOF count is used for coefficient vector sizing. Wrong count causes buffer overflows or truncated data.
- **Implementation**: Verify dofs_per_element() matches wrapped space for various interface element types

---

### 2. EnrichedSpace Comprehensive Tests (CRITICAL - No existing tests)

EnrichedSpace implements XFEM-style enrichment (V_enr = V_std + V_enrichment) but has **zero dedicated tests**. The DOF concatenation logic, combined evaluation, and interpolation are all untested.

#### 2.1 `EnrichedSpace_DofCountIsSum`
- [ ] Not started
- **What it tests**: `dofs_per_element()` returns sum of base and enrichment DOF counts
- **Why needed**: EnrichedSpace concatenates DOFs as [base_coeffs, enrichment_coeffs]. Incorrect count causes assembly indexing errors and corrupted solutions.
- **Implementation**: Create EnrichedSpace with H1Space(p=2) base and L2Space(p=1) enrichment, verify dofs_per_element() = base_dofs + enrichment_dofs

#### 2.2 `EnrichedSpace_EvaluateSumsComponents`
- [ ] Not started
- **What it tests**: `evaluate()` returns u_base + u_enrichment for split coefficient vector
- **Why needed**: The enriched solution is u = u_std + u_enr. If the coefficient split or summation is wrong, the enriched solution will be incorrect. This is the core functionality of XFEM.
- **Implementation**: Set coefficients so base gives f1 and enrichment gives f2, verify evaluate() returns f1+f2

#### 2.3 `EnrichedSpace_EvaluateWithZeroEnrichment`
- [ ] Not started
- **What it tests**: With zero enrichment coefficients, evaluate() matches base space alone
- **Why needed**: The common case is enrichment only active near discontinuities; elsewhere enrichment coefficients are zero. Result should match standard FEM.
- **Implementation**: Set enrichment coefficients to zero, verify result equals base space evaluation

#### 2.4 `EnrichedSpace_EvaluateWithZeroBase`
- [ ] Not started
- **What it tests**: With zero base coefficients, evaluate() matches enrichment space alone
- **Why needed**: Validates the enrichment term is correctly added when base contribution is zero.
- **Implementation**: Set base coefficients to zero, verify result equals enrichment space evaluation

#### 2.5 `EnrichedSpace_InterpolateProjectsToBase`
- [ ] Not started
- **What it tests**: `interpolate()` places coefficients in base space and zeros enrichment
- **Why needed**: Default interpolation should use the standard space; enrichment coefficients are determined by separate XFEM logic (e.g., level set intersection). Interpolate() should not activate enrichment.
- **Implementation**: Call interpolate() with smooth function, verify base coefficients are set and enrichment coefficients are zero

#### 2.6 `EnrichedSpace_PolynomialOrderIsMax`
- [ ] Not started
- **What it tests**: `polynomial_order()` returns max of base and enrichment orders
- **Why needed**: The effective polynomial order determines quadrature requirements. Must be the maximum to ensure sufficient integration accuracy.
- **Implementation**: Create EnrichedSpace with p=2 base and p=3 enrichment, verify polynomial_order() == 3

#### 2.7 `EnrichedSpace_MetadataFromBase`
- [ ] Not started
- **What it tests**: field_type, continuity, element_type come from base space
- **Why needed**: The base space defines the primary field properties. Enrichment modifies approximation capability but not field semantics.
- **Implementation**: Verify field_type(), continuity(), element_type() all match base space

#### 2.8 `EnrichedSpace_MismatchedElementTypesThrow`
- [ ] Not started
- **What it tests**: Constructor throws when base and enrichment have different element types
- **Why needed**: Base and enrichment must live on the same reference element. Mismatched element types indicate configuration error.
- **Implementation**: Attempt EnrichedSpace(H1Space(Quad4), H1Space(Triangle3)), verify exception

#### 2.9 `EnrichedSpace_MismatchedFieldTypesThrow`
- [ ] Not started
- **What it tests**: Constructor throws when base and enrichment have different field types
- **Why needed**: Cannot add scalar base to vector enrichment. Must have compatible field types for summation.
- **Implementation**: Attempt EnrichedSpace(H1Space scalar, HDivSpace vector), verify exception

#### 2.10 `EnrichedSpace_CoefficientVectorSizeValidation`
- [ ] Not started
- **What it tests**: evaluate() throws for coefficient vector with wrong size
- **Why needed**: Coefficient vector must have exactly dofs_per_element() entries. Wrong size would cause out-of-bounds access or incorrect splitting.
- **Implementation**: Call evaluate() with undersized coefficient vector, verify exception with informative message

---

### 3. AdaptiveSpace Comprehensive Tests (CRITICAL - No existing tests)

AdaptiveSpace manages multiple polynomial order levels for hp-adaptivity but has **zero dedicated tests**. Level management, active level switching, and DOF consistency are all untested.

#### 3.1 `AdaptiveSpace_AddLevelIncreasesCount`
- [ ] Not started
- **What it tests**: `add_level()` correctly adds levels and `num_levels()` returns correct count
- **Why needed**: Level management is the core functionality. If levels aren't stored correctly, hp-adaptivity cannot work.
- **Implementation**: Create AdaptiveSpace, add 3 levels, verify num_levels() == 3

#### 3.2 `AdaptiveSpace_SetActiveLevelByIndex`
- [ ] Not started
- **What it tests**: `set_active_level()` changes active level and `active_level()` returns correct level
- **Why needed**: During hp-refinement, the active level changes per-element. Incorrect level switching would use wrong polynomial order.
- **Implementation**: Add multiple levels, switch active index, verify active_level().order matches expected

#### 3.3 `AdaptiveSpace_SetActiveLevelByOrder`
- [ ] Not started
- **What it tests**: `set_active_level_by_order()` finds level with matching polynomial order
- **Why needed**: External code often specifies desired order, not index. Must find correct level by order value.
- **Implementation**: Add levels with orders [1, 2, 4], call set_active_level_by_order(2), verify active level has order 2

#### 3.4 `AdaptiveSpace_SetActiveLevelByOrderThrowsNotFound`
- [ ] Not started
- **What it tests**: `set_active_level_by_order()` throws when requested order doesn't exist
- **Why needed**: Requesting non-existent order is a configuration error. Clear exception message helps debugging.
- **Implementation**: Add levels with orders [1, 2], call set_active_level_by_order(3), verify exception

#### 3.5 `AdaptiveSpace_DelegatesToActiveLevel`
- [ ] Not started
- **What it tests**: All FunctionSpace methods delegate to currently active level
- **Why needed**: After switching levels, all operations (evaluate, interpolate, dofs_per_element) must use the new level's space. Incorrect delegation would produce wrong results.
- **Implementation**: Switch between levels, verify evaluate() produces results matching each level's space

#### 3.6 `AdaptiveSpace_DofsPerElementChangesWithLevel`
- [ ] Not started
- **What it tests**: `dofs_per_element()` returns correct count for active level
- **Why needed**: Higher polynomial orders have more DOFs. Count must update when level changes for correct assembly.
- **Implementation**: Create levels with p=1 (3 DOFs) and p=2 (6 DOFs) on Triangle, switch levels, verify dofs_per_element() changes

#### 3.7 `AdaptiveSpace_PolynomialOrderMatchesActiveLevel`
- [ ] Not started
- **What it tests**: `polynomial_order()` returns order of active level
- **Why needed**: Polynomial order is used for quadrature selection. Must match active level for correct integration.
- **Implementation**: Switch between levels with different orders, verify polynomial_order() updates

#### 3.8 `AdaptiveSpace_EvaluateUsesActiveLevel`
- [ ] Not started
- **What it tests**: `evaluate()` uses basis functions from active level
- **Why needed**: Evaluation must use the correct polynomial basis. Using wrong level's basis would produce completely wrong field values.
- **Implementation**: Create levels with linear and quadratic spaces, switch levels, verify evaluate() matches expected space

#### 3.9 `AdaptiveSpace_InterpolateUsesActiveLevel`
- [ ] Not started
- **What it tests**: `interpolate()` uses DOF structure from active level
- **Why needed**: Interpolation must produce coefficients for the active level's DOF layout. Wrong layout would be incompatible with assembly.
- **Implementation**: Switch to quadratic level, interpolate function, verify coefficient count matches quadratic DOFs

#### 3.10 `AdaptiveSpace_EmptySpaceThrowsOnAccess`
- [ ] Not started
- **What it tests**: Accessing active_level() on empty AdaptiveSpace throws
- **Why needed**: No levels added means no valid active level. Clear error prevents null pointer access.
- **Implementation**: Create empty AdaptiveSpace, call active_level(), verify exception

#### 3.11 `AdaptiveSpace_InvalidIndexThrows`
- [ ] Not started
- **What it tests**: `set_active_level()` throws for out-of-bounds index
- **Why needed**: Index must be valid. Out-of-bounds access would cause undefined behavior.
- **Implementation**: Add 2 levels, call set_active_level(5), verify exception

---

### 4. CompositeSpace Comprehensive Tests (CRITICAL - No existing tests)

CompositeSpace manages region-dependent function spaces but has **zero dedicated tests**. Region registration, lookup, and multi-region handling are all untested.

#### 4.1 `CompositeSpace_AddRegionIncreasesCount`
- [ ] Not started
- **What it tests**: `add_region()` registers spaces and `num_regions()` returns correct count
- **Why needed**: Region registration is the core functionality for multi-material problems. Must correctly store region-space associations.
- **Implementation**: Create CompositeSpace, add 3 regions, verify num_regions() == 3

#### 4.2 `CompositeSpace_RegionAccessByIndex`
- [ ] Not started
- **What it tests**: `region()` returns correct RegionSpace descriptor by index
- **Why needed**: External code may iterate over regions. Each region must return correct id and space pointer.
- **Implementation**: Add regions with ids [10, 20, 30], verify region(i).region_id and region(i).space match expected

#### 4.3 `CompositeSpace_SpaceForRegionLookup`
- [ ] Not started
- **What it tests**: `space_for_region()` finds space by region id
- **Why needed**: During assembly, elements query space by their region id. Lookup must return correct space for that region.
- **Implementation**: Add regions with different spaces, verify space_for_region(id) returns correct space

#### 4.4 `CompositeSpace_SpaceForRegionThrowsNotFound`
- [ ] Not started
- **What it tests**: `space_for_region()` throws for unregistered region id
- **Why needed**: Querying non-existent region is a mesh/setup error. Clear exception helps identify misconfigured regions.
- **Implementation**: Add regions [1, 2], call space_for_region(99), verify exception

#### 4.5 `CompositeSpace_TrySpaceForRegionReturnsNull`
- [ ] Not started
- **What it tests**: `try_space_for_region()` returns nullptr for unregistered region
- **Why needed**: Non-throwing lookup is useful for optional region handling. Must return nullptr, not garbage pointer.
- **Implementation**: Verify try_space_for_region(99) returns nullptr when region 99 not registered

#### 4.6 `CompositeSpace_TrySpaceForRegionReturnsValid`
- [ ] Not started
- **What it tests**: `try_space_for_region()` returns valid pointer for registered region
- **Why needed**: Must return actual space pointer when region exists, enabling optional lookup pattern.
- **Implementation**: Add region 42 with H1Space, verify try_space_for_region(42) returns non-null and matches

#### 4.7 `CompositeSpace_MetadataFromFirstRegion`
- [ ] Not started
- **What it tests**: Default metadata (field_type, continuity, etc.) comes from first registered region
- **Why needed**: CompositeSpace must provide FunctionSpace interface. Using first region's metadata is documented behavior.
- **Implementation**: Add H1Space as first region, L2Space as second, verify field_type() matches H1Space

#### 4.8 `CompositeSpace_EvaluateThrowsNotImplemented`
- [ ] Not started
- **What it tests**: `evaluate()` throws NotImplementedException as documented
- **Why needed**: CompositeSpace cannot evaluate without knowing which region to use. Exception clarifies this limitation.
- **Implementation**: Call evaluate(), verify NotImplementedException with informative message

#### 4.9 `CompositeSpace_EmptySpaceMetadataThrows`
- [ ] Not started
- **What it tests**: Accessing metadata on empty CompositeSpace throws
- **Why needed**: No regions means no valid metadata. Prevents undefined behavior from accessing empty vector.
- **Implementation**: Create empty CompositeSpace, call field_type(), verify exception

#### 4.10 `CompositeSpace_DuplicateRegionIdHandling`
- [ ] Not started
- **What it tests**: Adding same region id twice either throws or overwrites (document behavior)
- **Why needed**: Duplicate region ids are likely errors. Behavior should be consistent and documented.
- **Implementation**: Add region 1 twice with different spaces, verify either exception or overwrite behavior

---

### 5. HCurlSpace Trace and Orientation Tests (HIGH - Limited existing tests)

HCurlSpace has basic tests but `tangential_trace()`, `apply_edge_orientation()`, and `apply_face_orientation()` have minimal coverage.

#### 5.1 `HCurlSpace_TangentialTraceOrthogonalToNormal`
- [ ] Not started
- **What it tests**: `tangential_trace()` output is orthogonal to face normal at all points
- **Why needed**: Tangential trace is defined as n x (v x n), which must be perpendicular to n. Non-orthogonal result indicates incorrect computation.
- **Implementation**: Compute tangential trace, verify dot product with normal is zero (within tolerance)

#### 5.2 `HCurlSpace_TangentialTraceContinuityAcrossFace`
- [ ] Not started
- **What it tests**: Tangential trace is continuous when evaluated from two adjacent elements sharing a face
- **Why needed**: H(curl) conformity requires continuous tangential components. This is the defining property that makes Nedelec elements work for Maxwell equations.
- **Implementation**: Create two adjacent tetrahedra, evaluate tangential_trace() at shared face from both sides, verify values match

#### 5.3 `HCurlSpace_TangentialTraceOnTriangleFace`
- [ ] Not started
- **What it tests**: `tangential_trace()` works correctly on triangular faces of tetrahedra
- **Why needed**: Triangular faces have specific parameterization. Must verify 2D tangent plane projection is correct.
- **Implementation**: Evaluate on all 4 faces of tetrahedron, verify results are 2D vectors in face tangent plane

#### 5.4 `HCurlSpace_TangentialTraceOnQuadFace`
- [ ] Not started
- **What it tests**: `tangential_trace()` works correctly on quadrilateral faces of hexahedra
- **Why needed**: Quad faces have different structure than triangles. Tensor-product parameterization affects trace computation.
- **Implementation**: Evaluate on all 6 faces of hexahedron, verify consistency

#### 5.5 `HCurlSpace_ApplyEdgeOrientationSign`
- [ ] Not started
- **What it tests**: `apply_edge_orientation()` flips sign for negative orientation
- **Why needed**: Edge DOFs must be oriented consistently across elements. Negative orientation means edge direction is reversed relative to local element.
- **Implementation**: Apply orientation with Sign::Negative, verify all DOF values are negated

#### 5.6 `HCurlSpace_ApplyEdgeOrientationIdentity`
- [ ] Not started
- **What it tests**: `apply_edge_orientation()` with Sign::Positive leaves DOFs unchanged
- **Why needed**: Positive orientation means edge direction matches local element; no modification needed.
- **Implementation**: Apply orientation with Sign::Positive, verify DOF values unchanged

#### 5.7 `HCurlSpace_ApplyFaceOrientationPermutation`
- [ ] Not started
- **What it tests**: `apply_face_orientation()` correctly permutes face DOFs for rotated faces
- **Why needed**: Higher-order Nedelec elements have face-interior DOFs that must be permuted when face orientation differs between adjacent elements.
- **Implementation**: Apply various face rotations, verify DOF permutation matches expected

#### 5.8 `HCurlSpace_ApplyFaceOrientationReflection`
- [ ] Not started
- **What it tests**: `apply_face_orientation()` handles reflected (parity-reversed) faces
- **Why needed**: Face reflection changes DOF ordering and may flip signs. Essential for conforming assembly.
- **Implementation**: Apply reflection orientation, verify DOF transformation

#### 5.9 `HCurlSpace_HigherOrderEdgeDofOrientation`
- [ ] Not started
- **What it tests**: Edge DOF orientation works for higher-order (p > 1) elements
- **Why needed**: Higher-order edges have multiple DOFs with different orientation transformation rules.
- **Implementation**: Test orientation on order-3 Nedelec element edges

---

### 6. HDivSpace Trace and Orientation Tests (HIGH - Limited existing tests)

HDivSpace has basic tests but `normal_trace()` and `apply_face_orientation()` have minimal coverage.

#### 6.1 `HDivSpace_NormalTraceIsScalar`
- [ ] Not started
- **What it tests**: `normal_trace()` returns scalar values (v dot n) at evaluation points
- **Why needed**: Normal trace projects vector field onto normal direction, producing scalar flux. Result must be scalar, not vector.
- **Implementation**: Verify returned values are scalars with correct count matching evaluation points

#### 6.2 `HDivSpace_NormalTraceContinuityAcrossFace`
- [ ] Not started
- **What it tests**: Normal trace is continuous when evaluated from two adjacent elements sharing a face
- **Why needed**: H(div) conformity requires continuous normal components. This ensures mass conservation in mixed methods.
- **Implementation**: Create two adjacent elements, evaluate normal_trace() at shared face from both sides, verify values match

#### 6.3 `HDivSpace_NormalTraceMatchesDofValues`
- [ ] Not started
- **What it tests**: Normal trace at face DOF points matches corresponding DOF values (Kronecker property)
- **Why needed**: RT/BDM DOFs are defined as normal flux moments. The trace should reproduce these DOF definitions.
- **Implementation**: Evaluate normal_trace at DOF points, verify matches DOF values

#### 6.4 `HDivSpace_NormalTraceOnTriangleFace`
- [ ] Not started
- **What it tests**: `normal_trace()` works correctly on triangular faces
- **Why needed**: Triangle faces of tetrahedra are common in 3D mixed methods. Must verify correct normal projection.
- **Implementation**: Evaluate on triangular faces, verify scalar results

#### 6.5 `HDivSpace_NormalTraceOnQuadFace`
- [ ] Not started
- **What it tests**: `normal_trace()` works correctly on quadrilateral faces
- **Why needed**: Hex elements have quad faces with different integration structure.
- **Implementation**: Evaluate on all 6 hex faces, verify consistency

#### 6.6 `HDivSpace_ApplyFaceOrientationSign`
- [ ] Not started
- **What it tests**: `apply_face_orientation()` flips sign for reversed face normal
- **Why needed**: When face normal points opposite to element's outward normal, flux sign must be reversed.
- **Implementation**: Apply orientation with parity=true (flipped), verify DOF signs change

#### 6.7 `HDivSpace_ApplyFaceOrientationPermutation`
- [ ] Not started
- **What it tests**: `apply_face_orientation()` permutes DOFs for rotated faces
- **Why needed**: Higher-order H(div) elements have multiple face DOFs that must be reordered when face is rotated.
- **Implementation**: Apply various face rotations, verify DOF permutation

#### 6.8 `HDivSpace_ApplyEdgeOrientationIn2D`
- [ ] Not started
- **What it tests**: `apply_edge_orientation()` works for 2D RT elements
- **Why needed**: In 2D, "faces" are edges. Must verify edge orientation handling for RT on triangles/quads.
- **Implementation**: Test orientation on RT_1 triangle edges

#### 6.9 `HDivSpace_HigherOrderFaceDofOrientation`
- [ ] Not started
- **What it tests**: Face DOF orientation works for higher-order (k > 1) RT/BDM elements
- **Why needed**: Higher-order face DOFs have polynomial moments requiring sophisticated permutation.
- **Implementation**: Test on RT_2 and BDM_2 face DOFs

---

### 7. IsogeometricSpace Extended Tests (HIGH - Basic tests only)

IsogeometricSpace has only 70 lines of tests covering metadata and constant interpolation. B-spline integration, NURBS evaluation, and multi-patch handling are untested.

#### 7.1 `IsogeometricSpace_BSplineBasisIntegration`
- [ ] Not started
- **What it tests**: IsogeometricSpace correctly wraps BSplineBasis from Basis library
- **Why needed**: The primary purpose of IsogeometricSpace is to enable B-spline FEM. Current tests use Lagrange stand-ins.
- **Implementation**: Create BSplineBasis with proper knot vector, wrap in IsogeometricSpace, verify basic operations work

#### 7.2 `IsogeometricSpace_PartitionOfUnity`
- [ ] Not started
- **What it tests**: B-spline basis in IsogeometricSpace satisfies partition of unity
- **Why needed**: Sum of B-spline basis functions must equal 1 at any point. This is fundamental to IGA.
- **Implementation**: Evaluate all basis functions at random points, verify sum equals 1

#### 7.3 `IsogeometricSpace_LinearInterpolation`
- [ ] Not started
- **What it tests**: Linear function interpolation is exact
- **Why needed**: B-splines can exactly reproduce linear functions. This validates basic interpolation accuracy.
- **Implementation**: Interpolate f(x)=x, verify evaluate() matches f at random points

#### 7.4 `IsogeometricSpace_QuadraticInterpolation`
- [ ] Not started
- **What it tests**: Quadratic function interpolation is exact for p>=2 B-splines
- **Why needed**: Degree-p B-splines reproduce polynomials up to degree p. Essential for IGA convergence.
- **Implementation**: Create p=2 B-spline space, interpolate f(x)=x^2, verify exactness

#### 7.5 `IsogeometricSpace_GradientComputation`
- [ ] Not started
- **What it tests**: `evaluate_gradient()` produces correct derivatives for B-spline field
- **Why needed**: Gradients are essential for stiffness matrix assembly. Must work with B-spline basis.
- **Implementation**: Interpolate known polynomial, verify gradient matches analytical derivative

#### 7.6 `IsogeometricSpace_VectorFieldType`
- [ ] Not started
- **What it tests**: IsogeometricSpace with FieldType::Vector produces vector-valued fields
- **Why needed**: IGA for elasticity and fluid mechanics needs vector fields. Must verify multi-component handling.
- **Implementation**: Create vector IsogeometricSpace, verify value_dimension() and evaluation

#### 7.7 `IsogeometricSpace_QuadratureDimensionValidation`
- [ ] Not started
- **What it tests**: Constructor throws for mismatched basis/quadrature dimensions
- **Why needed**: Quadrature dimension must match basis dimension. Mismatch would cause integration errors.
- **Implementation**: Create 2D basis with 3D quadrature, verify exception

#### 7.8 `IsogeometricSpace_HighOrderBSpline`
- [ ] Not started
- **What it tests**: High-order (p=4,5) B-spline basis works correctly
- **Why needed**: IGA often uses high polynomial orders for smooth solutions. Must verify numerical stability.
- **Implementation**: Create p=5 B-spline IsogeometricSpace, verify partition of unity and interpolation

---

## Medium Priority Tests

### 8. MixedSpace Extended Tests (Currently 2 test cases)

MixedSpace has basic component access tests but evaluation, interpolation, and multi-component assembly are undertested.

#### 8.1 `MixedSpace_ComponentInterpolation`
- [ ] Not started
- **What it tests**: Each component can be interpolated independently
- **Why needed**: For velocity-pressure pairs, velocity and pressure are interpolated separately. Must verify component isolation.
- **Implementation**: Create MixedSpace with 2 components, interpolate function into each, verify coefficients placed correctly

#### 8.2 `MixedSpace_EvaluateComponentsReturnsAllValues`
- [ ] Not started
- **What it tests**: `evaluate_components()` returns values from all components at given point
- **Why needed**: Sometimes need all field values at once (e.g., for post-processing). Must aggregate correctly.
- **Implementation**: Set up MixedSpace, evaluate_components(), verify all component values present

#### 8.3 `MixedSpace_ComponentOffsetCorrectness`
- [ ] Not started
- **What it tests**: `component_offset()` returns correct DOF offset for each component
- **Why needed**: Assembly needs to know where each component's DOFs start in the element vector. Wrong offset causes assembly errors.
- **Implementation**: Verify offset(0)==0, offset(1)==component(0).dofs, etc.

#### 8.4 `MixedSpace_AddComponentIncreasesCount`
- [ ] Not started
- **What it tests**: `add_component()` and `num_components()` work correctly
- **Why needed**: Dynamic component addition is needed for flexible mixed formulation setup.
- **Implementation**: Create MixedSpace, add 3 components, verify num_components() == 3

#### 8.5 `MixedSpace_InfSupStablePairs`
- [ ] Not started
- **What it tests**: Common inf-sup stable pairs (RT/P0, P2/P1) can be created
- **Why needed**: These pairs are fundamental for incompressible flow. Must verify they can be assembled correctly.
- **Implementation**: Create RT_0/P_0 MixedSpace for Darcy, verify component properties

#### 8.6 `MixedSpace_TotalDofsPerElement`
- [ ] Not started
- **What it tests**: `dofs_per_element()` returns sum of all component DOFs
- **Why needed**: Total DOF count determines element matrix size. Must be accurate.
- **Implementation**: Create MixedSpace with known component DOF counts, verify total

---

### 9. TraceSpace Extended Tests (Missing some face types)

TraceSpace has good coverage but wedge/pyramid faces, curved element traces, and some edge cases are untested.

#### 9.1 `TraceSpace_WedgeTriangularFaces`
- [ ] Not started
- **What it tests**: TraceSpace correctly handles triangular faces of wedge elements
- **Why needed**: Wedge elements have mixed topology (2 triangular + 3 quad faces). Must verify trace on triangular faces.
- **Implementation**: Create TraceSpace for wedge triangular face, verify DOF count and evaluation

#### 9.2 `TraceSpace_WedgeQuadrilateralFaces`
- [ ] Not started
- **What it tests**: TraceSpace correctly handles quadrilateral faces of wedge elements
- **Why needed**: Wedge quad faces have different parameterization than triangular faces.
- **Implementation**: Create TraceSpace for wedge quad face, verify operations

#### 9.3 `TraceSpace_PyramidQuadBase`
- [ ] Not started
- **What it tests**: TraceSpace correctly handles quadrilateral base of pyramid
- **Why needed**: Pyramid base is a quad; must verify trace space setup and evaluation.
- **Implementation**: Create TraceSpace for pyramid face 0 (base), verify quad trace

#### 9.4 `TraceSpace_PyramidTriangularFaces`
- [ ] Not started
- **What it tests**: TraceSpace handles the 4 triangular faces of pyramid
- **Why needed**: Pyramid triangular faces converge at apex with special geometry.
- **Implementation**: Create TraceSpace for pyramid triangular faces, verify operations

#### 9.5 `TraceSpace_HigherOrderFaces`
- [ ] Not started
- **What it tests**: TraceSpace works for quadratic (p=2) face traces
- **Why needed**: Higher-order methods need accurate trace on curved faces.
- **Implementation**: Create TraceSpace from Triangle6/Quad9 face, verify DOF count

#### 9.6 `TraceSpace_EmbedFacePointAccuracy`
- [ ] Not started
- **What it tests**: `embed_face_point()` maps face coordinates to volume reference coordinates accurately
- **Why needed**: Face-to-volume coordinate mapping is essential for boundary condition application.
- **Implementation**: Map face centroid to volume, verify result lies on face

#### 9.7 `TraceSpace_LiftOperatorConsistency`
- [ ] Not started
- **What it tests**: `lift()` and `restrict()` are inverse operations
- **Why needed**: Lift extends face DOFs to volume; restrict extracts face DOFs. Must be consistent.
- **Implementation**: Verify lift(restrict(v)) projects v onto face DOF space

---

### 10. SpaceInterpolation Extended Tests (Limited coverage)

SpaceInterpolation has tests for basic projections but vector field transfer, p-refinement accuracy, and conservative interpolation are undertested.

#### 10.1 `SpaceInterpolation_VectorFieldL2Projection`
- [ ] Not started
- **What it tests**: L2 projection works for vector-valued spaces
- **Why needed**: Vector fields (velocity, stress) need projection between spaces.
- **Implementation**: Project vector field between ProductSpace instances, verify accuracy

#### 10.2 `SpaceInterpolation_HighToLowOrderProjection`
- [ ] Not started
- **What it tests**: L2 projection from p=4 to p=2 produces best L2 approximation
- **Why needed**: p-coarsening in hp-adaptivity requires accurate downward projection.
- **Implementation**: Create p=4 polynomial, project to p=2, verify L2 error is minimal

#### 10.3 `SpaceInterpolation_LowToHighOrderProjection`
- [ ] Not started
- **What it tests**: L2 projection from p=2 to p=4 is exact for polynomials
- **Why needed**: p-refinement must exactly reproduce lower-order solution. No accuracy loss.
- **Implementation**: Create p=2 polynomial, project to p=4, verify exactness

#### 10.4 `SpaceInterpolation_ConservativeInterpolationMassPreservation`
- [ ] Not started
- **What it tests**: `conservative_interpolation()` preserves total mass
- **Why needed**: For conservation laws, interpolation must preserve integrals. This is the defining property.
- **Implementation**: Compute integral before and after conservative interpolation, verify equality

#### 10.5 `SpaceInterpolation_NodalInterpolationExactness`
- [ ] Not started
- **What it tests**: `nodal_interpolation()` is exact at DOF points
- **Why needed**: Nodal interpolation should match function values exactly at nodes.
- **Implementation**: Interpolate function, verify values at nodes match

#### 10.6 `SpaceInterpolation_H1ToL2Projection`
- [ ] Not started
- **What it tests**: L2 projection from H1 space to L2 space works correctly
- **Why needed**: Mixed methods may need to project continuous fields to discontinuous spaces.
- **Implementation**: Create H1 field, project to L2, verify L2 optimality

---

### 11. SpaceCompatibility Extended Tests (Limited inf-sup coverage)

SpaceCompatibility has basic Stokes pair testing but other inf-sup stable pairs and conformity checks are undertested.

#### 11.1 `SpaceCompatibility_DarcyPair`
- [ ] Not started
- **What it tests**: RT_k / P_k pair is inf-sup stable for Darcy flow
- **Why needed**: Darcy flow requires H(div) velocity with discontinuous pressure. Must verify compatibility.
- **Implementation**: check_inf_sup() for RT_0/P_0 and RT_1/P_1 pairs

#### 11.2 `SpaceCompatibility_TaylorHoodPair`
- [ ] Not started
- **What it tests**: P_k / P_{k-1} Taylor-Hood pair is inf-sup stable
- **Why needed**: Taylor-Hood is the most common Stokes pair. Must verify.
- **Implementation**: check_inf_sup() for P2/P1 and P3/P2 pairs

#### 11.3 `SpaceCompatibility_UnstablePairDetection`
- [ ] Not started
- **What it tests**: check_inf_sup() returns false for unstable pairs like P1/P1
- **Why needed**: Must correctly identify spurious pressure modes. P1/P1 is the classic unstable pair.
- **Implementation**: Verify check_inf_sup() returns false for P1/P1 velocity-pressure

#### 11.4 `SpaceCompatibility_H1Conformity`
- [ ] Not started
- **What it tests**: `check_conformity()` validates H1 spaces have correct continuity
- **Why needed**: Conformity checks ensure spaces satisfy required regularity.
- **Implementation**: Verify H1Space passes H1 conformity check

#### 11.5 `SpaceCompatibility_HdivConformity`
- [ ] Not started
- **What it tests**: `check_conformity()` validates HDivSpace has normal continuity
- **Why needed**: H(div) conformity requires continuous normal components.
- **Implementation**: Verify HDivSpace passes H(div) conformity check

#### 11.6 `SpaceCompatibility_HcurlConformity`
- [ ] Not started
- **What it tests**: `check_conformity()` validates HCurlSpace has tangential continuity
- **Why needed**: H(curl) conformity requires continuous tangential components.
- **Implementation**: Verify HCurlSpace passes H(curl) conformity check

---

### 12. C1Space Extended Tests (Only 4 existing tests)

C1Space has basic cubic interpolation tests but higher-order Hermite, inter-element continuity, and 2D/3D extensions are untested.

#### 12.1 `C1Space_QuinticHermiteInterpolation`
- [ ] Not started
- **What it tests**: C1Space with order=5 (quintic Hermite) interpolates correctly
- **Why needed**: Quintic Hermite includes second derivative DOFs for C2 continuity. Tests higher-order C1 framework.
- **Implementation**: Create order-5 C1Space (if supported), verify interpolation of degree-5 polynomial

#### 12.2 `C1Space_DerivativeDofInterpretation`
- [ ] Not started
- **What it tests**: Derivative DOFs correctly capture slope at nodes
- **Why needed**: Hermite DOFs are [value, derivative] pairs. Derivative DOF must match function slope.
- **Implementation**: Interpolate linear function, verify derivative DOF equals slope

#### 12.3 `C1Space_C1ContinuityAcrossElements`
- [ ] Not started
- **What it tests**: Two adjacent C1 elements share C1 continuity at interface
- **Why needed**: The entire purpose of Hermite bases is inter-element smoothness. Must verify in multi-element context.
- **Implementation**: Create two adjacent Line elements, set shared DOFs, verify C1 continuity at interface

#### 12.4 `C1Space_2DQuadHermite`
- [ ] Not started
- **What it tests**: C1Space on Quad4 uses bicubic Hermite with mixed derivative DOFs
- **Why needed**: 2D Hermite has 4 DOFs per node: value, d/dx, d/dy, d^2/dxdy. Must verify 16-DOF element.
- **Implementation**: Create C1Space on Quad4, verify 16 DOFs per element

#### 12.5 `C1Space_2DMixedDerivativeDof`
- [ ] Not started
- **What it tests**: Mixed derivative DOF (d^2f/dxdy) is correctly handled in 2D
- **Why needed**: Mixed derivatives are the cross-term DOFs in 2D Hermite. Critical for smooth 2D interpolation.
- **Implementation**: Set mixed derivative DOF, verify contribution to interpolant

---

## Lower Priority Tests

### 13. ProductSpace Extended Tests

#### 13.1 `ProductSpace_VectorFieldInterpolation`
- [ ] Not started
- **What it tests**: ProductSpace correctly interpolates multi-component vector fields
- **Why needed**: ProductSpace creates vector fields from scalar bases. Must verify component handling.
- **Implementation**: Interpolate (x, y) vector field on 2D ProductSpace, verify accuracy

#### 13.2 `ProductSpace_GradientOfVectorField`
- [ ] Not started
- **What it tests**: Gradient computation for vector ProductSpace (returns matrix/tensor)
- **Why needed**: Vector field gradients are needed for stress computation in elasticity.
- **Implementation**: Evaluate gradient of vector field, verify 2x2 or 3x3 structure

#### 13.3 `ProductSpace_3ComponentElasticity`
- [ ] Not started
- **What it tests**: 3-component ProductSpace for 3D elasticity displacement
- **Why needed**: 3D displacement fields have 3 components. Must verify assembly compatibility.
- **Implementation**: Create 3D ProductSpace, verify dofs_per_element and evaluation

---

### 14. FaceRestriction Extended Tests (Well-tested but some edge cases)

#### 14.1 `FaceRestriction_Order3FaceDofs`
- [ ] Not started
- **What it tests**: Face DOF extraction for cubic (p=3) elements
- **Why needed**: Higher-order elements have face-interior DOFs. Must verify correct extraction.
- **Implementation**: Extract face DOFs from p=3 hexahedron, verify count and indices

#### 14.2 `FaceRestriction_WedgeMixedFaceTopology`
- [ ] Not started
- **What it tests**: FaceRestriction handles wedge with triangular and quad faces
- **Why needed**: Wedge faces have different DOF counts. Must handle mixed topology.
- **Implementation**: Extract DOFs from wedge triangular vs quad faces, verify counts differ

#### 14.3 `FaceRestriction_PyramidApexFaces`
- [ ] Not started
- **What it tests**: FaceRestriction handles pyramid triangular faces converging at apex
- **Why needed**: Pyramid triangular faces have special geometry near apex.
- **Implementation**: Verify face DOF extraction for all pyramid faces

---

### 15. DGOperators Extended Tests (Good coverage but some gaps)

#### 15.1 `DGOperators_CurvedFaceNormalJump`
- [ ] Not started
- **What it tests**: Jump operator uses correct normal for curved element faces
- **Why needed**: On curved faces, normal varies with position. Jump must use local normal.
- **Implementation**: Compute jump on curved face, verify uses position-dependent normal

#### 15.2 `DGOperators_HigherOrderPenalty`
- [ ] Not started
- **What it tests**: Penalty parameter scales correctly with polynomial order
- **Why needed**: DG penalty must increase with p to maintain coercivity. Usually scales as p^2.
- **Implementation**: Verify penalty_parameter() increases appropriately with order

#### 15.3 `DGOperators_3DGradientJump`
- [ ] Not started
- **What it tests**: Gradient normal jump works for 3D elements
- **Why needed**: 3D DG methods need gradient jumps for interior penalty formulations.
- **Implementation**: Compute gradient_normal_jump on hex face, verify result

---

### 16. Thread Safety Tests

#### 16.1 `SpaceCache_ConcurrentAccess`
- [ ] Not started
- **What it tests**: Multiple threads can access SpaceCache simultaneously
- **Why needed**: Parallel assembly accesses cache from multiple threads.
- **Implementation**: Spawn threads accessing same cache entries, verify no race conditions

#### 16.2 `SpaceWorkspace_ThreadLocalIsolation`
- [ ] Not started
- **What it tests**: SpaceWorkspace provides thread-local storage
- **Why needed**: Each thread needs its own workspace to avoid data races during assembly.
- **Implementation**: Verify different threads get different workspace buffers

#### 16.3 `FunctionSpace_ConcurrentEvaluation`
- [ ] Not started
- **What it tests**: FunctionSpace evaluation is thread-safe
- **Why needed**: Many threads evaluate same space during parallel assembly.
- **Implementation**: Spawn threads evaluating same space at different points

---

### 17. Error Handling Tests

#### 17.1 `FunctionSpace_InvalidCoefficientSizeThrows`
- [ ] Not started
- **What it tests**: evaluate() throws for wrong coefficient vector size
- **Why needed**: Coefficient count must match dofs_per_element(). Wrong size causes memory errors.
- **Implementation**: Call evaluate() with undersized vector, verify exception

#### 17.2 `SpaceFactory_UnsupportedElementTypeThrows`
- [ ] Not started
- **What it tests**: SpaceFactory throws for unsupported element types
- **Why needed**: Not all element types support all space types (e.g., C1 on simplices).
- **Implementation**: Request unsupported combination, verify informative exception

#### 17.3 `TraceSpace_InvalidFaceIdThrows`
- [ ] Not started
- **What it tests**: TraceSpace constructor throws for invalid face ID
- **Why needed**: Face ID must be valid for element type. Invalid ID indicates setup error.
- **Implementation**: Create TraceSpace with face_id=99 for tetrahedron, verify exception

#### 17.4 `OrientationManager_InvalidEdgeIndexThrows`
- [ ] Not started
- **What it tests**: OrientationManager throws for out-of-bounds edge index
- **Why needed**: Edge index must be valid. Out-of-bounds access would cause undefined behavior.
- **Implementation**: Request orientation for edge 99 of triangle, verify exception

---

## Checklist Summary

### Existing Coverage (Implemented)

| Category | Tests | Status |
|----------|-------|--------|
| FunctionSpaces (H1/L2/etc) basics | 15 | Complete |
| C1Space basics | 4 | Complete |
| TraceSpace operations | 12 | Complete |
| FaceRestriction DOF extraction | 20 | Complete |
| OrientationManager | 18 | Complete |
| DGOperators | 14 | Complete |
| VectorComponentExtractor | 8 | Complete |
| SpaceInterpolation basics | 5 | Complete |
| SpaceCache/Workspace | 6 | Complete |
| Vector space traces | 4 | Complete |
| Vector space operators | 2 | Complete |
| SpaceCompatibility basics | 2 | Complete |
| MixedSpace basics | 2 | Complete |
| FunctionSpace gradients | 2 | Complete |
| IsogeometricSpace basics | 2 | Complete |
| Integration test | 2 | Complete |
| **Subtotal** | **~118** | |

### Missing Tests (This Checklist)

| Category | Tests | Priority | Rationale |
|----------|-------|----------|-----------|
| **MortarSpace Comprehensive** | 7 | **CRITICAL** | Zero existing tests for interface coupling wrapper |
| **EnrichedSpace Comprehensive** | 10 | **CRITICAL** | Zero existing tests for XFEM enrichment |
| **AdaptiveSpace Comprehensive** | 11 | **CRITICAL** | Zero existing tests for hp-adaptivity |
| **CompositeSpace Comprehensive** | 10 | **CRITICAL** | Zero existing tests for multi-region spaces |
| **HCurlSpace Trace/Orientation** | 9 | High | Limited coverage of tangential trace and orientation |
| **HDivSpace Trace/Orientation** | 9 | High | Limited coverage of normal trace and orientation |
| **IsogeometricSpace Extended** | 8 | High | Only basic metadata tested; no B-spline validation |
| **MixedSpace Extended** | 6 | Medium | Limited component evaluation tests |
| **TraceSpace Extended** | 7 | Medium | Missing wedge/pyramid face coverage |
| **SpaceInterpolation Extended** | 6 | Medium | Limited vector field and conservation tests |
| **SpaceCompatibility Extended** | 6 | Medium | Limited inf-sup pair coverage |
| **C1Space Extended** | 5 | Medium | Only cubic Line tested |
| **ProductSpace Extended** | 3 | Lower | Basic tests exist |
| **FaceRestriction Extended** | 3 | Lower | Good existing coverage |
| **DGOperators Extended** | 3 | Lower | Good existing coverage |
| **Thread Safety** | 3 | Lower | Assembly parallelization |
| **Error Handling** | 4 | Lower | Robustness |
| **Subtotal** | **110** | | |

### Overall Summary

| Metric | Count |
|--------|-------|
| Existing Tests | ~118 |
| Missing Tests | 110 |
| **Grand Total** | ~228 |
| Current Coverage | ~52% |
| Target Coverage | 90%+ |

### Priority Ranking

1. **CRITICAL**: MortarSpace Comprehensive (7 tests) - Zero tests for interface coupling
2. **CRITICAL**: EnrichedSpace Comprehensive (10 tests) - Zero tests for XFEM
3. **CRITICAL**: AdaptiveSpace Comprehensive (11 tests) - Zero tests for hp-adaptivity
4. **CRITICAL**: CompositeSpace Comprehensive (10 tests) - Zero tests for multi-region
5. **HIGH**: HCurlSpace Trace/Orientation (9 tests) - H(curl) conformity validation
6. **HIGH**: HDivSpace Trace/Orientation (9 tests) - H(div) conformity validation
7. **HIGH**: IsogeometricSpace Extended (8 tests) - B-spline/NURBS validation
8. **MEDIUM**: MixedSpace Extended (6 tests) - Mixed formulation support
9. **MEDIUM**: TraceSpace Extended (7 tests) - Boundary condition support
10. **MEDIUM**: SpaceInterpolation Extended (6 tests) - Field transfer
11. **MEDIUM**: SpaceCompatibility Extended (6 tests) - Inf-sup stability
12. **MEDIUM**: C1Space Extended (5 tests) - Hermite completeness
13. **LOWER**: ProductSpace Extended (3 tests)
14. **LOWER**: FaceRestriction Extended (3 tests)
15. **LOWER**: DGOperators Extended (3 tests)
16. **LOWER**: Thread Safety (3 tests)
17. **LOWER**: Error Handling (4 tests)

---

## References

### Core Function Space Theory
1. Brenner, S.C., Scott, L.R. "The Mathematical Theory of Finite Element Methods" 3rd ed. - Function space theory, Sobolev spaces
2. Ern, A., Guermond, J.L. "Theory and Practice of Finite Elements" - Conforming and non-conforming spaces
3. Ciarlet, P.G. "The Finite Element Method for Elliptic Problems" - Abstract finite element analysis

### H(div) and H(curl) Spaces
4. Boffi, D., Brezzi, F., Fortin, M. "Mixed Finite Element Methods and Applications" - RT/BDM spaces, inf-sup stability
5. Monk, P. "Finite Element Methods for Maxwell's Equations" - Nedelec spaces, H(curl) conformity
6. Nedelec, J.C. "Mixed Finite Elements in R^3" (1980) - Original Nedelec element construction
7. Brezzi, F., Douglas, J., Marini, L.D. "Two Families of Mixed Finite Elements" (1985) - BDM elements

### Mortar and Interface Methods
8. Bernardi, C., Maday, Y., Patera, A.T. "A New Nonconforming Approach to Domain Decomposition: The Mortar Element Method" - Mortar method theory
9. Wohlmuth, B. "Discretization Methods and Iterative Solvers Based on Domain Decomposition" - Mortar coupling

### Enriched/Extended FEM
10. Belytschko, T., Black, T. "Elastic Crack Growth in Finite Elements with Minimal Remeshing" (1999) - XFEM foundations
11. Moes, N., Dolbow, J., Belytschko, T. "A Finite Element Method for Crack Growth without Remeshing" (1999) - XFEM formulation
12. Fries, T.P., Belytschko, T. "The Extended/Generalized Finite Element Method: An Overview" - XFEM review

### hp-Adaptivity
13. Szabo, B., Babuska, I. "Finite Element Analysis" - p-FEM and hp-adaptivity
14. Demkowicz, L. "Computing with hp-Adaptive Finite Elements" - hp-FEM implementation

### Isogeometric Analysis
15. Cottrell, J.A., Hughes, T.J.R., Bazilevs, Y. "Isogeometric Analysis: Toward Integration of CAD and FEA" - IGA foundations
16. de Boor, C. "A Practical Guide to Splines" - B-spline theory

### Hermite and C1 Continuity
17. Ciarlet, P.G. "Interpolation Theory over Curved Elements" - Hermite element theory
18. Argyris, J.H., Fried, I., Scharpf, D.W. "The TUBA Family of Plate Elements" (1968) - C1 plate elements

### Discontinuous Galerkin
19. Arnold, D.N., Brezzi, F., Cockburn, B., Marini, L.D. "Unified Analysis of Discontinuous Galerkin Methods" (2002)
20. Riviere, B. "Discontinuous Galerkin Methods for Solving Elliptic and Parabolic Equations" - DG theory and practice

---

## Implementation Notes

### Testing MortarSpace Delegation

```cpp
// Verify all operations delegate correctly to wrapped space
auto base_space = std::make_shared<H1Space>(ElementType::Triangle3, 2);
auto mortar = MortarSpace(base_space);

// Metadata delegation
EXPECT_EQ(mortar.space_type(), SpaceType::Mortar);  // MortarSpace-specific
EXPECT_EQ(mortar.field_type(), base_space->field_type());  // Delegated
EXPECT_EQ(mortar.polynomial_order(), base_space->polynomial_order());

// Operation delegation
std::vector<Real> coeffs(mortar.dofs_per_element(), 1.0);
auto result = mortar.evaluate({0.33, 0.33, 0.0}, coeffs);
auto expected = base_space->evaluate({0.33, 0.33, 0.0}, coeffs);
EXPECT_NEAR(result[0], expected[0], 1e-12);
```

### Testing EnrichedSpace DOF Splitting

```cpp
// Create enriched space with known DOF counts
auto base = std::make_shared<H1Space>(ElementType::Triangle3, 2);  // 6 DOFs
auto enrich = std::make_shared<L2Space>(ElementType::Triangle3, 0);  // 1 DOF
auto enriched = EnrichedSpace(base, enrich);

// Verify DOF count
EXPECT_EQ(enriched.dofs_per_element(), 7);  // 6 + 1

// Verify coefficient splitting in evaluate()
std::vector<Real> coeffs(7);
std::fill(coeffs.begin(), coeffs.begin() + 6, 1.0);  // Base = 1
coeffs[6] = 2.0;  // Enrichment = 2
auto result = enriched.evaluate({0.33, 0.33, 0.0}, coeffs);
// Result should be base_eval + enrich_eval
```

### Testing AdaptiveSpace Level Switching

```cpp
AdaptiveSpace adaptive;
adaptive.add_level(1, std::make_shared<H1Space>(ElementType::Triangle3, 1));  // 3 DOFs
adaptive.add_level(2, std::make_shared<H1Space>(ElementType::Triangle3, 2));  // 6 DOFs

// Verify level switching
adaptive.set_active_level(0);
EXPECT_EQ(adaptive.dofs_per_element(), 3);
EXPECT_EQ(adaptive.polynomial_order(), 1);

adaptive.set_active_level(1);
EXPECT_EQ(adaptive.dofs_per_element(), 6);
EXPECT_EQ(adaptive.polynomial_order(), 2);

// Verify set_active_level_by_order
adaptive.set_active_level_by_order(1);
EXPECT_EQ(adaptive.active_level().order, 1);
```

### Testing H(curl) Tangential Continuity

```cpp
// Create two adjacent tetrahedra sharing a face
// Set DOFs to define smooth field across interface
// Evaluate tangential_trace from both sides at shared face
// Verify values match (tangential continuity)

auto space = HCurlSpace(ElementType::Tetra4, 1);
auto face_normal = Vec3{0, 0, 1};  // Shared face normal

auto trace1 = space.tangential_trace(dofs_elem1, face_points, face_normal);
auto trace2 = space.tangential_trace(dofs_elem2, face_points, -face_normal);

for (size_t i = 0; i < face_points.size(); ++i) {
    EXPECT_NEAR((trace1[i] - trace2[i]).norm(), 0.0, 1e-10);
}
```

### Testing H(div) Normal Continuity

```cpp
// Similar to H(curl) but verify scalar normal trace continuity

auto space = HDivSpace(ElementType::Tetra4, 0);  // RT_0
auto face_normal = Vec3{0, 0, 1};

auto trace1 = space.normal_trace(dofs_elem1, face_points, face_normal);
auto trace2 = space.normal_trace(dofs_elem2, face_points, -face_normal);

for (size_t i = 0; i < face_points.size(); ++i) {
    // Note: sign flip because normal direction reversed
    EXPECT_NEAR(trace1[i] + trace2[i], 0.0, 1e-10);
}
```
