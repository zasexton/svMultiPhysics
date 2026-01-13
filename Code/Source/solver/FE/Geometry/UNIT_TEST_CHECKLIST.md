# Unit Test Checklist for FE Geometry Library

This document lists additional unit tests that should be added to improve coverage and ensure correctness of the Geometry subfolder. Tests are organized by priority and component.

**Last Updated:** 2026-01-13
**Status Legend:**
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Coverage Summary

The Geometry subfolder contains **15 header files** and approximately **1,200+ lines** of implementation code. Current test coverage is estimated at **75-80%** with significant gaps in metric tensor edge cases, surface/curve geometry element coverage, and inverse mapping robustness scenarios.

### Existing Test Files
| File | Lines | Primary Coverage |
|------|-------|------------------|
| test_LinearMapping.cpp | ~80 | LinearMapping basic operations |
| test_IsoparametricMapping.cpp | ~450 | IsoparametricMapping, MappingFactory, GeometryValidator |
| test_InverseMapping.cpp | ~100 | InverseMapping Newton solver |
| test_JacobianCacheGeometryQuadrature.cpp | ~400 | JacobianCache, GeometryQuadrature |
| test_HigherOrderMappings.cpp | ~200 | Higher-order element mappings |
| test_ElementTypeCoverage.cpp | ~100 | Parametrized element coverage |
| test_Pyramid14Mapping.cpp | ~150 | Pyramid14 rational geometry |
| test_MetricSurfacePushForward.cpp | ~270 | MetricTensor, SurfaceGeometry, CurveGeometry, PushForward |

**Total Existing Tests:** ~82 tests

---

## High Priority Tests

### 1. MetricTensor Extended Tests (CRITICAL)

The MetricTensor class has only **2 tests** covering identity and axis-aligned scaling. Non-diagonal metrics (skewed elements) and 1D cases are completely untested.

#### 1.1 `MetricTensor_SkewedQuadCovariantMetric`
- [ ] Not started
- **What it tests**: Covariant metric tensor G = J^T J for a parallelogram (non-orthogonal) quadrilateral
- **Why needed**: Skewed elements produce non-diagonal metric tensors with significant off-diagonal terms. These terms are critical for correct gradient computation in non-orthogonal coordinate systems. Current tests only cover diagonal metrics (axis-aligned scaling), leaving the more complex case untested.
- **Implementation**: Create Quad4 with nodes forming a parallelogram (e.g., {0,0}, {2,0}, {3,1}, {1,1}), verify G(0,1) = G(1,0) = J_0 · J_1 is non-zero and matches expected value

#### 1.2 `MetricTensor_SkewedQuadContravariantMetric`
- [ ] Not started
- **What it tests**: Contravariant metric (inverse) for skewed quadrilateral
- **Why needed**: The contravariant metric G^{-1} is used for computing physical gradients from reference gradients via (J^{-T}). For non-diagonal metrics, the inversion formula involves the off-diagonal terms. Incorrect inversion produces wrong gradient directions.
- **Implementation**: For the same parallelogram, verify G^{-1} * G = I (identity) to machine precision

#### 1.3 `MetricTensor_HighlySkewedElementConditionNumber`
- [ ] Not started
- **What it tests**: Metric tensor condition number for highly skewed elements (angle approaching 0 or 180 degrees)
- **Why needed**: Ill-conditioned metric tensors indicate poor element quality and can cause numerical instability in gradient computations. The ratio of eigenvalues of G measures this. Elements with angles < 10 degrees should produce large condition numbers.
- **Implementation**: Create quad with one angle ~5 degrees, compute eigenvalue ratio of G, verify it exceeds expected threshold (~10-20)

#### 1.4 `MetricTensor_1DLineCovariantMetric`
- [ ] Not started
- **What it tests**: Covariant metric for 1D line elements in 3D space
- **Why needed**: 1D elements have a 1x1 metric tensor (the squared length of the tangent vector). This special case uses different code paths than 2D/3D metrics. Currently completely untested - only 2D and 3D metrics are tested.
- **Implementation**: Create Line2 with length L, verify G(0,0) = (L/2)^2 (since reference element has length 2)

#### 1.5 `MetricTensor_1DLineContravariantMetric`
- [ ] Not started
- **What it tests**: Contravariant metric (inverse) for 1D line elements
- **Why needed**: The 1D contravariant metric is simply 1/G(0,0), but this trivial inversion must be verified to ensure the dimension-specific code path is correct.
- **Implementation**: Verify G^{-1}(0,0) * G(0,0) = 1 for various line lengths

#### 1.6 `MetricTensor_3DTetrahedronNonOrthogonalMetric`
- [ ] Not started
- **What it tests**: Metric tensor for a non-right-angled tetrahedron (all edges different lengths)
- **Why needed**: General tetrahedra have non-diagonal 3x3 metrics. The current 3D test only covers axis-aligned hexahedra with diagonal metrics. Simplex geometry produces fundamentally different metric structure.
- **Implementation**: Create Tetra4 with non-orthogonal edges, verify all 9 metric components match analytical J^T J computation

#### 1.7 `MetricTensor_SurfaceEmbeddedIn3DMetric`
- [ ] Not started
- **What it tests**: 2D metric tensor for a 2D surface embedded in 3D (dimension mismatch between reference and physical)
- **Why needed**: Surface elements (2D reference in 3D physical) have a 2x2 metric tensor computed from the 3x2 Jacobian via J^T J. This is the standard case for shell elements and boundary integrals. Currently only tested for planar surfaces; tilted surfaces are only tested for normals, not metrics.
- **Implementation**: Use the tilted quad (z = x + y) from existing tests, verify 2x2 metric tensor matches expected values for non-planar surface

---

### 2. SurfaceGeometry Extended Tests (CRITICAL)

SurfaceGeometry has only **3 tests** covering a single element type (Quad4). Multiple surface element types and curved surfaces are completely untested.

#### 2.1 `SurfaceGeometry_Triangle3PlanarNormals`
- [ ] Not started
- **What it tests**: Surface normal and area element for planar triangular surface
- **Why needed**: Triangular surfaces are fundamental for boundary integrals on tetrahedral meshes. The current tests only cover Quad4 surfaces. Triangle3 has different topology and Jacobian structure.
- **Implementation**: Create planar Triangle3, verify normal direction and area element = 0.5 * |v1 x v2|

#### 2.2 `SurfaceGeometry_Triangle6CurvedSurface`
- [ ] Not started
- **What it tests**: Surface normal and area element for curved quadratic triangular surface
- **Why needed**: Higher-order surfaces have position-dependent normals and area elements. The mid-edge nodes can create curvature even with planar vertices. This is critical for accurate boundary integrals on curved domains.
- **Implementation**: Create Triangle6 with mid-edge nodes displaced from planar positions, verify normal varies across the surface and area element differs from planar approximation

#### 2.3 `SurfaceGeometry_Quad8SerendipitySurface`
- [ ] Not started
- **What it tests**: Surface geometry for Quad8 serendipity surface element
- **Why needed**: Serendipity elements are commonly used for geometry representation. Quad8 can represent curved edges without interior node. The absence of center node affects the Jacobian computation.
- **Implementation**: Create Quad8 with curved edges (displaced mid-edge nodes), verify surface normal and area element computation

#### 2.4 `SurfaceGeometry_Quad9BiQuadraticSurface`
- [ ] Not started
- **What it tests**: Surface geometry for Quad9 biquadratic surface element
- **Why needed**: Quad9 includes center node allowing representation of doubly-curved surfaces. This is the highest-fidelity quad surface element and must be validated.
- **Implementation**: Create Quad9 with center node displaced to create dome-like surface, verify normal at center differs from edges

#### 2.5 `SurfaceGeometry_SphericalPatchNormalConsistency`
- [ ] Not started
- **What it tests**: Normal vectors point consistently outward on curved surface approximating sphere patch
- **Why needed**: On curved surfaces, normals must vary smoothly and consistently point outward. This validates the cross-product orientation convention for arbitrarily curved surfaces.
- **Implementation**: Create Quad9 nodes on sphere surface, evaluate normals at multiple points, verify all point radially outward from sphere center

#### 2.6 `SurfaceGeometry_TangentVectorIndependence`
- [ ] Not started
- **What it tests**: Tangent vectors tangent_u and tangent_v are linearly independent (non-parallel)
- **Why needed**: Parallel tangent vectors indicate a degenerate surface (collapsed to a curve or point). This should be caught or produce sensible results.
- **Implementation**: Verify tangent_u × tangent_v has non-zero magnitude at random interior points

#### 2.7 `SurfaceGeometry_3DMappingThrows`
- [ ] Not started
- **What it tests**: SurfaceGeometry rejects 3D volume mappings (dimension = 3)
- **Why needed**: SurfaceGeometry is only meaningful for 2D reference elements embedded in 3D. Using a Hex8 mapping should throw an informative exception.
- **Implementation**: Create Hex8 mapping, call SurfaceGeometry::evaluate(), verify FEException with message about dimension

---

### 3. CurveGeometry Extended Tests (CRITICAL)

CurveGeometry has only **1 test** covering Line2. Higher-order curves and various orientations are untested.

#### 3.1 `CurveGeometry_Line3QuadraticCurve`
- [ ] Not started
- **What it tests**: Tangent vector and line element for quadratic (curved) line element
- **Why needed**: Line3 elements with displaced midpoint create curved edges. The tangent vector varies along the curve, unlike linear Line2. This is critical for accurate line integrals on curved boundaries.
- **Implementation**: Create Line3 with midpoint displaced perpendicular to chord, verify tangent varies between endpoints

#### 3.2 `CurveGeometry_Line3LineElementIntegration`
- [ ] Not started
- **What it tests**: Integrating line_element over curved Line3 gives correct arc length
- **Why needed**: The line_element (|tangent|) is used for line integrals. For a curved line, integrating line_element should give the arc length, which is greater than the chord length.
- **Implementation**: Create semicircular Line3, integrate line_element with high-order quadrature, compare to known arc length

#### 3.3 `CurveGeometry_ArbitraryOrientationFrame`
- [ ] Not started
- **What it tests**: Normal frame (normal_1, normal_2) is orthonormal for arbitrarily oriented curves
- **Why needed**: The existing test uses a y-axis aligned line. For arbitrary orientations, the frame completion algorithm (Gram-Schmidt or similar) must produce consistent orthonormal frames.
- **Implementation**: Create Line2 with arbitrary 3D orientation (e.g., {1,2,3} to {4,5,6}), verify orthonormality

#### 3.4 `CurveGeometry_NearZAxisCurveFrame`
- [ ] Not started
- **What it tests**: Frame completion handles curves nearly parallel to z-axis
- **Why needed**: Frame completion typically uses cross product with a reference vector (often z-axis). When the curve is parallel to this reference, the algorithm must use a different reference to avoid singular behavior.
- **Implementation**: Create Line2 along z-axis (e.g., {0,0,0} to {0,0,1}), verify frame is valid and orthonormal

#### 3.5 `CurveGeometry_NearXAxisCurveFrame`
- [ ] Not started
- **What it tests**: Frame completion handles curves nearly parallel to x-axis
- **Why needed**: Tests robustness of frame completion for another axis-aligned case. Different algorithms may have different problematic directions.
- **Implementation**: Create Line2 along x-axis, verify frame orthonormality

#### 3.6 `CurveGeometry_2DMappingThrows`
- [ ] Not started
- **What it tests**: CurveGeometry rejects 2D surface mappings (dimension = 2)
- **Why needed**: CurveGeometry is only meaningful for 1D reference elements. Using a Triangle3 or Quad4 mapping should throw an informative exception.
- **Implementation**: Create Quad4 mapping, call CurveGeometry::evaluate(), verify FEException

#### 3.7 `CurveGeometry_HelixTangentAndFrame`
- [ ] Not started
- **What it tests**: Curve geometry for 3D helix-like curve (Line3 with out-of-plane curvature)
- **Why needed**: True 3D curves (not planar) exercise the full 3D frame computation. Helix is the simplest non-planar curve.
- **Implementation**: Create Line3 with nodes forming helix segment, verify frame varies along curve and remains orthonormal

---

### 4. InverseMapping Robustness Tests (HIGH)

InverseMapping has **6 tests** but many configuration options and edge cases remain untested.

#### 4.1 `InverseMapping_LineSearchActivation`
- [ ] Not started
- **What it tests**: Line search is actually activated and improves convergence for difficult mappings
- **Why needed**: The solve_robust() method enables line search, but there's no test verifying line search actually executes its backtracking loop. A mapping where standard Newton diverges but line search succeeds would validate this.
- **Implementation**: Create highly distorted quad where Newton overshoots, verify solve_robust() succeeds while solve() fails or takes many more iterations

#### 4.2 `InverseMapping_ArmijoConditionSatisfied`
- [ ] Not started
- **What it tests**: Each line search step satisfies the Armijo sufficient decrease condition
- **Why needed**: The Armijo condition (f(x + α*p) ≤ f(x) + c*α*∇f·p) ensures sufficient decrease. Incorrect implementation could accept steps that don't decrease the residual enough.
- **Implementation**: Create mapping requiring line search, instrument or verify residual decreases by at least armijo_c fraction of predicted decrease

#### 4.3 `InverseMapping_AllOptionsExercised`
- [ ] Not started
- **What it tests**: Each option in InverseMappingOptions affects solver behavior
- **Why needed**: Options like armijo_c, line_search_rho, max_line_search_iters, min_step_size are currently untested. Dead code or ignored parameters would not be caught.
- **Implementation**: Vary each option independently, verify output or behavior changes (e.g., more iterations, different convergence)

#### 4.4 `InverseMapping_MinStepSizeFailure`
- [ ] Not started
- **What it tests**: Solver throws when line search step size falls below min_step_size
- **Why needed**: This is a specific failure mode that should produce a clear error message about step size, not generic non-convergence.
- **Implementation**: Create pathological mapping where line search cannot find acceptable step, set min_step_size high enough to trigger, verify specific exception message

#### 4.5 `InverseMapping_MaxLineSearchItersExceeded`
- [ ] Not started
- **What it tests**: Solver handles max_line_search_iters limit gracefully
- **Why needed**: When line search exhausts its iteration budget without satisfying Armijo, the solver should either proceed with best step or throw.
- **Implementation**: Set max_line_search_iters = 1 on difficult mapping, verify behavior is sensible (throw or proceed)

#### 4.6 `InverseMapping_HighAspectRatioQuadConvergence`
- [ ] Not started
- **What it tests**: Inverse mapping converges for high aspect ratio (1000:1) quadrilateral
- **Why needed**: High aspect ratios create ill-conditioned Jacobians. The solver must still converge, potentially requiring more iterations or line search.
- **Implementation**: Create quad with aspect ratio 1000:1, verify solve() or solve_robust() succeeds

#### 4.7 `InverseMapping_HighlySkewedQuadConvergence`
- [ ] Not started
- **What it tests**: Inverse mapping converges for highly skewed quad (angle < 5 degrees)
- **Why needed**: Extreme skewness creates nearly singular Jacobians. Tests robustness of Newton iteration near singularity.
- **Implementation**: Create quad with minimum angle ~3 degrees, verify convergence

#### 4.8 `InverseMapping_NearApexPyramidConvergence`
- [ ] Not started
- **What it tests**: Inverse mapping converges for points very near pyramid apex
- **Why needed**: Pyramid bases have rational functions with potential singularity at apex. Points near apex (z = 0.999) may cause Newton instability.
- **Implementation**: Create Pyramid5, map point at z = 0.999 to physical, invert back, verify convergence

#### 4.9 `InverseMapping_DefaultInitialGuessQuality`
- [ ] Not started
- **What it tests**: Default initial guess (element centroid) provides good convergence
- **Why needed**: When no initial guess is provided, the solver uses element centroid. This should be close enough for Newton convergence for most elements.
- **Implementation**: Test inverse mapping with default guess on various element types and positions, verify all converge

#### 4.10 `InverseMapping_3DElementTypes`
- [ ] Not started
- **What it tests**: Inverse mapping works correctly for all 3D element types (Tetra4, Hex8, Wedge6, Pyramid5)
- **Why needed**: Each 3D element type has different reference geometry and Jacobian structure. Current tests focus on 2D elements and triangles.
- **Implementation**: Parametrized test over 3D element types, verify forward-inverse roundtrip

---

### 5. GeometryMapping Hessian Tests (HIGH)

The mapping_hessian() method exists but has **no direct tests**. Second derivatives are needed for curved element analysis and some advanced formulations.

#### 5.1 `IsoparametricMapping_HessianMatchesFiniteDifference`
- [ ] Not started
- **What it tests**: Analytical Hessian of mapping matches numerical second derivatives of Jacobian
- **Why needed**: The Hessian (∂²x/∂ξ∂η) is computed analytically from basis function Hessians. This must match finite differences on the Jacobian for correctness.
- **Implementation**: Create Quad9 mapping, compute Hessian at interior point, compare to central finite difference on Jacobian

#### 5.2 `IsoparametricMapping_HessianZeroForLinearElements`
- [ ] Not started
- **What it tests**: Hessian is exactly zero for linear (affine) mappings
- **Why needed**: Linear mappings have constant Jacobians, so second derivatives must be zero. This validates the base case.
- **Implementation**: Create Quad4 identity mapping, verify all Hessian components are zero

#### 5.3 `IsoparametricMapping_HessianSymmetry`
- [ ] Not started
- **What it tests**: Hessian is symmetric (∂²x_i/∂ξ∂η = ∂²x_i/∂η∂ξ)
- **Why needed**: Mixed partial derivatives must be equal by Schwarz's theorem. Asymmetric Hessian indicates implementation error.
- **Implementation**: Create curved Quad9, verify H(i,j,k) = H(i,k,j) for all components

#### 5.4 `IsoparametricMapping_HessianForCurvedTriangle6`
- [ ] Not started
- **What it tests**: Non-zero Hessian for curved Triangle6 element
- **Why needed**: Triangle6 with displaced midpoints creates curved geometry with non-zero second derivatives.
- **Implementation**: Create Triangle6 with curved edges, verify Hessian is non-zero and matches finite differences

#### 5.5 `IsoparametricMapping_HessianForHex27`
- [ ] Not started
- **What it tests**: 3D Hessian computation for Hex27 element
- **Why needed**: 3D Hessians have 3×3×3 = 27 unique components. The 3D code path must be validated separately.
- **Implementation**: Create curved Hex27, verify Hessian matches finite differences

---

## Medium Priority Tests

### 6. GeometryValidator Extended Tests

GeometryValidator has basic coverage but condition number accuracy and sampling strategy are untested.

#### 6.1 `GeometryValidator_ConditionNumberVsSVD`
- [ ] Not started
- **What it tests**: Condition number estimate is accurate compared to SVD-based condition number
- **Why needed**: GeometryValidator uses metric-based (2D) or infinity-norm (3D) condition number estimates. These approximations should be within factor of ~2-5 of the true SVD condition number.
- **Implementation**: Compute validator condition number and SVD condition number for various elements, verify ratio is bounded

#### 6.2 `GeometryValidator_MultipleSamplingPoints`
- [ ] Not started
- **What it tests**: Validator correctly reports worst quality across multiple sampling points
- **Why needed**: For non-affine elements, Jacobian varies across the element. The validator should sample at quadrature points and report the worst (minimum detJ, maximum condition number).
- **Implementation**: Create curved element with varying quality, verify min_detJ and max_condition_number reflect worst points

#### 6.3 `GeometryValidator_NegativeJacobianAtOnePoint`
- [ ] Not started
- **What it tests**: Validator detects negative Jacobian even if only at one quadrature point
- **Why needed**: An element may have positive Jacobian at centroid but negative at a corner. The validator must check all sampling points.
- **Implementation**: Create twisted hex positive at center but negative at corner, verify positive_jacobian = false

#### 6.4 `GeometryValidator_CustomQuadratureRule`
- [ ] Not started
- **What it tests**: Validator can use custom quadrature rule for sampling
- **Why needed**: Different applications may need different sampling densities. If validator accepts custom quadrature, it should use those points.
- **Implementation**: Pass custom high-order quadrature, verify validator samples at those points

#### 6.5 `GeometryValidator_WedgeQualityMetrics`
- [ ] Not started
- **What it tests**: Quality metrics are correct for wedge elements
- **Why needed**: Wedges have mixed topology with triangular and quadrilateral faces. Quality computation must handle both.
- **Implementation**: Create well-formed and distorted wedges, verify quality metrics are sensible

#### 6.6 `GeometryValidator_PyramidQualityMetrics`
- [ ] Not started
- **What it tests**: Quality metrics are correct for pyramid elements, especially near apex
- **Why needed**: Pyramid apex has degenerate geometry (Jacobian determinant approaches zero). Quality metrics must handle this gracefully.
- **Implementation**: Validate pyramid at various z-levels including near-apex, verify metrics are finite

---

### 7. JacobianCache Extended Tests

JacobianCache has good coverage but memory management and edge cases are untested.

#### 7.1 `JacobianCache_LargeCacheSizePerformance`
- [ ] Not started
- **What it tests**: Cache performs well with many entries (1000+)
- **Why needed**: Large meshes may have many unique mapping/quadrature combinations. Cache lookup should remain O(1) with hash table.
- **Implementation**: Create 1000 unique cache entries, verify lookup time is constant (not O(n))

#### 7.2 `JacobianCache_CacheEvictionStrategy`
- [ ] Not started
- **What it tests**: Cache eviction (if implemented) removes old entries appropriately
- **Why needed**: Unbounded cache growth could exhaust memory. If LRU or other eviction is implemented, it must be validated.
- **Implementation**: Fill cache to capacity, add new entries, verify old entries are evicted

#### 7.3 `JacobianCache_ClearDuringCompute`
- [ ] Not started
- **What it tests**: Calling clear() while compute is in progress doesn't cause crashes
- **Why needed**: In multi-threaded assembly, clear() might be called during adaptive remeshing while threads are still computing.
- **Implementation**: Spawn threads computing entries while main thread calls clear(), verify no crashes

#### 7.4 `JacobianCache_HashCollisionHandling`
- [ ] Not started
- **What it tests**: Cache correctly handles hash collisions for different quadrature rules
- **Why needed**: The quadrature_hash function may produce collisions. Cache must distinguish different quadratures even with same hash.
- **Implementation**: Create two quadrature rules with intentionally similar hashes (if possible), verify they get separate cache entries

#### 7.5 `JacobianCache_NaNHandling`
- [ ] Not started
- **What it tests**: Cache handles mappings that produce NaN Jacobian values
- **Why needed**: Degenerate mappings may produce NaN/Inf in Jacobian. Cache should not crash and should propagate or flag these values.
- **Implementation**: Create degenerate mapping (collapsed element), attempt to cache, verify sensible behavior

---

### 8. GeometryQuadrature Extended Tests

GeometryQuadrature has comprehensive volume integration tests but boundary integration is less covered.

#### 8.1 `GeometryQuadrature_SurfaceAreaIntegration`
- [ ] Not started
- **What it tests**: Integrating scaled_weights over surface elements gives correct surface area
- **Why needed**: Current tests focus on volume elements. Surface integration for boundary integrals uses different logic.
- **Implementation**: Create various surface elements, verify integrated weights equal surface area

#### 8.2 `GeometryQuadrature_LineIntegration`
- [ ] Not started
- **What it tests**: Integrating scaled_weights over line elements gives correct arc length
- **Why needed**: 1D quadrature for line integrals should produce arc length. Current tests don't cover 1D.
- **Implementation**: Create Line2 and Line3, verify integrated weights equal line length

#### 8.3 `GeometryQuadrature_CurvedSurfaceArea`
- [ ] Not started
- **What it tests**: Surface area of curved surface (e.g., sphere patch) is computed correctly
- **Why needed**: Curved surfaces have position-dependent area elements. Integrating these should give true surface area.
- **Implementation**: Create Quad9 approximating sphere patch of known area, verify integration matches analytical area

#### 8.4 `GeometryQuadrature_HighOrderQuadratureAccuracy`
- [ ] Not started
- **What it tests**: High-order quadrature exactly integrates polynomial functions over elements
- **Why needed**: Order 2p quadrature should exactly integrate degree 2p polynomials. This tests combined mapping and quadrature accuracy.
- **Implementation**: For polynomial of known integral, verify numerical integration matches exact value to machine precision

---

### 9. PushForward Extended Tests

PushForward is well-tested for basic transforms but element type coverage is incomplete.

#### 9.1 `PushForward_TriangleGradientTransform`
- [ ] Not started
- **What it tests**: Gradient transform for triangular elements
- **Why needed**: Current tests use Quad4 and Hex8. Triangle simplex geometry has different Jacobian structure.
- **Implementation**: Create scaled/rotated Triangle3, verify gradient transform matches expected

#### 9.2 `PushForward_TetrahedronGradientTransform`
- [ ] Not started
- **What it tests**: Gradient transform for tetrahedral elements
- **Why needed**: Tetra4 is the fundamental 3D simplex. Gradient transform must work correctly for FEM assembly.
- **Implementation**: Create Tetra4 with known transformation, verify gradient transform

#### 9.3 `PushForward_HigherOrderElementTransforms`
- [ ] Not started
- **What it tests**: Piola transforms for higher-order elements (Quad9, Hex27)
- **Why needed**: Higher-order elements have position-dependent Jacobians, so transforms vary across the element.
- **Implementation**: Evaluate transforms at multiple points on Quad9/Hex27, verify consistency

#### 9.4 `PushForward_IllConditionedJacobianStability`
- [ ] Not started
- **What it tests**: Push-forward operations remain stable for high aspect ratio elements
- **Why needed**: Ill-conditioned Jacobians can amplify numerical errors in J^{-T} computation.
- **Implementation**: Create 1000:1 aspect ratio element, verify transforms produce sensible (finite, non-NaN) results

#### 9.5 `PushForward_HdivPreservesNormalContinuity`
- [ ] Not started
- **What it tests**: H(div) Piola transform preserves normal component continuity across element interfaces
- **Why needed**: The contravariant Piola transform is specifically designed to preserve normal continuity. This is its defining property for mixed FEM.
- **Implementation**: Create two adjacent elements, verify normal flux is continuous at shared face

#### 9.6 `PushForward_HcurlPreservesTangentialContinuity`
- [ ] Not started
- **What it tests**: H(curl) Piola transform preserves tangential component continuity across element interfaces
- **Why needed**: The covariant Piola transform preserves tangential continuity for H(curl) conforming elements.
- **Implementation**: Create two adjacent elements, verify tangential component is continuous at shared edge

---

### 10. MappingFactory Extended Tests

MappingFactory is well-tested but some edge cases remain.

#### 10.1 `MappingFactory_AllElementTypesCreateSuccessfully`
- [ ] Not started
- **What it tests**: Factory creates valid mappings for all supported element types
- **Why needed**: Systematic verification that no element type is accidentally unsupported.
- **Implementation**: Loop over all ElementType values, verify factory creates mapping without throwing

#### 10.2 `MappingFactory_HighOrderGeometry`
- [ ] Not started
- **What it tests**: Factory creates correct high-order (cubic) geometry mappings
- **Why needed**: Current tests focus on linear and quadratic geometry. Cubic geometry is used for high-accuracy simulations.
- **Implementation**: Create geometry_order=3 mapping, verify correct node count and mapping behavior

#### 10.3 `MappingFactory_UseAffineOptimization`
- [ ] Not started
- **What it tests**: use_affine=true creates LinearMapping even for higher-order elements
- **Why needed**: LinearMapping is more efficient for affine geometry. The use_affine flag should force this optimization.
- **Implementation**: Create request with use_affine=true for Quad4 nodes, verify LinearMapping type is returned

#### 10.4 `MappingFactory_InvalidNodeCountThrows`
- [ ] Not started
- **What it tests**: Factory throws informative error for wrong number of nodes
- **Why needed**: Passing wrong node count should fail early with clear error, not produce garbage mapping.
- **Implementation**: Pass 5 nodes for Quad4, verify exception message identifies the problem

#### 10.5 `MappingFactory_CustomBasisValidation`
- [ ] Not started
- **What it tests**: Factory validates custom_basis is compatible with element_type
- **Why needed**: Custom basis must match element topology. Incompatible combinations should be rejected.
- **Implementation**: Pass Triangle basis for Quad element_type, verify exception

---

## Lower Priority Tests

### 11. LinearMapping Extended Tests

LinearMapping has good coverage but some simplex variants are less tested.

#### 11.1 `LinearMapping_Tetra4BarycentricMapping`
- [ ] Not started
- **What it tests**: LinearMapping correctly uses barycentric coordinates for Tetra4
- **Why needed**: Tetra4 uses barycentric coordinates (4 coordinates summing to 1) not Cartesian reference space.
- **Implementation**: Verify map_to_physical at barycentric centroid (0.25, 0.25, 0.25, 0.25) gives physical centroid

#### 11.2 `LinearMapping_Wedge6MixedTopology`
- [ ] Not started
- **What it tests**: LinearMapping handles wedge (triangular prism) topology correctly
- **Why needed**: Wedge6 combines triangular base with linear extrusion. The mapping has mixed barycentric/Cartesian structure.
- **Implementation**: Verify mapping at wedge centroid and various interior points

#### 11.3 `LinearMapping_Pyramid5RationalBasis`
- [ ] Not started
- **What it tests**: LinearMapping for Pyramid5 uses correct rational basis functions
- **Why needed**: Pyramid elements require rational basis functions for consistent mapping. This is different from polynomial bases.
- **Implementation**: Verify mapping reproduces vertex positions and maps interior points correctly

#### 11.4 `LinearMapping_NegativeOrientationHandling`
- [ ] Not started
- **What it tests**: LinearMapping behavior for negative-orientation (inverted) elements
- **Why needed**: Inverted elements have negative Jacobian determinant. The mapping should still compute correctly (negative detJ indicates inversion).
- **Implementation**: Create clockwise-oriented Triangle3, verify negative Jacobian but correct map_to_physical

---

### 12. Sub/SuperparametricMapping Tests

Sub/Superparametric mappings have minimal testing beyond factory creation.

#### 12.1 `SubparametricMapping_ReducedGeometryAccuracy`
- [ ] Not started
- **What it tests**: Subparametric mapping uses fewer geometry nodes than field discretization
- **Why needed**: Subparametric mappings (e.g., using linear geometry for quadratic field) save cost but reduce geometric accuracy. This tradeoff should be quantifiable.
- **Implementation**: Compare mapping accuracy of subparametric vs isoparametric for same nodes

#### 12.2 `SuperparametricMapping_EnhancedGeometryAccuracy`
- [ ] Not started
- **What it tests**: Superparametric mapping provides better geometry representation
- **Why needed**: Superparametric mappings use more geometry nodes than field nodes. This improves boundary representation.
- **Implementation**: Create superparametric mapping on curved boundary, verify improved normal accuracy

#### 12.3 `SubparametricMapping_ConsistentWithIsoparametric`
- [ ] Not started
- **What it tests**: Subparametric at full order equals isoparametric mapping
- **Why needed**: Subparametric(order=p) should be identical to isoparametric(order=p) for same nodes.
- **Implementation**: Create both mapping types with same nodes and order, verify identical results

---

### 13. Thread Safety Tests

#### 13.1 `IsoparametricMapping_ConcurrentEvaluation`
- [ ] Not started
- **What it tests**: Multiple threads can evaluate same mapping concurrently
- **Why needed**: Parallel assembly evaluates same element mapping from multiple threads.
- **Implementation**: Spawn 8 threads evaluating same mapping at different points, verify all results correct

#### 13.2 `PushForward_ThreadSafety`
- [ ] Not started
- **What it tests**: PushForward static methods are thread-safe
- **Why needed**: PushForward is called during parallel assembly for gradient transformation.
- **Implementation**: Spawn threads calling gradient(), hdiv_vector(), hcurl_vector() concurrently

#### 13.3 `MappingFactory_ConcurrentCreation`
- [ ] Not started
- **What it tests**: MappingFactory can create mappings from multiple threads
- **Why needed**: Initial element setup may be parallelized.
- **Implementation**: Spawn threads creating different mapping types concurrently

---

### 14. Negative and Error Case Tests

#### 14.1 `IsoparametricMapping_NullBasisThrows`
- [ ] Not started
- **What it tests**: IsoparametricMapping throws for null basis pointer
- **Why needed**: Null pointer would cause crash during evaluation; early detection is better.
- **Implementation**: Pass nullptr for basis, verify exception

#### 14.2 `IsoparametricMapping_EmptyNodesThrows`
- [ ] Not started
- **What it tests**: IsoparametricMapping throws for empty node vector
- **Why needed**: Empty nodes is definitely an error that should be caught at construction.
- **Implementation**: Pass empty vector for nodes, verify exception

#### 14.3 `GeometryQuadrature_EmptyQuadratureThrows`
- [ ] Not started
- **What it tests**: GeometryQuadrature throws for zero-point quadrature rule
- **Why needed**: Zero points means no integration; this is almost certainly an error.
- **Implementation**: Pass empty quadrature, verify exception

#### 14.4 `MappingFactory_UnknownElementTypeThrows`
- [ ] Not started
- **What it tests**: MappingFactory throws for ElementType::Unknown
- **Why needed**: Unknown element type cannot be mapped meaningfully.
- **Implementation**: Request mapping for Unknown type, verify exception

#### 14.5 `InverseMapping_ZeroDimensionThrows`
- [ ] Not started
- **What it tests**: InverseMapping throws for mapping with dimension = 0
- **Why needed**: Zero-dimensional mapping is degenerate and uninvertible.
- **Implementation**: Create degenerate mapping (if possible), verify appropriate exception

---

## Checklist Summary

### Existing Coverage (Implemented)

| Category | Tests | Status |
|----------|-------|--------|
| LinearMapping basics | 6 | Complete |
| IsoparametricMapping comprehensive | 23 | Complete |
| InverseMapping Newton solver | 6 | Complete |
| JacobianCache/GeometryQuadrature | 23 | Complete |
| HigherOrderMappings | 8 | Complete |
| ElementTypeCoverage | 1 | Complete |
| Pyramid14Mapping | 4 | Complete |
| MetricTensor/SurfaceGeometry/PushForward | 11 | Complete |
| **Subtotal** | **~82** | |

### Missing Tests (This Checklist)

| Category | Tests | Priority | Rationale |
|----------|-------|----------|-----------|
| **MetricTensor Extended** | 7 | **CRITICAL** | Non-diagonal metrics completely untested; affects gradient accuracy |
| **SurfaceGeometry Extended** | 7 | **CRITICAL** | Only Quad4 tested; curved surfaces untested |
| **CurveGeometry Extended** | 7 | **CRITICAL** | Only Line2 tested; higher-order curves untested |
| **InverseMapping Robustness** | 10 | High | Many options untested; edge cases missing |
| **GeometryMapping Hessian** | 5 | High | Hessian completely untested; affects curvature computation |
| **GeometryValidator Extended** | 6 | Medium | Condition number accuracy unvalidated |
| **JacobianCache Extended** | 5 | Medium | Memory/performance edge cases |
| **GeometryQuadrature Extended** | 4 | Medium | Boundary integration coverage |
| **PushForward Extended** | 6 | Medium | Element type coverage incomplete |
| **MappingFactory Extended** | 5 | Medium | Edge cases and validation |
| **LinearMapping Extended** | 4 | Lower | Simplex variant coverage |
| **Sub/SuperparametricMapping** | 3 | Lower | Minimal current testing |
| **Thread Safety** | 3 | Lower | Parallel execution |
| **Negative/Error Cases** | 5 | Lower | Robustness |
| **Subtotal** | **77** | | |

### Overall Summary

| Metric | Count |
|--------|-------|
| Existing Tests | ~82 |
| Missing Tests | 77 |
| **Grand Total** | ~159 |
| Current Coverage | ~52% |
| Target Coverage | 90%+ |

### Priority Ranking

1. **CRITICAL**: MetricTensor Extended (7 tests) - Non-diagonal metrics affect all gradient computations
2. **CRITICAL**: SurfaceGeometry Extended (7 tests) - Curved boundary integrals broken without this
3. **CRITICAL**: CurveGeometry Extended (7 tests) - Line integrals on curved edges broken
4. **HIGH**: InverseMapping Robustness (10 tests) - Newton solver edge cases
5. **HIGH**: GeometryMapping Hessian (5 tests) - Second derivatives completely untested
6. **MEDIUM**: GeometryValidator Extended (6 tests) - Quality metrics accuracy
7. **MEDIUM**: JacobianCache Extended (5 tests) - Memory management
8. **MEDIUM**: GeometryQuadrature Extended (4 tests) - Boundary integration
9. **MEDIUM**: PushForward Extended (6 tests) - Element type coverage
10. **MEDIUM**: MappingFactory Extended (5 tests) - Validation completeness
11. **LOWER**: LinearMapping Extended (4 tests) - Simplex variants
12. **LOWER**: Sub/SuperparametricMapping (3 tests) - Minimal testing
13. **LOWER**: Thread Safety (3 tests) - Parallel support
14. **LOWER**: Negative/Error Cases (5 tests) - Robustness

---

## References

### Core Geometry and Mapping References
1. Hughes, T.J.R. "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis" - Isoparametric mapping theory, Jacobian conditions
2. Zienkiewicz, O.C., Taylor, R.L., Zhu, J.Z. "The Finite Element Method: Its Basis and Fundamentals" 7th ed. - Element geometry, coordinate transformations
3. Ern, A., Guermond, J.L. "Theory and Practice of Finite Elements" - Piola transforms, H(div)/H(curl) mappings

### Metric Tensors and Differential Geometry
4. do Carmo, M.P. "Differential Geometry of Curves and Surfaces" - Metric tensor theory, curvature
5. Kreyszig, E. "Differential Geometry" - Covariant/contravariant metric tensors
6. Lovelock, D., Rund, H. "Tensors, Differential Forms, and Variational Principles" - Tensor calculus on manifolds

### Inverse Mapping and Newton Methods
7. Kelley, C.T. "Iterative Methods for Linear and Nonlinear Equations" - Newton's method, line search
8. Nocedal, J., Wright, S.J. "Numerical Optimization" 2nd ed. - Backtracking line search, Armijo condition
9. Dennis, J.E., Schnabel, R.B. "Numerical Methods for Unconstrained Optimization and Nonlinear Equations" - Newton convergence theory

### Element Quality Metrics
10. Knupp, P.M. "Algebraic Mesh Quality Metrics" - Jacobian-based quality measures
11. Shewchuk, J.R. "What Is a Good Linear Finite Element?" - Element quality theory
12. Field, D.A. "Qualitative Measures for Initial Meshes" - Condition number metrics

### Higher-Order Elements
13. Szabo, B., Babuska, I. "Finite Element Analysis" - p-FEM geometry representation
14. Karniadakis, G., Sherwin, S. "Spectral/hp Element Methods for CFD" 2nd ed. - Curved element mappings

### Special Element Types
15. Bedrosian, G. "Shape Functions and Integration Formulas for Three-Dimensional Finite Element Analysis" - Pyramid element geometry
16. Bergot, M., Cohen, G., Durufle, M. "Higher-Order Finite Elements for Hybrid Meshes Using New Nodal Pyramidal Elements" (2010) - Pyramid14 rational basis

---

## Implementation Notes

### Testing Non-Diagonal Metrics

When testing skewed elements, create parallelogram-shaped quads:

```cpp
// Parallelogram quad: vertices at (0,0), (2,0), (3,1), (1,1)
// This creates J with non-zero off-diagonal terms
std::vector<math::Vector<Real,3>> skewed_quad = {
    {Real(0), Real(0), Real(0)},   // (-1,-1)
    {Real(2), Real(0), Real(0)},   // (1,-1)
    {Real(3), Real(1), Real(0)},   // (1,1)
    {Real(1), Real(1), Real(0)}    // (-1,1)
};
// Expected metric: G = J^T J has non-zero G(0,1) = G(1,0)
```

### Testing Curved Surface Geometry

Create curved surfaces by displacing mid-edge/face nodes:

```cpp
// Quad9 with domed center: vertices planar, center displaced
auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad9, 2);
auto nodes = basis->nodes();
// Displace center node (index 8) in z-direction
nodes[8][2] = Real(0.5);  // Creates dome shape
```

### Testing Line Search Activation

Create difficult mappings that require line search:

```cpp
// Highly distorted quad where Newton overshoots
std::vector<math::Vector<Real,3>> difficult_quad = {
    {Real(0), Real(0), Real(0)},
    {Real(1), Real(0), Real(0)},
    {Real(0.9), Real(0.1), Real(0)},  // Nearly collapsed
    {Real(0.1), Real(0.1), Real(0)}
};
// Standard Newton may overshoot; line search should help
```

### Testing Frame Completion Edge Cases

Test curves aligned with each coordinate axis:

```cpp
// Z-axis aligned curve
std::vector<math::Vector<Real,3>> z_curve = {
    {Real(0), Real(0), Real(0)},
    {Real(0), Real(0), Real(1)}
};
// Frame completion must avoid cross product with z-axis
```
