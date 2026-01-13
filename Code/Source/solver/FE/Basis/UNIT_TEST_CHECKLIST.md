# Unit Test Checklist for FE Basis Library

This document lists additional unit tests that should be added to improve coverage and ensure correctness against established finite element literature. Tests are organized by priority and category.

**Last Updated:** 2026-01-13 (Added BSplineBasis local support unit coverage)
**Status Legend:**
- [ ] Not started
- [~] In progress
- [x] Completed

---

## High Priority Tests

### 1. Polynomial Reproduction Tests

These tests verify the fundamental approximation property: a degree-p basis should exactly reproduce polynomials up to degree p.

#### 1.1 `LagrangeBasis_LineInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Line2 exactly interpolates polynomials x^k for k = 0, 1, ..., p
- **Why needed**: Polynomial reproduction is the defining property that guarantees optimal convergence rates (Bramble-Hilbert lemma). Currently only Pyramid14 has this test.
- **Implementation**: Set nodal values to f(x_i) for f(x) = x^k, verify interpolant matches f at random interior points

#### 1.2 `LagrangeBasis_QuadInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Quad4 exactly interpolates tensor-product polynomials x^i * y^j for i,j <= p
- **Why needed**: Ensures the tensor-product construction preserves polynomial reproduction
- **Implementation**: Test monomials 1, x, y, xy, x^2, y^2, x^2*y, xy^2, x^2*y^2 for p=2

#### 1.3 `LagrangeBasis_TriangleInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Triangle3 exactly interpolates polynomials in P_p (complete polynomial space)
- **Why needed**: Simplex elements use different polynomial spaces than tensor-product elements
- **Implementation**: For p=2, test 1, x, y, x^2, xy, y^2 (6 basis functions)

#### 1.4 `LagrangeBasis_HexInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Hex8 exactly interpolates Q_p polynomials
- **Why needed**: 3D tensor-product verification
- **Implementation**: Test all monomials x^i * y^j * z^k for i,j,k <= p

#### 1.5 `LagrangeBasis_TetraInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Tetra4 exactly interpolates P_p polynomials
- **Why needed**: 3D simplex polynomial space verification
- **Implementation**: For p=2, test all 10 monomials in the complete quadratic space

#### 1.6 `LagrangeBasis_WedgeInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Wedge basis interpolates appropriate polynomial space (P_p triangle x Q_p line)
- **Why needed**: Mixed topology requires careful polynomial space definition
- **Implementation**: Test representative monomials from the wedge polynomial space

---

### 2. BDM Basis Comprehensive Tests

The BDM (Brezzi-Douglas-Marini) basis previously had minimal coverage. These tests bring coverage to parity with RT/Nedelec.

#### 2.1 `BDMBasis_TriangleEdgeNormalMomentKronecker`
- [x] Completed
- **What it tests**: BDM edge DOFs satisfy Kronecker property: integral of (v_i . n) * L_j over edge_k = delta_{ij} for edge DOFs
- **Why needed**: DOF definitions are fundamental to H(div) conformity and assembly
- **Implementation**: Integrate v . n weighted by `LagrangeBasis(Line2,k)` over each edge

#### 2.2 `BDMBasis_QuadEdgeNormalMomentKronecker`
- [x] Completed
- **What it tests**: Same as above for Quad elements
- **Why needed**: Tensor-product BDM construction verification

#### 2.3 `BDMBasis_DimensionFormulas`
- [x] Completed
- **What it tests**: dim(BDM_k) = (k+1)(k+2) on triangles, 2(k+1)^2 on quads
- **Why needed**: Correct DOF count is essential for assembly
- **Implementation**: Verify size() matches formula for k = 1, 2, 3

#### 2.4 `BDMBasis_DofAssociations`
- [x] Completed
- **What it tests**: DOF associations correctly identify edge vs interior DOFs
- **Why needed**: Required for global DOF numbering and inter-element continuity
- **Implementation**: Verify entity_type for each DOF matches expected location

#### 2.5 `BDMBasis_PolynomialReproduction`
- [x] Completed
- **What it tests**: BDM_k contains all vector polynomials of degree k
- **Why needed**: Defines approximation capability
- **Implementation**: Project vector polynomial onto BDM space, verify exact reproduction

#### 2.6 `BDMBasis_HexDivergenceAndSize`
- [x] Completed
- **What it tests**: BDM on hexahedra has correct divergence and DOF count
- **Why needed**: 3D BDM is used in mixed methods for Darcy flow
- **Implementation**: Similar to existing RT tests

---

### 3. Boundary and Edge Case Tests

These tests verify correct behavior at element boundaries and for degenerate inputs.

#### 3.1 `LagrangeBasis_EvaluationAtElementCorners`
- [x] Completed
- **What it tests**: Basis functions evaluate correctly at all corner nodes (xi = +/-1 for tensor-product, vertices for simplices)
- **Why needed**: Corner evaluation is critical for assembly and often involves special cases in implementation
- **Implementation**: Systematically test all corners for all element types

#### 3.2 `LagrangeBasis_EvaluationAtEdgeMidpoints`
- [x] Completed
- **What it tests**: Basis functions evaluate correctly at edge midpoints
- **Why needed**: Edge midpoints are quadratic nodes and face special handling
- **Implementation**: For quadratic elements, verify edge node basis functions

#### 3.3 `VectorBasis_EvaluationAtFaceCenters`
- [x] Completed
- **What it tests**: RT/Nedelec/BDM evaluate correctly at face centers
- **Why needed**: Face centers are important for flux evaluation
- **Implementation**: Evaluate at face centroids, verify against analytical expressions

#### 3.4 `LagrangeBasis_GradientAtElementBoundary`
- [x] Completed
- **What it tests**: Gradients are well-defined and continuous as xi approaches +/-1
- **Why needed**: Numerical issues can arise at element boundaries
- **Implementation**: Evaluate gradients at xi = 1-epsilon for small epsilon

#### 3.5 `PyramidBasis_EvaluationNearApex`
- [x] Completed
- **What it tests**: Pyramid basis functions remain well-behaved near the apex (z -> 1)
- **Why needed**: Pyramid bases involve rational functions that can be singular at apex
- **Implementation**: Evaluate at z = 0.99, 0.999, verify values remain bounded

---

### 4. Literature Validation Tests

These tests compare computed values against published reference values.

#### 4.1 `LagrangeBasis_QuadraticTriangleMatchesZienkiewicz`
- [x] Completed
- **What it tests**: Shape functions match values in Zienkiewicz-Taylor Table 6.2
- **Why needed**: External validation against authoritative FE textbook
- **Implementation**: Evaluate N_i at specific points, compare to tabulated values

#### 4.2 `LagrangeBasis_LinearHexMatchesHughes`
- [x] Completed
- **What it tests**: Trilinear hex shape functions match Hughes "The Finite Element Method" formulas
- **Why needed**: Standard reference for isoparametric elements
- **Implementation**: Verify N_i = (1/8)(1 + xi_i*xi)(1 + eta_i*eta)(1 + zeta_i*zeta)

#### 4.3 `RaviartThomasBasis_RT0TriangleMatchesBoffiBrezziFortin`
- [x] Completed
- **What it tests**: RT0 triangle basis matches formulas in "Mixed Finite Element Methods and Applications"
- **Why needed**: Authoritative reference for mixed methods
- **Implementation**: Verify v_i = (x - x_i) / (2 * area) for opposite vertex formulation

#### 4.4 `NedelecBasis_ND0TetraMatchesMonk`
- [x] Completed
- **What it tests**: Lowest-order Nedelec tetrahedron matches Monk "Finite Element Methods for Maxwell's Equations"
- **Why needed**: Standard reference for H(curl) elements
- **Implementation**: Compare edge basis functions against published formulas

---

## Medium Priority Tests

### 5. High-Order Polynomial Tests (p > 4)

#### 5.1 `LagrangeBasis_Order5And6KroneckerProperty`
- [x] Completed
- **What it tests**: Kronecker property holds for p = 5, 6
- **Why needed**: High-order methods are increasingly used; tests catch potential overflow/precision issues
- **Implementation**: Same as existing Kronecker tests but for higher orders

#### 5.2 `LagrangeBasis_Order5And6PartitionOfUnity`
- [x] Completed
- **What it tests**: Partition of unity holds for p = 5, 6
- **Why needed**: Higher-order bases involve larger coefficient magnitudes
- **Implementation**: Same as existing tests for higher orders

#### 5.3 `LagrangeBasis_Order5And6GradientSumZero`
- [x] Completed
- **What it tests**: Gradient sum property holds for p = 5, 6
- **Why needed**: Gradient computation involves derivative of basis coefficients
- **Implementation**: Same as existing tests for higher orders

#### 5.4 `HierarchicalBasis_HighOrderConditionNumber`
- [x] Completed
- **What it tests**: Gram matrix condition number growth rate for p = 2, 3, ..., 8
- **Why needed**: Hierarchical bases should have better conditioning than nodal; verify this
- **Implementation**: Compute Gram matrix, estimate condition number, verify sub-exponential growth

---

### 6. SpectralBasis Expanded Tests

#### 6.1 `SpectralBasis_QuadGradientsMatchFiniteDifference`
- [x] Completed
- **What it tests**: 2D spectral basis gradients match numerical derivatives
- **Why needed**: Only 1D gradient test exists currently
- **Implementation**: Central difference comparison for both partial derivatives

#### 6.2 `SpectralBasis_HexGradientsMatchFiniteDifference`
- [x] Completed
- **What it tests**: 3D spectral basis gradients match numerical derivatives
- **Why needed**: 3D gradient computation verification
- **Implementation**: Central difference comparison for all three partials

#### 6.3 `SpectralBasis_HessiansMatchFiniteDifference`
- [x] Completed
- **What it tests**: Spectral basis Hessians (if implemented) match numerical second derivatives
- **Why needed**: Second derivatives needed for some formulations
- **Implementation**: Central difference on gradients

#### 6.4 `SpectralBasis_GLLNodesMatchLiterature`
- [x] Completed
- **What it tests**: GLL node locations match published tables (e.g., Canuto et al.)
- **Why needed**: GLL nodes are roots of (1-x^2)P'_n(x); verify numerical accuracy
- **Implementation**: Compare against tabulated values for n = 2, 3, 4, 5

#### 6.5 `SpectralBasis_GradientSumZero`
- [x] Completed
- **What it tests**: Sum of gradients equals zero (partition of unity derivative)
- **Why needed**: Verify spectral basis satisfies same property as Lagrange
- **Implementation**: Same as Lagrange gradient sum test

---

### 7. Serendipity Basis Geometry Mode Tests

#### 7.1 `SerendipityBasis_GeometryModeHex20`
- [x] Completed
- **What it tests**: geometry_mode=true produces correct reduced basis for geometry mapping
- **Why needed**: Isoparametric elements often use different basis for geometry vs field
- **Implementation**: Verify size and properties match geometry representation needs

#### 7.2 `SerendipityBasis_GeometryModeWedge15`
- [x] Completed
- **What it tests**: Same as above for wedge elements
- **Why needed**: Mixed topology geometry representation

#### 7.3 `SerendipityBasis_GeometryModePyramid13`
- [x] Completed
- **What it tests**: Same as above for pyramid elements
- **Why needed**: Pyramid geometry representation

---

### 8. Bernstein Basis Expanded Tests

#### 8.1 `BernsteinBasis_NonNegativity`
- [x] Completed
- **What it tests**: All Bernstein basis functions are >= 0 everywhere in reference element
- **Why needed**: Non-negativity is a defining property of Bernstein polynomials
- **Implementation**: Evaluate at random points, verify all values >= 0

#### 8.2 `BernsteinBasis_ConvexHullProperty`
- [x] Completed
- **What it tests**: Interpolant lies within convex hull of control points
- **Why needed**: Key property for geometric modeling and stability
- **Implementation**: Generate random control values, verify interpolant bounded by min/max

#### 8.3 `BernsteinBasis_EndpointInterpolation`
- [x] Completed
- **What it tests**: B_0(0. = 1, B_n(1) = 1 (endpoint basis functions)
- **Why needed**: Bernstein polynomials interpolate at endpoints
- **Implementation**: Verify first and last basis functions equal 1 at respective endpoints

#### 8.4 `BernsteinBasis_GradientsMatchNumerical`
- [x] Completed
- **What it tests**: Analytical gradients match finite differences
- **Why needed**: Gradient tests missing for Bernstein (only partition of unity tested)
- **Implementation**: Central difference comparison

#### 8.5 `BernsteinBasis_HessiansMatchNumerical`
- [x] Completed
- **What it tests**: Analytical Hessians match finite differences
- **Why needed**: Second derivatives for Bernstein not currently tested
- **Implementation**: Central difference on gradients

---

### 9. BatchEvaluator Expanded Tests

#### 9.1 `BatchEvaluator_HessianBatchMatchesPointwise`
- [x] Completed
- **What it tests**: Batched Hessian evaluation matches point-by-point evaluation
- **Why needed**: Hessian batch operations not currently tested
- **Implementation**: Compare batched vs loop over quadrature points

#### 9.2 `BatchEvaluator_LargeQuadratureRule`
- [x] Completed
- **What it tests**: Batch evaluation works correctly for high-order quadrature (many points)
- **Why needed**: Memory allocation and indexing edge cases
- **Implementation**: Use order 10+ quadrature rule

#### 9.3 `BatchEvaluator_SmallQuadratureRule`
- [x] Completed
- **What it tests**: Batch evaluation works correctly for minimal quadrature (1-2 points)
- **Why needed**: Edge case for small problems
- **Implementation**: Use order 1 quadrature rule

#### 9.4 `BatchEvaluator_HighOrderBasis`
- [x] Completed
- **What it tests**: Batch evaluation works for p = 4, 5 bases
- **Why needed**: Many basis functions per point tests memory layout
- **Implementation**: Use high-order Lagrange basis

---

### 10. ModalTransform Expanded Tests

#### 10.1 `ModalTransform_ConditionNumberVsOrder`
- [x] Completed
- **What it tests**: Condition number growth rate as polynomial order increases
- **Why needed**: Numerical stability for high-order hp-FEM
- **Implementation**: Compute condition numbers for p = 1, 2, ..., 8, verify growth rate

#### 10.2 `ModalTransform_TriangleRoundTrip`
- [x] Completed
- **What it tests**: Modal <-> nodal transform round-trips correctly on triangles
- **Why needed**: Simplex transforms may have different numerical properties
- **Implementation**: Same as line/quad tests but for triangles

#### 10.3 `ModalTransform_HexRoundTrip`
- [x] Completed
- **What it tests**: Modal <-> nodal transform round-trips correctly on hexahedra
- **Why needed**: 3D tensor-product verification
- **Implementation**: Same as line/quad tests but for hexahedra

---

## Lower Priority Tests

### 11. Thread Safety and Concurrency Tests

#### 11.1 `BasisCache_ConcurrentDifferentBases`
- [x] Completed
- **What it tests**: Multiple threads requesting different basis types simultaneously
- **Why needed**: Real applications evaluate different bases concurrently
- **Implementation**: Spawn threads requesting Lagrange, Hierarchical, Spectral simultaneously

#### 11.2 `BasisCache_HighContentionSingleEntry`
- [x] Completed
- **What it tests**: Many threads (32+) requesting same cache entry
- **Why needed**: Stress test mutex implementation
- **Implementation**: Spawn 32 threads all requesting same basis/quadrature combination

#### 11.3 `BasisCache_ConcurrentClearAndAccess`
- [x] Completed
- **What it tests**: Cache clear during concurrent access doesn't cause crashes
- **Why needed**: Cache may be cleared during runtime in some applications
- **Implementation**: One thread clearing cache while others access it

---

### 12. BasisFactory Comprehensive Tests

#### 12.1 `BasisFactory_AllSupportedCombinations`
- [x] Completed
- **What it tests**: Factory successfully creates all documented basis/element/order combinations
- **Why needed**: Ensures factory coverage matches implementation
- **Implementation**: Iterate over all valid combinations, verify creation succeeds

#### 12.2 `BasisFactory_InvalidCombinationsThrow`
- [x] Completed
- **What it tests**: Factory throws appropriate exceptions for unsupported combinations
- **Why needed**: Clear error messages for users
- **Implementation**: Test known invalid combinations (e.g., Serendipity on Tetra)

#### 12.3 `BasisFactory_ContinuitySelection`
- [x] Completed
- **What it tests**: Factory selects correct basis type based on continuity requirement
- **Why needed**: C0 vs C1 vs H(div) vs H(curl) logic verification
- **Implementation**: Verify Hermite selected for C1, RT for H(div), Nedelec for H(curl)

---

### 13. TensorBasis Tests

#### 13.1 `TensorBasis_AnisotropicHessiansMatchNumerical`
- [x] Completed
- **What it tests**: Hessians for anisotropic tensor-product basis match finite differences
- **Why needed**: Anisotropic orders tested for values/gradients but not Hessians
- **Implementation**: Central difference on gradients for px != py case

#### 13.2 `TensorBasis_3DAnisotropicConstruction`
- [x] Completed
- **What it tests**: 3D tensor product with different orders in each direction
- **Why needed**: hp-FEM often uses anisotropic refinement
- **Implementation**: Construct with px, py, pz different, verify size and properties

---

### 14. Numerical Stability Tests

#### 14.1 `LagrangeBasis_NumericalPrecisionHighOrder`
- [x] Completed
- **What it tests**: Basis values remain accurate (not polluted by round-off) for p = 6, 7, 8
- **Why needed**: High-order Lagrange on equispaced nodes is ill-conditioned
- **Implementation**: Verify partition of unity holds to reasonable tolerance

#### 14.2 `HierarchicalBasis_StabilityHighOrder`
- [x] Completed
- **What it tests**: Hierarchical basis remains well-conditioned for p = 6, 7, 8
- **Why needed**: Modal bases should be more stable; verify this
- **Implementation**: Compare condition numbers vs Lagrange

#### 14.3 `SpectralBasis_StabilityHighOrder`
- [x] Completed
- **What it tests**: GLL spectral basis stability for p = 8, 10, 12
- **Why needed**: Spectral methods often use high order
- **Implementation**: Verify Kronecker property still holds accurately

---

### 15. Quadrature Integration Tests

#### 15.1 `LagrangeBasis_MassMatrixSymmetry`
- [x] Completed
- **What it tests**: Numerically integrated mass matrix is symmetric
- **Why needed**: Symmetry is essential for many solvers
- **Implementation**: Compute M_ij = integral(N_i * N_j), verify M = M^T

#### 15.2 `LagrangeBasis_MassMatrixPositiveDefinite`
- [x] Completed
- **What it tests**: Mass matrix is positive definite
- **Why needed**: Ensures well-posed problems
- **Implementation**: Compute mass matrix, verify all eigenvalues positive

#### 15.3 `LagrangeBasis_StiffnessMatrixSymmetry`
- [x] Completed
- **What it tests**: Stiffness matrix is symmetric for symmetric material tensor
- **Why needed**: Fundamental property for elliptic problems
- **Implementation**: Compute K_ij = integral(grad N_i . D . grad N_j), verify K = K^T

---

## CRITICAL: Missing Tests (Currently No Coverage)

### 16. BSplineBasis Tests (CRITICAL - minimal existing coverage)

The BSplineBasis implementation (172 lines) previously had no unit tests. While there is now basic coverage, this category remains critical because B-splines are the foundation for IsoGeometric Analysis (IGA) workflows. Additional tests are still needed to validate the Cox-de Boor recursion, knot vector handling, and derivative computation.

#### 16.1 `BSplineBasis_PartitionOfUnity`
- [x] Completed
- **What it tests**: Sum of all B-spline basis functions equals 1 at any point in the parameter domain
- **Why needed**: Partition of unity is a fundamental property of B-splines required for geometric exactness in IGA. Without this property, NURBS curves/surfaces would not reproduce control point positions correctly.
- **Implementation**: For various knot vectors and polynomial degrees (p=1,2,3,4), evaluate all basis functions at random interior points and verify sum equals 1 to machine precision.

#### 16.2 `BSplineBasis_LocalSupport`
- [x] Completed
- **What it tests**: Each B-spline basis function N_{i,p} is non-zero only on [t_i, t_{i+p+1})
- **Why needed**: Local support is essential for sparse matrix assembly in IGA. If basis functions have incorrect support, stiffness matrices will have wrong sparsity patterns and incorrect entries.
- **Implementation**: Evaluate N_{i,p} at points outside its knot span, verify values are exactly zero.

#### 16.3 `BSplineBasis_NonNegativity`
- [ ] Not started
- **What it tests**: All B-spline basis functions are >= 0 everywhere in their support
- **Why needed**: Non-negativity is required for convex hull property and variation diminishing property. Negative values would indicate incorrect Cox-de Boor implementation.
- **Implementation**: Evaluate at many random points within each knot span, verify all values are non-negative.

#### 16.4 `BSplineBasis_CoxDeBoorRecursion`
- [ ] Not started
- **What it tests**: Basis values match analytical formulas for known cases (linear, quadratic B-splines on uniform knots)
- **Why needed**: Direct validation of the Cox-de Boor recursion implementation against closed-form expressions from de Boor's "A Practical Guide to Splines".
- **Implementation**: For uniform knot vector and p=1,2, compare computed values against analytical expressions at specific points.

#### 16.5 `BSplineBasis_DerivativesMatchFiniteDifference`
- [ ] Not started
- **What it tests**: Analytical first derivatives match central finite differences
- **Why needed**: Derivatives are needed for gradient computation in IGA. Incorrect derivatives would cause wrong stiffness matrices and non-convergent solutions.
- **Implementation**: Compute analytical derivative, compare to (N(x+h) - N(x-h))/(2h) for small h.

#### 16.6 `BSplineBasis_ClampedKnotVector`
- [ ] Not started
- **What it tests**: Clamped (open) knot vectors produce interpolating basis at endpoints
- **Why needed**: Clamped knot vectors are standard in CAD/IGA where curves must pass through first and last control points. This is achieved by repeating end knots p+1 times.
- **Implementation**: For clamped knot vector, verify N_0(t_0) = 1 and N_n(t_m) = 1 where t_0, t_m are domain endpoints.

#### 16.7 `BSplineBasis_RepeatedKnots`
- [ ] Not started
- **What it tests**: Repeated interior knots reduce continuity correctly (multiplicity m reduces continuity to C^{p-m})
- **Why needed**: Knot repetition is used to introduce C^0 or C^1 discontinuities at specific points (e.g., for sharp corners). Incorrect handling breaks geometric fidelity.
- **Implementation**: Create knot vector with repeated interior knot, verify basis function derivatives have reduced continuity at that knot.

#### 16.8 `BSplineBasis_KnotVectorValidation`
- [ ] Not started
- **What it tests**: Constructor throws for invalid knot vectors (non-monotonic, insufficient knots for degree)
- **Why needed**: Early error detection prevents cryptic downstream failures. Knot vector must be non-decreasing and have at least n+p+2 knots for n+1 basis functions of degree p.
- **Implementation**: Pass invalid knot vectors (decreasing, too short), verify appropriate exceptions are thrown.

#### 16.9 `BSplineBasis_HighDegree`
- [ ] Not started
- **What it tests**: B-spline basis functions remain accurate for high degree (p=5,6,7)
- **Why needed**: High-degree B-splines are used in IGA for smooth solutions. High degree involves many levels of Cox-de Boor recursion that can accumulate numerical error.
- **Implementation**: Verify partition of unity holds to reasonable tolerance for p=5,6,7 on various knot vectors.

#### 16.10 `BSplineBasis_BezierExtraction`
- [ ] Not started
- **What it tests**: B-spline basis on a single Bezier span (no interior knots) matches Bernstein basis
- **Why needed**: This is the limiting case connecting B-splines to Bernstein polynomials. Serves as cross-validation between the two basis implementations.
- **Implementation**: Create B-spline with no interior knots, compare values against BernsteinBasis of same degree.

---

### 17. NodeOrderingConventions Tests (No direct tests)

The NodeOrderingConventions utility (399 lines) provides VTK-compatible node ordering for all element types but has no dedicated tests. It is only tested indirectly through LagrangeBasis tests. Direct tests are needed to ensure node coordinates and permutations are correct for all element types.

#### 17.1 `NodeOrderingConventions_NumNodesFormulas`
- [ ] Not started
- **What it tests**: num_nodes() returns correct count for all element types and orders
- **Why needed**: Incorrect node counts cause buffer overflows or incomplete data during assembly. Formulas differ between tensor-product and simplex elements.
- **Implementation**: Verify num_nodes(element_type, order) matches known formulas: Line: p+1, Tri: (p+1)(p+2)/2, Quad: (p+1)^2, Tet: (p+1)(p+2)(p+3)/6, Hex: (p+1)^3, etc.

#### 17.2 `NodeOrderingConventions_VTKCompatibility`
- [ ] Not started
- **What it tests**: Node orderings match VTK conventions for Lagrange elements
- **Why needed**: Interoperability with VTK-based visualization (ParaView). Incorrect ordering causes garbled visualizations and wrong interpolation.
- **Implementation**: For each element type, compare node ordering against VTK documentation (Lagrange{Tri,Quad,Hex,Wedge,Pyramid,Tet} orderings).

#### 17.3 `NodeOrderingConventions_VertexNodesFirst`
- [ ] Not started
- **What it tests**: Vertex nodes always come first in the ordering (indices 0 to num_vertices-1)
- **Why needed**: Many algorithms assume vertex nodes are at the beginning. This is the standard convention for Lagrange elements.
- **Implementation**: For all element types, verify get_node_coords() returns vertex positions for first num_vertices indices.

#### 17.4 `NodeOrderingConventions_EdgeNodeOrdering`
- [ ] Not started
- **What it tests**: Edge nodes follow vertices in correct edge order
- **Why needed**: Edge ordering affects inter-element connectivity in DG and CG methods. Incorrect edge ordering breaks conforming assembly.
- **Implementation**: For quadratic+ elements, verify nodes on each edge appear in expected sequence after vertices.

#### 17.5 `NodeOrderingConventions_FaceNodeOrdering`
- [ ] Not started
- **What it tests**: Face-interior nodes appear after edge nodes in correct face order
- **Why needed**: Face DOFs matter for H(div)/H(curl) element assembly and DG methods.
- **Implementation**: For cubic+ 3D elements, verify face-interior nodes appear in expected sequence.

#### 17.6 `NodeOrderingConventions_ReferenceCoordinatesValid`
- [ ] Not started
- **What it tests**: All returned reference coordinates lie within the reference element
- **Why needed**: Out-of-bounds reference coordinates indicate incorrect node placement formulas.
- **Implementation**: For all element types and orders p=1..4, verify all node coordinates satisfy element containment constraints.

#### 17.7 `NodeOrderingConventions_PermutationInverse`
- [ ] Not started
- **What it tests**: Permutation functions satisfy P(P^{-1}(i)) = i for all valid indices
- **Why needed**: Permutations are used for node reordering between conventions. Non-invertible permutations cause data corruption.
- **Implementation**: For all element types, compose permutation with inverse, verify identity.

---

### 18. HermiteBasis Extended Tests

Current HermiteBasis tests only cover cubic (order 3) on Line and Quad elements. Higher-order Hermite bases and 3D extensions need testing when implemented.

#### 18.1 `HermiteBasis_QuinticLine`
- [ ] Not started
- **What it tests**: Quintic (order 5) Hermite basis on Line element with function, first, and second derivative DOFs
- **Why needed**: Quintic Hermite is used in beam/plate formulations requiring C^2 continuity. Higher-order Hermite bases extend the C^1 framework.
- **Implementation**: Verify DOF interpretation (f, f', f'' at each node), partition of unity for value modes, derivative consistency.

#### 18.2 `HermiteBasis_SepticLine`
- [ ] Not started
- **What it tests**: Septic (order 7) Hermite basis on Line element with DOFs through third derivatives
- **Why needed**: Very high-order Hermite bases are used in some beam formulations and spectral methods.
- **Implementation**: Same as quintic but with additional derivative DOFs.

#### 18.3 `HermiteBasis_ContinuityAcrossElements`
- [ ] Not started
- **What it tests**: Two adjacent elements share C^1 continuity at their common node
- **Why needed**: The entire purpose of Hermite bases is to achieve inter-element smoothness. This must be verified in a multi-element context.
- **Implementation**: Create two adjacent line elements, set DOFs for a smooth function, verify C^1 continuity at interface.

#### 18.4 `HermiteBasis_MixedDerivativeDOFs`
- [ ] Not started
- **What it tests**: 2D bicubic Hermite correctly handles mixed derivative DOF (d^2f/dxdy at corner nodes)
- **Why needed**: Mixed derivatives are the cross-term DOFs in tensor-product Hermite bases. Incorrect handling breaks C^1 continuity in 2D.
- **Implementation**: Set mixed derivative DOF, verify correct basis function contribution.

#### 18.5 `HermiteBasis_3DHex`
- [ ] Not started (pending implementation)
- **What it tests**: Tricubic Hermite on Hex element with 8 DOFs per corner (f and all first derivatives)
- **Why needed**: 3D Hermite bases are used in smooth geometry representation and high-order plate/shell elements.
- **Implementation**: Verify 64 DOFs total, correct DOF interpretation at each corner.

---

### 19. OrthogonalPolynomials Extended Tests

Current tests cover basic orthogonality and derivatives for low orders. Additional tests are needed for comprehensive validation.

#### 19.1 `OrthogonalPolynomials_LegendreThreeTermRecurrence`
- [ ] Not started
- **What it tests**: Legendre polynomials satisfy the three-term recurrence: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
- **Why needed**: The recurrence relation is typically used for efficient evaluation. Verifying it holds validates the implementation.
- **Implementation**: For n=2..10, verify recurrence at random points.

#### 19.2 `OrthogonalPolynomials_JacobiParameterRanges`
- [ ] Not started
- **What it tests**: Jacobi polynomials P_n^{(α,β)} work correctly for various (α,β) combinations
- **Why needed**: Different (α,β) values are used in different contexts: (0,0) = Legendre, (-1/2,-1/2) = Chebyshev, (1,1) = ultraspherical. All must work.
- **Implementation**: Test α,β ∈ {-0.5, 0, 0.5, 1, 2} combinations, verify orthogonality.

#### 19.3 `OrthogonalPolynomials_HighOrderDerivatives`
- [ ] Not started
- **What it tests**: Second and third derivatives of Legendre/Jacobi polynomials match analytical formulas
- **Why needed**: Higher derivatives are needed for Hessian computation and some formulations. Currently only first derivatives tested.
- **Implementation**: Compare analytical higher derivatives against finite differences on first derivative.

#### 19.4 `OrthogonalPolynomials_DubinerSimplexOrthogonality`
- [ ] Not started
- **What it tests**: Dubiner basis functions are orthogonal over the reference triangle with appropriate weight
- **Why needed**: Dubiner basis is the simplex analog of tensor-product Legendre. Orthogonality enables efficient modal methods.
- **Implementation**: Numerically integrate product of different Dubiner modes over triangle, verify orthogonality.

#### 19.5 `OrthogonalPolynomials_ProriolTetraOrthogonality`
- [ ] Not started
- **What it tests**: Proriol (3D Dubiner) basis functions are orthogonal over reference tetrahedron
- **Why needed**: 3D modal methods on tetrahedra use Proriol polynomials. Orthogonality is essential for conditioning.
- **Implementation**: Numerically integrate product of different Proriol modes over tetrahedron.

#### 19.6 `OrthogonalPolynomials_IntegratedLegendreValues`
- [ ] Not started
- **What it tests**: Integrated Legendre polynomials L_n(x) = ∫P_{n-1}(t)dt satisfy known properties
- **Why needed**: Integrated Legendre polynomials are used in hierarchical bases. They must satisfy L_n(±1) = 0 for bubble functions.
- **Implementation**: Verify L_n(-1) = L_n(1) = 0 for n >= 2, verify derivative equals P_{n-1}.

---

### 20. BDMBasis 3D Extended Tests

BDM implementation exists for Tetra and Hex but tests only cover 2D elements. 3D BDM is critical for mixed methods in 3D Darcy flow, elasticity, and Stokes problems.

#### 20.1 `BDMBasis_TetraDivergenceLinear`
- [ ] Not started
- **What it tests**: BDM_1 on tetrahedron has correct divergence (polynomial in P_0)
- **Why needed**: Divergence correctness is essential for mass conservation in mixed methods. 3D BDM currently untested.
- **Implementation**: Evaluate divergence at random interior points, verify it matches expected polynomial.

#### 20.2 `BDMBasis_TetraDimensionFormula`
- [ ] Not started
- **What it tests**: dim(BDM_k) on tetrahedron equals k(k+1)(k+3)/2 for k >= 1
- **Why needed**: Correct DOF count is essential for assembly and solver setup.
- **Implementation**: Verify size() matches formula for k = 1, 2, 3.

#### 20.3 `BDMBasis_TetraDofAssociations`
- [ ] Not started
- **What it tests**: DOF associations correctly identify face vs interior DOFs on tetrahedron
- **Why needed**: Face DOFs must match between adjacent elements for H(div) conformity.
- **Implementation**: Verify entity_type for each DOF; k=1 has only face DOFs, k>=2 has interior DOFs.

#### 20.4 `BDMBasis_TetraFaceNormalMomentKronecker`
- [ ] Not started
- **What it tests**: Face DOFs satisfy Kronecker property: integral of (v_i · n) L_j over face_k = δ_{ij} for face k DOFs
- **Why needed**: This is the defining DOF property for H(div) elements. Incorrect implementation breaks conforming assembly.
- **Implementation**: Numerically integrate v · n weighted by face Lagrange basis over each face.

#### 20.5 `BDMBasis_HexDivergenceHighOrder`
- [ ] Not started
- **What it tests**: BDM_k (k=2,3) on hexahedron has correct divergence
- **Why needed**: Higher-order BDM on hex is used in high-accuracy mixed methods but currently not tested for k > 1.
- **Implementation**: Verify divergence polynomial degree matches expected (k-1).

#### 20.6 `BDMBasis_HexFaceNormalMomentKronecker`
- [ ] Not started
- **What it tests**: Face DOFs on hex satisfy Kronecker property
- **Why needed**: Same reasoning as tetra - essential for H(div) conformity.
- **Implementation**: Numerically integrate v · n weighted by face Lagrange basis over each hex face.

#### 20.7 `BDMBasis_ComparisonWithRT`
- [ ] Not started
- **What it tests**: BDM and RT spaces have correct containment relationship: RT_k ⊂ BDM_{k+1}
- **Why needed**: Mathematical verification that the implemented spaces satisfy the theoretical relationship.
- **Implementation**: For vector polynomial in RT_k, verify it can be exactly represented in BDM_{k+1}.

---

### 21. Negative and Error Case Tests

Current tests focus on correct behavior; few tests verify error handling for invalid inputs.

#### 21.1 `LagrangeBasis_InvalidOrderThrows`
- [ ] Not started
- **What it tests**: Constructor throws for order < 0 or order > max_supported_order
- **Why needed**: Clear error messages help users identify configuration mistakes quickly.
- **Implementation**: Attempt construction with order = -1, order = 100, verify exceptions with informative messages.

#### 21.2 `LagrangeBasis_InvalidElementTypeThrows`
- [ ] Not started
- **What it tests**: Constructor throws for invalid element type enum values
- **Why needed**: Corrupted input should fail early, not produce garbage output.
- **Implementation**: Pass invalid element type, verify exception.

#### 21.3 `VectorBasis_InvalidOrderForElement`
- [ ] Not started
- **What it tests**: RT/Nedelec/BDM throw for unsupported order on specific element types
- **Why needed**: Not all element/order combinations are implemented; users need clear feedback.
- **Implementation**: Test known unsupported combinations (e.g., BDM order 4 on pyramid).

#### 21.4 `BasisFactory_UnsupportedContinuityThrows`
- [ ] Not started
- **What it tests**: Factory throws for unsupported continuity requirements (e.g., C^2 on standard elements)
- **Why needed**: C^2 continuity requires special elements (Argyris, etc.) not yet implemented.
- **Implementation**: Request C^2 basis, verify informative exception.

#### 21.5 `BatchEvaluator_EmptyQuadratureThrows`
- [ ] Not started
- **What it tests**: BatchEvaluator throws for empty quadrature rule (0 points)
- **Why needed**: Empty quadrature is almost certainly a user error; fail loudly.
- **Implementation**: Pass quadrature with 0 points, verify exception.

#### 21.6 `ModalTransform_SingularBasisThrows`
- [ ] Not started
- **What it tests**: ModalTransform throws or warns for ill-conditioned (near-singular) Vandermonde matrix
- **Why needed**: Very high-order nodal bases have poor conditioning; user should be warned before getting garbage results.
- **Implementation**: Create extremely high-order (p=15+) modal transform, verify warning/exception about conditioning.

#### 21.7 `BSplineBasis_EmptyKnotVectorThrows`
- [ ] Not started
- **What it tests**: Constructor throws for empty or single-element knot vector
- **Why needed**: Knot vector must have at least p+2 elements for a single basis function of degree p.
- **Implementation**: Pass empty vector, single-element vector, verify exceptions.

#### 21.8 `BSplineBasis_DecreasingKnotVectorThrows`
- [ ] Not started
- **What it tests**: Constructor throws for non-monotonic (decreasing) knot vector
- **Why needed**: Knot vector must be non-decreasing by definition.
- **Implementation**: Pass [0, 0.5, 0.3, 1.0], verify exception with message about monotonicity.

---

### 22. ModalTransform Additional Tests

ModalTransform currently has only 4 tests. Additional tests are needed for comprehensive coverage of this numerically sensitive component.

#### 22.1 `ModalTransform_TetraRoundTrip`
- [ ] Not started
- **What it tests**: Modal ↔ nodal transform round-trips correctly on tetrahedra
- **Why needed**: 3D simplex transforms have different structure than tensor-product; need separate validation.
- **Implementation**: Transform random coefficient vector modal→nodal→modal, verify recovery.

#### 22.2 `ModalTransform_WedgeRoundTrip`
- [ ] Not started
- **What it tests**: Round-trip on wedge elements (mixed topology)
- **Why needed**: Wedge combines triangle (simplex) and line (tensor-product) which may have different transform properties.
- **Implementation**: Same as other round-trip tests.

#### 22.3 `ModalTransform_ConditionNumberScaling`
- [ ] Not started
- **What it tests**: Condition number growth rate follows expected pattern (polynomial in p, not exponential)
- **Why needed**: Hierarchical/modal bases should have bounded condition number growth. Exponential growth indicates implementation issues.
- **Implementation**: Compute condition numbers for p=1..10, fit growth curve, verify sub-exponential.

#### 22.4 `ModalTransform_HighOrderStability`
- [ ] Not started
- **What it tests**: Round-trip error remains bounded for high polynomial order (p=8,9,10)
- **Why needed**: Numerical stability at high order is essential for hp-FEM.
- **Implementation**: Compute round-trip error vs order, verify error remains below reasonable threshold.

#### 22.5 `ModalTransform_AllElementTypes`
- [ ] Not started
- **What it tests**: Transform works for all supported element types (Line, Tri, Quad, Tet, Hex, Wedge, Pyramid)
- **Why needed**: Systematic coverage of all element types at moderate order.
- **Implementation**: For each element type, verify round-trip at p=3.

---

### 23. Integration and Cross-Module Tests

These tests verify basis functions work correctly when combined with other FE library components.

#### 23.1 `BasisQuadrature_ExactIntegration`
- [ ] Not started
- **What it tests**: Quadrature rule of sufficient order exactly integrates basis function products
- **Why needed**: This is the foundation of FE assembly. If quadrature doesn't integrate basis products exactly, mass/stiffness matrices will be wrong.
- **Implementation**: For degree p basis, use order 2p quadrature, verify integral(N_i * N_j) matches symbolic result.

#### 23.2 `BasisQuadrature_GradientIntegration`
- [ ] Not started
- **What it tests**: Quadrature exactly integrates gradient products for stiffness matrix
- **Why needed**: Stiffness matrix assembly requires integral of grad(N_i) · grad(N_j).
- **Implementation**: For degree p basis, use order 2(p-1) quadrature, verify integral of gradient products.

#### 23.3 `BasisMixedMethods_RTLagrangeCompatibility`
- [ ] Not started
- **What it tests**: RT basis and Lagrange basis are compatible for mixed methods (inf-sup stable pairing)
- **Why needed**: Mixed methods require compatible velocity-pressure pairs. RT_k / P_k is a standard inf-sup stable pair.
- **Implementation**: Verify RT_k divergence lies in P_k space (same space as Lagrange pressure).

#### 23.4 `BasisMixedMethods_NedelecLagrangeCompatibility`
- [ ] Not started
- **What it tests**: Nedelec and Lagrange are compatible for Maxwell problems
- **Why needed**: H(curl) conforming elements must be paired with appropriate H^1 or L^2 elements.
- **Implementation**: Verify Nedelec curl maps to appropriate discontinuous space.

---

## Checklist Summary

### Existing Tests (Implemented)

| Category | Tests | Priority | Status |
|----------|-------|----------|--------|
| Polynomial Reproduction | 6 | High | ✅ Complete |
| BDM Comprehensive (2D) | 6 | High | ✅ Complete |
| Boundary/Edge Cases | 5 | High | ✅ Complete |
| Literature Validation | 4 | High | ✅ Complete |
| High-Order (p > 4) | 4 | Medium | ✅ Complete |
| SpectralBasis Expanded | 5 | Medium | ✅ Complete |
| Serendipity Geometry Mode | 3 | Medium | ✅ Complete |
| Bernstein Expanded | 5 | Medium | ✅ Complete |
| BatchEvaluator Expanded | 4 | Medium | ✅ Complete |
| ModalTransform Expanded | 3 | Medium | ✅ Complete |
| Thread Safety | 3 | Lower | ✅ Complete |
| BasisFactory Comprehensive | 3 | Lower | ✅ Complete |
| TensorBasis | 2 | Lower | ✅ Complete |
| Numerical Stability | 3 | Lower | ✅ Complete |
| Quadrature Integration | 3 | Lower | ✅ Complete |
| BSplineBasis | 2 | CRITICAL | 🚧 In Progress |
| **Subtotal (Implemented)** | **61** | | |

### Missing Tests (Remaining)

| Category | Tests | Priority | Status | Rationale |
|----------|-------|----------|--------|-----------|
| **BSplineBasis** | 8 | **CRITICAL** | 🚧 In Progress | Foundation for IGA; basic properties tested but major coverage gaps remain |
| **NodeOrderingConventions** | 7 | High | ❌ Not Started | 399 lines with no direct tests; only tested implicitly |
| **HermiteBasis Extended** | 5 | High | ❌ Not Started | Only cubic 1D/2D tested; higher order and 3D missing |
| **OrthogonalPolynomials Extended** | 6 | Medium | ❌ Not Started | Limited order/parameter coverage; 3D orthogonality missing |
| **BDMBasis 3D Extended** | 7 | High | ❌ Not Started | Implementation exists for Tet/Hex but no tests |
| **Negative/Error Cases** | 8 | Medium | ❌ Not Started | Minimal invalid input testing across all bases |
| **ModalTransform Additional** | 5 | Medium | ❌ Not Started | Only 4 existing tests for critical numerical component |
| **Integration/Cross-Module** | 4 | Lower | ❌ Not Started | Basis+Quadrature combined validation |
| **Subtotal (Missing)** | **50** | | | |

### Overall Summary

| Metric | Count |
|--------|-------|
| Total Existing Tests | 61 |
| Total Missing Tests | 50 |
| **Grand Total** | **111** |
| Current Coverage | 55% |

### Priority Ranking for Missing Tests

1. **CRITICAL**: BSplineBasis (8 remaining tests) - IGA foundation needs deeper validation
2. **HIGH**: BDMBasis 3D Extended (7 tests) - 3D mixed methods unusable without tests
3. **HIGH**: NodeOrderingConventions (7 tests) - VTK compatibility unvalidated
4. **HIGH**: HermiteBasis Extended (5 tests) - C^1 continuity incomplete
5. **MEDIUM**: Negative/Error Cases (8 tests) - Robustness gaps
6. **MEDIUM**: OrthogonalPolynomials Extended (6 tests) - Modal method foundations
7. **MEDIUM**: ModalTransform Additional (5 tests) - Numerical stability concerns
8. **LOWER**: Integration/Cross-Module (4 tests) - System-level validation

---

## References

### Core Finite Element References
1. Zienkiewicz, O.C., Taylor, R.L., Zhu, J.Z. "The Finite Element Method: Its Basis and Fundamentals" 7th ed.
2. Hughes, T.J.R. "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis"
3. Ern, A., Guermond, J.L. "Theory and Practice of Finite Elements"
4. Szabo, B., Babuska, I. "Finite Element Analysis"

### Mixed and Vector Finite Elements
5. Boffi, D., Brezzi, F., Fortin, M. "Mixed Finite Element Methods and Applications"
6. Monk, P. "Finite Element Methods for Maxwell's Equations"
7. Nedelec, J.C. "Mixed Finite Elements in R^3" (1980)
8. Brezzi, F., Douglas, J., Marini, L.D. "Two Families of Mixed Finite Elements for Second Order Elliptic Problems" (1985)

### Spectral and High-Order Methods
9. Canuto, C., et al. "Spectral Methods: Fundamentals in Single Domains"
10. Karniadakis, G., Sherwin, S. "Spectral/hp Element Methods for CFD" 2nd ed.

### B-Splines and IsoGeometric Analysis
11. de Boor, C. "A Practical Guide to Splines" - Revised ed. (2001) - Primary reference for B-spline implementation
12. Cottrell, J.A., Hughes, T.J.R., Bazilevs, Y. "Isogeometric Analysis: Toward Integration of CAD and FEA" (2009)
13. Piegl, L., Tiller, W. "The NURBS Book" 2nd ed. (1997)

### Orthogonal Polynomials
14. Abramowitz, M., Stegun, I.A. "Handbook of Mathematical Functions" - Legendre, Jacobi polynomial formulas
15. Dubiner, M. "Spectral Methods on Triangles and Other Domains" (1991) - Simplex orthogonal polynomials
16. Proriol, J. "Sur une Famille de Polynomes à Deux Variables Orthogonaux" (1957) - 3D extension

### Visualization Compatibility
17. VTK File Formats Documentation - https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
18. VTK Lagrange Element Ordering - https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
